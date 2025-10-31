from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc
from tqdm import tqdm

from ..base_stage import BaseStage
from ...models import VGAE, MLPDenoiser, GaussianDiffusion, get_named_beta_schedule, GradualWarmupScheduler

class FilterSamplesStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)
        self.num_nodes = dataset.x.shape[0]

        self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                        hidden_dim=self.config.vae.hidden_sizes[0],
                        latent_dim=self.config.vae.hidden_sizes[1],
                        adj=self.adj).to(self.device)

        self.mlpdenoiser = MLPDenoiser(x_dim=self.config.vae.hidden_sizes[-1],
                                            emb_dim = self.config.diffusion.cdim,
                                            hidden_dim = self.config.diffusion.hidden_dim,
                                            num_classes = self.config.num_classes,
                                            layers = self.config.diffusion.layers,
                                            dtype = torch.float32).to(self.device)
        self.betas = get_named_beta_schedule(schedule_name=self.config.diffusion.schedule_name, num_diffusion_timesteps=self.config.diffusion.T)
        self.diffusion = GaussianDiffusion(dtype = self.config.dtype,
                                    model=self.mlpdenoiser,
                                    betas = self.betas,
                                    w = self.config.diffusion.w,
                                    v = self.config.diffusion.v,
                                    device = self.device,
                                    config = self.config)



    def _get_checkpoints_load_path(self):
        self.checkpoints_load_path1 = os.path.join(self.checkpoints_root, 'checkpoint_vae_train.pth')
        self.checkpoints_load_path2 = os.path.join(self.checkpoints_root, 'checkpoint_diff_train.pth')
        self.checkpoints_load_path3 = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
    
    def _get_checkpoints_save_path(self):
        save_dir = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.checkpoints_save_path = os.path.join(save_dir, 'checkpoint_filter_samples.pth')
        

    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoints_load_path1):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path1}")
        if not os.path.exists(self.checkpoints_load_path2):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path2}")
        if not os.path.exists(self.checkpoints_load_path3):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path3}")
        checkpoint1 = torch.load(self.checkpoints_load_path1, weights_only=True, map_location=self.device)
        self.VGAE.load_state_dict(checkpoint1['vae_stage_dict'])
        checkpoint2 = torch.load(self.checkpoints_load_path2, weights_only=True, map_location=self.device)
        self.diffusion.model.load_state_dict(checkpoint2['diffusion_model_state'])
        checkpoint3 = torch.load(self.checkpoints_load_path3, weights_only=True, map_location=self.device)
        self.latents = checkpoint3['latents'].to(self.device)
        self.labels = checkpoint3['labels'].to(self.device)

    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "filter_samples",
            'generated_samples': self.filtered_samples,
            'generated_labels': self.filtered_labels,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: filter_samples | Checkpoint saved: {self.checkpoints_save_path}")

    def _empty_memory(self):
        if hasattr(self, 'VGAE'):
            del self.VGAE
        if hasattr(self, 'mlpdenoiser'):
            del self.mlpdenoiser
        if hasattr(self, 'betas'):
            del self.betas
        if hasattr(self, 'diffusion'):
            del self.diffusion
        if hasattr(self, 'filtered_samples'):
            del self.filtered_samples
        if hasattr(self, 'filtered_labels'):
            del self.filtered_labels
        gc.collect()
        torch.cuda.empty_cache()

    def _filter_topk(self):
        self.VGAE.eval()
        self.diffusion.model.eval()
        counts = torch.bincount(self.labels)
        align_counts = counts.max() - counts
        align_counts = (align_counts * self.config.diffusion.generate_ratio).ceil().int()   # generate samples controled by ratio
        if self.config.diffusion.generate_ratio == -1:
            align_counts = (counts * 1.0).ceil().int()
        print("align_counts: ",align_counts)

        filtered_samples = []
        filtered_labels = []

        for cls in range(self.config.num_classes):
            if align_counts[cls] == 0:
                continue
            # num of samples for each class is 100 * align_counts[cls]
            count = align_counts[cls] * 100
            per_count = count // 100
            loops_count = (count / per_count).ceil().int().item()
            print("per_count: ", per_count, "count: ", count, "loops_count: ", loops_count)

            sampled_latents_list = []
            reconstructed_latents_list = []
            for _ in tqdm(range(loops_count),desc = f"class {cls}"):
                # sample
                cls_labels = torch.ones(per_count, dtype = torch.long) * cls
                cls_labels_one_hot = torch.nn.functional.one_hot(cls_labels, num_classes=self.config.num_classes).float().to(self.device)
                genshape = (per_count, self.latents.shape[-1])
                samples = self.diffusion.sample(genshape, cemb=cls_labels_one_hot)
                sampled_latents_list.append(samples)

                # reconstruct
                aug_latents = torch.cat([self.latents, samples], dim=0)
                re_feats, adj = self.VGAE.decode(aug_latents)
                re_feats[:self.num_nodes] = self.features
                adj[:self.num_nodes, :self.num_nodes] = self.adj
                adj = (adj > self.config.vae.threshold).float().to(self.device)
                self.VGAE.reset_adj(adj)

                _ =self.VGAE.encode(re_feats)
                re_latents = self.VGAE.mean[self.num_nodes:]
                reconstructed_latents_list.append(re_latents)
            # calculate mse
            sampled_latents = torch.cat(sampled_latents_list, dim=0)
            reconstructed_latents = torch.cat(reconstructed_latents_list, dim=0)
            mse = torch.mean((sampled_latents - reconstructed_latents) ** 2, dim=1)

            # select the top k samples
            filter_index = torch.argsort(mse)
            filtered_samples.append(sampled_latents[filter_index[:align_counts[cls]]])
            filtered_labels.append(torch.ones(align_counts[cls],dtype = torch.int) * cls)
        
        return filtered_samples, filtered_labels
    def _filter_threshold(self):
        self.VGAE.eval()
        self.diffusion.model.eval()
        re_feats, _ = self.VGAE.decode(self.latents)
        self.VGAE.set_device(self.device)
        _ = self.VGAE.encode(re_feats)
        re_latents = self.VGAE.mean

        thresholds_classes = []

        for i in range(self.config.num_classes):
            class_index = torch.where(self.labels == i)[0]
            class_latents = self.latents[class_index]
            class_re_latents = re_latents[class_index]

            mse = torch.mean((class_latents - class_re_latents) ** 2, dim=1)
            thresholds_classes.append(mse.mean().item())
        print("thresholds_classes:", thresholds_classes)

        counts = torch.bincount(self.labels)
        align_counts = counts.max() - counts
        align_counts = (align_counts * self.config.diffusion.generate_ratio).ceil().int()   # generate samples controled by ratio
        if self.config.diffusion.generate_ratio == -1:
            align_counts = (counts * 1.0).ceil().int()
        print("align_counts: ",align_counts)

        filtered_samples = []
        filtered_labels = []

        # main process: sample and filter
        for cls in range(self.config.num_classes):
            count = align_counts[cls].clone()
            if count == 0:
                continue
            fix_count = 100
            print("fix_count: ", fix_count)
            genshape = (fix_count, self.latents.shape[-1])
            cls_labels = torch.ones(fix_count, dtype = torch.long) * cls
            cls_labels_one_hot = torch.nn.functional.one_hot(cls_labels, num_classes=self.config.num_classes).float().to(self.device)
            ori_cls_index = torch.where(self.labels == cls)[0]

            sup_samples = []
            loop = 0
            while count > 0:
                print(f"class {cls} | loop {loop} | need {count} samples")
                # sample
                perm = torch.randperm(ori_cls_index.shape[0])[:fix_count]
                generated_candidate = self.diffusion.sample(genshape, cemb=cls_labels_one_hot)
                aug_latents = torch.cat([self.latents, generated_candidate], dim=0)
                aug_labels = torch.cat([self.labels, cls_labels], dim=0)
                # reconstruct
                re_feats, aug_adj = self.VGAE.decode(aug_latents)
                re_feats[:self.num_nodes] = self.features
                aug_adj[:self.num_nodes, :self.num_nodes] = self.adj
                aug_adj = (aug_adj > self.config.vae.threshold).float().to(self.device)
                self.VGAE.reset_adj(aug_adj)

                _ = self.VGAE.encode(re_feats)
                re_latents = self.VGAE.mean
                # calculate mse
                mse = torch.mean((generated_candidate - re_latents[self.num_nodes:]) ** 2, dim=1)
                filter_index = torch.where(mse <= thresholds_classes[cls])[0]

                sup_samples.append(generated_candidate[filter_index])
                count -= filter_index.shape[0]
                loop += 1

            sup_samples = torch.cat(sup_samples, dim=0)[:align_counts[cls]]
            filtered_samples.append(sup_samples)
            filtered_labels.append(torch.ones(align_counts[cls],dtype = torch.int) * cls)
            print(f"generate {align_counts[cls]} samples for class {cls} finished!",sup_samples.shape)
        return filtered_samples, filtered_labels

    def run(self):
        """
        two strategy to select the samples:
                1. generate certain amounts of samples and select the top k samples
                2. set a threshold and repeat the process until get k samples, where the threshold could be 
                   the MSE of the original latents and their reconstructed latents
            now adopt the first strategy
        """
        self._load_checkpoints()

        if self.config.diffusion.filter_strategy == "topk":
            filtered_samples, filtered_labels = self._filter_topk()
        elif self.config.diffusion.filter_strategy == "threshold":
            filtered_samples, filtered_labels = self._filter_threshold()
        else:
            raise ValueError(f"Invalid filter strategy: {self.config.diffusion.filter_strategy}")

        self.filtered_samples = torch.cat(filtered_samples, dim=0).to(self.device)
        self.filtered_labels = torch.cat(filtered_labels, dim=0).to(self.device)

        self._save_checkpoints()
        self._empty_memory()




