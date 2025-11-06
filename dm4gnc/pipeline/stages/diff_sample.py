import math
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import gc
import os

from ..base_stage import BaseStage
from ...models import MLPDenoiser, GaussianDiffusion, get_named_beta_schedule, GradualWarmupScheduler


class DiffSampleStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)
        self.train_index = torch.nonzero(dataset.train_mask, as_tuple=True)[0].to(self.device)
        self.val_index = torch.nonzero(dataset.val_mask, as_tuple=True)[0].to(self.device)
        self.test_index = torch.nonzero(dataset.test_mask, as_tuple=True)[0].to(self.device)

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
        self.optimizer_diffusion = torch.optim.Adam(self.diffusion.model.parameters(), 
                                                    lr=self.config.diffusion.lr,
                                                    weight_decay=self.config.diffusion.weight_decay)


    def _get_checkpoints_load_path(self):
        self.checkpoints_load_path1 = os.path.join(self.checkpoints_root, 'checkpoint_diff_train.pth')
        self.checkpoints_load_path2 = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
    
    def _get_checkpoints_save_path(self):
        save_dir = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.checkpoints_save_path = os.path.join(save_dir, 'checkpoint_diff_sample.pth')
            
    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoints_load_path1):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path1}")
        checkpoint = torch.load(self.checkpoints_load_path1, weights_only=True, map_location=self.device)
        self.diffusion.model.load_state_dict(checkpoint['diffusion_model_state'])
        self.optimizer_diffusion.load_state_dict(checkpoint['optimizer_diffusion_state'])

        if not os.path.exists(self.checkpoints_load_path2):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path2}")
        checkpoint = torch.load(self.checkpoints_load_path2, weights_only=True, map_location=self.device)
        self.latents = checkpoint['latents']
        self.labels = checkpoint['labels']
    
    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "diff_sample",
            'generated_samples': self.generated_samples,
            'generated_labels': self.generated_labels,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: diff_sample | Checkpoint saved: {self.checkpoints_save_path}")

    def _empty_memory(self):
        if hasattr(self, 'mlpdenoiser'):
            del self.mlpdenoiser
        if hasattr(self, 'diffusion'):
            del self.diffusion
        if hasattr(self, 'optimizer_diffusion'):
            del self.optimizer_diffusion
        if hasattr(self, 'generated_samples'):
            del self.generated_samples
        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        self._load_checkpoints()

        # align number of samples for per class
        counts = torch.bincount(self.labels)
        align_counts = counts.max() - counts
        align_counts = (align_counts * self.config.diffusion.generate_ratio).ceil().int()   # generate samples controled by ratio

        # test quality of generated nodes
        if self.config.diffusion.generate_ratio == -1:
            align_counts = (counts * 1.0).ceil().int()

        sample_sum = align_counts.sum()
        print(f"Generating {sample_sum} samples for balancing...")
        print(f"generate samples for per class: {align_counts}")
        supplemented_labels = torch.cat([torch.full((int(n.item()),), i) for i, n in enumerate(align_counts) if n > 0]).to(self.device)
        supplemented_labels_one_hot = torch.nn.functional.one_hot(supplemented_labels, num_classes=self.config.num_classes).float().to(self.device)
        self.generated_labels = supplemented_labels
        generated_samples = []
        numloop = math.ceil(sample_sum / self.config.diffusion.genbatch)

        for loop in range(numloop):
            print(f"Generation loop: {loop+1}/{numloop}")
            index_length = self.config.diffusion.genbatch if (loop+1) * self.config.diffusion.genbatch <= sample_sum else sample_sum - loop*self.config.diffusion.genbatch
            genshape = (index_length, self.latents.shape[-1])
            head = loop*self.config.diffusion.genbatch
            print("genshape:",genshape)
            generated = self.diffusion.sample(genshape, cemb=supplemented_labels_one_hot[head:head+index_length])
            generated_samples.append(generated)

        self.generated_samples = torch.cat(generated_samples, dim=0)

        self.log_metrics({
            'total_generated_samples': int(sample_sum.item()),
            'generate_ratio': self.config.diffusion.generate_ratio,
            'samples_per_class': align_counts.tolist()
        })

        self._save_checkpoints()
        self._empty_memory()
