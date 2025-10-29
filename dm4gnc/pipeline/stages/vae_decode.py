from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc

from ..base_stage import BaseStage
from ...models import VGAE

class VAEDecodeStage(BaseStage):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)

        self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                        hidden_dim=self.config.vae.hidden_sizes[0],
                        latent_dim=self.config.vae.hidden_sizes[1],
                        adj=self.adj).to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                            lr=self.config.vae.lr, 
                                            weight_decay=self.config.vae.weight_decay)
        self.train_index = torch.nonzero(dataset.train_mask, as_tuple=True)[0].to(self.device)
        

    def _get_checkpoints_load_path(self):
        self.checkpoints_load_path1 = os.path.join(self.checkpoints_root, 'checkpoint_vae_train.pth')
        self.checkpoints_load_path2 = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
        if self.config.diffusion.filter:
            self.checkpoints_load_path3 = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}", 'checkpoint_filter_samples.pth')
        else:
            self.checkpoints_load_path3 = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}", 'checkpoint_diff_sample.pth')
        
    def _get_checkpoints_save_path(self):
        save_dir = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.checkpoints_save_path = os.path.join(save_dir, 'checkpoint_vae_decode.pth')
        
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
        self.optimizer_vae.load_state_dict(checkpoint1['optimizer_vae_stage_dict'])
        checkpoint2 = torch.load(self.checkpoints_load_path2, weights_only=True, map_location=self.device)
        self.latents = checkpoint2['latents']
        self.labels = checkpoint2['labels']
        checkpoint3 = torch.load(self.checkpoints_load_path3, weights_only=True, map_location=self.device)
        self.generated_samples = checkpoint3['generated_samples'].to(self.device)
        self.generated_labels = checkpoint3['generated_labels'].to(self.device)
    
    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "vae_decode",
            'aug_adj': self.aug_adj,
            'aug_feats': self.aug_feats,
            'aug_labels': self.aug_labels,
            'aug_train_index': self.aug_train_index,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: vae_decode | Checkpoint saved: {self.checkpoints_save_path}")

    def _empty_memory(self):
        if hasattr(self, 'VGAE'):
            del self.VGAE
        if hasattr(self, 'optimizer_vae'):
            del self.optimizer_vae
        if hasattr(self, 'latents'):
            del self.latents
        if hasattr(self, 'labels'):
            del self.labels
        if hasattr(self, 'generated_samples'):
            del self.generated_samples
        if hasattr(self, 'generated_labels'):
            del self.generated_labels
        if hasattr(self, 'aug_train_index'):
            del self.aug_train_index
        if hasattr(self, 'aug_adj'):
            del self.aug_adj
        if hasattr(self, 'aug_feats'):
            del self.aug_feats
        if hasattr(self, 'aug_labels'):
            del self.aug_labels
        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        self._load_checkpoints()

        self.labels_one_hot = torch.nn.functional.one_hot(self.labels, num_classes=self.config.num_classes).float().to(self.device)
        aug_latents = torch.cat([self.latents,self.generated_samples],dim=0)
        aug_labels = torch.cat([self.labels,self.generated_labels],dim=0)

        self.VGAE.eval()
        pred_feats, pred_adj = self.VGAE.decode(aug_latents)
        assert self.adj.device == pred_adj.device
        assert self.features.device == pred_feats.device
        h,w = self.adj.shape
        pred_adj[:h,:w] = self.adj
        pred_feats[:h,:] = self.features

        len_ori,len_aug = self.labels.shape[0], self.generated_labels.shape[0]
        self.aug_train_index = torch.cat([self.train_index, torch.arange(len_ori, len_ori+len_aug).to(self.device)]).to(self.device)
        self.aug_adj = pred_adj
        self.aug_feats = pred_feats
        self.aug_labels = aug_labels

        self._save_checkpoints()
        self._empty_memory()