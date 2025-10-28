from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc

from ..base_stage import BaseStage
from ...models import VGAE
from ...utils import sparse_to_tuple

class VAEEncodeStage(BaseStage):
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

    def _get_checkpoints_load_path(self):
        self.checkpoints_load_path = os.path.join(self.checkpoints_root, 'checkpoint_vae_train.pth')
    
    def _get_checkpoints_save_path(self):
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
        
    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoints_load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path}")
        checkpoint = torch.load(self.checkpoints_load_path, weights_only=True, map_location=self.device)
        self.VGAE.load_state_dict(checkpoint['vae_stage_dict'])
        self.optimizer_vae.load_state_dict(checkpoint['optimizer_vae_stage_dict'])
        adj_norm = checkpoint['adj_norm']
        self.features = checkpoint['features']
        self.VGAE.reset_adj(adj_norm)

    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "vae_encode",
            'latents': self.z,
            'labels': self.labels
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: vae_encode | Checkpoint saved: {self.checkpoints_save_path}")
    
    def _empty_memory(self):
        if hasattr(self, 'VGAE'):
            del self.VGAE
        if hasattr(self, 'optimizer_vae'):
            del self.optimizer_vae
        if hasattr(self, 'z'):
            del self.z
        if hasattr(self, 'pred_adj'):
            del self.pred_adj
        if hasattr(self, 'labels'):
            del self.labels
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self, 'edge_index'):
            del self.edge_index
        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def run(self):
        self._load_checkpoints()

        self.VGAE.eval()
        with torch.no_grad():
            self.z = self.VGAE.encode(self.features)

        _, self.pred_adj = self.VGAE.decode(self.z)

        self._save_checkpoints()

        self.metrics()

        print(f"stage: vae_encode | Run successfully")
        self._empty_memory()

    def metrics(self):
        pass
