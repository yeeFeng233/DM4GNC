from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc

from ..base_stage import BaseStage
from ...models import VGAE, VGAE_class, VGAE_class_v2, VGAE_DEC
from ...utils import sparse_to_tuple

class VAEEncodeStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)

        self._init_model()

    def _init_model(self):
        if self.config.vae.name == "normal_vae":
            self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                            hidden_dim=self.config.vae.hidden_sizes[0],
                            latent_dim=self.config.vae.hidden_sizes[1],
                            adj=None).to(self.device)
            self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                                lr=self.config.vae.lr)
        elif self.config.vae.name == "vae_class":
            self.VGAE = VGAE_class(feat_dim=self.config.feat_dim,
                                hidden_dim=self.config.vae.hidden_sizes[0],
                                latent_dim=self.config.vae.hidden_sizes[1],
                                adj=None,
                                num_classes = self.config.num_classes).to(self.device)
            self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                                lr=self.config.vae.lr)
        elif self.config.vae.name == "vae_class_v2":
            self.VGAE = VGAE_class_v2(feat_dim=self.config.feat_dim,
                                hidden_dim=self.config.vae.hidden_sizes[0],
                                latent_dim=self.config.vae.hidden_sizes[1],
                                adj=None,
                                num_classes = self.config.num_classes).to(self.device)
            self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                                lr=self.config.vae.lr)
        elif self.config.vae.name == "vae_dec":
            self.VGAE = VGAE_DEC(feat_dim=self.config.feat_dim,
                                hidden_dim=self.config.vae.hidden_sizes[0],
                                latent_dim=self.config.vae.hidden_sizes[1],
                                adj=None,
                                n_clusters = self.config.num_classes).to(self.device)
            self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                                lr=self.config.vae.lr)
        else:
            raise ValueError(f"Invalid vae name: {self.config.vae.name}")

    
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
        self.VGAE.reset_adj(adj_norm)

    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "vae_encode",
            'latents': self.latents,
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

        features = self.features.cpu().detach().numpy()
        features = sp.lil_matrix(features)
        features = sparse_to_tuple(features.tocoo())
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2])).to(self.device)

        self.VGAE.eval()
        with torch.no_grad():
            if self.config.vae.name == "normal_vae":
                self.z = self.VGAE.encode(features)
            elif self.config.vae.name == "vae_class":
                self.z = self.VGAE.encode(features)
            elif self.config.vae.name == "vae_class_v2":
                self.z, class_pred = self.VGAE.encode(features)
            elif self.config.vae.name == "vae_dec":
                self.z = self.VGAE.encode(features)
            else:
                raise ValueError(f"Invalid vae name: {self.config.vae.name}")

        if self.config.vae.name == "normal_vae":
            feat_pred, A_pred = self.VGAE.decode(self.z)
        elif self.config.vae.name == "vae_class":
            feat_pred, A_pred, class_pred = self.VGAE.decode(self.z)
        elif self.config.vae.name == "vae_class_v2":
            feat_pred, A_pred = self.VGAE.decode(self.z)
        elif self.config.vae.name == "vae_dec":
            feat_pred, A_pred = self.VGAE.decode(self.z)
        else:
            raise ValueError(f"Invalid vae name: {self.config.vae.name}")

        self.latents = self.VGAE.mean

        self._save_checkpoints()

        self.metrics()

        print(f"stage: vae_encode | Run successfully")
        self._empty_memory()

    def metrics(self):
        pass
