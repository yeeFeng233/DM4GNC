from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np
import gc

from ..base_stage import BaseStage
from ...utils.visualization import t_SNE, visualize_umap_comparison

class VisualizeDataStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)
        self.num_nodes = dataset.x.shape[0]
        self.output_dir = os.path.join(self.config.output_dir, 'visualization', self.config.dataset)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        stage_to_visualize = self.config.stage_to_visualize
        if stage_to_visualize == 'vae_encode':
            self._vae_encode_visualization()
        else:
            raise ValueError(f"Invalid stage to visualize: {stage_to_visualize}")

        
    def _get_checkpoints_load_path(self):
        pass
    def _get_checkpoints_save_path(self):
        pass
    def _load_checkpoints(self):
        pass
    def _save_checkpoints(self):
        pass

    def _load_vae_encode_checkpoints(self):
        path = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode', f'{self.config.vae.name}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.latents = checkpoint['latents']
        self.labels = checkpoint['labels']

    def _vae_encode_visualization(self):
        self._load_vae_encode_checkpoints()
        save_path_root = os.path.join(self.output_dir, 'vae_encode')
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        save_path_tsne = os.path.join(save_path_root, f'{self.config.vae.name}_tsne.png')
        save_path_umap = os.path.join(save_path_root, f'{self.config.vae.name}_umap.png')
        t_SNE(self.latents, self.labels, save_path=save_path_tsne)
        visualize_umap_comparison(self.latents, self.labels, save_path=save_path_umap)

        self.logger.log_image_path("vae_encode_visualization", "tsne", save_path_tsne)
        self.logger.log_image_path("vae_encode_visualization", "umap", save_path_umap)
