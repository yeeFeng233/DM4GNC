from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np
import gc

from ..base_stage import BaseStage
from ...utils.visualization import t_SNE, visualize_umap_comparison, analyze_neighbor_class_distribution
from ...models import VGAE, VGAE_class, VGAE_class_v2

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
        elif stage_to_visualize == 'diff_sample':
            self._diff_sample_visualization()
        elif stage_to_visualize == 'filter_samples':
            self._filter_samples_visualization()
        elif stage_to_visualize == 'neighbor_distribution':
            self._neighbor_distribution()
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
        else:
            raise ValueError(f"Invalid vae name: {self.config.vae.name}")


    def _load_vae_encode_checkpoints(self):
        path = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.latents = checkpoint['latents'].to(self.device)
        self.labels = checkpoint['labels'].to(self.device)
    
    def _load_diff_sample_checkpoints(self):
        path = os.path.join(self.checkpoints_root,f"{self.config.diffusion.generate_ratio}", 'checkpoint_diff_sample.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.generated_samples = checkpoint['generated_samples'].to(self.device)
        self.generated_labels = checkpoint['generated_labels'].to(self.device)
    
    def _load_filter_samples_checkpoints(self):
        path = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}", f"{self.config.diffusion.filter_strategy}", 'checkpoint_filter_samples.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.filtered_samples = checkpoint['generated_samples'].to(self.device)
        self.filtered_labels = checkpoint['generated_labels'].to(self.device)
    
    def _load_vae_train_checkpoints(self):
        self._init_model()
        path = os.path.join(self.checkpoints_root, 'checkpoint_vae_train.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.VGAE.load_state_dict(checkpoint['vae_stage_dict'])
        self.optimizer_vae.load_state_dict(checkpoint['optimizer_vae_stage_dict'])

    def _vae_encode_visualization(self):
        self._load_vae_encode_checkpoints()
        save_path_root = os.path.join(self.output_dir, 'vae_encode_kl_ratio_{self.config.vae.coef_kl}')
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        save_path_tsne = os.path.join(save_path_root, f'{self.config.vae.name}_tsne.png')
        save_path_umap = os.path.join(save_path_root, f'{self.config.vae.name}_umap.png')
        t_SNE(self.latents, self.labels, save_path=save_path_tsne)
        visualize_umap_comparison(self.latents, self.labels, save_path=save_path_umap)

        self.logger.log_image_path("vae_encode_visualization", "tsne", save_path_tsne)
        self.logger.log_image_path("vae_encode_visualization", "umap", save_path_umap)
    
    def _diff_sample_visualization(self):
        self._load_vae_encode_checkpoints()
        self._load_diff_sample_checkpoints()
        save_path_root = os.path.join(self.output_dir, 'diff_sample')
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        save_path_tsne = os.path.join(save_path_root, f'{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_tsne.png')
        save_path_umap = os.path.join(save_path_root, f'{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_umap.png')
        t_SNE(self.latents, self.labels, x_syn=self.generated_samples, y_syn=self.generated_labels, save_path=save_path_tsne)
        visualize_umap_comparison(self.latents, self.labels, x_syn=self.generated_samples, y_syn=self.generated_labels, save_path=save_path_umap)
        
        save_path_tsne2 = os.path.join(save_path_root, f'samples_{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_tsne.png')
        save_path_umap2 = os.path.join(save_path_root, f'sample_{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_umap.png')
        t_SNE(self.generated_samples, self.generated_labels, save_path=save_path_tsne2)
        visualize_umap_comparison(self.generated_samples, self.generated_labels, save_path=save_path_umap2)

    def _filter_samples_visualization(self):
        self._load_vae_encode_checkpoints()
        self._load_filter_samples_checkpoints()
        save_path_root = os.path.join(self.output_dir, 'filter_samples')
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        save_path_tsne = os.path.join(save_path_root, f'{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_tsne.png')
        save_path_umap = os.path.join(save_path_root, f'{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_umap.png')
        t_SNE(self.latents, self.labels, x_syn=self.filtered_samples, y_syn=self.filtered_labels, save_path=save_path_tsne)
        visualize_umap_comparison(self.latents, self.labels, x_syn=self.filtered_samples, y_syn=self.filtered_labels, save_path=save_path_umap)

        save_path_tsne2 = os.path.join(save_path_root, f'samples_{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_tsne.png')
        save_path_umap2 = os.path.join(save_path_root, f'sample_{self.config.vae.name}_generate_ratio_{self.config.diffusion.generate_ratio}_umap.png')
        t_SNE(self.filtered_samples, self.filtered_labels, save_path=save_path_tsne2)
        visualize_umap_comparison(self.filtered_samples, self.filtered_labels, save_path=save_path_umap2)

    def _neighbor_distribution(self):
        self._load_vae_train_checkpoints()
        self._load_vae_encode_checkpoints()
        
        if self.config.diffusion.filter:
            self._load_filter_samples_checkpoints()
            self.VGAE.eval()
            with torch.no_grad():
                if self.config.vae.name != "vae_class":
                    pred_feats, pred_adj = self.VGAE.decode(self.filtered_samples)
                else:
                    pred_feats, pred_adj, _ = self.VGAE.decode(self.filtered_samples)
            save_path_root = os.path.join(self.output_dir,'neighbor_distribution', f'{self.config.vae.name}','filtered_samples', f'{self.config.diffusion.filter_strategy}')
        else:
            self._load_diff_sample_checkpoints()
            self.VGAE.eval()
            with torch.no_grad():
                if self.config.vae.name != "vae_class":
                    pred_feats, pred_adj = self.VGAE.decode(self.generated_samples)
                else:
                    pred_feats, pred_adj, _ = self.VGAE.decode(self.generated_samples)
            save_path_root = os.path.join(self.output_dir,'neighbor_distribution', f'{self.config.vae.name}','generated_samples')
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        thresholds = [i/100.0+0.90 for i in range(1,10,1)]
        analyze_neighbor_class_distribution(self.adj, pred_adj, self.labels, self.filtered_labels, save_path_root,thresholds=thresholds)