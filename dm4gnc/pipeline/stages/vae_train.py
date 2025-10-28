from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np
import gc

from ..base_stage import BaseStage
from ...models import VGAE
from ...evaluation import get_acc, get_scores
from ...utils import preprocess_graph, mask_test_edges, sparse_to_tuple

class VAETrainStage(BaseStage):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)

        self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                        hidden_dim=self.config.vae.hidden_sizes[0],
                        latent_dim=self.config.vae.hidden_sizes[1],
                        adj=self.adj).to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                            lr=self.config.vae.lr, 
                                            weight_decay=self.config.vae.weight_decay)

    def _get_checkpoints_load_path(self):
        return None
    
    def _get_checkpoints_save_path(self):
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, 'checkpoint_vae_train.pth')
        
    def _load_checkpoints(self):
        return None

    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "vae_train",
            'vae_stage_dict': self.best_vae_state,
            'optimizer_vae_stage_dict': self.best_optimizer_vae_state,
            'adj_norm': self.adj_norm,
            'features': self.features,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: vae_train | Checkpoint saved: {self.checkpoints_save_path}")

    def _empty_memory(self):
        if hasattr(self, 'VGAE'):
            del self.VGAE
        if hasattr(self, 'optimizer_vae'):
            del self.optimizer_vae
        if hasattr(self, 'adj'):
            del self.adj
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self, 'edge_index'):
            del self.edge_index
        if hasattr(self, 'adj_norm'):
            del self.adj_norm
        if hasattr(self, 'train_mask_vae'):
            del self.train_mask_vae
        if hasattr(self, 'val_mask_vae'):
            del self.val_mask_vae
        if hasattr(self, 'best_vae_state'):
            del self.best_vae_state
        if hasattr(self, 'best_optimizer_vae_state'):
            del self.best_optimizer_vae_state       
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        

    def run(self):
        self._load_checkpoints()
        # data preprecess
        adj = self.adj.cpu().detach().numpy()
        features = self.features.cpu().detach().numpy()
        adj = sp.csr_array(adj)
        features = sp.lil_matrix(features)

        adj_orig = adj.copy()
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        # split train and val, only for vae training
        total_size = self.features.shape[0]
        train_size = int(total_size * 0.8)
        index = torch.randperm(total_size)
        self.train_mask_vae = index[:train_size]
        self.val_mask_vae = index[train_size:]

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_orig, val_ratio=0.1)
        adj = adj_train
        adj_norm = preprocess_graph(adj)

        num_nodes = adj.shape[0]
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        self.adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                    torch.FloatTensor(adj_norm[1]), 
                                    torch.Size(adj_norm[2])).to(self.device)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]), 
                                    torch.Size(adj_label[2])).to(self.device)
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                    torch.FloatTensor(features[1]), 
                                    torch.Size(features[2])).to(self.device)

        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(self.device) 
        weight_tensor[weight_mask] = pos_weight

        self.VGAE.reset_adj(self.adj_norm)
        self.features = features

        self.best_vae_state = None
        self.best_optimizer_vae_state = None
        best_epoch = 0
        best_score = 0

        # train
        for epoch in range(self.config.vae.epoch):
            self.VGAE.train()
            self.optimizer_vae.zero_grad()
            feat_pred, A_pred = self.VGAE(features)

            feat_loss = F.mse_loss(feat_pred[self.train_mask_vae], features.to_dense()[self.train_mask_vae])
            link_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            kl_loss =  -0.5/ A_pred.size(0) * (1 + 2*self.VGAE.log_std - self.VGAE.mean**2 - torch.exp(self.VGAE.log_std)**2).sum(1).mean()

            loss = self.config.vae.coef_link * link_loss + self.config.vae.coef_feat * feat_loss + self.config.vae.coef_kl * kl_loss
            
            loss.backward()
            self.optimizer_vae.step()
            
            # val
            train_acc = get_acc(A_pred, adj_label)
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)

            print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | kl_loss: {kl_loss.item():.5f} | train_acc: {train_acc:.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
            
            if val_roc + val_ap > best_score:
                best_score = val_roc + val_ap
                best_epoch = epoch
                self.best_vae_state = {k: v.cpu().clone() for k, v in self.VGAE.state_dict().items()}
                self.best_optimizer_vae_state = {k: v.cpu().clone() if torch.is_tensor(v) else v for k, v in self.optimizer_vae.state_dict().items()}
            else:
                if epoch - best_epoch > self.config.vae.patience:
                    break
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
        print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))

        self._save_checkpoints()
        self._empty_memory()



    