from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np

from ..base_stage import BaseStage
from ...models import VGAE
from ...evaluation import get_acc, get_scores

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj,val_ratio=0.1):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

class VAETrainStage(BaseStage):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)

        self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                        hidden_dim=self.config.hidden_sizes[0],
                        latent_dim=self.config.hidden_sizes[1],
                        adj=self.adj).to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                            lr=self.config.vae.lr, 
                                            weight_decay=self.config.vae.weight_decay)

    def _get_checkpoints_load_path(self):
        return None
    
    def _get_checkpoints_save_path(self):
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, 'vae_checkpoint.pth')
        
    def _load_checkpoints(self):
        return None

    def _save_checkpoints(self):
        checkpoint = {
            'stage': "vae_train",
            'config': self.config,
            'vae_stage_dict': self.best_vae_state,
            'optimizer_vae_stage_dict': self.best_optimizer_vae_state,
            'adj_norm': self.adj_norm,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: vae_train | Checkpoint saved: {self.checkpoints_save_path}")

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
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight

        self.vae.reset_adj(adj_norm)

        best_loss = float('inf')
        self.best_vae_state = None
        self.best_optimizer_vae_state = None
        best_epoch = 0
        best_score = 0

        # train
        for epoch in range(self.config.vae.epoch):
            self.VGAE.train()
            self.optimizer_vae.zero_grad()
            feat_pred, A_pred = self.vae(features)

            feat_loss = F.mse_loss(feat_pred[self.train_mask_vae], features.to_dense()[self.train_mask_vae])
            link_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            kl_loss =  -0.5/ A_pred.size(0) * (1 + 2*self.vae.log_std - self.vae.mean**2 - torch.exp(self.vae.log_std)**2).sum(1).mean()

            loss = self.config.coef_link * link_loss + self.config.coef_feat * feat_loss + self.config.coef_kl * kl_loss
            
            loss.backward()
            self.optimizer_vae.step()
            
            # val
            train_acc = get_acc(A_pred, adj_label)
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)

            print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | kl_loss: {kl_loss.item():.5f} | train_acc: {train_acc:.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
            
            if val_roc + val_ap > best_score:
                best_score = val_roc + val_ap
                best_epoch = epoch
                best_vae_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
                best_optimizer_vae_state = {k: v.cpu().clone() if torch.is_tensor(v) else v for k, v in self.optimizer_vae.state_dict().items()}
            else:
                if epoch - best_epoch > self.config.patience_vae:
                    break
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
        print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))

        self._save_checkpoints()
        torch.cuda.empty_cache()




    