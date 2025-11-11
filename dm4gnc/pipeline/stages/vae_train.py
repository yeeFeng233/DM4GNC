from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np
import gc
from torch.amp import GradScaler, autocast

from ..base_stage import BaseStage
from ...models import VGAE, VGAE_class, VGAE_class_v2, VGAE_DEC, VGAE_DEC_class
from ...evaluation import get_acc, get_scores
from ...utils import preprocess_graph, mask_test_edges, mask_test_edges_fast, sparse_to_tuple
from ..model_factory import _init_VGAE

class VAETrainStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)
        self.VGAE, self.optimizer_vae = _init_VGAE(config)
        
        self.large_graph_threshold = 10000
        self.neg_sample_ratio = 5
        self.val_interval = getattr(config.vae, 'val_interval', 5)  # Validate every N epochs
        self.scaler = GradScaler(device='cuda')

    def compute_kl_loss_stable(self, mean, log_std):
        """
        Numerically stable KL divergence computation
        KL(N(μ,σ²)||N(0,1)) = 0.5 * sum(μ² + σ² - 1 - log(σ²))
        """
        # Clamp log_std to prevent exp explosion
        log_std_clamped = torch.clamp(log_std, min=-10, max=2)
        
        # Compute KL divergence
        kl_per_dim = 0.5 * (
            mean.pow(2) + 
            torch.exp(2 * log_std_clamped) - 
            1 - 
            2 * log_std_clamped
        )
        
        # Mean over all dimensions
        kl_loss = torch.mean(kl_per_dim)
        
        return kl_loss
    
    def _get_checkpoints_load_path(self):
        return None
    
    def _get_checkpoints_save_path(self):
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, "checkpoint_vae_train.pth")
        
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
        if hasattr(self, 'pos_edges'):
            del self.pos_edges
        if hasattr(self, 'neg_edges'):
            del self.neg_edges
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _sample_negative_edges_gpu(self, num_nodes, pos_edges, num_neg_samples):
        pos_edge_indices = pos_edges[:, 0] * num_nodes + pos_edges[:, 1]
        pos_edge_set_tensor = torch.zeros(num_nodes * num_nodes, dtype=torch.bool, device=self.device)
        pos_edge_set_tensor[pos_edge_indices] = True
        
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 5
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # 批量采样（提高效率）
            batch_size = min(num_neg_samples * 3, num_neg_samples - len(neg_edges) + 1000)
            src = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            dst = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            
            # 向量化过滤：去除自环和正边
            valid_mask = (src != dst)  # 不是自环
            edge_indices = src * num_nodes + dst
            valid_mask &= ~pos_edge_set_tensor[edge_indices]  # 不是正边
            
            # 提取有效的负边
            valid_src = src[valid_mask]
            valid_dst = dst[valid_mask]
            
            # 添加到结果（转换为列表避免重复）
            if len(valid_src) > 0:
                new_edges = torch.stack([valid_src, valid_dst], dim=1)
                neg_edges.append(new_edges)
                
                # 标记已采样的边，避免重复
                new_indices = edge_indices[valid_mask]
                pos_edge_set_tensor[new_indices] = True
            
            attempts += batch_size
        
        if len(neg_edges) == 0:
            # Fallback: 如果没有采样到任何边
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        
        # 合并所有批次的结果
        neg_edges = torch.cat(neg_edges, dim=0)
        
        # 截断到所需数量
        if len(neg_edges) > num_neg_samples:
            neg_edges = neg_edges[:num_neg_samples]
        
        return neg_edges

    def _compute_loss_small_graph(self, A_pred, adj_label, weight_tensor, norm):
        link_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        return link_loss

    def _compute_loss_large_graph(self, A_pred, pos_edges, neg_edges):

        pos_pred = A_pred[pos_edges[:, 0], pos_edges[:, 1]]
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred), reduction='mean')

        neg_pred = A_pred[neg_edges[:, 0], neg_edges[:, 1]]
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred), reduction='mean')
        
        link_loss = (pos_loss + neg_loss) / 2
        return link_loss

    def run(self):
        import time
        start_time = time.time()
        
        self._load_checkpoints()
        
        num_nodes = self.adj.shape[0]
        is_large_graph = num_nodes > self.large_graph_threshold
        
        print(f"\n{'='*60}")
        print(f"VAE Training Preparation")
        print(f"{'='*60}")
        print(f"Graph size: {num_nodes} nodes, {self.adj.sum().item()//2:.0f} edges")
        print(f"Mode: {'LARGE GRAPH (Negative Sampling)' if is_large_graph else 'SMALL GRAPH (Full Matrix)'}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        

        adj_cpu = self.adj.cpu().detach().numpy()
        adj_cpu = sp.csr_array(adj_cpu)
        labels_one_hot = F.one_hot(self.labels, num_classes=self.config.num_classes).float()

        adj_orig = adj_cpu.copy()
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_fast(
            adj_orig, val_ratio=0.05, test_ratio=0.1, seed=self.config.seed)
        adj_norm = preprocess_graph(adj_train)
        self.adj_norm = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm[0].T), 
            torch.FloatTensor(adj_norm[1]), 
            torch.Size(adj_norm[2])
        ).to(self.device)
        
        if self.features.is_sparse:
            features = self.features.coalesce().to(self.device)
        else:
            features = self.features.to_sparse().coalesce().to(self.device)
        features_dense = features.to_dense()


        total_size = num_nodes
        train_size = int(total_size * 0.8)
        index = torch.randperm(total_size)
        self.train_mask_vae = index[:train_size].to(self.device)
        self.val_mask_vae = index[train_size:].to(self.device)
 
        if is_large_graph:

            pos_edges_np = np.array(adj_train.nonzero()).T
            self.pos_edges = torch.tensor(pos_edges_np, dtype=torch.long, device=self.device)
            num_pos = self.pos_edges.shape[0] // 2  # 无向图，除以2
            num_neg = num_pos * self.neg_sample_ratio
            
            print(f"Positive edges: {num_pos}, Negative samples per epoch: {num_neg}")
        else:
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label_tuple = sparse_to_tuple(adj_label)
            adj_label = torch.sparse.FloatTensor(
                torch.LongTensor(adj_label_tuple[0].T),
                torch.FloatTensor(adj_label_tuple[1]),
                torch.Size(adj_label_tuple[2])
            ).to(self.device)
            
            pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
            norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
            
            weight_mask = adj_label.to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0), device=self.device)
            weight_tensor[weight_mask] = pos_weight
            
            print(f"Pos weight: {pos_weight:.2f}, Norm: {norm:.4f}")

        self.VGAE.reset_adj(self.adj_norm)
        
        self.best_vae_state = None
        self.best_optimizer_vae_state = None
        best_epoch = 0
        best_score = 0
        if self.config.vae.name in ["vae_dec", "vae_dec_class"]:
            pretrain_epochs = min(200, self.config.vae.epoch // 3)
            print(f"Pretraining for {pretrain_epochs} epochs...")
            
            for epoch in range(pretrain_epochs):
                self.VGAE.train()
                self.optimizer_vae.zero_grad()
                
                if self.config.vae.name == "vae_dec":
                    feat_pred, A_pred, z = self.VGAE(features)
                elif self.config.vae.name == "vae_dec_class":
                    feat_pred, A_pred, class_pred, z = self.VGAE(features)
                # Use mixed precision if enabled
                with autocast(device_type='cuda'):
                    feat_loss = F.mse_loss(feat_pred[self.train_mask_vae], features_dense[self.train_mask_vae])

                
                if is_large_graph:
                    neg_edges = self._sample_negative_edges_gpu(num_nodes, self.pos_edges, num_neg)
                    link_loss = self._compute_loss_large_graph(A_pred, self.pos_edges, neg_edges)
                else:
                    link_loss = self._compute_loss_small_graph(A_pred, adj_label, weight_tensor, norm)
                
                # Use numerically stable KL loss
                kl_loss = self.compute_kl_loss_stable(self.VGAE.mean, self.VGAE.log_std)
                
                loss = (self.config.vae.coef_link * link_loss + 
                       self.config.vae.coef_feat * feat_loss + 
                       self.config.vae.coef_kl * kl_loss)
                
                if self.config.vae.name == "vae_dec_class":
                    class_loss = F.cross_entropy(class_pred[self.train_mask_vae], self.labels[self.train_mask_vae])
                    loss += class_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer_vae)
                torch.nn.utils.clip_grad_norm_(self.VGAE.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer_vae)
                self.scaler.update()
                
                if epoch % 10 == 0:
                    with torch.no_grad():
                        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)
                    print(f"Pretrain epoch {epoch} | loss: {loss.item():.5f} | "
                          f"val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
            

            print("\nInitializing cluster centers with K-means...")
            with torch.no_grad():
                z = self.VGAE.encode(features)
                y_pred = self.VGAE.init_cluster_centers(z)
                
                # 计算初始聚类质量
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(z.cpu().numpy(), y_pred)
                print(f"Initial silhouette score: {silhouette:.4f}")
                
                # 打印聚类分布
                unique, counts = np.unique(y_pred, return_counts=True)
                print(f"Initial cluster distribution: {dict(zip(unique, counts))}")
            
            print(f"\nPhase 2: Fine-tuning with clustering loss...")
            print("="*50 + "\n")
            
            # 重置最佳模型追踪
            best_epoch = pretrain_epochs
            best_score = 0

        # ============ 训练循环 ============
        for epoch in range(self.config.vae.epoch):
            self.VGAE.train()
            self.optimizer_vae.zero_grad()
            if self.config.vae.name in ["normal_vae", "vae_sig"]:
                feat_pred, A_pred = self.VGAE(features)
            elif self.config.vae.name == "vae_class":
                feat_pred, A_pred, class_pred = self.VGAE(features)
            elif self.config.vae.name == "vae_class_v2":
                feat_pred, A_pred, class_pred = self.VGAE(features)
            elif self.config.vae.name == "vae_dec":
                feat_pred, A_pred, z = self.VGAE(features)
            elif self.config.vae.name == "vae_dec_class":
                feat_pred, A_pred, class_pred, z = self.VGAE(features)
            else:
                raise ValueError(f"Invalid vae name: {self.config.vae.name}")
            # Use automatic mixed precision for faster training

            with autocast(device_type='cuda'):
                feat_loss = F.mse_loss(feat_pred[self.train_mask_vae], features_dense[self.train_mask_vae])

            # Link loss
            # Note: This is computed outside autocast as it doesn't benefit much from FP16
            if is_large_graph:
                neg_edges = self._sample_negative_edges_gpu(
                    num_nodes, 
                    self.pos_edges, 
                    num_neg
                )
                link_loss = self._compute_loss_large_graph(A_pred, self.pos_edges, neg_edges)
            else:
                link_loss = self._compute_loss_small_graph(A_pred, adj_label, weight_tensor, norm)
            
            kl_loss = self.compute_kl_loss_stable(self.VGAE.mean, self.VGAE.log_std)
            
            # Total loss
            loss = (self.config.vae.coef_link * link_loss + 
                   self.config.vae.coef_feat * feat_loss + 
                   self.config.vae.coef_kl * kl_loss)
            
            if self.config.vae.name in ["vae_class", "vae_class_v2", "vae_dec_class"]:
                class_loss = F.cross_entropy(class_pred[self.train_mask_vae], self.labels[self.train_mask_vae])
                loss += class_loss * 0.5

            if self.config.vae.name in ["vae_dec", "vae_dec_class"]:
                total_epochs = self.config.vae.epoch
                warmup_epochs = min(20, total_epochs // 5)
                if epoch < warmup_epochs:
                    cluster_weight = 0.5 + (5.0 - 0.5) * (epoch / warmup_epochs)
                else:
                    cluster_weight = max(5.0 + 5.0 * ((epoch - warmup_epochs) / (total_epochs - warmup_epochs)), 0.1)
                
                cluster_loss, q = self.VGAE.clustering_loss(z)
                # loss += cluster_loss * cluster_weight
                loss += cluster_loss

                if epoch % 10 == 0:
                    with torch.no_grad():
                        max_probs, pred_labels = q.max(dim=1)
                        avg_confidence = max_probs.mean().item()
                        cluster_entropy = -(q * (q + 1e-10).log()).sum(dim=1).mean().item()
                        cluster_counts = pred_labels.bincount(minlength=self.VGAE.n_clusters)


            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer_vae)
            torch.nn.utils.clip_grad_norm_(self.VGAE.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer_vae)
            self.scaler.update()

            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)

                if self.config.vae.name == "vae_dec":
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"cluster_loss: {cluster_loss.item():.5f} (w:{cluster_weight:.2f}) | "
                        f"kl_loss: {kl_loss.item():.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}", end="")
                    print(f"Cluster quality: confidence={avg_confidence:.4f} | "
                        f"entropy={cluster_entropy:.4f} | dist={cluster_counts.tolist()}")
                elif self.config.vae.name in ["vae_class", "vae_class_v2"]:
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"class_loss: {class_loss.item():.5f} | "
                        f"kl_loss: {kl_loss.item():.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
                elif self.config.vae.name == "vae_dec_class":
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"class_loss: {class_loss.item():.5f} | "
                        f"cluster_loss: {cluster_loss.item():.5f} (w:{cluster_weight:.2f}) | "
                        f"kl_loss: {kl_loss.item():.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}", end="")
                    print(f"Cluster quality: confidence={avg_confidence:.4f} | "
                        f"entropy={cluster_entropy:.4f} | dist={cluster_counts.tolist()}")
                else:
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"kl_loss: {kl_loss.item():.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
                # else:
                #     print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                #         f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                #         f"class_loss: {class_loss.item():.5f} | "
                #         f"kl_loss: {kl_loss.item():.5f} | train_acc: {train_acc:.5f} | "
                #         f"val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
            

                if val_roc >= 0 and val_roc + val_ap > best_score:
                    best_score = val_roc + val_ap
                    best_epoch = epoch
                    self.best_vae_state = {k: v.cpu().clone() for k, v in self.VGAE.state_dict().items()}
                    self.best_optimizer_vae_state = {
                        k: v.cpu().clone() if torch.is_tensor(v) else v 
                        for k, v in self.optimizer_vae.state_dict().items()
                    }
                else:
                    if epoch - best_epoch > self.config.vae.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # 测试集评估
        with torch.no_grad():
            test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred, adj_orig)
        print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))
        
        # 记录关键指标到logger
        self.log_metrics({
            'best_val_roc': best_score / 2,
            'best_epoch': best_epoch,
            'test_roc': test_roc,
            'test_ap': test_ap,
            'is_large_graph': is_large_graph
        })
        
        self._save_checkpoints()
        self._empty_memory()



    