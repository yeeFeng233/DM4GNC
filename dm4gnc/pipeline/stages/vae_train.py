from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import numpy as np
import gc

from ..base_stage import BaseStage
from ...models import VGAE, VGAE_class, VGAE_class_v2
from ...evaluation import get_acc, get_scores
from ...utils import preprocess_graph, mask_test_edges, sparse_to_tuple

class VAETrainStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)

        self.adj = dataset.adj.to(self.device)
        self.features = dataset.x.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        self.labels = dataset.y.to(self.device)
        self._init_model()
        
        self.large_graph_threshold = 20000
        self.neg_sample_ratio = 20

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

    def _get_checkpoints_load_path(self):
        return None
    
    def _get_checkpoints_save_path(self):
        if not os.path.exists(os.path.join(self.checkpoints_root, "checkpoint_vae_train")):
            os.makedirs(os.path.join(self.checkpoints_root, "checkpoint_vae_train"))
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, "checkpoint_vae_train", f'{self.config.vae.name}.pth')
        
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
        """
        在GPU上采样负边
        Args:
            num_nodes: 节点数量
            pos_edges: 正边张量 [num_pos, 2]
            num_neg_samples: 需要采样的负边数量
        Returns:
            neg_edges: 负边张量 [num_neg_samples, 2]
        """
        # 将正边转换为集合（在GPU上进行哈希）
        pos_edge_set = set()
        pos_edges_cpu = pos_edges.cpu().numpy()
        for i in range(pos_edges_cpu.shape[0]):
            e = tuple(pos_edges_cpu[i])
            pos_edge_set.add(e)
            pos_edge_set.add((e[1], e[0]))  # 无向图
        
        neg_edges = []
        max_attempts = num_neg_samples * 100  # 防止无限循环
        attempts = 0
        
        # 批量采样提高效率
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # 一次采样多个候选边
            batch_size = min(num_neg_samples * 2, num_neg_samples - len(neg_edges) + 100)
            src = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            dst = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            
            # 移到CPU进行集合检查（更快）
            src_cpu = src.cpu().numpy()
            dst_cpu = dst.cpu().numpy()
            
            for i in range(batch_size):
                if len(neg_edges) >= num_neg_samples:
                    break
                s, d = int(src_cpu[i]), int(dst_cpu[i])
                if s != d and (s, d) not in pos_edge_set:
                    neg_edges.append([s, d])
                    pos_edge_set.add((s, d))  # 避免重复采样
            
            attempts += batch_size
        
        return torch.tensor(neg_edges, dtype=torch.long, device=self.device)

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
        self._load_checkpoints()
        
        num_nodes = self.adj.shape[0]
        is_large_graph = num_nodes > self.large_graph_threshold
        
        print(f"Graph size: {num_nodes} nodes")
        print(f"Mode: {'LARGE GRAPH (Negative Sampling)' if is_large_graph else 'SMALL GRAPH (Full Matrix)'}")
        
        # ============ 数据预处理（尽量在GPU上进行） ============
        
        # 将adj转到CPU进行稀疏矩阵操作（scipy只支持CPU）
        adj_cpu = self.adj.cpu().detach().numpy()
        adj_cpu = sp.csr_array(adj_cpu)
        labels_one_hot = F.one_hot(self.labels, num_classes=self.config.num_classes).float()
        
        # 去除自环
        adj_orig = adj_cpu.copy()
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        
        # 划分训练/验证/测试边
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_orig, val_ratio=0.1)
        
        # 归一化邻接矩阵
        adj_norm = preprocess_graph(adj_train)
        
        # 转换为PyTorch稀疏张量（GPU）
        self.adj_norm = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm[0].T), 
            torch.FloatTensor(adj_norm[1]), 
            torch.Size(adj_norm[2])
        ).to(self.device)
        
        # 处理特征矩阵
        features = self.features.cpu().detach().numpy()
        features = sp.lil_matrix(features)
        features = sparse_to_tuple(features.tocoo())
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                    torch.FloatTensor(features[1]), 
                                    torch.Size(features[2])).to(self.device)


        # 特征重构的训练/验证划分
        total_size = num_nodes
        train_size = int(total_size * 0.8)
        index = torch.randperm(total_size)
        self.train_mask_vae = index[:train_size].to(self.device)
        self.val_mask_vae = index[train_size:].to(self.device)
        
        # ============ 根据图大小选择训练策略 ============
        
        if is_large_graph:
            # 大图模式：负采样
            # 提取正边并转为GPU张量
            pos_edges_np = np.array(adj_train.nonzero()).T
            self.pos_edges = torch.tensor(pos_edges_np, dtype=torch.long, device=self.device)
            
            # 预采样负边（每个epoch重新采样）
            num_pos = self.pos_edges.shape[0] // 2  # 无向图，除以2
            num_neg = num_pos * self.neg_sample_ratio
            
            print(f"Positive edges: {num_pos}, Negative samples per epoch: {num_neg}")
            
        else:
            # 小图模式：全矩阵
            # 构建标签邻接矩阵
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label_tuple = sparse_to_tuple(adj_label)
            adj_label = torch.sparse.FloatTensor(
                torch.LongTensor(adj_label_tuple[0].T),
                torch.FloatTensor(adj_label_tuple[1]),
                torch.Size(adj_label_tuple[2])
            ).to(self.device)
            
            # 计算权重
            pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
            norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
            
            weight_mask = adj_label.to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0), device=self.device)
            weight_tensor[weight_mask] = pos_weight
            
            print(f"Pos weight: {pos_weight:.2f}, Norm: {norm:.4f}")
        
        # 更新VGAE的邻接矩阵
        self.VGAE.reset_adj(self.adj_norm)
        
        # 最佳模型追踪
        self.best_vae_state = None
        self.best_optimizer_vae_state = None
        best_epoch = 0
        best_score = 0
        
        # ============ 训练循环 ============
        for epoch in range(self.config.vae.epoch):
            self.VGAE.train()
            self.optimizer_vae.zero_grad()
            
            # 前向传播
            if self.config.vae.name == "normal_vae":
                feat_pred, A_pred = self.VGAE(features)
            elif self.config.vae.name == "vae_class":
                feat_pred, A_pred, class_pred = self.VGAE(features)
            elif self.config.vae.name == "vae_class_v2":
                feat_pred, A_pred, class_pred = self.VGAE(features)
            else:
                raise ValueError(f"Invalid vae name: {self.config.vae.name}")
            
            # 特征重构损失
            feat_loss = F.mse_loss(feat_pred[self.train_mask_vae], features.to_dense()[self.train_mask_vae])
            
            # 链接预测损失（根据图大小选择策略）
            if is_large_graph:
                # 每个epoch重新采样负边
                neg_edges = self._sample_negative_edges_gpu(
                    num_nodes, 
                    self.pos_edges, 
                    num_neg
                )
                link_loss = self._compute_loss_large_graph(A_pred, self.pos_edges, neg_edges)
            else:
                link_loss = self._compute_loss_small_graph(A_pred, adj_label, weight_tensor, norm)
            
            kl_loss = -0.5 / A_pred.size(0) * (1 + 2 * self.VGAE.log_std - self.VGAE.mean**2 - torch.exp(self.VGAE.log_std)**2).sum(1).mean()
            
            # 总损失
            loss = (self.config.vae.coef_link * link_loss + 
                   self.config.vae.coef_feat * feat_loss + 
                   self.config.vae.coef_kl * kl_loss)
            
            if self.config.vae.name == "vae_class":
                class_loss = F.cross_entropy(class_pred[self.train_mask_vae], self.labels[self.train_mask_vae])
                loss += class_loss
            elif self.config.vae.name == "vae_class_v2":
                class_loss = F.cross_entropy(class_pred[self.train_mask_vae], self.labels[self.train_mask_vae])
                loss += class_loss
            else:
                class_loss = torch.tensor(0.0, device=self.device)
            
            # 反向传播
            loss.backward()
            self.optimizer_vae.step()
            
            # 验证（使用采样的验证集）
            with torch.no_grad():
                if is_large_graph:
                    # 大图模式：只在采样边上计算准确率
                    train_acc = 0.0  # 大图模式下跳过全矩阵准确率计算
                else:
                    train_acc = get_acc(A_pred, adj_label)
                
                val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred, adj_orig)
            
            # 打印日志
            if epoch%10 == 0:
                if is_large_graph:
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"class_loss: {class_loss.item():.5f} | "
                        f"kl_loss: {kl_loss.item():.5f} | val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
                else:
                    print(f"VAE training: epoch {epoch} | train_loss: {loss.item():.5f} | "
                        f"feat_loss: {feat_loss.item():.5f} | link_loss: {link_loss.item():.5f} | "
                        f"class_loss: {class_loss.item():.5f} | "
                        f"kl_loss: {kl_loss.item():.5f} | train_acc: {train_acc:.5f} | "
                        f"val_roc: {val_roc:.5f} | val_ap: {val_ap:.5f}")
            
            # 早停机制
            if val_roc + val_ap > best_score:
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



    