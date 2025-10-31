from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc

from ..base_stage import BaseStage
from ...models import GCN_node_sparse
from ...utils import adj2edgeindex
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

class ClassifierTrainStage(BaseStage):
    def __init__(self, config, dataset, logger=None):
        super().__init__(config, dataset, logger=logger)
        self.train_index = torch.nonzero(dataset.train_mask,as_tuple=True)[0]
        self.val_index = torch.nonzero(dataset.val_mask,as_tuple=True)[0]
        self.test_index = torch.nonzero(dataset.test_mask,as_tuple=True)[0]

        self.features = dataset.x.to(self.device)
        self.labels = dataset.y.to(self.device)
        self.adj = dataset.adj.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)

        self.classifier = GCN_node_sparse(n_feat = self.config.feat_dim,
                                n_hidden = self.config.classifier.hidden_dim,
                                n_class = self.config.num_classes,
                                n_layer = self.config.classifier.n_layer,
                                dropout = self.config.classifier.dropout).to(self.device)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), 
                                                    lr=self.config.classifier.lr,
                                                    weight_decay=self.config.classifier.weight_decay)


    def _get_checkpoints_load_path(self):
        self.checkpoint_load_path = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}", 'checkpoint_vae_decode.pth')
    
    def _get_checkpoints_save_path(self):
        self.checkpoint_save_path = os.path.join(self.checkpoints_root, 'checkpoint_classifier_train.pth')
        
    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoint_load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_load_path}")
        checkpoint = torch.load(self.checkpoint_load_path)
        self.aug_adj = checkpoint['aug_adj'].to(self.device)
        self.aug_feats = checkpoint['aug_feats'].to(self.device)
        self.aug_labels = checkpoint['aug_labels'].to(self.device)
        self.aug_train_index = checkpoint['aug_train_index'].to(self.device)
    
    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': 'classifier_train',
            'classifier_state': self.classifier.state_dict(),
            'optimizer_classifier_state': self.optimizer_classifier.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_save_path)
        print(f"Stage: classifier_train | Checkpoint saved: {self.checkpoint_save_path}")

    def run(self):
        self._load_checkpoints()

        self.aug_adj = (self.aug_adj > self.config.vae.threshold).int().to(self.device)
        self.best_acc = 0
        self.best_epoch = 0
        self.best_classifier_state = None
        self.best_optimizer_classifier_state = None
        self.aug_edge_index = adj2edgeindex(self.aug_adj)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.config.classifier.epoch):
            self.classifier.train()
            self.optimizer_classifier.zero_grad()
            out = self.classifier(self.aug_feats, self.aug_edge_index)
            loss = criterion(out[self.aug_train_index], self.aug_labels[self.aug_train_index])
            loss.backward()
            self.optimizer_classifier.step()

            if epoch % 10 == 0:
                val_acc = self.classifier_eval(metric="accuracy")
                print(f"Classifier training: epoch {epoch} | loss: {loss.item():.5f} | acc_val: {val_acc:.5f} ")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.best_classifier_state = {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()}
                self.best_optimizer_classifier_state = {k: v.cpu().clone() if torch.is_tensor(v) else v for k, v in self.optimizer_classifier.state_dict().items()}
                
            if epoch - self.best_epoch > self.config.classifier.patience:
                print(f"Classifier training: early stopping at epoch {epoch}")
                break

        print(f"Classifier training finished: best_val_acc={self.best_acc:.4f} at epoch {self.best_epoch}")
        
        # 记录关键指标到logger
        self.log_metrics({
            'best_val_accuracy': self.best_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch
        })

        self._save_checkpoints()
        self._empty_memory()

    def _empty_memory(self):
        if hasattr(self, 'classifier'):
            del self.classifier
        if hasattr(self, 'optimizer_classifier'):
            del self.optimizer_classifier
        if hasattr(self, 'aug_adj'):
            del self.aug_adj
        if hasattr(self, 'aug_feats'):
            del self.aug_feats
        if hasattr(self, 'aug_labels'):
            del self.aug_labels
        if hasattr(self, 'aug_train_index'):
            del self.aug_train_index
        if hasattr(self, 'aug_edge_index'):
            del self.aug_edge_index
        if hasattr(self, 'best_classifier_state'):
            del self.best_classifier_state
        if hasattr(self, 'best_optimizer_classifier_state'):
            del self.best_optimizer_classifier_state
        gc.collect()
        torch.cuda.empty_cache()

    def classifier_eval(self, metric="accuracy"):
        """ Evaluate the model on the validation or test set using the selected metric. """
        self.classifier.eval()
        val_labels = self.labels[self.val_index].cpu().numpy()

        with torch.no_grad():
            out = self.classifier(self.aug_feats, self.aug_edge_index)
            predictions = out[self.val_index].argmax(dim=1).cpu().numpy()

        if metric == "accuracy":
            return accuracy_score(val_labels, predictions)
        elif metric == "bacc":
            return balanced_accuracy_score(val_labels, predictions)
        elif metric == "macro_f1":
            return f1_score(val_labels, predictions, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")