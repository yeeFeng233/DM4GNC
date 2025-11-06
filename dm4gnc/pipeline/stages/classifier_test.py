from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
import gc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from ..base_stage import BaseStage
from ...models import GCN_node_sparse

class ClassifierTestStage(BaseStage):
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
        if self.config.diffusion.filter:
                self.checkpoint_load_path = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}",f"{self.config.diffusion.filter_strategy}", "checkpoint_classifier_train.pth")
        else:
            self.checkpoint_load_path = os.path.join(self.checkpoints_root, f"{self.config.diffusion.generate_ratio}", "checkpoint_classifier_train.pth")
    
    def _get_checkpoints_save_path(self):
        return None
        
    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoint_load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_load_path}")
        checkpoint = torch.load(self.checkpoint_load_path)
        self.classifier.load_state_dict(checkpoint['classifier_state'])
        self.optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state'])
    
    def _save_checkpoints(self):
        return None
    
    def _empty_memory(self):
        if hasattr(self, 'classifier'):
            del self.classifier
        if hasattr(self, 'optimizer_classifier'):
            del self.optimizer_classifier
        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        self._load_checkpoints()

        test_labels = self.labels[self.test_index].cpu().numpy()

        self.classifier.eval()
        with torch.no_grad():
            out = self.classifier(self.features, self.edge_index)
            predictions = out[self.test_index].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(out[self.test_index], dim=1).cpu().numpy()

        accuracy = accuracy_score(test_labels, predictions)
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        bacc = balanced_accuracy_score(test_labels, predictions)
        auc_roc = roc_auc_score(test_labels, probabilities, multi_class='ovr', average='macro')
        
        print(f"Test Results: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}, bacc={bacc:.4f}, auc_roc={auc_roc:.4f}")
        
        self.log_metrics({
            'test_accuracy': accuracy,
            'test_macro_f1': macro_f1,
            'test_balanced_accuracy': bacc,
            'test_auc_roc': auc_roc
        })

        self._empty_memory()

