from abc import ABC, abstractmethod
import torch
import os

class BaseStage(ABC):
    def __init__(self, config, dataset = None):
        self.config = config
        if dataset is not None:
            self.dataset = dataset
        self.device = torch.device(config.device)
        self._check_checkpoints_root()

    def _check_checkpoints_root(self):
        self.checkpoints_root = os.path.join(self.config.output_dir, 'checkpoints', self.config.dataset)
        if not os.path.exists(self.checkpoints_root):
            os.makedirs(self.checkpoints_root)

    @abstractmethod
    def _get_checkpoints_load_path(self):
        pass
    
    @abstractmethod
    def _get_checkpoints_save_path(self):
        pass
        
    @abstractmethod
    def _load_checkpoints(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _save_checkpoints(self):
        pass