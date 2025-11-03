from abc import ABC, abstractmethod
import torch
import os

class BaseStage(ABC):
    def __init__(self, config, dataset = None, logger=None):
        self.config = config
        if dataset is not None:
            self.dataset = dataset
        self.device = torch.device(config.device)
        self._check_checkpoints_root()

        self.logger = logger
        self.stage_name = self.__class__.__name__.replace('Stage', '').replace('VAE', 'vae_').replace('Diff', 'diff_').replace('Classifier', 'classifier_').lower()
        self.stage_name = self.stage_name.replace('train', 'train').replace('encode', 'encode').replace('decode', 'decode').replace('sample', 'sample').replace('test', 'test')

    def _check_checkpoints_root(self):
        self.checkpoints_root = os.path.join(self.config.output_dir, 'checkpoints', self.config.dataset)
        if not os.path.exists(self.checkpoints_root):
            os.makedirs(self.checkpoints_root)
    
    def log_metrics(self, metrics: dict):
        if self.logger is not None:
            self.logger.log_stage_metrics(self.stage_name, metrics)
    
    def log_image(self, image_name: str, image_path: str):

        if self.logger is not None:
            self.logger.log_image_path(self.stage_name, image_name, image_path)

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