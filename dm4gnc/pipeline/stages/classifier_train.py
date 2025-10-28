from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp

from dm4gnc.pipeline.base_stage import BaseStage

class ClassifierTrainStage(BaseStage):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
    def _get_checkpoints_load_path(self):
        return None
    
    def _get_checkpoints_save_path(self):
        pass
        
    def _load_checkpoints(self):
        pass
    
    def _save_checkpoints(self):
        pass

    def run(self):
        pass
