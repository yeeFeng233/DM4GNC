from compileall import compile_file
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import torch


@dataclass
class VAEConfig:
    name: str = "normal_vae"
    weight_decay: float = 0.0005
    lr: float = 0.01
    dropout: float = 0.5

    epoch: int = 1500
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    coef_kl: float = 1.0
    coef_feat: float = 1.0
    coef_link: float = 1.0
    neighbor_map_dim: int = 2708
    shuffle: bool = True
    neg_ratio: int = 20
    patience: int = 100
    threshold: float = 0.9

    add_kl_loss: bool = True


@dataclass
class DiffusionConfig:
    lr: float = 0.00001
    weight_decay: float = 0.0005
    epoch: int = 1500
    batch_size: int = 512
    patience: int = 500
    multiplier: float = 2.5

    T: int = 1000
    cdim: int = 128
    hidden_dim: int = 1024
    layers: int = 6
    droprate: float = 0.1
    schedule_name: str = "linear"

    w: float = 2.5
    v: float = 0.3
    genbatch: int = 256
    generate_ratio: float = 0.1
    dropout_condition: float = 0.1
    step: int = 10
    filter: bool = False
    filter_strategy: str = "topk"
    
    # Distance filter strategy parameters
    distance_threshold_factor: float = 0.5  # Factor for threshold: mean + factor * std
    distance_metric: str = "euclidean"  # Distance metric: "euclidean", "manhattan", "cosine"
    distance_batch_multiplier: int = 3  # Initial batch size multiplier (target * multiplier)


@dataclass
class ClassifierConfig:
    n_layer: int = 2
    hidden_dim: int = 128
    lr: float = 0.01
    dropout: float = 0.5
    weight_decay: float = 0.0005
    epoch: int = 500
    patience: int = 50


@dataclass
class Config:
    # basic information
    algorithm: str = 'dm4gnc'
    task: str = 'node'
    dataset: str = 'Cora'
    imb_level: str = 'low'
    device: str = 'cuda:0'
    seed: int = 42
    data_path = 'data'
    dtype: torch.dtype = torch.float32
    
    # pipeline control
    stage_start: str = 'vae_train'
    stage_end: str = 'classifier_test'
    stage_to_visualize: str = 'vae_encode'
    
    # path configuration
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    
    # sub-configuration
    vae: VAEConfig = field(default_factory=VAEConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    num_classes: Optional[int] = None
    feat_dim: Optional[int] = None
    neighbor_map_dim: Optional[int] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """load config from yaml file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # create sub-configuration objects
        if 'vae' in data:
            data['vae'] = VAEConfig(**data['vae'])
        if 'diffusion' in data:
            data['diffusion'] = DiffusionConfig(**data['diffusion'])
        if 'classifier' in data:
            data['classifier'] = ClassifierConfig(**data['classifier'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    def validate(self):
        """validate the configuration"""
        # validate stage order
        all_stages = ['vae_train', 'vae_encode', 'diff_train', 
                     'diff_sample', 'vae_decode', 'filter_samples',
                     'classifier_train', 'classifier_test']
        
        if self.stage_start not in all_stages:
            raise ValueError(f"Invalid stage_start: {self.stage_start}")
        if self.stage_end not in all_stages:
            raise ValueError(f"Invalid stage_end: {self.stage_end}")
        
        start_idx = all_stages.index(self.stage_start)
        end_idx = all_stages.index(self.stage_end)
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid stage order: {self.stage_start} -> {self.stage_end}")
        
        # validate dataset
        valid_datasets = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 
                         'Computers', 'ogbn-arxiv', 'Actor', 
                         'Chameleon', 'Squirrel']
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        
        # validate imbalance level
        valid_levels = ['low', 'mid', 'high']
        if self.imb_level not in valid_levels:
            raise ValueError(f"Invalid imb_level: {self.imb_level}")