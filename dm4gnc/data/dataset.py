import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from torch_geometric.data import Data
import os

from .loader import load_node_data
from .split import split_nodes_by_degree

class Dataset:
    data_dict = {'Cora':'planetoid','CiteSeer':'planetoid','PubMed':'planetoid',
                'Photo':'amazon','Computers':'amazon','Actor':'Actor',
                'Chameleon':'WikipediaNetwork','Squirrel':'WikipediaNetwork','ogbn-arxiv':'ogbn'} 
    def __init__(self, 
        data_path: str, 
        name: str, 
        imb_level: str,
        shuffle_seed: 10):
        
        self.data_path = os.path.join(data_path)
        self.name = name
        self.imb_level = imb_level

        dataset = load_node_data(self.name, self.data_path)
        dataset.data_name = self.name

        dataset = split_nodes_by_degree(dataset, self.imb_level, shuffle_seed)
        self.dataset = dataset

    
    def _check(self):
        if self.name not in self.data_dict:
            raise ValueError(f"Invalid dataset name: {self.name}")
        if self.imb_level not in ['low', 'mid', 'high']:
            raise ValueError(f"Invalid imbalance level: {self.imb_level}")

    def load_dataset(self):
        return self.dataset
