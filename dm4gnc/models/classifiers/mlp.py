import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    MLP分类器
    用于对节点嵌入进行分类
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = False
    ):
        super(MLPClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
            ])
        else:
            self.bns = None
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            if self.bns is not None:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
