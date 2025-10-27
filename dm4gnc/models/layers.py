
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter

# Graph Convolution Layers
class GraphConvolution(nn.Module):
    """
    For dense adjacency matrix
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    """
    For sparse adjacency matrix
    Used in VAE
    """
    def __init__(self, input_dim, output_dim, adj, activation=F.relu):
        super(GraphConvSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = torch.mm(inputs, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


# Time Embedding Layers
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# FiLM Layers
class FiLMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_dim):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.film_generator = nn.Linear(cond_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.should_project = input_dim != hidden_dim
        
        if self.should_project:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, cond):
        residual = self.residual_proj(x) if self.should_project else x
        x_processed = self.main_path(self.norm(x))
        gamma, beta = self.film_generator(cond).chunk(2, dim=-1)
        output = x_processed * gamma + beta + residual
        
        return output

# Attention
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)
