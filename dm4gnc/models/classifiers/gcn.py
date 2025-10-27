import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_node_sparse(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layer, dropout=0.5, batch_norm=False):
        super(GCN_node_sparse, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNConv(n_feat, n_hidden))

        for _ in range(n_layer - 2):
            self.graph_encoders.append(GCNConv(n_hidden, n_hidden))

        self.graph_encoders.append(GCNConv(n_hidden, n_class))

        if batch_norm:
            self.bn = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layer - 1)])
        else:
            self.bn = None

    def forward(self, x, edge_index, edge_weight=None):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = encoder(x, edge_index, edge_weight)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        for conv in self.graph_encoders:
            conv.reset_parameters()
        if self.bn is not None:
            for b in self.bn:
                b.reset_parameters()