import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import GraphConvSparse

class GraphEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj):
        super(GraphEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.adj = adj

        self.base_gcn = GraphConvSparse(feat_dim, hidden_dim, adj, activation=F.relu)
        
        self.gcn_mean = GraphConvSparse(hidden_dim, latent_dim, adj, activation=lambda x: x)
        self.gcn_logstd = GraphConvSparse(hidden_dim, latent_dim, adj, activation=lambda x: x)
        
        self.mean = None
        self.log_std = None

    def reset_adj(self, adj):
        self.adj = adj
        self.base_gcn.adj = adj
        self.gcn_mean.adj = adj
        self.gcn_logstd.adj = adj

    def set_device(self,device):
        self.adj = self.adj.to(device)
        self.base_gcn.adj = self.base_gcn.adj.to(device)
        self.gcn_mean.adj = self.gcn_mean.adj.to(device)
        self.gcn_logstd.adj = self.gcn_logstd.adj.to(device)    

    def forward(self, x):
        hidden = self.base_gcn(x)
        self.mean = self.gcn_mean(hidden)
        self.log_std = self.gcn_logstd(hidden)

        gaussian_noise = torch.randn(x.size(0), self.latent_dim, device = x.device)
        z = self.mean + torch.exp(self.log_std) * gaussian_noise
        return z


class GraphEncoder_class(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj, num_classes):
        super(GraphEncoder_class, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.adj = adj
        self.num_classes = num_classes

        self.base_gcn = GraphConvSparse(feat_dim, hidden_dim, adj, activation=F.relu)
        
        self.gcn_mean = GraphConvSparse(hidden_dim, latent_dim, adj, activation=lambda x: x)
        self.gcn_logstd = GraphConvSparse(hidden_dim, latent_dim, adj, activation=lambda x: x)

        self.gcn_class1 = GraphConvSparse(hidden_dim, hidden_dim//2, adj, activation=F.relu)
        self.gcn_class2 = GraphConvSparse(hidden_dim//2, num_classes, adj, activation=F.relu)
        
        self.mean = None
        self.log_std = None

    def reset_adj(self, adj):
        self.adj = adj
        self.base_gcn.adj = adj
        self.gcn_mean.adj = adj
        self.gcn_logstd.adj = adj

        self.gcn_class1.adj = adj
        self.gcn_class2.adj = adj

    def set_device(self,device):
        self.adj = self.adj.to(device)
        self.base_gcn.adj = self.base_gcn.adj.to(device)
        self.gcn_mean.adj = self.gcn_mean.adj.to(device)
        self.gcn_logstd.adj = self.gcn_logstd.adj.to(device)    

    def forward(self, x):
        hidden = self.base_gcn(x)
        self.mean = self.gcn_mean(hidden)
        self.log_std = self.gcn_logstd(hidden)

        hidden_class1 = self.gcn_class1(hidden)
        hidden_class2 = self.gcn_class2(hidden_class1)
        class_pred = hidden_class2

        gaussian_noise = torch.randn(x.size(0), self.latent_dim, device = x.device)
        z = self.mean + torch.exp(self.log_std) * gaussian_noise
        return z, class_pred