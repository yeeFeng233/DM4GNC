import torch
import torch.nn as nn
from .encoder import GraphEncoder, GraphEncoder_class
from .decoder import GraphDecoder, GraphDecoder_class

class VGAE(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj):
        super(VGAE, self).__init__()
        self.adj = adj
        self.encoder = GraphEncoder(feat_dim, hidden_dim, latent_dim, adj)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, feat_dim)

        self.mean = None
        self.log_std = None

    def reset_adj(self, adj):
        self.adj = adj
        self.encoder.reset_adj(adj)

    def set_device(self,device):
        self.adj = self.adj.to(device)
        self.encoder.set_device(device)
        self.decoder.set_device(device)

    def encode(self, x):
        z = self.encoder(x)
        self.mean, self.log_std = self.encoder.mean, self.encoder.log_std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        feat, A_pred = self.decode(z)
        return feat, A_pred


class VGAE_class(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj, num_classes):
        super(VGAE_class, self).__init__()
        self.adj = adj
        self.encoder = GraphEncoder(feat_dim, hidden_dim, latent_dim, adj)
        self.decoder = GraphDecoder_class(latent_dim, hidden_dim, feat_dim, num_classes)

        self.mean = None
        self.log_std = None

    def reset_adj(self, adj):
        self.adj = adj
        self.encoder.reset_adj(adj)

    def set_device(self,device):
        self.adj = self.adj.to(device)
        self.encoder.set_device(device)
        self.decoder.set_device(device)

    def encode(self, x,):
        z = self.encoder(x)
        self.mean, self.log_std = self.encoder.mean, self.encoder.log_std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)

        feat, A_pred, class_pred = self.decode(z)
        return feat, A_pred, class_pred

class VGAE_class_v2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj, num_classes):
        super(VGAE_class_v2, self).__init__()
        self.adj = adj
        self.encoder = GraphEncoder_class(feat_dim, hidden_dim, latent_dim, adj, num_classes)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, feat_dim)

        self.mean = None
        self.log_std = None

    def reset_adj(self, adj):
        self.adj = adj
        self.encoder.reset_adj(adj)

    def set_device(self,device):
        self.adj = self.adj.to(device)
        self.encoder.set_device(device)
        self.decoder.set_device(device)

    def encode(self, x,):
        z, class_pred = self.encoder(x)
        self.mean, self.log_std = self.encoder.mean, self.encoder.log_std
        return z, class_pred
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, class_pred = self.encode(x)

        feat, A_pred = self.decode(z)
        return feat, A_pred, class_pred