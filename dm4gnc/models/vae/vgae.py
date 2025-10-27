import torch
import torch.nn as nn
from .encoder import GraphEncoder
from .decoder import GraphDecoder

class VGAE(nn.modele):
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

    def encode(self, x, sample=True):
        return self.encoder(x, sample=sample)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, sample=True):
        z = self.encode(x, sample=sample)
        feat, A_pred = self.decode(z)
        return feat, A_pred