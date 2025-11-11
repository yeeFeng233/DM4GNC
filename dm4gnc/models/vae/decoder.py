import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feat_dim):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        
        self.feat_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.feat_decode2 = nn.Linear(hidden_dim, feat_dim)
    
    def set_device(self,device):
        self.feat_decode1.to(device)
        self.feat_decode2.to(device)
    
    def forward(self, z):
        feat = F.relu(self.feat_decode1(z))
        feat = self.feat_decode2(feat)
        
        A_pred = dot_product_decode(z)
        
        return feat, A_pred

class GraphDecoder_class(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feat_dim, num_classes):
        super(GraphDecoder_class, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        
        self.feat_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.feat_decode2 = nn.Linear(hidden_dim, feat_dim)

        self.class_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.class_decode2 = nn.Linear(hidden_dim, num_classes)
    
    def set_device(self,device):
        self.feat_decode1.to(device)
        self.feat_decode2.to(device)

        self.class_decode1.to(device)
        self.class_decode2.to(device)
    
    def forward(self, z):
        feat = F.relu(self.feat_decode1(z))
        feat = self.feat_decode2(feat)
        
        A_pred = dot_product_decode(z)

        class_pred = F.relu(self.class_decode1(z))
        class_pred = self.class_decode2(class_pred)

        return feat, A_pred, class_pred

class SparseDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feat_dim):
        super(SparseDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim

        init_r = torch.empty(latent_dim).uniform_(-6.0, 0.0)
        self.rk = nn.Parameter(torch.sigmoid(init_r))
        
        self.feat_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.feat_decode2 = nn.Linear(hidden_dim, feat_dim)

    def set_device(self,device):
        self.feat_decode1.to(device)
        self.feat_decode2.to(device)
        self.rk.to(device)

    def forward(self, z):
        x = z @ torch.diag(self.rk) @ z.t()
        x = torch.clamp(x, min=-20.0, max=20.0)
        A_pred = 1.0 - torch.exp(-torch.exp(x))
        A_pred = torch.clamp(A_pred, min=1e-10, max=1.0 - 1e-10)

        feat = F.relu(self.feat_decode1(z))
        feat = self.feat_decode2(feat)

        return feat, A_pred