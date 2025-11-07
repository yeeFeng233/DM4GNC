import torch
import torch.nn as nn
import torch.nn.functional as F
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


class VGAE_DEC(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, adj, n_clusters=7, alpha=1.0):
        super(VGAE_DEC, self).__init__()
        self.adj = adj
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        self.encoder = GraphEncoder(feat_dim, hidden_dim, latent_dim, adj)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, feat_dim)
        
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        
        self.mean = None
        self.log_std = None
        self._cluster_initialized = False

    def reset_adj(self, adj):
        self.adj = adj
        self.encoder.reset_adj(adj)

    def set_device(self, device):
        self.adj = self.adj.to(device)
        self.encoder.set_device(device)
        self.decoder.set_device(device)

    def init_cluster_centers(self, z):
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z.detach().cpu().numpy())
        
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=z.device)
        self.cluster_centers.data = cluster_centers
        self._cluster_initialized = True
        
        return y_pred

    def soft_assignment(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        weight = q ** 2 / q.sum(dim=0)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p
    
    def clustering_loss(self, z):
        q = self.soft_assignment(z)
        p = self.target_distribution(q.detach())
        
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss, q

    def encode(self, x):
        z = self.encoder(x)
        self.mean, self.log_std = self.encoder.mean, self.encoder.log_std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        feat, A_pred = self.decode(z)
        return feat, A_pred, z