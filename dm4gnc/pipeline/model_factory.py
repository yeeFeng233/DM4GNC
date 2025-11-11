import torch 
from ..models import VGAE, VGAE_class, VGAE_class_v2, VGAE_DEC, VGAE_DEC_class, VGAE_SIG

def _init_VGAE(config):
        device = torch.device(config.device)
        if config.vae.name == "normal_vae":
            vgae = VGAE(feat_dim=config.feat_dim,
                            hidden_dim=config.vae.hidden_sizes[0],
                            latent_dim=config.vae.hidden_sizes[1],
                            adj=None).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        elif config.vae.name == "vae_class":
            vgae = VGAE_class(feat_dim=config.feat_dim,
                                hidden_dim=config.vae.hidden_sizes[0],
                                latent_dim=config.vae.hidden_sizes[1],
                                adj=None,
                                num_classes = config.num_classes).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        elif config.vae.name == "vae_class_v2":
            vgae = VGAE_class_v2(feat_dim=config.feat_dim,
                                hidden_dim=config.vae.hidden_sizes[0],
                                latent_dim=config.vae.hidden_sizes[1],
                                adj=None,
                                num_classes = config.num_classes).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        elif config.vae.name == "vae_dec":
            vgae = VGAE_DEC(feat_dim=config.feat_dim,
                                hidden_dim=config.vae.hidden_sizes[0],
                                latent_dim=config.vae.hidden_sizes[1],
                                adj=None,
                                n_clusters = config.num_classes).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        elif config.vae.name == "vae_dec_class":
            vgae = VGAE_DEC_class(feat_dim=config.feat_dim,
                                hidden_dim=config.vae.hidden_sizes[0],
                                latent_dim=config.vae.hidden_sizes[1],
                                adj=None,
                                n_clusters = config.num_classes).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        elif config.vae.name == "vae_sig":
            vgae = VGAE_SIG(feat_dim=config.feat_dim,
                                hidden_dim=config.vae.hidden_sizes[0],
                                latent_dim=config.vae.hidden_sizes[1],
                                adj=None).to(device)
            optimizer_vae = torch.optim.Adam(vgae.parameters(), 
                                                lr=config.vae.lr)
        else:
            raise ValueError(f"Invalid vae name: {config.vae.name}")
        
        return vgae, optimizer_vae
