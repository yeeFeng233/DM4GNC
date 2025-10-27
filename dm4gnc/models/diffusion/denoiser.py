import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import SinusoidalTimestepEmbedding, FiLMBlock


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        x_dim: int = 128,
        emb_dim: int = 64,
        hidden_dim: int = 1024,
        num_classes: int = 7,
        layers: int = 5,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.x_dim = x_dim
        self.dtype = dtype
        act = nn.SiLU()

        # condition network
        self.time_proj = nn.Sequential(
            SinusoidalTimestepEmbedding(dim=emb_dim),
            nn.Linear(emb_dim, emb_dim), 
            act,
        )

        self.label_proj = nn.Sequential(
            nn.Linear(num_classes, emb_dim), 
            # act,
        )
        cond_dim = emb_dim * 2
        
        # main network
        self.initial_proj = nn.Linear(x_dim, hidden_dim)

        self.backbone = nn.ModuleList(
            FiLMBlock(hidden_dim, hidden_dim, cond_dim) for _ in range(layers)
        )

        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, x_dim),
        )     

    def forward(self, x: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        need_unsqueeze = False
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            need_unsqueeze = True
        x = x.to(self.dtype)

        t_emb = self.time_proj(t)
        y_emb = self.label_proj(cemb)
        cond = torch.cat([t_emb, y_emb], dim=-1)

        x = self.initial_proj(x)
        for blocks in self.backbone:
            x = blocks(x, cond)
        out = self.final_proj(x)

        if need_unsqueeze:
            out = out.unsqueeze(1)
        return out
