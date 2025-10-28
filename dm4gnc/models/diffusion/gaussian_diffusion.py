import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        model: nn.Module,
        betas: np.ndarray,
        w: float,
        v: float,
        device: torch.device,
        config = None
    ):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas, dtype=self.dtype)
        self.w = w  # CFG strength
        self.v = v  # variance interpolation factor
        self.T = len(betas)
        self.device = device
        self.config = config

        self._precompute_coefficients()

    def _precompute_coefficients(self):
        # basic coefficients
        self.alphas = 1. - self.betas
        self.log_alphas = torch.log(self.alphas)
        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim=0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)
        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1], [1,0], 'constant', value=0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # coefficients for q(x_t|x_0)
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)
        # coefficients for q(x_{t-1}|x_t,x_0)
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = self.tilde_betas
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar

        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar-self.log_sqrt_alphas_bar)

    @staticmethod
    def _extract(coef:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        coef = coef.to(t.device)
        chosen = coef[t]

        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor) :
        """
        calculate the parameters of q(x_t|x_0)
        """
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var
    
    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor):
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps, eps
    
    def q_posterior_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor):
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max, neo_posterior_var

    def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs):
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        
        # Standard CFG: eps = (1-w)*eps_uncond + w*eps_cond
        if hasattr(self, 'w') and self.w > 0 and ('cemb' in model_kwargs) and (model_kwargs['cemb'] is not None):
            cemb_cond = model_kwargs['cemb']
            cemb_uncond = torch.zeros_like(cemb_cond)
            pred_eps_cond = self.model(x_t, t, cemb=cemb_cond)
            pred_eps_uncond = self.model(x_t, t, cemb=cemb_uncond)
            # print("L2 diff:", (pred_eps_cond - pred_eps_uncond).norm(dim=1).mean().item())

            # pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
            pred_eps = pred_eps_uncond + self.w * (pred_eps_cond - pred_eps_uncond)
        else:
            pred_eps = self.model(x_t, t, **model_kwargs)
    
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.sqrt_recip_alphas_bar, t = t, x_shape = x_t.shape) \
            * (x_t - self._extract(coef = self.sqrt_one_minus_alphas_bar, t = t, x_shape = x_t.shape) * eps)

    def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        return self._extract(coef = self.coef1, t = t, x_shape = x_t.shape) * x_t - \
            self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def sample(self, shape:tuple, **model_kwargs) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        # print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T

        # for _ in tqdm(range(self.T), dynamic_ncols=True, desc="sampling: "):
        for _ in range(self.T):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        # x_t = torch.clamp(x_t, -1, 1)
        # print('ending sampling process...')
        return x_t

    def trainloss(self, x_0:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)

        pred_eps = self.model(x_t, t, **model_kwargs)
        if pred_eps.shape != eps.shape:
            eps = eps.reshape(pred_eps.shape)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        # loss = nn.MSELoss(pred_eps, eps)
        return loss

    def reconstruct_latents(self, x_0: torch.Tensor, steps = 999, **model_kwargs) -> torch.Tensor:

        # local_rank = get_rank()
        local_rank = 0
        if local_rank == 0:
            print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.ones([x_0.shape[0]], device = self.device, dtype=torch.int) * steps

        x_t, eps = self.q_sample(x_0, t)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * steps
        _denom = max(1, torch.cuda.device_count())
        _disable_bar = (local_rank % _denom != 0)
        for _ in tqdm(range(steps), dynamic_ncols=True, disable=_disable_bar):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        # x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process...')
        return x_t

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        
        # 移动所有预计算的系数
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and not attr_name.startswith('_'):
                setattr(self, attr_name, attr.to(device))
        
        return self
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  T={self.T},\n"
            f"  w={self.w},\n"
            f"  device={self.device},\n"
            f"  model={self.model.__class__.__name__}\n"
            f")"
        )

