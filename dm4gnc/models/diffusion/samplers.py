import numpy as np

def get_named_beta_schedule(schedule_name: str='linear', num_diffusion_timesteps=1000) -> np.array:
    if schedule_name == 'linear':
        return linear_beta_schedule(num_diffusion_timesteps)
    elif schedule_name == 'cosine':
        return cosine_beta_schedule(num_diffusion_timesteps)
    elif schedule_name == 'sqrt':
        return sqrt_beta_schedule(num_diffusion_timesteps)
    else:
        raise ValueError(f"Unsupported schedule: {schedule_name}")

def linear_beta_schedule(time_steps):
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(
        beta_start, beta_end, time_steps, dtype=np.float64
    )

def cosine_beta_schedule(time_steps, s=0.008):
    steps = time_steps + 1
    x = np.linspace(0, time_steps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / time_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

def sqrt_beta_schedule(time_steps, beta_start=0.0001, beta_end=0.02):
    return np.sqrt(np.linspace(beta_start ** 2, beta_end ** 2, time_steps, dtype=np.float64))
