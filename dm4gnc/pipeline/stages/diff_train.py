import gc
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from ..base_stage import BaseStage
from ...models import MLPDenoiser, GaussianDiffusion, get_named_beta_schedule, GradualWarmupScheduler

class DiffTrainStage(BaseStage):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_index = torch.nonzero(dataset.train_mask, as_tuple=True)[0].to(self.device)
        self.val_index = torch.nonzero(dataset.val_mask, as_tuple=True)[0].to(self.device)
        self.test_index = torch.nonzero(dataset.test_mask, as_tuple=True)[0].to(self.device)

        self.mlpdenoiser = MLPDenoiser(x_dim=self.config.vae.hidden_sizes[-1],
                                    emb_dim = self.config.diffusion.cdim,
                                    hidden_dim = self.config.diffusion.hidden_dim,
                                    num_classes = self.config.num_classes,
                                    layers = self.config.diffusion.layers,
                                    dtype = torch.float32).to(self.device)

        self.betas = get_named_beta_schedule(schedule_name=self.config.diffusion.schedule_name, num_diffusion_timesteps=self.config.diffusion.T)
        self.diffusion = GaussianDiffusion(dtype = self.config.dtype,
                                    model=self.mlpdenoiser,
                                    betas = self.betas,
                                    w = self.config.diffusion.w,
                                    v = self.config.diffusion.v,
                                    device = self.device,
                                    config = self.config)
        self.optimizer_diffusion = torch.optim.Adam(self.diffusion.model.parameters(), 
                                                    lr=self.config.diffusion.lr,
                                                    weight_decay=self.config.diffusion.weight_decay)
        self.cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    optimizer = self.optimizer_diffusion, 
                                                    T_max=self.config.diffusion.epoch, 
                                                    eta_min=0,
                                                    last_epoch=-1)
        self.warmUpScheduler = GradualWarmupScheduler(optimizer = self.optimizer_diffusion, 
                                                    multiplier = self.config.diffusion.multiplier, 
                                                    warm_epoch = max(1, int(self.config.diffusion.epoch * 0.1)), 
                                                    after_scheduler = self.cosineScheduler,
                                                    last_epoch = 0)
        

    def _get_checkpoints_load_path(self):
        self.checkpoints_load_path = os.path.join(self.checkpoints_root, 'checkpoint_vae_encode.pth')
    
    def _get_checkpoints_save_path(self):
        self.checkpoints_save_path = os.path.join(self.checkpoints_root, 'checkpoint_diff_train.pth')
        
    def _load_checkpoints(self):
        self._get_checkpoints_load_path()
        if not os.path.exists(self.checkpoints_load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoints_load_path}")
        checkpoint = torch.load(self.checkpoints_load_path, weights_only=True, map_location=self.device)
        self.latents = checkpoint['latents']
        self.labels = checkpoint['labels']

    
    def _save_checkpoints(self):
        self._get_checkpoints_save_path()
        checkpoint = {
            'stage': "vae_encode",
            'diffusion_model_state': self.best_diffusion_state,
            'optimizer_diffusion_state': self.best_optimizer_diffusion_state,
        }
        torch.save(checkpoint, self.checkpoints_save_path)
        print(f"stage: diff_train | Checkpoint saved: {self.checkpoints_save_path}")

    def run(self):
        self._load_checkpoints()

        labels_one_hot = labels_one_hot = torch.nn.functional.one_hot(self.labels, num_classes=self.config.num_classes).float().to(self.device)
        diff_dataset = Diffusion_Emb_Dataset(self.latents, labels_one_hot, self.device)
        train_loader = DataLoader(diff_dataset, 
                                 batch_size=self.config.diffusion.batch_size, 
                                 sampler = None,
                                 shuffle=True,
                                 num_workers=0,)
        val_loader = DataLoader(diff_dataset, 
                            batch_size=self.config.diffusion.batch_size, 
                            sampler = SubsetRandomSampler(self.val_index.tolist()),
                            num_workers=0,)

        self.best_loss = float('inf')
        self.best_diffusion_state = None
        self.best_optimizer_diffusion_state = None
        self.best_epoch = 0

        for epc in range(self.config.diffusion.epoch):
            # train
            self.diffusion.model.train()
            total_loss = .0
            
            for x_0, label, idx in train_loader:
                self.optimizer_diffusion.zero_grad()
                idx = idx.to(self.device)
                
                mask_train = torch.isin(idx, self.train_index)
                label[~mask_train] = torch.zeros_like(label[~mask_train])
                
                loss = self.diffusion.trainloss(x_0, cemb=label)
                loss.backward()
                self.optimizer_diffusion.step()
                total_loss += loss.item()
            
            total_loss /= len(train_loader)
            self.warmUpScheduler.step()
            
            # val
            if epc % 10 == 0 or epc == self.config.diffusion.epoch - 1:
                self.diffusion.model.eval()
                total_val_loss = 0
                for x_0_val, label_val, _ in val_loader:

                    with torch.no_grad():
                        loss_val = self.diffusion.trainloss(x_0_val, cemb=label_val)
                    total_val_loss += loss_val.item()
                total_val_loss /= len(val_loader)

                print(f"Diffusion training: epoch {epc} | train_loss: {total_loss:.5f} | val_loss: {total_val_loss:.5f}")
                
                if total_val_loss < self.best_loss:
                    self.best_loss = total_val_loss
                    self.best_epoch = epc
                    self.best_diffusion_state = {k: v.cpu().clone() for k, v in self.diffusion.model.state_dict().items()}
                    self.best_optimizer_diffusion_state = {k: v.cpu().clone() if torch.is_tensor(v) else v for k, v in self.optimizer_diffusion.state_dict().items()}
                    
                else:
                    if epc - self.best_epoch >= self.config.diffusion.patience:
                        print("Early stopping at epoch: ", epc)
                        break
        self._save_checkpoints()
        self._empty_memory()

    def _empty_memory(self):
        if hasattr(self, 'mlpdenoiser'):
            del self.mlpdenoiser
        if hasattr(self, 'diffusion'):
            del self.diffusion
        if hasattr(self, 'optimizer_diffusion'):
            del self.optimizer_diffusion
        if hasattr(self, 'cosineScheduler'):
            del self.cosineScheduler
        if hasattr(self, 'warmUpScheduler'):
            del self.warmUpScheduler
        if hasattr(self, 'best_diffusion_state'):
            del self.best_diffusion_state
        if hasattr(self, 'best_optimizer_diffusion_state'):
            del self.best_optimizer_diffusion_state
        if hasattr(self, 'best_epoch'):
            del self.best_epoch
        if hasattr(self, 'best_loss'):
            del self.best_loss
        if hasattr(self, 'train_index'):
            del self.train_index
        if hasattr(self, 'val_index'):
            del self.val_index
        if hasattr(self, 'test_index'):
            del self.test_index
        if hasattr(self, 'latents'):
            del self.latents
        if hasattr(self, 'labels'):
            del self.labels
        gc.collect()
        torch.cuda.empty_cache()

class Diffusion_Emb_Dataset(Dataset):
    def __init__(self, emb, labels, device):
        self.emb = emb.to(device)       
        self.max = self.emb.max()
        self.min = self.emb.min()
        self.labels = labels.to(device)
        B, D = self.emb.shape
        self.emb = self.emb.unsqueeze(1)
        # self.labels = self.labels.squeeze(1)
        # if len(self.labels.shape) > 1:
        #     self.labels = self.labels.argmax(dim=1)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        emb = self.emb[idx]
        label = self.labels[idx]

        return emb, label, idx
    
    def transback(self, data):
        return data*(self.max - self.min) + self.min