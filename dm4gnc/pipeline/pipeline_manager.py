from .stages import (VAETrainStage, VAEEncodeStage, VAEDecodeStage, 
            DiffTrainStage, DiffSampleStage, 
            ClassifierTrainStage, ClassifierTestStage,
            FilterSamplesStage)
from ..models import (VGAE, GaussianDiffusion, MLPDenoiser, get_named_beta_schedule, 
                        GCN_node_sparse, GradualWarmupScheduler)

import torch
import os

class PipelineManager:
    all_stages = ['vae_train', 'vae_encode', 'diff_train', 'diff_sample', 
                     'vae_decode', 'classifier_train', 'classifier_test']
    stage_classes = {
        'vae_train': VAETrainStage,
        'vae_encode': VAEEncodeStage,
        'diff_train': DiffTrainStage,
        'diff_sample': DiffSampleStage,
        'vae_decode': VAEDecodeStage,
        'filter_samples': FilterSamplesStage,
        'classifier_train': ClassifierTrainStage,
        'classifier_test': ClassifierTestStage,
    }
    def __init__(self, config, dataset):
        self.config = config
        if dataset is not None:
            self.dataset = dataset
        self.device = torch.device(config.device)
        self.stage_start = config.stage_start
        self.stage_end = config.stage_end

        self._check_stages()

        self.features = dataset.x.to(self.device)
        self.labels = dataset.y.to(self.device)
        self.num_classes = self.labels.max().item() + 1
        self.adj = dataset.adj.to(self.device)
        self.edge_index = dataset.edge_index.to(self.device)
        
        self.train_index = torch.nonzero(dataset.train_mask,as_tuple=True)[0]
        self.val_index = torch.nonzero(dataset.val_mask,as_tuple=True)[0]
        self.test_index = torch.nonzero(dataset.test_mask,as_tuple=True)[0]

        self._update_config()
        
    def _check_stages(self):
        if self.stage_start not in self.all_stages:
            raise ValueError(f"Invalid stage: {self.stage_start}")
        if self.stage_end not in self.all_stages:
            raise ValueError(f"Invalid stage: {self.stage_end}")
        start_idx = self.all_stages.index(self.stage_start)
        end_idx = self.all_stages.index(self.stage_end)
        if start_idx > end_idx:
            raise ValueError(f"Invalid stage order: {self.stage_start} -> {self.stage_end}")


    def _update_config(self):
        self.config.num_classes = self.num_classes
        self.config.feat_dim = self.features.shape[1]
        self.config.neighbor_map_dim = self.adj.shape[0]
    
    def _init_models(self):
        self.VGAE = VGAE(feat_dim=self.config.feat_dim,
                        hidden_dim=self.config.hidden_sizes[0],
                        latent_dim=self.config.hidden_sizes[1],
                        adj=self.adj).to(self.device)
        self.optimizer_vae = torch.optim.Adam(self.VGAE.parameters(), 
                                            lr=self.config.vae.lr, 
                                            weight_decay=self.config.vae.weight_decay)

        self.mlpdenoiser = MLPDenoiser(x_dim=self.config.hidden_sizes[-1],
                                    emb_dim = self.config.diffusion.cdim,
                                    hidden_dim = self.config.diffusion.hidden_dim,
                                    num_classes = self.num_classes,
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
        
        self.classifier = GCN_node_sparse(n_feat = self.config.feat_dim,
                                n_hidden = self.config.classifier.hidden_dim,
                                n_class = self.num_classes,
                                n_layer = self.config.classifier.n_layer,
                                dropout = self.config.classifier.dropout).to(self.device)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), 
                                                    lr=self.config.classifier.lr,
                                                    weight_decay=self.config.classifier.weight_decay)

    def run(self):
        start_idx = self.all_stages.index(self.stage_start)
        end_idx = self.all_stages.index(self.stage_end)
        stages_to_run = self.all_stages[start_idx:end_idx+1]
        if 'diff_sample' in stages_to_run and self.config.diffusion.filter:
            index = stages_to_run.index('diff_sample')
            stages_to_run[index] = 'filter_samples'
            
        self.stages = {}
        for stage_name in stages_to_run:
            self.stages[stage_name] = self.stage_classes[stage_name](self.config, self.dataset)
            self.stages[stage_name].run()