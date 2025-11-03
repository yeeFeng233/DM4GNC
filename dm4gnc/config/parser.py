import argparse
from typing import Dict, Any
from .config import Config

class ConfigParser:
    """config parser"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """create command line argument parser"""
        parser = argparse.ArgumentParser(description='dm4gnc training pipeline')
        
        # config file
        parser.add_argument('--config_path', type=str, required=True,
                           help='config file path')
        # pipeline control
        parser.add_argument('--stage_start', type=str,
                           choices=['vae_train', 'vae_encode', 'diff_train', 
                                   'diff_sample', 'vae_decode', 'filter_samples',
                                   'classifier_train', 'classifier_test'],
                           help='start stage')
        parser.add_argument('--stage_end', type=str,
                           choices=['vae_train', 'vae_encode', 'diff_train', 
                                   'diff_sample', 'vae_decode', 'filter_samples',
                                   'classifier_train', 'classifier_test'],
                           help='end stage')
        parser.add_argument('--stage_to_visualize', type=str,
                           choices=['vae_encode', 'diff_sample', 'vae_decode', 'filter_samples', 'classifier_test'],
                           help='stage to visualize')
        
        # basic parameters
        parser.add_argument('--dataset', type=str,
                           help='dataset name')
        parser.add_argument('--device', type=str,
                           help='device')
        parser.add_argument('--seed', type=int,
                           help='random seed')
        
        # VAE parameters
        parser.add_argument('--vae_lr', type=float,
                           help='VAE learning rate')
        parser.add_argument('--vae_epoch', type=int,
                           help='VAE training epochs')
        
        # diffusion model parameters
        parser.add_argument('--diff_lr', type=float,
                           help='diffusion model learning rate')
        parser.add_argument('--diff_T', type=int,
                           help='diffusion steps')
        
        # classifier parameters
        parser.add_argument('--cls_lr', type=float,
                           help='classifier learning rate')
        parser.add_argument('--cls_epoch', type=int,
                           help='classifier training epochs')
        
        return parser
    
    @staticmethod
    def parse_args_and_merge_config(args: argparse.Namespace) -> Config:
        """parse args and merge config"""
        config = Config.from_file(args.config_path)
        
        if args.stage_start:
            config.stage_start = args.stage_start
        if args.stage_end:
            config.stage_end = args.stage_end
        if args.dataset:
            config.dataset = args.dataset
        if args.device:
            config.device = args.device
        if args.seed:
            config.seed = args.seed
        
        if args.vae_lr:
            config.vae.lr = args.vae_lr
        if args.vae_epoch:
            config.vae.epoch = args.vae_epoch
        
        if args.diff_lr:
            config.diffusion.lr = args.diff_lr
        if args.diff_T:
            config.diffusion.T = args.diff_T
        
        if args.cls_lr:
            config.classifier.lr = args.cls_lr
        if args.cls_epoch:
            config.classifier.epoch = args.cls_epoch
        
        return config
