#!/usr/bin/env python3
"""
This script runs multiple experiments with different hyperparameter configurations.
It focuses on the classifier_train and classifier_test stages and records all results
in a single log file for comparison.
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dm4gnc.config.config import Config
from dm4gnc.utils.random_seed import random_seed
from dm4gnc.utils.logger import MultiExperimentLogger
from dm4gnc.data.dataset import Dataset
from dm4gnc.pipeline.pipeline_manager import PipelineManager


def define_hyperparameter_sets():
    """
    Define multiple sets of hyperparameters to test.
    
    Each set is a dictionary containing the hyperparameters to modify.
    You can customize this function to test different combinations.
    
    Returns:
        list: List of dictionaries, each containing hyperparameters for one experiment
    """
    
    # Example hyperparameter sets
    # You can modify these based on your experimental needs
    generate_ratios = [0.1,0.2]
    thresholds = [ i/100.0+0.90 for i in range(1,10,1)]
    filters = [True]
    filter_strategies = ["distance"]
    hyperparam_sets = []
    vae_names = ["vae_class"]
    for generate_ratio in generate_ratios:
        for threshold in thresholds:
            for filter in filters:
                for filter_strategy in filter_strategies:
                    for vae_name in vae_names:
                        hyperparam_sets.append({
                            'name': f'generate_ratio_{generate_ratio}_threshold_{threshold}_filter_{filter}_filter_strategy_{filter_strategy}_vae_name_{vae_name}',
                            'generate_ratio': generate_ratio,
                            'vae_threshold': threshold,
                            'filter': filter,
                            'filter_strategy': filter_strategy,
                            'vae_name': vae_name,
                        })
    return hyperparam_sets

def apply_hyperparameters(config, hyperparams):
    """
    Apply hyperparameters to the config object.
    
    Args:
        config: Config object to modify
        hyperparams: Dictionary of hyperparameters to apply
    
    Returns:
        Config: Modified config object
    """
    # Create a deep copy to avoid modifying the original
    config_copy = deepcopy(config)
    
    # vae
    if "vae_threshold" in hyperparams:
        config_copy.vae.threshold = hyperparams['vae_threshold']
    # diffusion
    if "generate_ratio" in hyperparams:
        config_copy.diffusion.generate_ratio = hyperparams['generate_ratio']
    if "filter" in hyperparams:
        config_copy.diffusion.filter = hyperparams['filter']
    if "filter_strategy" in hyperparams:
        config_copy.diffusion.filter_strategy = hyperparams['filter_strategy']
    return config_copy

def extract_metrics_from_logger(logger):
    """
    Extract metrics from the logger after running the pipeline.
    
    Args:
        logger: ExperimentLogger object
    
    Returns:
        dict: Dictionary of metrics from classifier_train and classifier_test stages
    """
    metrics = {}
    
    # Extract from classifier_train stage
    if 'classifier_train' in logger.log_data['stages']:
        train_metrics = logger.log_data['stages']['classifier_train'].get('metrics', {})
        for key, value in train_metrics.items():
            metrics[f'train_{key}'] = value
    
    # Extract from classifier_test stage
    if 'classifier_test' in logger.log_data['stages']:
        test_metrics = logger.log_data['stages']['classifier_test'].get('metrics', {})
        for key, value in test_metrics.items():
            metrics[key] = value
    
    return metrics


def main():
    """Main function for running multiple experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multiple experiments with different hyperparameters')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Base configuration file path')
    parser.add_argument('--dataset', type=str,
                       help='Dataset name (optional, overrides config)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Load base configuration
    print("="*80)
    print("Loading base configuration...")
    base_config = Config.from_file(args.config_path)
    
    # Override with command line arguments
    if args.dataset:
        base_config.dataset = args.dataset
    base_config.device = args.device
    base_config.seed = args.seed
    
    # Force stages to be classifier_train and classifier_test
    base_config.stage_start = 'classifier_train'
    base_config.stage_end = 'classifier_test'
    
    base_config.validate()
    
    # Set random seed
    random_seed(base_config.seed)
    
    # Initialize multi-experiment logger
    multi_logger = MultiExperimentLogger(base_config)
    
    # Load dataset (only once, reused for all experiments)
    print("\n" + "="*80)
    print("Loading dataset...")
    dataset_builder = Dataset(
        data_path=base_config.data_path,
        name=base_config.dataset,
        imb_level=base_config.imb_level,
        shuffle_seed=base_config.seed
    )
    dataset = dataset_builder.load_dataset()
    print(dataset)
    
    # Get hyperparameter sets
    hyperparam_sets = define_hyperparameter_sets()
    print("\n" + "="*80)
    print(f"Total number of experiments to run: {len(hyperparam_sets)}")
    print("="*80)
    
    # Run experiments
    for exp_id, hyperparams in enumerate(hyperparam_sets):
        # Apply hyperparameters to config
        config = apply_hyperparameters(base_config, hyperparams)
        
        # Create a temporary logger for this experiment
        from dm4gnc.utils import ExperimentLogger
        temp_logger = ExperimentLogger(config)
        
        try:
            # Run pipeline (classifier_train and classifier_test)
            pipeline_manager = PipelineManager(config, dataset, logger=temp_logger)
            pipeline_manager.run()
            
            # Finalize temporary logger
            temp_logger.finalize()
            
            # Extract metrics
            metrics = extract_metrics_from_logger(temp_logger)
            
            # Create hyperparameter dictionary for logging (exclude 'name' field)
            hyperparam_for_log = {k: v for k, v in hyperparams.items() if k != 'name'}
            
            # Record to multi-experiment logger
            multi_logger.add_experiment(
                exp_id=exp_id,
                hyperparams=hyperparam_for_log,
                metrics=metrics
            )
            
            print(f"\n[Experiment {exp_id}] Completed successfully!")
            
        except Exception as e:
            print(f"\n[Experiment {exp_id}] Failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failed experiment
            multi_logger.add_experiment(
                exp_id=exp_id,
                hyperparams={k: v for k, v in hyperparams.items() if k != 'name'},
                metrics={'error': str(e), 'status': 'failed'}
            )
    
    # Finalize multi-experiment logger
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    multi_logger.finalize()
    
    print("\nResults have been saved and can be analyzed.")


if __name__ == "__main__":
    main()
