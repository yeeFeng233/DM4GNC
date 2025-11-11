import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union


class ExperimentLogger:
    """A simple experiment logger
    
    Content:
    - Configuration information
    - Execution date
    - Metrics of each stage
    """
    
    def __init__(self, config, log_dir: str = None):
        """initialize the logger
        
        Args:
            config: config object
            log_dir: log directory, default: outputs/logs
        """
        self.config = config
        
        # set the log directory
        if log_dir is None:
            log_dir = os.path.join(config.output_dir, 'logs')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # create the timestamp of the experiment
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.timestamp}_{config.dataset}_{config.vae.name}"
        
        # log file path
        self.log_file = self.log_dir / f"{self.experiment_name}.json"
        
        # initialize the log data
        self.log_data = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': self._config_to_dict(),
            'stages': {}
        }
        
        # save the initial log
        self._save_log()
        
        print(f"[Logger] Experiment logger initialized: {self.log_file}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """convert the config object to a dictionary"""
        config_dict = {}
        
        # get all attributes of the config
        for key in dir(self.config):
            if not key.startswith('_'):
                value = getattr(self.config, key)
                # skip methods
                if callable(value):
                    continue
                # handle nested config objects (e.g. vae, diffusion)
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, list, dict, bool, type(None))):
                    config_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                else:
                    # convert special types
                    if isinstance(value, Path):
                        config_dict[key] = str(value)
                    elif isinstance(value, (str, int, float, list, dict, bool, type(None))):
                        config_dict[key] = value
                        
        return config_dict
    
    def log_stage_start(self, stage_name: str):
        """record the start of the stage"""
        if stage_name not in self.log_data['stages']:
            self.log_data['stages'][stage_name] = {
                'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'metrics': {}
            }
        print(f"[Logger] Stage '{stage_name}' start")
    
    def log_stage_metrics(self, stage_name: str, metrics: Dict[str, Any]):
        """record the metrics of the stage
        
        Args:
            stage_name: stage name
            metrics: dictionary of metrics, e.g. {'test_acc': 0.85, 'test_f1': 0.82}
        """
        if stage_name not in self.log_data['stages']:
            self.log_stage_start(stage_name)
        
        # update the metrics
        self.log_data['stages'][stage_name]['metrics'].update(metrics)
        
        # record the end time
        self.log_data['stages'][stage_name]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # save the log
        self._save_log()
        
        # print the metrics
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        print(f"[Logger] Stage '{stage_name}' metrics: {metrics_str}")
    
    def log_image_path(self, stage_name: str, image_name: str, image_path: str):
        """record the path of the image
        
        Args:
            stage_name: stage name
            image_name: image name (e.g. 'tsne_visualization')
            image_path: path to save the image
        """
        if stage_name not in self.log_data['stages']:
            self.log_stage_start(stage_name)
        
        if 'images' not in self.log_data['stages'][stage_name]:
            self.log_data['stages'][stage_name]['images'] = {}
        
        self.log_data['stages'][stage_name]['images'][image_name] = str(image_path)
        self._save_log()
        
        print(f"[Logger] Stage '{stage_name}' image saved to: {image_path}")
    
    def _save_log(self):
        """save the log to a JSON file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> str:
        """get the summary of the experiment"""
        summary = f"\n{'='*60}\n"
        summary += f"Experiment summary: {self.experiment_name}\n"
        summary += f"{'='*60}\n"
        summary += f"Date: {self.log_data['date']}\n"
        summary += f"Dataset: {self.config.dataset}\n"
        summary += f"Log file: {self.log_file}\n\n"
        
        summary += "Stage execution results:\n"
        for stage_name, stage_data in self.log_data['stages'].items():
            summary += f"\n  [{stage_name}]\n"
            if 'metrics' in stage_data and stage_data['metrics']:
                for metric_name, metric_value in stage_data['metrics'].items():
                    if isinstance(metric_value, float):
                        summary += f"    - {metric_name}: {metric_value:.4f}\n"
                    else:
                        summary += f"    - {metric_name}: {metric_value}\n"
            if 'images' in stage_data:
                for img_name, img_path in stage_data['images'].items():
                    summary += f"    - image {img_name}: {img_path}\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def finalize(self):
        """complete the logging, print the summary"""
        print(self.get_summary())
        print(f"[Logger] Complete log saved to: {self.log_file}")


class MultiExperimentLogger:
    """A logger for managing multiple experiments with different hyperparameters
    
    This logger records results from multiple runs with different hyperparameter
    configurations and saves them all to a single log file for comparison.
    """
    
    def __init__(self, base_config, log_dir: str = None):
        """Initialize the multi-experiment logger
        
        Args:
            base_config: Base config object
            log_dir: Log directory, default: outputs/logs
        """
        self.base_config = base_config
        
        # Set log directory
        if log_dir is None:
            log_dir = os.path.join(base_config.output_dir, 'logs')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this batch of experiments
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_batch_name = f"multi_exp_{base_config.dataset}_{self.timestamp}"
        
        # Log file path
        self.log_file = self.log_dir / f"{self.experiment_batch_name}.json"
        
        # Initialize log data
        self.log_data = {
            'experiment_batch_name': self.experiment_batch_name,
            'timestamp': self.timestamp,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset': base_config.dataset,
            'base_config': self._config_to_dict(base_config),
            'experiments': []
        }
        
        # Save initial log
        self._save_log()
        
        print(f"[MultiExperimentLogger] Batch experiment logger initialized: {self.log_file}")
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        config_dict = {}
        
        for key in dir(config):
            if not key.startswith('_'):
                value = getattr(config, key)
                if callable(value):
                    continue
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, list, dict, bool, type(None))):
                    config_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                else:
                    if isinstance(value, Path):
                        config_dict[key] = str(value)
                    elif isinstance(value, (str, int, float, list, dict, bool, type(None))):
                        config_dict[key] = value
                        
        return config_dict
    
    def add_experiment(self, exp_id: int, hyperparams: Dict[str, Any], metrics: Dict[str, Any]):
        """Add an experiment result
        
        Args:
            exp_id: Experiment ID (typically the index in the loop)
            hyperparams: Dictionary of hyperparameters used in this experiment
            metrics: Dictionary of metrics (results) from this experiment
        """
        experiment_record = {
            'exp_id': exp_id,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.log_data['experiments'].append(experiment_record)
        self._save_log()
        
        # Print experiment result
        print(f"\n[MultiExperimentLogger] Experiment {exp_id} recorded:")
        print(f"  Hyperparams: {hyperparams}")
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        print(f"  Metrics: {metrics_str}")
    
    def _save_log(self):
        """Save log to JSON file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> str:
        """Get summary of all experiments"""
        summary = f"\n{'='*80}\n"
        summary += f"Multi-Experiment Summary: {self.experiment_batch_name}\n"
        summary += f"{'='*80}\n"
        summary += f"Date: {self.log_data['date']}\n"
        summary += f"Dataset: {self.log_data['dataset']}\n"
        summary += f"Total experiments: {len(self.log_data['experiments'])}\n"
        summary += f"Log file: {self.log_file}\n\n"
        
        # Find best experiment by test accuracy
        if self.log_data['experiments']:
            best_exp = max(self.log_data['experiments'], 
                          key=lambda x: x['metrics'].get('test_accuracy', 0))
            
            summary += "="*80 + "\n"
            summary += "Best Experiment (by test_accuracy):\n"
            summary += f"  Exp ID: {best_exp['exp_id']}\n"
            summary += f"  Hyperparams: {best_exp['hyperparams']}\n"
            summary += f"  Metrics:\n"
            for metric_name, metric_value in best_exp['metrics'].items():
                if isinstance(metric_value, float):
                    summary += f"    - {metric_name}: {metric_value:.4f}\n"
                else:
                    summary += f"    - {metric_name}: {metric_value}\n"
            summary += "="*80 + "\n\n"
        
        summary += "All Experiments:\n"
        summary += "-"*80 + "\n"
        for exp in self.log_data['experiments']:
            summary += f"\n[Experiment {exp['exp_id']}]\n"
            summary += f"  Hyperparams: {exp['hyperparams']}\n"
            summary += f"  Metrics:\n"
            for metric_name, metric_value in exp['metrics'].items():
                if isinstance(metric_value, float):
                    summary += f"    - {metric_name}: {metric_value:.4f}\n"
                else:
                    summary += f"    - {metric_name}: {metric_value}\n"
        
        summary += "\n" + "="*80 + "\n"
        return summary
    
    def finalize(self):
        """Finalize logging and print summary"""
        print(self.get_summary())
        print(f"[MultiExperimentLogger] Complete log saved to: {self.log_file}")
        
        # Generate CSV for easy analysis
        self._export_to_csv()
    
    def _export_to_csv(self):
        """Export results to CSV file for easy analysis"""
        if not self.log_data['experiments']:
            return
        
        csv_file = self.log_file.with_suffix('.csv')
        
        try:
            import csv
            
            # Collect all unique hyperparameter and metric keys
            all_hyperparam_keys = set()
            all_metric_keys = set()
            
            for exp in self.log_data['experiments']:
                all_hyperparam_keys.update(exp['hyperparams'].keys())
                all_metric_keys.update(exp['metrics'].keys())
            
            hyperparam_keys = sorted(all_hyperparam_keys)
            metric_keys = sorted(all_metric_keys)
            
            # Write CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['exp_id'] + hyperparam_keys + metric_keys
                writer.writerow(header)
                
                # Data rows
                for exp in self.log_data['experiments']:
                    row = [exp['exp_id']]
                    row.extend([exp['hyperparams'].get(k, '') for k in hyperparam_keys])
                    row.extend([exp['metrics'].get(k, '') for k in metric_keys])
                    writer.writerow(row)
            
            print(f"[MultiExperimentLogger] Results exported to CSV: {csv_file}")
        except Exception as e:
            print(f"[MultiExperimentLogger] Failed to export CSV: {e}")
