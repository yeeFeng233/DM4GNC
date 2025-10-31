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
        self.experiment_name = f"{config.dataset}_{self.timestamp}"
        
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
