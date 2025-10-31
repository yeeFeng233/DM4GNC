#!/usr/bin/env python3
"""
简单测试脚本，验证 ExperimentLogger 功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dm4gnc.config.config import Config
from dm4gnc.utils import ExperimentLogger

def test_logger():
    """测试logger基本功能"""
    print("=" * 60)
    print("测试 ExperimentLogger 功能")
    print("=" * 60)
    
    # 加载配置
    config_path = project_root / "configs" / "dm4gnc" / "cora.yml"
    config = Config.from_file(str(config_path))
    
    # 设置基本参数
    config.dataset = "Cora"
    config.output_dir = str(project_root / "outputs")
    config.seed = 42
    config.device = "cuda"
    
    # 初始化logger
    print("\n1. 初始化 Logger...")
    logger = ExperimentLogger(config)
    
    # 模拟记录stage信息
    print("\n2. 模拟记录 VAE Train Stage...")
    logger.log_stage_start("vae_train")
    logger.log_stage_metrics("vae_train", {
        'best_val_roc': 0.9234,
        'best_epoch': 234,
        'test_roc': 0.9156,
        'test_ap': 0.8891
    })
    
    print("\n3. 模拟记录 Diffusion Train Stage...")
    logger.log_stage_start("diff_train")
    logger.log_stage_metrics("diff_train", {
        'best_val_loss': 0.0234,
        'best_epoch': 567,
        'total_epochs': 1000
    })
    
    print("\n4. 模拟记录 Classifier Test Stage...")
    logger.log_stage_start("classifier_test")
    logger.log_stage_metrics("classifier_test", {
        'test_accuracy': 0.8520,
        'test_macro_f1': 0.8234,
        'test_balanced_accuracy': 0.8456,
        'test_auc_roc': 0.9012
    })
    
    print("\n5. 模拟记录图片路径...")
    logger.log_image_path("classifier_test", "tsne_visualization", 
                          str(project_root / "outputs" / "visualizations" / "tsne.png"))
    
    # 完成并打印总结
    print("\n6. 生成实验总结...")
    logger.finalize()
    
    print("\n" + "=" * 60)
    print("✅ Logger 测试完成！")
    print(f"📝 日志文件已保存至: {logger.log_file}")
    print("=" * 60)

if __name__ == "__main__":
    test_logger()
