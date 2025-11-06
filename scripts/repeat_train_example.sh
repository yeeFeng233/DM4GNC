#!/bin/bash
# repeat_train.py 使用示例

# 使用方式 1: 基本使用
# 运行多组超参数实验，使用配置文件中的默认数据集
python scripts/repeat_train.py --config_path configs/dm4gnc/cora.yml

# 使用方式 2: 指定数据集
# 可以覆盖配置文件中的数据集设置
python scripts/repeat_train.py \
    --config_path configs/dm4gnc/cora.yml \
    --dataset CiteSeer

# 使用方式 3: 指定设备和随机种子
# 可以控制使用的GPU设备和随机种子
python scripts/repeat_train.py \
    --config_path configs/dm4gnc/cora.yml \
    --device cuda:0 \
    --seed 42

# 使用方式 4: 完整参数示例
python scripts/repeat_train.py \
    --config_path configs/dm4gnc/cora.yml \
    --dataset Cora \
    --device cuda:0 \
    --seed 42

# 说明:
# 1. 脚本会自动运行 classifier_train 和 classifier_test 阶段
# 2. 需要确保之前的阶段(vae_train, vae_encode, diff_train, diff_sample, vae_decode)已经完成
# 3. 所有实验结果会保存在同一个日志文件中: outputs/logs/multi_exp_<dataset>_<timestamp>.json
# 4. 同时会生成CSV文件方便分析: outputs/logs/multi_exp_<dataset>_<timestamp>.csv 