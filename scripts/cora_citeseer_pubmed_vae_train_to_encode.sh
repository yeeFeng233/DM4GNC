#!/bin/bash

# CiteSeer和PubMed数据集的VAE训练和编码
# CiteSeer在cuda:1上运行，PubMed在cuda:2上运行
# 每个数据集运行normal_vae和vae_class两种模型

set -e

echo "========================================="
echo "开始VAE训练流程"
echo "========================================="

# Cora (cuda:0)
{
    echo "[Cora] 开始训练..."
    
    # normal_vae
    echo "[Cora - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_train --stage_end vae_encode --device cuda:0 --vae_name normal_vae
    echo "[Cora - normal_vae] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode --device cuda:0 --vae_name normal_vae
    
    # vae_class
    echo "[Cora - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_train --stage_end vae_encode --device cuda:0 --vae_name vae_class
    echo "[Cora - vae_class] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode --device cuda:0 --vae_name vae_class
    
    # vae_class_v2
    echo "[Cora - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_train --stage_end vae_encode --device cuda:0 --vae_name vae_class_v2
    echo "[Cora - vae_class_v2] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode --device cuda:0 --vae_name vae_class_v2

    echo "[Cora] 完成"
} &

# CiteSeer数据集 (cuda:1)
{
    echo "[CiteSeer] 开始训练..."
    
    # normal_vae
    echo "[CiteSeer - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start vae_train --stage_end vae_encode --device cuda:1 --vae_name normal_vae
    echo "[CiteSeer - normal_vae] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize vae_encode --device cuda:1 --vae_name normal_vae
    
    # vae_class
    echo "[CiteSeer - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start vae_train --stage_end vae_encode --device cuda:1 --vae_name vae_class
    echo "[CiteSeer - vae_class] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize vae_encode --device cuda:1 --vae_name vae_class
    
    # vae_class_v2
    echo "[CiteSeer - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start vae_train --stage_end vae_encode --device cuda:1 --vae_name vae_class_v2
    echo "[CiteSeer - vae_class_v2] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize vae_encode --device cuda:1 --vae_name vae_class_v2
    
    echo "[CiteSeer] 完成"
} &

# PubMed数据集 (cuda:2)
{
    echo "[PubMed] 开始训练..."
    
    # normal_vae
    echo "[PubMed - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start vae_train --stage_end vae_encode --device cuda:2 --vae_name normal_vae
    echo "[PubMed - normal_vae] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize vae_encode --device cuda:2 --vae_name normal_vae
    
    # vae_class
    echo "[PubMed - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start vae_train --stage_end vae_encode --device cuda:2 --vae_name vae_class
    echo "[PubMed - vae_class] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize vae_encode --device cuda:2 --vae_name vae_class
    
    # vae_class_v2
    echo "[PubMed - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start vae_train --stage_end vae_encode --device cuda:2 --vae_name vae_class_v2
    echo "[PubMed - vae_class_v2] 可视化中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize vae_encode --device cuda:2 --vae_name vae_class_v2
    
    echo "[PubMed] 完成"
} &

# 等待两个数据集都完成
wait

echo "========================================="
echo "所有任务完成"
echo "========================================="
