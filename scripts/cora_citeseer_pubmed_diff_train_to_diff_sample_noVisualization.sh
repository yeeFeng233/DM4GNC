#!/bin/bash
set -e

echo "========================================="
echo "Training from diff_train to diff_sample"
echo "========================================="

# Cora (cuda:0)
{
    echo "[Cora] 开始训练..."
    
    # normal_vae
    echo "[Cora - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start diff_train --stage_end diff_sample --device cuda:0 --vae_name normal_vae --diff_generate_ratio -1
   
    # vae_class
    echo "[Cora - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start diff_train --stage_end diff_sample --device cuda:0 --vae_name vae_class --diff_generate_ratio -1
    
    # vae_class_v2
    echo "[Cora - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start diff_train --stage_end diff_sample --device cuda:0 --vae_name vae_class_v2 --diff_generate_ratio -1
   
    echo "[Cora] 完成"
} &

# CiteSeer数据集 (cuda:1)
{
    echo "[CiteSeer] 开始训练..."
    
    # normal_vae
    echo "[CiteSeer - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start diff_train --stage_end diff_sample --device cuda:1 --vae_name normal_vae --diff_generate_ratio -1
    
    # vae_class
    echo "[CiteSeer - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start diff_train --stage_end diff_sample --device cuda:1 --vae_name vae_class --diff_generate_ratio -1
    
    # vae_class_v2
    echo "[CiteSeer - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/citeseer.yml --stage_start diff_train --stage_end diff_sample --device cuda:1 --vae_name vae_class_v2 --diff_generate_ratio -1
    
    echo "[CiteSeer] 完成"
} &

# PubMed数据集 (cuda:2)
{
    echo "[PubMed] 开始训练..."
    
    # normal_vae
    echo "[PubMed - normal_vae] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start diff_train --stage_end diff_sample --device cuda:2 --vae_name normal_vae --diff_generate_ratio -1
   
    # vae_class
    echo "[PubMed - vae_class] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start diff_train --stage_end diff_sample --device cuda:2 --vae_name vae_class --diff_generate_ratio -1
   
    # vae_class_v2
    echo "[PubMed - vae_class_v2] 训练中..."
    python scripts/train.py --config_path configs/dm4gnc/pubmed.yml --stage_start diff_train --stage_end diff_sample --device cuda:2 --vae_name vae_class_v2 --diff_generate_ratio -1
    
    echo "[PubMed] 完成"
} &

wait

echo "========================================="
echo "所有任务完成"
echo "========================================="
