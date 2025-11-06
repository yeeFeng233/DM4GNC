#!/bin/bash
set -e

echo "========================================="
echo "Visualization from diff_sample"
echo "========================================="

# # Cora (cuda:0)
# {
#     echo "[Cora] 开始可视化..."
    
#     # normal_vae
#     echo "[Cora - normal_vae] 可视化中..."
#     python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:0 --vae_name normal_vae
   
#     # vae_class
#     echo "[Cora - vae_class] 可视化中..."
#     python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:0 --vae_name vae_class
    
#     # vae_class_v2
#     echo "[Cora - vae_class_v2] 可视化中..."
#     python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:0 --vae_name vae_class_v2
   
#     echo "[Cora] 完成"
# } &

# CiteSeer数据集 (cuda:1)
{
    echo "[CiteSeer] 开始训练..."
    
    # normal_vae
    echo "[CiteSeer - normal_vae] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:1 --vae_name normal_vae
   
    # vae_class
    echo "[CiteSeer - vae_class] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:1 --vae_name vae_class
    
    # vae_class_v2
    echo "[CiteSeer - vae_class_v2] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/citeseer.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:1 --vae_name vae_class_v2
    
    echo "[CiteSeer] 完成"
} &

# PubMed数据集 (cuda:2)
{
    echo "[PubMed] 开始训练..."
    
    # normal_vae
    echo "[PubMed - normal_vae] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:2 --vae_name normal_vae
   
    # vae_class
    echo "[PubMed - vae_class] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:2 --vae_name vae_class
   
    # vae_class_v2
    echo "[PubMed - vae_class_v2] 训练中..."
    python scripts/visualize_results.py --config_path configs/dm4gnc/pubmed.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --device cuda:2 --vae_name vae_class_v2
    
    echo "[PubMed] 完成"
} &

wait

echo "========================================="
echo "所有任务完成"
echo "========================================="
