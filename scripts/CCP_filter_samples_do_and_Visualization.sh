#!/bin/bash
set -e

echo "========================================="
echo "Filter Samples and Visualization Pipeline"
echo "========================================="

# 定义数据集、GPU和VAE模型配置
declare -A DATASETS=(
    ["cora"]="cuda:0"
    ["citeseer"]="cuda:1"
    ["pubmed"]="cuda:2"
)

VAE_MODELS=("normal_vae" "vae_class" "vae_class_v2")

# 为每个数据集启动并行任务
for dataset in "${!DATASETS[@]}"; do
    device="${DATASETS[$dataset]}"
    
    {
        echo "[${dataset^}] 开始过滤样本..."
        
        # 遍历每种VAE模型
        for vae_model in "${VAE_MODELS[@]}"; do
            echo "[${dataset^} - ${vae_model}] 开始处理..."
            
            # 1. 执行过滤样本
            python scripts/train.py \
                --config_path "configs/dm4gnc/${dataset}.yml" \
                --stage_start diff_sample \
                --stage_end diff_sample \
                --diff_filter true \
                --diff_generate_ratio -1 \
                --filter_strategy "distance" \
                --vae_name "${vae_model}"
            
            echo "[${dataset^} - ${vae_model}] 过滤样本完成"
            
            # 2. 执行可视化
            python scripts/visualize_results.py \
                --config_path "configs/dm4gnc/${dataset}.yml" \
                --stage_to_visualize filter_samples \
                --diff_generate_ratio -1 \
                --device "${device}" \
                --filter_strategy "distance" \
                --vae_name "${vae_model}"
            
            echo "[${dataset^} - ${vae_model}] 可视化完成"
        done
        
        echo "[${dataset^}] 所有模型处理完成"
    } &
done

# 等待所有并行任务完成
wait

echo "========================================="
echo "所有任务完成"
echo "========================================="
