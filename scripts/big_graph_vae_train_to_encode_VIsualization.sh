#!/bin/bash
set -e

echo "Big Graph VAE Training: Computers, Photo | 4 VAEs on 4 GPUs"

# 定义VAE模型和对应的GPU
declare -A VAE_GPU_MAP=(
    ["normal_vae"]="cuda:0"
    ["vae_class"]="cuda:1"
    ["vae_class_v2"]="cuda:2"
    ["vae_dec"]="cuda:3"
)

DATASETS=("computers" "photo")

# 检查配置文件
for dataset in "${DATASETS[@]}"; do
    [ ! -f "configs/dm4gnc/${dataset}.yml" ] && echo "Error: configs/dm4gnc/${dataset}.yml not found" && exit 1
done

# 存储任务PID
declare -A TASK_PIDS
declare -a TASK_NAMES

# 并行启动4个GPU任务
for vae_model in "${!VAE_GPU_MAP[@]}"; do
    device="${VAE_GPU_MAP[$vae_model]}"
    
    {
        set -e
        set -o pipefail
        
        for dataset in "${DATASETS[@]}"; do
            echo "[$(date '+%H:%M:%S')] ${vae_model} - ${dataset^} (${device})"
            
            # 训练和编码
            python scripts/train.py \
                --config_path "configs/dm4gnc/${dataset}.yml" \
                --stage_start vae_train \
                --stage_end vae_encode \
                --device "${device}" \
                --vae_name "${vae_model}" > /dev/null 2>&1 || exit 1
            
            # 可视化
            python scripts/visualize_results.py \
                --config_path "configs/dm4gnc/${dataset}.yml" \
                --stage_to_visualize vae_encode \
                --device "${device}" \
                --vae_name "${vae_model}" > /dev/null 2>&1 || exit 1
            
            echo "[$(date '+%H:%M:%S')] ✓ ${vae_model} - ${dataset^}"
        done
        
    } &
    
    TASK_PIDS["${vae_model}"]=$!
    TASK_NAMES+=("${vae_model}")
done

echo "Waiting for all GPUs to complete..."

# 等待所有任务
FAILED_TASKS=()
for task_name in "${TASK_NAMES[@]}"; do
    if wait "${TASK_PIDS[$task_name]}"; then
        echo "✓ ${task_name}"
    else
        echo "✗ ${task_name} FAILED"
        FAILED_TASKS+=("${task_name}")
    fi
done

# 检查失败任务
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "Failed tasks: ${FAILED_TASKS[*]}"
    exit 1
fi

echo "All tasks completed successfully"
