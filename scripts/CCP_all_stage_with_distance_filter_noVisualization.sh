#!/bin/bash
set -e

echo "========================================="
echo "Training from diff_sample to vae_decode"
echo "with distance filter and distance filter strategy"
echo "and generate ratio 0.1"
echo "========================================="

# 定义数据集、GPU和VAE模型配置
declare -A DATASETS=(
    ["cora"]="cuda:0"
    # ["citeseer"]="cuda:1"
    # ["pubmed"]="cuda:2"
)

VAE_MODELS=("normal_vae" "vae_class" "vae_class_v2" "vae_dec")
GENERATE_RATIOS=(0.1)

# 检查配置文件是否存在
for dataset in "${!DATASETS[@]}"; do
    config_file="configs/dm4gnc/${dataset}.yml"
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件 $config_file 不存在"
        exit 1
    fi
done

# 存储每个数据集任务的PID和名称
declare -A TASK_PIDS
declare -a TASK_NAMES

# 为每个数据集启动并行任务
for dataset in "${!DATASETS[@]}"; do
    device="${DATASETS[$dataset]}"
    
    {
        set -e
        set -o pipefail
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^}] 开始训练..."
        
        # 遍历每种VAE模型
        for vae_model in "${VAE_MODELS[@]}"; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^} - ${vae_model}] 训练中..."
            
            # 遍历每个生成比率
            for ratio in "${GENERATE_RATIOS[@]}"; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^} - ${vae_model}] 生成比率: ${ratio}"
                
                if ! python scripts/train.py \
                    --config_path "configs/dm4gnc/${dataset}.yml" \
                    --stage_start vae_train \
                    --stage_end classifier_test \
                    --device "${device}" \
                    --vae_name "${vae_model}" \
                    --diff_filter true \
                    --filter_strategy "distance" \
                    --diff_generate_ratio "${ratio}"; then
                    echo "[ERROR] [${dataset^} - ${vae_model}] 生成比率 ${ratio} 失败"
                    exit 1
                fi
                
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^} - ${vae_model}] 生成比率 ${ratio} 完成"
            done
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^} - ${vae_model}] 完成"
        done
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^}] 所有模型训练完成"
    } &
    
    # 记录PID和任务名
    TASK_PIDS["${dataset}"]=$!
    TASK_NAMES+=("${dataset}")
done

# 等待所有并行任务完成并检查退出状态
echo ""
echo "等待所有任务完成..."
echo ""

FAILED_TASKS=()
for task_name in "${TASK_NAMES[@]}"; do
    pid="${TASK_PIDS[$task_name]}"
    if wait "$pid"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${task_name^}] 任务成功完成"
    else
        exit_code=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] [${task_name^}] 任务失败 (退出码: $exit_code)"
        FAILED_TASKS+=("${task_name}")
    fi
done

# 检查是否有失败的任务
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo ""
    echo "========================================="
    echo "部分任务失败:"
    for failed_task in "${FAILED_TASKS[@]}"; do
        echo "  - ${failed_task^}"
    done
    echo "========================================="
    exit 1
fi

echo ""
echo "========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务完成"
echo "========================================="
