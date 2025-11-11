#!/bin/bash
set -e
set -o pipefail

# ---------------------------
# 模型与数据集配置
# ---------------------------
VAE_MODELS=(
    "normal_vae"
    "vae_class"
    "vae_class_v2"
    "vae_dec"
    "vae_dec_class"
    "vae_sig"
)

DATASETS=("cora" "citeseer" "pubmed" "computers" "photo")

declare -A GPU_MAP=(
    ["cora"]="cuda:0"
    ["citeseer"]="cuda:1"
    ["pubmed"]="cuda:2"
    ["computers"]="cuda:3"
    ["photo"]="cuda:4"
)

# ---------------------------
# 检查配置文件是否存在
# ---------------------------
for dataset in "${DATASETS[@]}"; do
    if [ ! -f "configs/dm4gnc/${dataset}.yml" ]; then
        echo "Error: configs/dm4gnc/${dataset}.yml not found"
        exit 1
    fi
done

# ---------------------------
# 执行函数
# ---------------------------
run_dataset_model() {
    local dataset=$1
    local model=$2
    local device=$3

    echo "[$(date '+%H:%M:%S')] ▶ Starting ${dataset} on ${device} with model ${model}"

    # --- 训练阶段 ---
    python scripts/train.py \
        --config_path "configs/dm4gnc/${dataset}.yml" \
        --stage_start vae_train \
        --stage_end vae_encode \
        --device "${device}" \
        --vae_name "${model}"

    # --- 可视化阶段 ---
    python scripts/visualize_results.py \
        --config_path "configs/dm4gnc/${dataset}.yml" \
        --stage_to_visualize vae_encode \
        --device "${device}" \
        --vae_name "${model}"

    echo "[$(date '+%H:%M:%S')] ✓ Completed ${dataset} (${model}) on ${device}"
}

# ---------------------------
# 主循环：模型 × 数据集 并行运行
# ---------------------------
FAILED_TASKS=()

for model in "${VAE_MODELS[@]}"; do
    echo "Running model: ${model}"

    declare -A PIDS
    declare -a TASK_NAMES

    for dataset in "${DATASETS[@]}"; do
        device="${GPU_MAP[$dataset]}"
        {
            run_dataset_model "${dataset}" "${model}" "${device}"
        } &
        PIDS["${dataset}"]=$!
        TASK_NAMES+=("${dataset}")
        sleep 2  # 防止GPU资源竞争
    done

    echo ">>> Waiting for all datasets of model '${model}' to finish..."
    for dataset in "${TASK_NAMES[@]}"; do
        if wait "${PIDS[$dataset]}"; then
            echo "✓ ${dataset} (${model}) completed successfully"
        else
            echo "✗ ${dataset} (${model}) FAILED"
            FAILED_TASKS+=("${dataset}_${model}")
        fi
    done

    echo ">>> Model ${model} finished all datasets."
done

# ---------------------------
# 检查失败任务
# ---------------------------
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "============================================================"
    echo "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo " - ${task}"
    done
    echo "============================================================"
    exit 1
else
    echo "All models and datasets completed successfully!"
fi
