#!/bin/bash
set -e

echo "VAE_DEC_class Training: Cora, Citeseer, Pubmed, Computers, Photo"
echo "Cora and Citeseer on cuda:0 (sequential)"
echo "Pubmed, Computers, Photo on cuda:1, cuda:2, cuda:3 (parallel)"

VAE_MODEL="vae_dec_class"

# 定义数据集和GPU映射
declare -A DATASET_GPU_MAP=(
    ["cora"]="cuda:0"
    ["citeseer"]="cuda:0"
    ["pubmed"]="cuda:1"
    ["computers"]="cuda:2"
    ["photo"]="cuda:3"
)

# 串行执行的数据集（cuda:0）
SERIAL_DATASETS=("cora" "citeseer")

# 并行执行的数据集
PARALLEL_DATASETS=("pubmed" "computers" "photo")

# 检查配置文件
ALL_DATASETS=("${SERIAL_DATASETS[@]}" "${PARALLEL_DATASETS[@]}")
for dataset in "${ALL_DATASETS[@]}"; do
    [ ! -f "configs/dm4gnc/${dataset}.yml" ] && echo "Error: configs/dm4gnc/${dataset}.yml not found" && exit 1
done

# 执行单个数据集的训练、编码和可视化
run_dataset() {
    local dataset=$1
    local device=$2
    
    echo "[$(date '+%H:%M:%S')] Starting ${dataset^} on ${device}"
    
    # 训练和编码
    python scripts/train.py \
        --config_path "configs/dm4gnc/${dataset}.yml" \
        --stage_start vae_train \
        --stage_end vae_encode \
        --device "${device}" \
        --vae_name "${VAE_MODEL}" || return 1
    
    # 可视化
    python scripts/visualize_results.py \
        --config_path "configs/dm4gnc/${dataset}.yml" \
        --stage_to_visualize vae_encode \
        --device "${device}" \
        --vae_name "${VAE_MODEL}" || return 1
    
    echo "[$(date '+%H:%M:%S')] ✓ Completed ${dataset^} on ${device}"
    return 0
}

# 串行执行 Cora 和 Citeseer 在 cuda:0
echo "=== Sequential execution on cuda:0 ==="
FAILED_DATASETS=()
for dataset in "${SERIAL_DATASETS[@]}"; do
    device="${DATASET_GPU_MAP[$dataset]}"
    if ! run_dataset "${dataset}" "${device}"; then
        echo "✗ ${dataset} FAILED"
        FAILED_DATASETS+=("${dataset}")
    fi
done

# 并行执行 Pubmed, Computers, Photo
echo "=== Parallel execution on cuda:1, cuda:2, cuda:3 ==="
declare -A PARALLEL_PIDS
declare -a PARALLEL_NAMES

for dataset in "${PARALLEL_DATASETS[@]}"; do
    device="${DATASET_GPU_MAP[$dataset]}"
    
    {
        set -e
        set -o pipefail
        run_dataset "${dataset}" "${device}" || exit 1
    } &
    
    PARALLEL_PIDS["${dataset}"]=$!
    PARALLEL_NAMES+=("${dataset}")
done

# 等待并行任务完成
echo "Waiting for parallel tasks to complete..."
for dataset in "${PARALLEL_NAMES[@]}"; do
    if wait "${PARALLEL_PIDS[$dataset]}"; then
        echo "✓ ${dataset} completed"
    else
        echo "✗ ${dataset} FAILED"
        FAILED_DATASETS+=("${dataset}")
    fi
done

# 检查失败任务
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "Failed datasets: ${FAILED_DATASETS[*]}"
    exit 1
fi

echo "All tasks completed successfully"
