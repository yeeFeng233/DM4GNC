#!/bin/bash
set -e

echo "========================================="
echo "Running Repeat Classifier Training"
echo "on Cora, CiteSeer, and PubMed datasets"
echo "with multiple hyperparameter configurations"
echo "========================================="

# 定义数据集和GPU映射
declare -A DATASETS=(
    ["cora"]="cuda:0"
    ["citeseer"]="cuda:1"
    ["pubmed"]="cuda:2"
)

# 可选: 设置随机种子
SEED=42

# 检查配置文件是否存在
for dataset in "${!DATASETS[@]}"; do
    config_file="configs/dm4gnc/${dataset}.yml"
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件 $config_file 不存在"
        exit 1
    fi
done

# 检查 repeat_train.py 是否存在
if [ ! -f "scripts/repeat_train.py" ]; then
    echo "错误: scripts/repeat_train.py 不存在"
    exit 1
fi

# 存储每个数据集任务的PID和名称
declare -A TASK_PIDS
declare -a TASK_NAMES

echo ""
echo "开始在所有数据集上运行重复分类器训练..."
echo ""

# 为每个数据集启动并行任务
for dataset in "${!DATASETS[@]}"; do
    device="${DATASETS[$dataset]}"
    
    {
        set -e
        set -o pipefail
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^}] 开始重复训练实验..."
        
        if ! python scripts/repeat_train.py \
                    --config_path "configs/dm4gnc/${dataset}.yml" \
                    --device "${device}" \
            --seed "${SEED}"; then
            echo "[ERROR] [${dataset^}] 重复训练实验失败"
                    exit 1
                fi
                
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${dataset^}] 所有实验完成"
    } &
    
    # 记录PID和任务名
    TASK_PIDS["${dataset}"]=$!
    TASK_NAMES+=("${dataset}")
done

# 等待所有并行任务完成并检查退出状态
echo ""
echo "等待所有数据集任务完成..."
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
echo "查看日志文件了解详细实验结果"
echo "========================================="