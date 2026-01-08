#!/bin/bash
# 多卡训练启动脚本（优化版）

# 检查可用的GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 张GPU"

# 默认使用所有GPU
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$GPU_COUNT
fi

echo "使用 $NUM_GPUS 张GPU进行训练"

# 设置环境变量（优化）
export OMP_NUM_THREADS=$NUM_GPUS
export MKL_NUM_THREADS=$NUM_GPUS
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # 根据GPU架构调整
export CUDA_LAUNCH_BLOCKING=0

# 启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -- \
    main_distributed.py \
    --multi_gpu \
    "$@"
