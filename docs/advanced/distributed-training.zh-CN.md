# 分布式训练指南

## 概述

本指南介绍如何使用PyTorch的DistributedDataParallel (DDP)进行SASRec/TiSASRec模型的多卡训练。

## 环境要求

- PyTorch 2.0+
- CUDA 11.0+
- 多张GPU（已在2x、4x V100/A100上测试）
- NCCL后端（推荐）或Gloo

## 快速开始

### 方法1：使用启动脚本（推荐）

```bash
cd python

# 使用所有可用GPU
./train_distributed.sh ml-1m tisasrec_dist

# 使用指定GPU数量
NUM_GPUS=2 ./train_distributed.sh ml-1m tisasrec_2gpu

# 自定义设置
BATCH_SIZE=256 ./train_distributed.sh ml-1m tisasrec_large
```

### 方法2：使用torch.distributed.launch

```bash
cd python

# 4卡训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_4gpu \
    --batch_size=512 \
    --use_amp \
    --multi_gpu

# 2卡训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_2gpu \
    --batch_size=256 \
    --use_amp \
    --multi_gpu
```

### 方法3：使用torchrun（PyTorch 2.0+）

```bash
cd python

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_torchrun \
    --batch_size=512 \
    --use_amp \
    --multi_gpu
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--multi_gpu` | False | 启用分布式训练 |
| `--batch_size` | 128 | 总batch大小（自动分配到各卡） |
| `--use_amp` | False | 启用自动混合精度 |
| `--backend` | nccl | 通信后端（nccl/gloo） |
| `--master_port` | 29500 | 主节点通信端口 |
| `--num_workers` | 3 | 每卡数据加载线程数 |

## 模型选择

所有模型变体都支持DDP：

| 模式 | 命令 | 说明 |
|------|------|------|
| TiSASRec + mHC | 默认 | 时序感知 + 流形约束HC |
| TiSASRec | `--no_mhc` | 仅时序感知 |
| SASRec + mHC | `--no_time` | 标准注意力 + mHC |
| SASRec | `--no_time --no_mhc` | 基准模型 |

## 性能优化

### Batch Size缩放

| 配置 | 单卡 | 4卡 |
|------|------|-----|
| batch_size | 128 | 512 |
| lr | 0.001 | 0.002-0.004 |
| 预期加速 | 1x | ~3.5x |

**注意**：学习率应随batch size大致缩放。

### AMP（自动混合精度）

AMP可减少内存使用并可能提升速度：

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_amp \
    --batch_size=1024 \
    --use_amp \
    --multi_gpu
```

### 内存优化

对于大模型或小显存GPU：

```bash
python main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_lowmem \
    --batch_size=64 \
    --maxlen=100 \
    --use_amp \
    --mhc_no_amp
```

## 分布式训练架构

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式训练                                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │ GPU 0 (Rank 0) │───▶ 模型副本 (DDP)                       │
│  │ 主进程        │                                          │
│  │ - 协调NCCL   │                                          │
│  │ - 保存检查点  │                                          │
│  └─────────────┘                                            │
│         │                                                   │
│  ┌──────┴──────┐                                            │
│  │  NCCL/Gloo  │─── 梯度同步                                 │
│  │   后端      │    - 梯度的AllReduce                        │
│  └──────┬──────┘    - 参数同步                              │
│         │                                                   │
│  ┌──────┼──────┬──────┐                                     │
│  ▼      ▼      ▼      ▼                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                 │
│ │GPU0│ │GPU1│ │GPU2│ │GPU3│                                 │
│ │Rank│ │Rank│ │Rank│ │Rank│                                 │
│ │ 0  │ │ 1  │ │ 2  │ │ 3  │                                 │
│ └────┘ └────┘ └────┘ └────┘                                 │
│  - 相同模型，不同数据批次                                      │
│  - 同步梯度                                                  │
└─────────────────────────────────────────────────────────────┘
```

## 监控训练

### GPU使用

```bash
# 查看GPU使用情况
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

### TensorBoard

```bash
# 安装tensorboard（如果需要）
pip install tensorboard

# 启动tensorboard
tensorboard --logdir=ml-1m_tisasrec_4gpu
```

### 自定义日志

训练日志同时输出到控制台和文件：

```bash
# 训练日志
tail -f ml-1m_tisasrec_4gpu/train.log

# 验证结果
tail -f ml-1m_tisasrec_4gpu/val.log
```

## 检查点管理

### 自动检查点

自动保存检查点：
- 每轮：`checkpoint_*.pth`
- 最佳模型：`best.pth`
- 最终模型：`final.pth`

### 恢复训练

```bash
python main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_resume \
    --state_dict_path=ml-1m_tisasrec_4gpu/checkpoint_10.pth \
    --inference_only=false
```

## 预期加速效果

| 配置 | 相对加速 | 说明 |
|------|---------|------|
| 1x V100 | 1.0x | 基准 |
| 2x V100 | ~1.8x | 良好扩展 |
| 4x V100 | ~3.5x | 近似线性 |
| 8x A100 | ~7.0x | 优秀扩展 |

**注意**：实际加速取决于：
- 数据加载效率
- 模型架构
- GPU互联（NVLink有帮助）
- Batch size

## 常见问题

### NCCL错误

```bash
# 错误：ncclUnhandle
# 解决方案：设置环境变量
NCCL_DEBUG=INFO python main.py --multi_gpu ...

# 或使用gloo后端（较慢但更兼容）
python main.py --multi_gpu --backend=gloo
```

### 显存不足

```bash
# 减小batch size
--batch_size=256

# 减小序列长度
--maxlen=100

# 使用AMP
--use_amp

# 如果不需要mHC
--no_mhc
```

### 训练不稳定

```bash
# 减小学习率
--lr=0.0005

# 增加预热步数
--warmup_steps=500

# 使用梯度裁剪
#（已在main.py中启用）
```

### 数据加载瓶颈

```bash
# 增加数据加载线程数
--num_workers=8

# 使用pin_memory
#（默认启用）
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_DEBUG` | NCCL调试级别（INFO/WARNING） |
| `NCCL_SOCKET_IFNAME` | NCCL网络接口 |
| `CUDA_VISIBLE_DEVICES` | GPU选择 |
| `OMP_NUM_THREADS` | 每进程CPU线程数 |

## 多节点训练

对于多节点训练：

```bash
# 节点0（主节点）
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_multinode \
    --batch_size=2048

# 节点1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_multinode \
    --batch_size=2048
```

## 最佳实践

1. **使用NCCL后端**：在NVIDIA GPU上比Gloo更快
2. **启用AMP**：节省内存并可能加速训练
3. **学习率随Batch Size缩放**：lr_new = lr_base × (batch_new / batch_base)
4. **使用梯度检查点**：如果内存紧张（尚未实现）
5. **监控GPU利用率**：确保GPU被充分利用
6. **频繁保存检查点**：用于长时间训练

## 故障排除清单

- [ ] 验证NCCL正常工作：`NCCL_DEBUG=INFO python test_distributed.py`
- [ ] 检查GPU显存：`nvidia-smi`
- [ ] 验证数据加载：检查数据加载瓶颈
- [ ] 监控训练损失：查找NaN或发散
- [ ] 检查磁盘IO：慢磁盘可能成为数据加载瓶颈
- [ ] 验证网络连接：用于多节点训练
