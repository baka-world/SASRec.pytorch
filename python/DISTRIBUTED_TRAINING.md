# SASRec 多卡训练指南

## 环境要求

- PyTorch 1.9+
- CUDA 11.0+
- 4x NVIDIA V100 32GB

## 快速开始

### 方法1：使用启动脚本（推荐）

```bash
# 使用所有4张GPU
cd python
chmod +x train_distributed.sh
./train_distributed.sh ml-1m tisasrec_dist

# 使用2张GPU
NUM_GPUS=2 ./train_distributed.sh ml-1m tisasrec_dist
```

### 方法2：直接使用 torch.distributed.launch

```bash
cd python

# 4卡训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_4gpu \
    --batch_size=512 \
    --num_epochs=200 \
    --use_amp \
    --multi_gpu

# 2卡训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_2gpu \
    --batch_size=256 \
    --use_amp \
    --multi_gpu
```

### 方法3：使用 torchrun（PyTorch 2.0+）

```bash
cd python

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_torchrun \
    --batch_size=512 \
    --use_amp \
    --multi_gpu
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 总batch大小，会自动分配到各卡 | 128 |
| `--use_amp` | 启用自动混合精度，节省显存 | True |
| `--multi_gpu` | 启用分布式训练 | False |
| `--num_workers` | 数据加载线程数 | 3 |

## 性能优化建议

1. **增加batch size**：4x32GB V100可以承载更大的batch
   ```bash
   --batch_size=1024
   ```

2. **启用AMP**：节省显存并加速训练
   ```bash
   --use_amp
   ```

3. **增加学习率**：batch增大时可以适当增大学习率
   ```bash
   --lr=0.002
   ```

4. **完整训练命令示例**：
   ```bash
   python -m torch.distributed.launch \
       --nproc_per_node=4 \
       --master_port=29500 \
       main_distributed.py \
       --dataset=ml-1m \
       --train_dir=tisasrec_4gpu_large \
       --batch_size=1024 \
       --lr=0.002 \
       --num_epochs=300 \
       --num_heads=4 \
       --num_blocks=3 \
       --hidden_units=100 \
       --maxlen=100 \
       --dropout_rate=0.2 \
       --use_amp \
       --multi_gpu
   ```

## 监控训练

```bash
# 查看GPU使用情况
nvidia-smi

# 使用TensorBoard监控（如果已安装）
tensorboard --logdir=ml-1m_tisasrec_4gpu
```

## 预期加速效果

| 配置 | 相对单卡加速 |
|------|-------------|
| 1x V100 | 1.0x (baseline) |
| 2x V100 | ~1.8x |
| 4x V100 | ~3.5x |

> 注：实际加速比取决于数据加载效率和模型架构。SASRec由于序列依赖，可能无法达到完美的线性加速。

## 常见问题

### Q: 训练开始时报错 "ncclUnhandle"
A: 确保使用 `NCCL_DEBUG=INFO` 环境变量检查，或尝试使用 `--backend=gloo`

### Q: 显存使用不均衡
A: 这在首次迭代中是正常的，各卡会逐步平衡显存使用

### Q: 如何中断后继续训练
A: 使用 `--state_dict_path` 参数指定checkpoint路径
