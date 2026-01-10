# Distributed Training Guide

## Overview

This guide covers multi-GPU training using PyTorch's DistributedDataParallel (DDP) for SASRec/TiSASRec models.

## Requirements

- PyTorch 2.0+
- CUDA 11.0+
- Multiple GPUs (tested with 2x, 4x V100/A100)
- NCCL backend (recommended) or Gloo

## Quick Start

### Method 1: Using Launch Script (Recommended)

```bash
cd python

# Use all available GPUs
./train_distributed.sh ml-1m tisasrec_dist

# Use specific GPU count
NUM_GPUS=2 ./train_distributed.sh ml-1m tisasrec_2gpu

# Custom settings
BATCH_SIZE=256 ./train_distributed.sh ml-1m tisasrec_large
```

### Method 2: Using torch.distributed.launch

```bash
cd python

# 4 GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_4gpu \
    --batch_size=512 \
    --use_amp \
    --multi_gpu

# 2 GPU training
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_2gpu \
    --batch_size=256 \
    --use_amp \
    --multi_gpu
```

### Method 3: Using torchrun (PyTorch 2.0+)

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

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multi_gpu` | False | Enable distributed training |
| `--batch_size` | 128 | Total batch size (auto-split across GPUs) |
| `--use_amp` | False | Enable automatic mixed precision |
| `--backend` | nccl | Communication backend (nccl/gloo) |
| `--master_port` | 29500 | Master port for communication |
| `--num_workers` | 3 | Data loader threads per GPU |

## Model Selection with DDP

All model variants work with DDP:

| Mode | Command | Description |
|------|---------|-------------|
| TiSASRec + mHC | Default | Time-aware + manifold-constrained HC |
| TiSASRec | `--no_mhc` | Time-aware only |
| SASRec + mHC | `--no_time` | Standard attention + mHC |
| SASRec | `--no_time --no_mhc` | Baseline model |

## Performance Optimization

### Batch Size Scaling

| Configuration | Single GPU | 4x GPU |
|---------------|------------|--------|
| batch_size | 128 | 512 |
| lr | 0.001 | 0.002-0.004 |
| Expected Speedup | 1x | ~3.5x |

**Note**: Learning rate should scale roughly with batch size.

### AMP (Automatic Mixed Precision)

AMP reduces memory usage and can improve speed:

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

### Memory Optimization

For large models or small GPUs:

```bash
# Reduce memory usage
python main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_lowmem \
    --batch_size=64 \
    --maxlen=100 \
    --use_amp \
    --mhc_no_amp
```

## Distributed Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed Training                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │ GPU 0 (Rank 0) │───▶ Model Replica (DDP)                 │
│  │ Master Process │                                          │
│  │ - Coordinates NCCL                                       │
│  │ - Saves checkpoints                                      │
│  └─────────────┘                                            │
│         │                                                   │
│  ┌──────┴──────┐                                            │
│  │  NCCL/Gloo  │─── Gradient Synchronization                │
│  │   Backend   │    - AllReduce for gradients               │
│  └──────┴──────┘    - Parameter synchronization            │
│         │                                                   │
│  ┌──────┼──────┬──────┐                                     │
│  ▼      ▼      ▼      ▼                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                 │
│ │GPU0│ │GPU1│ │GPU2│ │GPU3│                                 │
│ │Rank│ │Rank│ │Rank│ │Rank│                                 │
│ │ 0  │ │ 1  │ │ 2  │ │ 3  │                                 │
│ └────┘ └────┘ └────┘ └────┘                                 │
│  - Same model, different data batches                        │
│  - Synchronized gradients                                    │
└─────────────────────────────────────────────────────────────┘
```

## Monitoring Training

### GPU Usage

```bash
# View GPU usage
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

### TensorBoard

```bash
# Install tensorboard if needed
pip install tensorboard

# Launch tensorboard
tensorboard --logdir=ml-1m_tisasrec_4gpu
```

### Custom Logging

The training script logs to both console and file:

```bash
# Training logs
tail -f ml-1m_tisasrec_4gpu/train.log

# Validation results
tail -f ml-1m_tisasrec_4gpu/val.log
```

## Checkpoint Management

### Automatic Checkpoints

Checkpoints are saved automatically:
- Every epoch: `checkpoint_*.pth`
- Best model: `best.pth`
- Final model: `final.pth`

### Resume Training

```bash
python main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_resume \
    --state_dict_path=ml-1m_tisasrec_4gpu/checkpoint_10.pth \
    --inference_only=false
```

## Expected Acceleration

| Configuration | Relative Speedup | Notes |
|---------------|------------------|-------|
| 1x V100 | 1.0x | Baseline |
| 2x V100 | ~1.8x | Good scaling |
| 4x V100 | ~3.5x | Near-linear |
| 8x A100 | ~7.0x | Excellent scaling |

**Note**: Actual speedup depends on:
- Data loading efficiency
- Model architecture
- GPU interconnect (NVLink helps)
- Batch size

## Common Issues

### NCCL Errors

```bash
# Error: ncclUnhandle
# Solution: Set environment variable
NCCL_DEBUG=INFO python main.py --multi_gpu ...

# Or use gloo backend (slower but more compatible)
python main.py --multi_gpu --backend=gloo
```

### Out of Memory

```bash
# Reduce batch size
--batch_size=256

# Reduce sequence length
--maxlen=100

# Use AMP
--use_amp

# Disable mHC if not needed
--no_mhc
```

### Training Instability

```bash
# Reduce learning rate
--lr=0.0005

# Increase warmup steps
--warmup_steps=500

# Use gradient clipping
# (already enabled in main.py)
```

### Data Loading Bottleneck

```bash
# Increase data loader workers
--num_workers=8

# Use pin_memory
# (enabled by default)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NCCL_DEBUG` | NCCL debug level (INFO/WARNING) |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `OMP_NUM_THREADS` | CPU threads per process |

## Multi-Node Training

For multi-node training:

```bash
# Node 0 (master)
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

# Node 1
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

## Best Practices

1. **Use NCCL Backend**: Faster than Gloo on NVIDIA GPUs
2. **Enable AMP**: Saves memory and can speed up training
3. **Scale LR with Batch Size**: lr_new = lr_base × (batch_new / batch_base)
4. **Use Gradient Checkpointing**: If memory is tight (not implemented yet)
5. **Monitor GPU Utilization**: Ensure GPUs are fully utilized
6. **Save Checkpoints Frequently**: For long training runs

## Troubleshooting Checklist

- [ ] Verify NCCL is working: `NCCL_DEBUG=INFO python test_distributed.py`
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Verify data loading: Check for data loading bottlenecks
- [ ] Monitor training loss: Look for NaN or divergence
- [ ] Check disk I/O: Slow disk can bottleneck data loading
- [ ] Verify network connectivity: For multi-node training
