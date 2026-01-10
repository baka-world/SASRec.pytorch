# Manifold-Constrained Hyper-Connections (mHC) - Quick Start

> **Note**: This document provides a quick overview. For detailed documentation, see [mHC Guide](../advanced/mhc-guide.md)

## Overview

mHC extends the traditional residual connection paradigm by projecting the residual connection space onto a constrained manifold to restore the identity mapping property.

## Key Benefits

- **Training Stability**: Doubly stochastic constraint ensures signal norm preservation
- **Scalability**: Stable gradients across deep networks
- **Performance**: Improved representation learning through multi-stream information exchange

## Usage

### Enable mHC

mHC is enabled by default for TiSASRec:

```bash
# TiSASRec + mHC (default)
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc

# SASRec + mHC (no time features)
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

# Disable mHC
python main.py --dataset=ml-1m --train_dir=no_mhc --no_mhc
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mhc_expansion_rate` | 4 | Expansion factor n (2-8) |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn-Knopp iterations (10-50) |
| `--mhc_init_gate` | 0.01 | Initial gating factor α |
| `--mhc_no_amp` | False | Disable mHC AMP calculation |

## Architecture

The mHC residual connection follows:

```
x_{l+1} = H_l^{res} × x_l + H_l^{post}^T × F(H_l^{pre} × x_l, W_l)
```

Where:
- `×` denotes matrix multiplication
- `H_l^{res}` is projected to a doubly stochastic matrix
- `H_l^{pre}` and `H_l^{post}` are non-negative (Sigmoid activated)
- `F` is the residual function (attention + FFN)

## Troubleshooting

### NaN During Training

1. Enable mHC no-AMP:
   ```bash
   python main.py --mhc_no_amp
   ```

2. Reduce expansion rate:
   ```bash
   python main.py --mhc_expansion_rate=2
   ```

3. Increase Sinkhorn iterations:
   ```bash
   python main.py --mhc_sinkhorn_iter=50
   ```

### Memory Issues

```bash
# Use smaller batch size
--batch_size=64

# Use AMP
--use_amp

# Reduce sequence length
--maxlen=100
```

## More Information

For detailed documentation, see:
- [mHC Guide](../advanced/mhc-guide.md) - Complete mHC documentation
- [Architecture](../architecture.md) - System architecture overview
- [Distributed Training](../advanced/distributed-training.md) - Multi-GPU training

## Reference

```bibtex
@misc{deepseek2025mhcsurvey,
      title={mHC: Manifold-Constrained Hyper-Connections},
      author={DeepSeek-AI},
      year={2025}
}
```
