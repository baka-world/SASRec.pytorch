# Manifold-Constrained Hyper-Connections (mHC) Implementation

This directory contains an implementation of Manifold-Constrained Hyper-Connections (mHC) for the SASRec model, based on the paper "mHC: Manifold-Constrained Hyper-Connections" by DeepSeek-AI.

## Overview

mHC extends the traditional residual connection paradigm by:

1. **Expanding the residual stream width** by a factor of `n` (default: 4)
2. **Introducing three learnable mappings**:
   - `H_pre`: Input projection (with Sigmoid activation for non-negativity)
   - `H_post`: Output projection (with Sigmoid activation for non-negativity)  
   - `H_res`: Residual stream mixing (projected to doubly stochastic matrices)

3. **Constraining `H_res` to the Birkhoff polytope** using the Sinkhorn-Knopp algorithm
4. **Preserving the identity mapping property** for stable training at scale

## Key Benefits

- **Training Stability**: Doubly stochastic constraint ensures signal norm preservation
- **Scalability**: Stable gradients across deep networks
- **Performance**: Improved representation learning through multi-stream information exchange

## Usage

### Standard SASRec (without mHC)
```bash
python main.py --dataset=Beauty --train_dir=baseline
```

### SASRec with mHC
```bash
python main_mhc.py --dataset=Beauty --train_dir=mhc_test --use_mhc --mhc_expansion_rate=4
```

### Key Arguments

- `--use_mhc`: Enable mHC (default: False)
- `--mhc_expansion_rate`: Expansion factor `n` (default: 4)
- `--mhc_init_gate`: Initial gating factor α (default: 0.01)
- `--mhc_sinkhorn_iter`: Sinkhorn-Knopp iterations (default: 20)

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

## Reference

@misc{deepseek2025mhcsurvey,
      title={mHC: Manifold-Constrained Hyper-Connections},
      author={DeepSeek-AI},
      year={2025}
}
