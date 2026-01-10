# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │───▶│ Sampler  │───▶│  Model   │───▶│  Loss    │  │
│  │  Loader  │    │ (Warp)   │    │ (Forward)│    │ Compute  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                               │                │        │
│       ▼                               ▼                ▼        │
│  ┌──────────┐                 ┌──────────┐    ┌──────────┐      │
│  │  Dataset │                 │ Backward │    │ Optimizer│      │
│  │  (Utils) │                 │ (Autograd)│   │ (Adam)   │      │
│  └──────────┘                 └──────────┘    └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
python/
├── main.py              # Unified training entry point (single/multi-GPU)
├── main_distributed.py  # Distributed training entry point (DDP)
├── run_experiments_v2.py # Experiment manager with GPU scheduling
│
├── model.py             # Base models (SASRec, TiSASRec)
├── model_mhc.py         # Models with mHC variant
│   ├── SASRec_mHC       # SASRec with Manifold-Constrained HC
│   ├── TiSASRec_mHC     # TiSASRec with Manifold-Constrained HC
│   └── mHCResidual      # mHC residual connection implementation
│
├── utils.py             # Core utilities
│   ├── DataPartition    # Train/Valid/Test split
│   ├── WarpSampler      # Negative sampling
│   ├── WarpSamplerWithTime  # Time-aware sampling
│   ├── evaluate()       # SASRec evaluation
│   ├── evaluate_tisasrec()  # TiSASRec evaluation
│   └── get_user_timestamps() # Timestamp extraction
│
├── convert_ml1m.py      # MovieLens 1M data converter
└── test/                # Unit tests
```

## Model Architecture

### SASRec

```
Input: (batch, seq_len) item indices
    │
    ▼
┌───────────────────┐
│   Item Embedding  │──▶ (batch, seq_len, hidden_units)
│   (n_items, C)    │
└───────────────────┘
    │
    ▼
┌───────────────────┐
│  Positional Emb   │──▶ (1, max_len, C)
│   (learned)       │
└───────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   Transformer Blocks                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Block i:                                         │   │
│  │  1. Multi-Head Self-Attention                    │   │
│  │     Q, K, V = Linear(C→C) × 3                    │   │
│  │     Attention = softmax(QK^T / √d) V             │   │
│  │                                                  │   │
│  │  2. Add & LayerNorm                              │   │
│  │                                                  │   │
│  │  3. Feed-Forward Network                         │   │
│  │     FFN(x) = W2(ReLU(W1 x + b1)) + b2           │   │
│  │                                                  │   │
│  │  4. Add & LayerNorm                              │   │
│  └─────────────────────────────────────────────────┘   │
│                          :                              │
│                    (num_blocks)                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Final LayerNorm                         │
│         +                                │
│  Linear(C → n_items) → logits           │
└─────────────────────────────────────────┘
```

### TiSASRec (Time-Aware)

```
SASRec + Time Matrix Injection in Attention:

Standard Attention:
    A_ij = softmax(Q_i · K_j^T)

TiSASRec Attention:
    A_ij = softmax(Q_i · K_j^T 
                   + Q_i · abs_pos_K_i^T 
                   + time_matrix_K_j · Q_i)

Additional Components:
┌─────────────────────────────────────────┐
│  Time Matrix Embedding                  │
│  (time_span, hidden_units)              │
│  - Encodes relative time intervals      │
│  - Shared across attention heads        │
└─────────────────────────────────────────┘
```

### mHC (Manifold-Constrained Hyper-Connections)

Replaces standard residual connections:

```
Standard Residual:
    x_{l+1} = x_l + F(x_l)

mHC Residual:
    x_{l+1} = H_res × x_l + H_post^T × F(H_pre × x_l)

Where:
    - H_pre: Input projection (nC → n), sigmoid activated
    - H_post: Output projection (nC → n), sigmoid activated  
    - H_res: Residual mixing (nC → n²), Sinkhorn-Knopp constrained

mHC Module Structure:
┌──────────────────────────────────────────────────────────────┐
│ Input: x ∈ (batch, seq, C)                                   │
│                                                              │
│ 1. Expand: repeat x → x_exp ∈ (batch, seq, n, C)            │
│                                                              │
│ 2. H_pre: (batch, seq, n) ← sigmoid(Linear(x, n))           │
│                                                              │
│ 3. H_post: (batch, seq, n) ← sigmoid(Linear(x, n))          │
│                                                              │
│ 4. H_res: (batch, seq, n, n)                                │
│    - Initialize as doubly stochastic matrix                  │
│    - Sinkhorn-Knopp iterations (default: 20)                 │
│                                                              │
│ 5. Residual: x_res = H_res × x_exp → (batch, seq, n, C)     │
│                                                              │
│ 6. Output path: f_out = H_post × F(H_pre × x) → (batch, seq, C)
│                                                              │
│ 7. Combine: sum_n(x_res) + f_out                            │
└──────────────────────────────────────────────────────────────┘
```

#### Sinkhorn-Knopp Algorithm

```python
def sinkhorn_knopp(M, max_iter=20):
    """Project matrix onto doubly stochastic manifold"""
    n = M.shape[-1]
    M = torch.exp(M)  # M^(0) = exp(H_l^{res})
    for _ in range(max_iter):
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + 1e-12)  # Row normalization
        
        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + 1e-12)  # Column normalization
    
    return M  # Doubly stochastic matrix
```

#### mHC Integration Points

mHC replaces standard residual connections at two points in each Transformer block:

```
Standard Block:
    x = LayerNorm(x + Attention(x))
    x = LayerNorm(x + FFN(x))

mHC Block (in model_mhc.py):
    x = mHCResidual(x, Attention(x))    # Attention residual
    x = mHCResidual(x, FFN(x))          # FFN residual
```

## Data Flow

```
Training:
┌─────────────────────────────────────────────────────────────┐
│  raw_data.txt (UserID, MovieID, Timestamp)                  │
│                      ↓                                      │
│  data_partition() → user_train, user_valid, user_test       │
│                      ↓                                      │
│  WarpSampler/WarpSamplerWithTime → batches                  │
│      (u, seq, pos, neg, [time_mat])                         │
│                      ↓                                      │
│  model.forward() → (pos_logits, neg_logits)                 │
│                      ↓                                      │
│  BCEWithLogitsLoss + L2 regularization                      │
│                      ↓                                      │
│  Adam/AdamW optimizer update                                 │
└─────────────────────────────────────────────────────────────┘

Inference:
┌─────────────────────────────────────────────────────────────┐
│  user_history (sequence)                                     │
│                      ↓                                      │
│  model.predict() → item scores for all items                │
│                      ↓                                      │
│  Top-K evaluation (NDCG@10, HR@10)                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Unified Training Script**: Single `main.py` supports all model combinations via flags:
   - `--no_time`: Use SASRec instead of TiSASRec
   - `--no_mhc`: Disable mHC module
   - `--multi_gpu`: Enable DDP distributed training

2. **Negative Sampling**: In-batch negative sampling with uniform item distribution

3. **Time Encoding**: Relative time intervals discretized to `[1, time_span]`

4. **mHC Integration**: Drop-in replacement for standard residual connections

5. **Evaluation**: Full ranking over all items (not sampled), report NDCG@10 and HR@10

6. **AMP Support**: Automatic mixed precision training for memory efficiency

## Distributed Training

The project supports multi-GPU training using PyTorch DDP:

```
Single GPU:
    python main.py --dataset=ml-1m --train_dir=exp1

Multi-GPU (4x):
    python -m torch.distributed.launch --nproc_per_node=4 \
        main.py --dataset=ml-1m --train_dir=exp_dist --multi_gpu

Multi-GPU with AMP:
    python -m torch.distributed.launch --nproc_per_node=4 \
        main.py --dataset=ml-1m --train_dir=exp_amp --multi_gpu --use_amp
```

### DDP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed Training                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │ GPU 0 (Rank 0) │───▶ Model Replica (DDP)                 │
│  │ Master Process │                                          │
│  └─────────────┘                                            │
│         │                                                   │
│  ┌──────┴──────┐                                            │
│  │  NCCL/Gloo  │─── Gradient Synchronization                │
│  │   Backend   │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│  ┌──────┼──────┬──────┐                                     │
│  ▼      ▼      ▼      ▼                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                 │
│ │GPU0│ │GPU1│ │GPU2│ │GPU3│                                 │
│ └────┘ └────┘ └────┘ └────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

## Dependencies

```
torch >= 2.0.0 (recommended)
numpy
tqdm (optional, for progress bars)
rich (optional, for experiment manager)
```

## Experiment Management

The `run_experiments_v2.py` script provides:

- **GPU Auto-Scheduling**: Automatically assigns experiments to available GPUs
- **Memory Monitoring**: Waits for available GPU memory before starting
- **Concurrent Execution**: Runs multiple experiments in parallel
- **Result Collection**: Parses and stores NDCG@10 and HR@10 results

```python
# Example usage
from run_experiments_v2 import ExperimentManager

manager = ExperimentManager()
manager.add_experiment("sasrec_base", gpu=0, cmd="python main.py --no_time --no_mhc")
manager.add_experiment("tisasrec_mhc", gpu=-1, cmd="python main.py")  # Auto-assign GPU
manager.run_all()
```

## LayerNorm Strategy

This implementation supports two LayerNorm strategies:

### Pre-LN (`--norm_first`)
```
x = x + Dropout(Attention(LN(x)))
x = x + Dropout(FFN(LN(x)))
```
LayerNorm is applied **before** the sub-layer.

### Post-LN (default, `--norm_first=False`)
```
x = LN(x + Dropout(Attention(x)))
x = LN(x + Dropout(FFN(x)))
```
LayerNorm is applied **after** the residual connection.

**Recommendation**: Post-LN generally performs better for SASRec/TiSASRec with num_blocks <= 3.
