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
├── main.py              # Unified training entry point
│
├── model.py             # Base models
│   ├── SASRec           # Standard self-attentive sequential rec
│   └── TiSASRec         # Time-aware self-attention
│
├── model_mhc.py         # Models with mHC variant
│   ├── SASRec_mHC       # SASRec with Manifold-Constrained HC
│   └── TiSASRec_mHC     # TiSASRec with Manifold-Constrained HC
│
├── utils.py             # Core utilities
│   ├── DataPartition    # Train/Valid/Test split
│   ├── WarpSampler      # Negative sampling
│   ├── WarpSamplerWithTime  # Time-aware sampling
│   ├── evaluate()       # SASRec evaluation
│   └── evaluate_tisasrec()  # TiSASRec evaluation
│
└── convert_ml1m.py      # MovieLens 1M data converter
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

## Data Flow

```
Training:
┌─────────────────────────────────────────────────────────────┐
│  raw_data.txt (UserID, MovieID, Timestamp)                  │
│                      ↓                                      │
│  data_partition() → user_train, user_valid, user_test       │
│                      ↓                                      │
│  WarpSampler → (u, seq, pos, neg, [time_mat]) batches      │
│                      ↓                                      │
│  model.forward() → (pos_logits, neg_logits)                 │
│                      ↓                                      │
│  BCEWithLogitsLoss + L2 regularization                      │
│                      ↓                                      │
│  Adam optimizer update                                       │
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

1. **Unified Training Script**: Single `main.py` supports all model combinations via flags

2. **Negative Sampling**: In-batch negative sampling with uniform item distribution

3. **Time Encoding**: Relative time intervals discretized to `[1, time_span]`

4. **mHC Integration**: Drop-in replacement for standard residual connections

5. **Evaluation**: Full ranking over all items (not sampled), report NDCG@10 and HR@10

## Dependencies

```
torch >= 1.9.0
numpy
tqdm (optional, for progress bars)
```
