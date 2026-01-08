# SASRec.pytorch

PyTorch implementation of sequential recommendation models: SASRec, TiSASRec, and their mHC (Manifold-Constrained Hyper-Connections) variants.

## Models

| Model | Description | Time-Aware | mHC Support |
|-------|-------------|------------|-------------|
| SASRec | Self-Attentive Sequential Recommendation | No | Yes |
| TiSASRec | Time Interval Aware Self-Attention | Yes | Yes |

## Quick Start

```bash
# Install dependencies
pip install torch numpy

# Prepare data (MovieLens 1M)
cd data && wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && cd ..
python convert_ml1m.py

# Train SASRec (baseline)
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc

# Train SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

# Train TiSASRec (default, uses time intervals)
python main.py --dataset=ml-1m --train_dir=tisasrec

# Train TiSASRec + mHC (default, time-aware with hyper-connections)
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset name |
| `--train_dir` | required | Output directory |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--lr_decay_step` | 1000 | LR decay step (per epoch) |
| `--lr_decay_rate` | 0.95 | LR decay rate |
| `--warmup_steps` | 100 | Warmup steps (0 to disable) |
| `--maxlen` | 200 | Max sequence length |
| `--hidden_units` | 50 | Hidden dimension |
| `--num_blocks` | 2 | Transformer blocks |
| `--num_heads` | 2 | Attention heads |
| `--dropout_rate` | 0.2 | Dropout rate |
| `--l2_emb` | 0.0 | L2 regularization |
| `--device` | cuda | cuda or cpu |
| `--no_time` | False | Disable TiSASRec (use SASRec) |
| `--no_mhc` | False | Disable mHC module |
| `--mhc_expansion_rate` | 4 | mHC expansion factor |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn iterations |
| `--time_span` | 100 | Time interval range |
| `--time_unit` | hour | Time unit (second/minute/hour/day) |

## Experimental Settings (Baseline Comparison)

Default parameters match the original SASRec paper for fair comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_units` | 50 | Embedding dimension |
| `num_blocks` | 2 | Transformer layers |
| `batch_size` | 128 | Training batch size |
| `maxlen` | 200 | Max sequence length |
| `dropout_rate` | 0.2 | Dropout ratio |
| `lr` | 0.001 | Learning rate |
| `num_epochs` | 1000 | Training epochs |

For comparison with `Result_Norm.md` baseline:
- SASRec baseline: `--no_time --no_mhc`
- TiSASRec: `--no_mhc`
- TiSASRec + mHC: (default)

## Hyperparameter Tuning Guide

This guide explains key hyperparameters and how to tune them for optimal performance.

### 1. Model Capacity Parameters

| Parameter | Range | Impact | Recommended Start |
|-----------|-------|--------|-------------------|
| `--hidden_units` | 50-512 | Embedding dimension, more = higher capacity | 50 (baseline) |
| `--num_blocks` | 1-6 | Transformer layers, more = deeper | 2 (baseline) |
| `--num_heads` | 1-8 | Attention heads, more = finer attention | 2 (baseline) |

**Tuning Strategy**:
- Start with baseline (50, 2, 2) to verify training
- Increase `hidden_units` first for more capacity
- Increase `num_blocks` for deeper networks
- Adjust `num_heads` proportionally to `hidden_units` (hidden_units % num_heads == 0)

### 2. Training Parameters

| Parameter | Range | Impact | Recommended Start |
|-----------|-------|--------|-------------------|
| `--batch_size` | 32-256 | Batch size, affects gradient stability | 128 (baseline) |
| `--lr` | 0.0001-0.005 | Learning rate | 0.001 (baseline) |
| `--warmup_steps` | 0-500 | Warmup iterations | 100 (baseline) |
| `--num_epochs` | 200-2000 | Training duration | 1000 (baseline) |

**Tuning Strategy**:
- Larger batch_size → more stable gradients, can use larger lr
- If loss oscillates: reduce lr or increase batch_size
- If loss decreases too slowly: increase lr (max 0.005)
- If loss NaN: reduce lr, check data

### 3. Regularization Parameters

| Parameter | Range | Impact | Recommended Start |
|-----------|-------|--------|-------------------|
| `--dropout_rate` | 0.0-0.5 | Dropout ratio | 0.2 (baseline) |
| `--l2_emb` | 0.0-0.001 | L2 regularization | 0.0 (baseline) |

**Tuning Strategy**:
- High overfitting (Train ↓, Valid ↑): increase dropout_rate to 0.3-0.4
- Low overfitting: reduce dropout_rate to 0.1-0.2
- Large models need more regularization

### 4. mHC Parameters (if using mHC)

| Parameter | Range | Impact | Recommended Start |
|-----------|-------|--------|-------------------|
| `--mhc_expansion_rate` | 2-8 | Manifold expansion factor | 4 (baseline) |
| `--mhc_sinkhorn_iter` | 10-50 | Sinkhorn iterations | 20 (baseline) |
| `--mhc_init_gate` | 0.001-0.1 | Initial gate value | 0.01 (baseline) |

**Tuning Strategy**:
- More GPU memory → increase `expansion_rate` to 6-8
- Better precision needed → increase `sinkhorn_iter` to 30-50
- Training instability → reduce `init_gate` to 0.005

### 5. TiSASRec Parameters (if using time features)

| Parameter | Range | Impact | Recommended Start |
|-----------|-------|--------|-------------------|
| `--time_span` | 50-500 | Time interval discretization range | 100 (baseline) |
| `--time_unit` | second/minute/hour/day | Time unit | hour (baseline) |

**Tuning Strategy**:
- Dense interactions (short intervals): increase `time_span` to 200-500
- Sparse interactions (long intervals): decrease `time_span` to 50-100

### 6. Learning Rate Scheduling

| Parameter | Impact | Recommended |
|-----------|--------|-------------|
| `--lr_decay_step` | Decay frequency (epochs) | 1000 (baseline) |
| `--lr_decay_rate` | Decay factor per step | 0.95 (baseline) |

**Alternative Schedulers**:
```bash
# Cosine annealing (smoother decay)
# Note: Not built-in, requires code modification

# Step decay (aggressive)
python main.py --dataset=ml-1m --train_dir=xxx \
    --lr_decay_step=200 --lr_decay_rate=0.9
```

### 7. Common Issues and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss NaN | Learning rate too high | Reduce `--lr` to 0.0005 |
| Loss oscillates | Batch size too small | Increase `--batch_size` |
| Loss stuck >0.5 | Learning rate too low | Increase `--lr` to 0.002 |
| Overfitting (valid ↓) | Too much capacity | Increase `--dropout_rate` |
| Underfitting (both ↓) | Too little capacity | Increase `--hidden_units` |
| GPU OOM | Model too large | Use `--mhc_no_amp`, reduce `--batch_size` |
| Slow convergence | LR decay too fast | Increase `--lr_decay_step` |

### 8. Recommended Tuning Workflow

**Phase 1: Baseline Verification (1-2 hours)**
```bash
# Default settings, verify training works
python main.py --dataset=ml-1m --train_dir=baseline_test
```

**Phase 2: Capacity Search (4-8 hours)**
```bash
# Test different model sizes
python main.py --dataset=ml-1m --train_dir=model_50  --hidden_units=50
python main.py --dataset=ml-1m --train_dir=model_128 --hidden_units=128 --num_blocks=3
python main.py --dataset=ml-1m --train_dir=model_256 --hidden_units=256 --num_blocks=3
```

**Phase 3: Learning Rate Fine-tuning (2-4 hours)**
```bash
# Test different learning rates
python main.py --dataset=ml-1m --train_dir=lr_0005 --lr=0.0005
python main.py --dataset=ml-1m --train_dir=lr_002  --lr=0.002
```

**Phase 4: Regularization Tuning (2-4 hours)**
```bash
# If overfitting observed
python main.py --dataset=ml-1m --train_dir=dropout_03 --dropout_rate=0.3
python main.py --dataset=ml-1m --train_dir=dropout_04 --dropout_rate=0.4
```

### 9. Quick Reference

| Goal | Command |
|------|---------|
| Baseline comparison | `--no_time --no_mhc` |
| Time-aware only | `--no_mhc` |
| Time + mHC (default) | (no flags) |
| High capacity | `--hidden_units=128 --num_blocks=3` |
| Very high capacity | `--hidden_units=256 --num_blocks=3` |
| Low memory | `--batch_size=32 --mhc_no_amp` |
| Fast exploration | `--num_epochs=200` |

## Memory Issues

If CUDA out of memory:

| GPU Memory | Recommended Config |
|------------|-------------------|
| 8 GB | `--batch_size 32 --hidden_units 50 --maxlen 100 --mhc_no_amp` |
| 16 GB | `--batch_size 64 --hidden_units 50` (default config) |
| 24+ GB | `--batch_size 128 --hidden_units 128` |

Other options:
- Reduce batch_size: `--batch_size=64`
- Reduce sequence length: `--maxlen=100`
- Reduce model size: `--num_blocks=1`
- Disable mHC AMP: `--mhc_no_amp`
- Use CPU: `--device=cpu`

## File Structure

```
SASRec.pytorch/
├── python/
│   ├── main.py         # Unified training script
│   ├── model.py        # SASRec/TiSASRec
│   ├── model_mhc.py    # SASRec/TiSASRec + mHC
│   ├── utils.py        # Utilities
│   └── convert_ml1m.py # Data converter
├── data/
├── docs/
└── README.md
```

## References

- [SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)](https://arxiv.org/abs/1808.09781)
- [TiSASRec: Time Interval Aware Self-Attention (Li et al., 2020)](https://arxiv.org/abs/2004.11780)
- [mHC: Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.03977)
