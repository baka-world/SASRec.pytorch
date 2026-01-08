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

## Training Tips

For better performance, try larger models:

```bash
# Medium model
python main.py --dataset=ml-1m --train_dir=sasrec_medium \
    --hidden_units=128 --num_blocks=3 --batch_size=128

# Large model
python main.py --dataset=ml-1m --train_dir=sasrec_large \
    --hidden_units=256 --num_blocks=3 --batch_size=128
```

If loss stagnates around 0.45:
- Increase model capacity (`--hidden_units`, `--num_blocks`)
- Use learning rate decay (`--lr_decay_step=500 --lr_decay_rate=0.95`)
- Add warmup (`--warmup_steps=100`)

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
