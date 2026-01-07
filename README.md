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

# Train SASRec
python main.py --dataset=ml-1m --train_dir=sasrec_base

# Train SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --use_mhc

# Train TiSASRec
python main.py --dataset=ml-1m --train_dir=tisasrec --use_time

# Train TiSASRec + mHC
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc --use_time --use_mhc
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
| `--hidden_units` | 256 | Hidden dimension |
| `--num_blocks` | 3 | Transformer blocks |
| `--num_heads` | 2 | Attention heads |
| `--dropout_rate` | 0.2 | Dropout rate |
| `--l2_emb` | 0.0 | L2 regularization |
| `--device` | cuda | cuda or cpu |
| `--use_time` | False | Enable TiSASRec |
| `--use_mhc` | False | Enable mHC |
| `--mhc_expansion_rate` | 4 | mHC expansion factor |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn iterations |
| `--time_span` | 100 | Time interval range |
| `--time_unit` | hour | Time unit (second/minute/hour/day) |

## Training Tips

For better performance, try larger models:

```bash
# Large model
python main.py --dataset=ml-1m --train_dir=sasrec_large \
    --hidden_units=256 --num_blocks=3 --num_heads=2

# Extra large model
python main.py --dataset=ml-1m --train_dir=sasrec_xl \
    --hidden_units=512 --num_blocks=4 --num_heads=4 --batch_size=256
```

If loss stagnates around 0.45:
- Increase model capacity (`--hidden_units`, `--num_blocks`)
- Use learning rate decay (`--lr_decay_step=500 --lr_decay_rate=0.95`)
- Add warmup (`--warmup_steps=100`)

## Memory Issues

If CUDA out of memory:

| GPU Memory | Recommended Config |
|------------|-------------------|
| 8 GB | `--batch_size 32 --hidden_units 64 --maxlen 100` |
| 16 GB | `--batch_size 64 --hidden_units 128` |
| 24+ GB | `--batch_size 128 --hidden_units 256` (default) |

Other options:
- Reduce batch_size: `--batch_size=64`
- Reduce sequence length: `--maxlen=100`
- Reduce model size: `--hidden_units=128`
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
