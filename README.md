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
| `--maxlen` | 200 | Max sequence length |
| `--hidden_units` | 50 | Hidden dimension |
| `--num_blocks` | 2 | Transformer blocks |
| `--num_heads` | 1 | Attention heads |
| `--dropout_rate` | 0.2 | Dropout rate |
| `--device` | cuda | cuda or cpu |
| `--use_time` | False | Enable TiSASRec |
| `--use_mhc` | False | Enable mHC |
| `--mhc_expansion_rate` | 4 | mHC expansion factor |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn iterations |
| `--time_span` | 100 | Time interval range |
| `--time_unit` | hour | Time unit (second/minute/hour/day) |

## Memory Issues

If CUDA out of memory:
- Reduce batch_size: `--batch_size=64`
- Use CPU: `--device=cpu`
- Reduce model size: `--num_blocks=1`

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
