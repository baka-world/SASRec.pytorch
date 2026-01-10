# Documentation Changelog

## [Unreleased]

### Added

- **Unified Documentation Structure** (`docs/`):
  - Created `docs/README.md` as documentation index
  - Created `docs/concepts/background.md` - Background knowledge for sequential recommendation
  - Created `docs/advanced/mhc-guide.md` - mHC usage and troubleshooting guide
  - Created `docs/advanced/distributed-training.md` - Distributed training guide

### Updated

- **`ARCHITECTURE.md`**:
  - Added mHC module structure and Sinkhorn-Knopp algorithm details
  - Added DDP distributed training architecture
  - Added experiment manager description (`run_experiments_v2.py`)
  - Added LayerNorm strategy comparison (Pre-LN vs Post-LN)
  - Updated directory structure to reflect latest code organization

### Removed

- Deprecated `docs/mHC_manifold_constrained_hyper_connections.md` (original paper, not implementation guide)
  - Content moved to `docs/advanced/mhc-guide.md`

### Fixed

- Synchronized all documentation with latest code implementation
- Updated parameter defaults to match `main.py`:
  - `--lr_decay_rate`: 0.98 (was 0.95)
  - `--warmup_steps`: 200 (was 100)
  - `--num_epochs`: 300 (was 1000)
  - `--mhc_sinkhorn_iter`: Added (default 20)
  - `--mhc_no_amp`: Added flag

## [Previous Documentation] - Before Reorganization

### Original Documentation Structure

```
SASRec.pytorch/
├── README.md                 # Main documentation
├── ARCHITECTURE.md           # Architecture overview
├── Result_Norm.md           # Results normalization
├── docs/
│   ├── MHC_README.md       # mHC quick start
│   └── mHC_paper.md        # Original paper (65KB)
├── python/
│   ├── BACKGROUND_KNOWLEDGE.md  # Background concepts
│   └── DISTRIBUTED_TRAINING.md  # Distributed training
└── latex/
    └── README.md
```

### Issues Identified

1. **Documentation Fragmentation**: 6+ locations for documentation
2. **Content Duplication**: README and docs/ content overlap
3. **Outdated Information**: Parameters not matching code
4. **Missing Content**: No experiment management docs
5. **Organization**: No clear hierarchy or index

## Key Changes Summary

### Parameter Updates

| Parameter | Old Value | New Value | Source |
|-----------|-----------|-----------|--------|
| `--lr_decay_rate` | 0.95 | 0.98 | main.py:113 |
| `--warmup_steps` | 100 | 200 | main.py:115 |
| `--num_epochs` | 1000 | 300 | main.py:122 |
| `--mhc_sinkhorn_iter` | N/A | 20 | main.py:179 |
| `--mhc_init_gate` | N/A | 0.01 | main.py:176 |

### New Features Documented

1. **run_experiments_v2.py**: Experiment manager with GPU scheduling
2. **AMP Support**: `--use_amp` flag
3. **AdamW Support**: `--use_adamw` and `--weight_decay`
4. **Early Stopping**: `--early_stop_patience` and `--early_stop_threshold`
5. **mHC No-AMP**: `--mhc_no_amp` for numerical stability

### Model Architecture Updates

- Added `mHCResidual` class details
- Added Sinkhorn-Knopp algorithm pseudocode
- Updated attention mechanism diagrams
- Added DDP architecture diagrams

## Migration Guide

### For Users with Old Documentation

1. **Check `docs/README.md`** for current documentation structure
2. **Refer to `docs/advanced/mhc-guide.md`** for mHC usage
3. **Use `docs/advanced/distributed-training.md`** for multi-GPU training
4. **Check `ARCHITECTURE.md`** for system design

### For Contributors

1. All new documentation should go in `docs/`
2. Update `docs/README.md` when adding new documents
3. Update this changelog when modifying existing docs
4. Ensure docs match code by running relevant tests

## Future Documentation Plans

- [ ] Add API reference documentation
- [ ] Add Jupyter notebook tutorials
- [ ] Add performance benchmark results
- [ ] Add debugging guide
- [ ] Add model comparison tables
- [ ] Translate key docs to Chinese (已有中文版本: README.zh-CN.md, ARCHITECTURE.zh-CN.md)
