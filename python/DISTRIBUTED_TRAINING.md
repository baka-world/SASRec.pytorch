# SASRec 多卡训练指南

> **注意**: 此文档已迁移到统一文档目录。
> 
> **完整文档**: [Distributed Training Guide](../../docs/advanced/distributed-training.md)

## 快速开始

```bash
cd python
chmod +x train_distributed.sh
./train_distributed.sh ml-1m tisasrec_dist
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--multi_gpu` | False | 启用分布式训练 |
| `--batch_size` | 128 | 总batch大小 |
| `--use_amp` | False | 启用自动混合精度 |

## 相关文档

- [docs/README.md](../../docs/README.md) - 文档目录
- [docs/advanced/distributed-training.md](../../docs/advanced/distributed-training.md) - 完整分布式训练指南
- [docs/advanced/mhc-guide.md](../../docs/advanced/mhc-guide.md) - mHC使用指南
- [docs/concepts/background.md](../../docs/concepts/background.md) - 背景知识
