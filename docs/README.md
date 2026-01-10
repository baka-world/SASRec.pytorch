# SASRec.pytorch Documentation

## 文档目录

### 快速入门

| 文档 | 说明 |
|------|------|
| [README.md](../README.md) | 项目主文档，包含快速开始和完整参数说明 |
| [getting-started/README.md](getting-started/README.md) | 快速上手指南（待创建） |

### 核心概念

| 文档 | 说明 |
|------|------|
| [concepts/background.md](concepts/background.md) | 序列推荐背景知识（整合自python/BACKGROUND_KNOWLEDGE.md） |
| [concepts/models.md](concepts/models.md) | 模型架构详解（SASRec, TiSASRec, mHC） |
| [concepts/training.md](concepts/training.md) | 训练策略和技巧 |

### 高级功能

| 文档 | 说明 |
|------|------|
| [advanced/mhc-guide.md](advanced/mhc-guide.md) | mHC（流形约束超连接）使用指南 |
| [advanced/distributed-training.md](advanced/distributed-training.md) | 分布式训练指南（整合自python/DISTRIBUTED_TRAINING.md） |
| [advanced/hyperparameter-tuning.md](advanced/hyperparameter-tuning.md) | 超参数调优指南 |

### 架构设计

| 文档 | 说明 |
|------|------|
| [architecture.md](architecture.md) | 系统架构和组件说明 |
| [architecture.zh-CN.md](architecture.zh-CN.md) | 系统架构中文说明 |

### 参考

| 文档 | 说明 |
|------|------|
| [references/papers.md](references/papers.md) | 相关论文引用 |
| [references/changelog.md](references/changelog.md) | 文档更新日志 |

## 项目结构

```
SASRec.pytorch/
├── python/                    # 主要代码目录
│   ├── main.py               # 统一训练入口
│   ├── main_distributed.py   # 分布式训练入口
│   ├── model.py              # SASRec/TiSASRec模型
│   ├── model_mhc.py          # mHC变体模型
│   ├── utils.py              # 工具函数
│   └── run_experiments_v2.py # 实验管理脚本
├── data/                      # 数据目录
├── docs/                      # 文档目录（本文档所在）
├── rust/                      # Rust扩展（可选）
└── README.md                  # 项目主文档
```

## 文档更新日志

见 [references/changelog.md](references/changelog.md)

## 贡献指南

1. 所有新文档应放在 `docs/` 目录下
2. 文档使用Markdown格式
3. 保持文档与代码实现同步更新
4. 重要更新应记录在changelog中
