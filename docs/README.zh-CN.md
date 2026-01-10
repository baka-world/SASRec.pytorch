# SASRec.pytorch 文档

## 文档目录

### 快速入门

| 文档 | 说明 |
|------|------|
| [README.md](../README.md) | 项目主文档，包含快速开始和完整参数说明 |
| [getting-started/README.md](getting-started/README.md) | 快速上手指南 |

### 核心概念

| 文档 | 说明 |
|------|------|
| [concepts/background.zh-CN.md](concepts/background.zh-CN.md) | 序列推荐背景知识 |
| [concepts/background.md](concepts/background.md) | 序列推荐背景知识（英文） |
| [concepts/models.md](concepts/models.md) | 模型架构详解 |

### 高级功能

| 文档 | 说明 |
|------|------|
| [advanced/mhc-guide.zh-CN.md](advanced/mhc-guide.zh-CN.md) | mHC（流形约束超连接）使用指南 |
| [advanced/mhc-guide.md](advanced/mhc-guide.md) | mHC使用指南（英文） |
| [advanced/distributed-training.zh-CN.md](advanced/distributed-training.zh-CN.md) | 分布式训练指南 |
| [advanced/distributed-training.md](advanced/distributed-training.md) | 分布式训练指南（英文） |

### 架构设计

| 文档 | 说明 |
|------|------|
| [architecture.zh-CN.md](architecture.zh-CN.md) | 系统架构说明（中文） |
| [architecture.md](architecture.md) | 系统架构说明（英文） |

### 参考

| 文档 | 说明 |
|------|------|
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

## 支持的模型

| 模型 | 说明 | mHC支持 |
|------|------|---------|
| SASRec | 标准自注意力序列推荐 | ✅ |
| TiSASRec | 时序感知自注意力推荐 | ✅ |

## 快速开始

```bash
# 1. 安装依赖
pip install torch numpy

# 2. 准备数据
cd data && wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && cd ..
python python/convert_ml1m.py

# 3. 训练模型
cd python
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time
python main.py --dataset=ml-1m --train_dir=tisasrec
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc

# 4. 多卡训练
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --dataset=ml-1m --train_dir=tisasrec_dist --batch_size=512 --use_amp --multi_gpu
```

## 文档更新日志

见 [references/changelog.md](references/changelog.md)

## 贡献指南

1. 所有新文档应放在 `docs/` 目录下
2. 文档使用Markdown格式
3. 保持文档与代码实现同步更新
4. 重要更新应记录在changelog中
5. 建议同时提供中英文版本
