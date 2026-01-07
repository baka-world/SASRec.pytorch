# SASRec.pytorch

基于PyTorch的序列推荐模型实现，包括SASRec、TiSASRec及其mHC（流形约束超连接）变体。

## 模型支持

| 模型 | 描述 | 时序感知 | mHC支持 |
|------|------|---------|---------|
| SASRec | 自注意力序列推荐 | 否 | 是 |
| TiSASRec | 时间间隔感知自注意力 | 是 | 是 |

## 快速开始

```bash
# 安装依赖
pip install torch numpy

# 准备数据（MovieLens 1M）
cd data && wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && cd ..
python convert_ml1m.py

# 训练 SASRec
python main.py --dataset=ml-1m --train_dir=sasrec_base

# 训练 SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --use_mhc

# 训练 TiSASRec
python main.py --dataset=ml-1m --train_dir=tisasrec --use_time

# 训练 TiSASRec + mHC
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc --use_time --use_mhc
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集名称 |
| `--train_dir` | 必填 | 输出目录 |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--maxlen` | 200 | 最大序列长度 |
| `--hidden_units` | 50 | 隐藏层维度 |
| `--num_blocks` | 2 | Transformer块数 |
| `--num_heads` | 1 | 注意力头数 |
| `--dropout_rate` | 0.2 | Dropout比例 |
| `--device` | cuda | 训练设备 |
| `--use_time` | False | 启用TiSASRec |
| `--use_mhc` | False | 启用mHC |
| `--mhc_expansion_rate` | 4 | mHC扩展因子 |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn迭代次数 |
| `--time_span` | 100 | 时间间隔范围 |
| `--time_unit` | hour | 时间单位 |

## 显存不足

遇到CUDA内存溢出时：
- 减小批次：`--batch_size=64`
- 使用CPU：`--device=cpu`
- 减小模型：`--num_blocks=1`

## 目录结构

```
SASRec.pytorch/
├── python/
│   ├── main.py         # 统一训练脚本
│   ├── model.py        # SASRec/TiSASRec
│   ├── model_mhc.py    # SASRec/TiSASRec + mHC
│   ├── utils.py        # 工具函数
│   └── convert_ml1m.py # 数据转换
├── data/
├── docs/
└── README.md
```

## mHC参数调优

| 参数 | 默认值 | 建议 |
|------|--------|------|
| `--mhc_expansion_rate` | 4 | 显存充足可用8 |
| `--mhc_sinkhorn_iter` | 20 | 精度优先用50 |

## 引用

- [SASRec: 自注意力序列推荐 (Kang & McAuley, 2018)](https://arxiv.org/abs/1808.09781)
- [TiSASRec: 时间间隔感知自注意力 (Li et al., 2020)](https://arxiv.org/abs/2004.11780)
- [mHC: 流形约束超连接 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.03977)
