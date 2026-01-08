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

# 训练 SASRec（基准对比）
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc

# 训练 SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

# 训练 TiSASRec（默认，使用时间间隔）
python main.py --dataset=ml-1m --train_dir=tisasrec

# 训练 TiSASRec + mHC（默认，时序感知+超连接）
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集名称 |
| `--train_dir` | 必填 | 输出目录 |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--lr_decay_step` | 1000 | 学习率衰减步长（按epoch） |
| `--lr_decay_rate` | 0.95 | 学习率衰减率 |
| `--warmup_steps` | 100 | Warmup步数（0表示不使用） |
| `--maxlen` | 200 | 最大序列长度 |
| `--hidden_units` | 50 | 隐藏层维度 |
| `--num_blocks` | 2 | Transformer块数 |
| `--num_heads` | 2 | 注意力头数 |
| `--dropout_rate` | 0.2 | Dropout比例 |
| `--l2_emb` | 0.0 | L2正则化系数 |
| `--device` | cuda | 训练设备 |
| `--no_time` | False | 禁用TiSASRec（使用SASRec） |
| `--no_mhc` | False | 禁用mHC模块 |
| `--mhc_expansion_rate` | 4 | mHC扩展因子 |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn迭代次数 |
| `--time_span` | 100 | 时间间隔范围 |
| `--time_unit` | hour | 时间单位 |

## 实验设置（基准对比）

默认参数与原始SASRec论文一致，用于公平对比：

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_units` | 50 | 嵌入维度 |
| `num_blocks` | 2 | Transformer层数 |
| `batch_size` | 128 | 训练批次大小 |
| `maxlen` | 200 | 最大序列长度 |
| `dropout_rate` | 0.2 | Dropout比例 |
| `lr` | 0.001 | 学习率 |
| `num_epochs` | 1000 | 训练轮数 |

与 `Result_Norm.md` 基准对比：
- SASRec 基准：`--no_time --no_mhc`
- TiSASRec：`--no_mhc`
- TiSASRec + mHC：（默认）

## 训练建议

尝试更大的模型以获得更好的性能：

```bash
# 中等模型
python main.py --dataset=ml-1m --train_dir=sasrec_medium \
    --hidden_units=128 --num_blocks=3 --batch_size=128

# 大模型
python main.py --dataset=ml-1m --train_dir=sasrec_large \
    --hidden_units=256 --num_blocks=3 --batch_size=128
```

如果loss卡在0.45左右：
- 增加模型容量（`--hidden_units`、`--num_blocks`）
- 使用学习率衰减（`--lr_decay_step=500 --lr_decay_rate=0.95`）
- 添加warmup（`--warmup_steps=100`）

## 显存不足

遇到CUDA内存溢出时：

| GPU显存 | 推荐配置 |
|---------|----------|
| 8 GB | `--batch_size 32 --hidden_units 50 --maxlen 100 --mhc_no_amp` |
| 16 GB | `--batch_size 64 --hidden_units 50`（默认配置） |
| 24+ GB | `--batch_size 128 --hidden_units 128` |

其他选项：
- 减小批次：`--batch_size=64`
- 减小序列长度：`--maxlen=100`
- 减小模型：`--num_blocks=1`
- 禁用mHC AMP：`--mhc_no_amp`
- 使用CPU：`--device=cpu`

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
