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

# 多卡训练（4x V100示例）
python -m torch.distributed.launch --nproc_per_node=4 main_distributed.py \
    --dataset=ml-1m --train_dir=tisasrec_dist --batch_size=512 --use_amp
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
| `--norm_first` | False | 使用Pre-LN（True）或Post-LN（False）结构 |
| `--multi_gpu` | False | 启用分布式训练 |
| `--num_workers` | 3 | 数据加载线程数 |

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

## 调参指南

本指南介绍关键超参数及其调优方法，帮助你获得最佳性能。

### 1. 模型容量参数

| 参数 | 范围 | 影响 | 推荐起始值 |
|------|------|------|-----------|
| `--hidden_units` | 50-512 | 嵌入维度，越大容量越高 | 50（基准） |
| `--num_blocks` | 1-6 | Transformer层数，越深越复杂 | 2（基准） |
| `--num_heads` | 1-8 | 注意力头数，越多越精细 | 2（基准） |

**调参策略**：
- 先用基准配置（50, 2, 2）验证训练正常
- 首先增加 `hidden_units` 以提高容量
- 其次增加 `num_blocks` 以增加深度
- `num_heads` 应与 `hidden_units` 成比例（hidden_units % num_heads == 0）

### 2. 训练参数

| 参数 | 范围 | 影响 | 推荐起始值 |
|------|------|------|-----------|
| `--batch_size` | 32-256 | 批次大小，影响梯度稳定性 | 128（基准） |
| `--lr` | 0.0001-0.005 | 学习率 | 0.001（基准） |
| `--warmup_steps` | 0-500 | 预热步数 | 100（基准） |
| `--num_epochs` | 200-2000 | 训练轮数 | 1000（基准） |

**调参策略**：
- 批次越大 → 梯度越稳定，可以使用更大的学习率
- 如果loss震荡：减小lr或增大batch_size
- 如果loss下降太慢：增大lr（最大0.005）
- 如果loss出现NaN：减小lr，检查数据

### 3. 正则化参数

| 参数 | 范围 | 影响 | 推荐起始值 |
|------|------|------|-----------|
| `--dropout_rate` | 0.0-0.5 | Dropout比例 | 0.2（基准） |
| `--l2_emb` | 0.0-0.001 | L2正则化系数 | 0.0（基准） |

**调参策略**：
- 过拟合严重（训练↓，验证↑）：增加dropout_rate到0.3-0.4
- 过拟合轻微：减小dropout_rate到0.1-0.2
- 大模型需要更多正则化

### 4. mHC参数（使用mHC时）

| 参数 | 范围 | 影响 | 推荐起始值 |
|------|------|------|-----------|
| `--mhc_expansion_rate` | 2-8 | 流形扩展因子 | 4（基准） |
| `--mhc_sinkhorn_iter` | 10-50 | Sinkhorn迭代次数 | 20（基准） |
| `--mhc_init_gate` | 0.001-0.1 | 门控初始值 | 0.01（基准） |

**调参策略**：
- 显存充足 → 增加`expansion_rate`到6-8
- 需要更高精度 → 增加`sinkhorn_iter`到30-50
- 训练不稳定 → 减小`init_gate`到0.005

### 5. TiSASRec参数（使用时序特征时）

| 参数 | 范围 | 影响 | 推荐起始值 |
|------|------|------|-----------|
| `--time_span` | 50-500 | 时间间隔离散化范围 | 100（基准） |
| `--time_unit` | second/minute/hour/day | 时间单位 | hour（基准） |

**调参策略**：
- 密集交互（间隔短）：增加`time_span`到200-500
- 稀疏交互（间隔长）：减小`time_span`到50-100

### 6. 学习率调度参数

| 参数 | 影响 | 推荐值 |
|------|------|--------|
| `--lr_decay_step` | 衰减频率（轮数） | 1000（基准） |
| `--lr_decay_rate` | 每步衰减因子 | 0.95（基准） |

**替代调度器**：
```bash
# 阶梯衰减（激进）
python main.py --dataset=ml-1m --train_dir=xxx \
    --lr_decay_step=200 --lr_decay_rate=0.9
```

### 7. 常见问题与解决方案

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss NaN | 学习率过高 | 减小 `--lr` 到 0.0005 |
| Loss震荡 | 批次太小 | 增加 `--batch_size` |
| Loss卡住 >0.5 | 学习率过低 | 增加 `--lr` 到 0.002 |
| 过拟合（验证集↓） | 容量过大 | 增加 `--dropout_rate` |
| 欠拟合（两者都↓） | 容量过小 | 增加 `--hidden_units` |
| GPU显存不足 | 模型过大 | 使用 `--mhc_no_amp`，减小 `--batch_size` |
| 收敛慢 | LR衰减太快 | 增加 `--lr_decay_step` |

### 8. 推荐调参流程

**第一阶段：基准验证（1-2小时）**
```bash
# 默认设置，验证训练正常
python main.py --dataset=ml-1m --train_dir=baseline_test
```

**第二阶段：容量搜索（4-8小时）**
```bash
# 测试不同模型大小
python main.py --dataset=ml-1m --train_dir=model_50  --hidden_units=50
python main.py --dataset=ml-1m --train_dir=model_128 --hidden_units=128 --num_blocks=3
python main.py --dataset=ml-1m --train_dir=model_256 --hidden_units=256 --num_blocks=3
```

**第三阶段：学习率微调（2-4小时）**
```bash
# 测试不同学习率
python main.py --dataset=ml-1m --train_dir=lr_0005 --lr=0.0005
python main.py --dataset=ml-1m --train_dir=lr_002  --lr=0.002
```

**第四阶段：正则化调优（2-4小时）**
```bash
# 如果观察到过拟合
python main.py --dataset=ml-1m --train_dir=dropout_03 --dropout_rate=0.3
python main.py --dataset=ml-1m --train_dir=dropout_04 --dropout_rate=0.4
```

### 9. 快速参考

| 目标 | 命令 |
|------|------|
| 基准对比 | `--no_time --no_mhc` |
| 仅时序感知 | `--no_mhc` |
| 时序+mHC（默认） | （无参数） |
| 高容量 | `--hidden_units=128 --num_blocks=3` |
| 更高容量 | `--hidden_units=256 --num_blocks=3` |
| 低显存 | `--batch_size=32 --mhc_no_amp` |
| 快速探索 | `--num_epochs=200` |

## 显存不足

遇到CUDA内存溢出时：

| GPU显存 | 推荐配置 |
|---------|----------|
| 8 GB | `--batch_size 32 --hidden_units 50 --maxlen 100 --mhc_no_amp` |
| 16 GB | `--batch_size 64 --hidden_units 50`（默认配置） |
| 24+ GB | `--batch_size 128 --hidden_units 128` |
| 4x V100 32GB | `--batch_size 512 --use_amp --multi_gpu` |

其他选项：
- 减小批次：`--batch_size=64`
- 减小序列长度：`--maxlen=100`
- 减小模型：`--num_blocks=1`
- 禁用mHC AMP：`--mhc_no_amp`
- 使用CPU：`--device=cpu`
- 启用多卡获得更大有效批次：`--multi_gpu`

## 目录结构

```
SASRec.pytorch/
├── python/
│   ├── main.py              # 统一训练脚本（单卡）
│   ├── main_distributed.py  # 分布式训练脚本（多卡）
│   ├── model.py             # SASRec/TiSASRec
│   ├── model_mhc.py         # SASRec/TiSASRec + mHC
│   ├── utils.py             # 工具函数
│   ├── convert_ml1m.py      # 数据转换
│   ├── train_distributed.sh # 多卡启动脚本
│   ├── test_distributed.py  # 分布式环境测试
│   └── DISTRIBUTED_TRAINING.md # 多卡训练指南
├── data/
├── docs/
└── README.md
```

## 多卡训练

本实现支持使用PyTorch DistributedDataParallel (DDP)在多GPU上进行分布式训练。

### 环境要求

- PyTorch 1.9+
- CUDA 11.0+
- 多GPU（已在4x V100 32GB上测试）

### 训练命令

```bash
# 4卡训练（推荐用于4x V100）
python -m torch.distributed.launch --nproc_per_node=4 \
    main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_4gpu \
    --batch_size=512 \
    --use_amp \
    --multi_gpu

# 2卡训练
python -m torch.distributed.launch --nproc_per_node=2 \
    main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_2gpu \
    --batch_size=256 \
    --use_amp \
    --multi_gpu

# 使用 torchrun（PyTorch 2.0+）
torchrun --nproc_per_node=4 main_distributed.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_torchrun \
    --batch_size=512 \
    --use_amp \
    --multi_gpu
```

### 使用启动脚本

```bash
# 使用全部4张GPU
cd python && ./train_distributed.sh ml-1m tisasrec_dist

# 指定GPU数量
NUM_GPUS=2 ./train_distributed.sh ml-1m tisasrec_2gpu
```

### 多卡参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--multi_gpu` | False | 启用分布式训练 |
| `--backend` | nccl | 通信后端（nccl/gloo） |
| `--master_port` | 29500 | 通信端口 |
| `--batch_size` | 128 | 总批次大小（自动分配） |
| `--use_amp` | True | 启用自动混合精度 |

### 性能优化

| 配置 | 单卡 | 4卡 |
|------|------|-----|
| batch_size | 128 | 512 |
| lr | 0.001 | 0.002 |
| 预期加速 | 1x | ~3.5x |

### 环境测试

```bash
# 测试分布式环境
python test_distributed.py
```

### 预期加速效果

| 配置 | 相对加速 |
|------|----------|
| 1x V100 | 1.0x（基准） |
| 2x V100 | ~1.8x |
| 4x V100 | ~3.5x |

注意：实际加速比取决于数据加载效率和模型架构。

## 引用

- [SASRec: 自注意力序列推荐 (Kang & McAuley, 2018)](https://arxiv.org/abs/1808.09781)
- [TiSASRec: 时间间隔感知自注意力 (Li et al., 2020)](https://arxiv.org/abs/2004.11780)
- [mHC: 流形约束超连接 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.03977)
