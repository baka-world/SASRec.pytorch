update on 05/23/2025: thx to [Wentworth1028](https://github.com/Wentworth1028) and [Tiny-Snow](https://github.com/Tiny-Snow), we have LayerNorm update, for higher NDCG&HR, and here's the [doc](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)👍.

---

## TiSASRec 时序感知序列推荐（新增功能）

本仓库现在支持**TiSASRec（Time Interval Aware Self-Attentive Sequential Recommendation）**，这是对原始SASRec的扩展，引入了时间间隔信息来增强推荐效果。

### 核心创新

TiSASRec在标准自注意力的基础上融入了时间间隔信息，让模型能够学习到：

- **越近的行为越相关**：时间间隔越短，行为之间的关联性越强
- **用户兴趣随时间的演变**：捕捉用户兴趣的变化趋势
- **不同时间尺度的影响**：区分短期和长期的用户偏好

### 注意力机制对比

**SASRec标准注意力：**
```
A_ij = softmax(Q_i * K_j^T)
```

**TiSASRec时序感知注意力：**
```
A_ij = softmax(Q_i * K_j^T + Q_i * abs_pos_K_i^T + time_matrix_K_j * Q_i)
```

第三项 `time_matrix_K_j * Q_i` 是核心创新点，将时间间隔信息融入注意力权重计算。

### 模型架构

| 组件 | SASRec | TiSASRec |
|------|--------|----------|
| 物品嵌入层 | ✓ | ✓ |
| 位置嵌入层 | ✓ | ✓ |
| 时间矩阵嵌入层 | ✗ | ✓（新增） |
| 标准多头注意力 | ✓ | ✗ |
| 时序感知多头注意力 | ✗ | ✓（新增） |

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install torch numpy datasets
```

### 训练SASRec（标准版本）

```bash
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

### 训练TiSASRec（时序感知版本）

使用HuggingFace数据集（推荐，包含时间戳信息）：

```bash
python main_tisasrec.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_default \
    --use_time \
    --time_span=100 \
    --use_hf \
    --device=cuda
```

### 推理测试

```bash
python main_tisasrec.py \
    --device=cuda \
    --dataset=ml-1m \
    --train_dir=tisasrec_default \
    --state_dict_path=[YOUR_CKPT_PATH] \
    --inference_only=true \
    --maxlen=200
```

---

## 命令行参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集名称 |
| `--train_dir` | 必填 | 训练结果保存目录 |
| `--batch_size` | 128 | 每个批次的样本数量 |
| `--lr` | 0.001 | 学习率 |
| `--maxlen` | 200 | 序列最大长度 |
| `--hidden_units` | 50 | 隐藏层维度 |
| `--num_blocks` | 2 | Transformer编码器块数量 |
| `--num_heads` | 1 | 多头注意力头数 |
| `--dropout_rate` | 0.2 | Dropout比率 |
| `--device` | cuda | 训练设备 |
| `--num_epochs` | 1000 | 训练轮数 |

### TiSASRec特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_time` | False | 是否启用时序感知机制 |
| `--time_span` | 100 | 时间间分离散化范围（将连续时间间隔映射到[1, time_span]） |
| `--time_unit` | hour | 时间单位（second/minute/hour/day） |
| `--use_hf` | False | 是否使用HuggingFace数据集 |

### 参数调优建议

1. **`time_span`**：时间间分离散化范围
   - 数据集时间跨度大（跨月/年）：建议100-200
   - 数据集时间跨度小（跨天/周）：建议50-100

2. **`time_unit`**：时间单位选择
   - 交互频繁（分钟级）：使用`minute`
   - 交互一般（小时级）：使用`hour`（推荐）
   - 交互稀疏（天级）：使用`day`

---

## HuggingFace数据集支持

本仓库支持从HuggingFace加载包含时间信息的ml-1m数据集（cep-ter/ML-1M）。

### 数据集字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `uid` | int64 | 用户ID |
| `iid` | int64 | 物品ID |
| `timestamp` | int64 | Unix时间戳（秒） |
| `time` | int64 | 一天中的小时（0-23） |
| `rating` | int64 | 评分（1-5） |
| `genres` | string | 电影类型 |
| `label` | int64 | 二元标签 |

### 数据处理流程

```
原始时间戳 → 计算时间间隔 → 离散化到[1, time_span] → 嵌入为向量 → 融入注意力计算
```

---

## 输出示例

```
============================================================================================
Training TiSASRec on dataset: ml-1m
Use Time Information: True
Time Span: 100, Time Unit: hour
============================================================================================
Loading dataset from HuggingFace...
Dataset loaded in 2.34s
average sequence length: 165.32
Evaluating epoch:20, time: 125.6(s), valid (NDCG@10: 0.6012, HR@10: 0.8356), test (NDCG@10: 0.5956, HR@10: 0.8312)
```

---

## 引用信息

如果您在研究中使用了本代码，请考虑引用：

**SASRec原始论文：**
```
@article{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018}
}
```

**TiSASRec论文：**
```
@article{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Jiacheng and Wang, Yujie and McAuley, Julian},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={322--330},
  year={2020}
}
```

**本仓库：**
```
@misc{Huang_SASRec_pytorch,
  author = {Huang, Zan},
  title = {{SASRec.pytorch}},
  url = {https://github.com/pmixer/SASRec.pytorch},
  year={2020}
}
```

---

## 文件结构

```
SASRec.pytorch/
├── python/
│   ├── main.py              # SASRec训练脚本
│   ├── main_tisasrec.py     # TiSASRec训练脚本（新增）
│   ├── model.py             # 模型定义（含TiSASRec，新增）
│   ├── utils.py             # 工具函数（含时序采样，新增）
│   └── dataset_hf.py        # HuggingFace数据集加载（新增）
├── data/                    # 数据目录
├── latex/                   # 论文源码
└── README.md                # 本文档
```

---

## 常见问题

**Q: TiSASRec与SASRec相比有什么优势？**
A: TiSASRec通过融入时间间隔信息，能够更好地捕捉用户兴趣的演变，通常在具有时间信息的真实数据集上表现更优。

**Q: 如何选择合适的time_span值？**
A: 建议根据数据集中的实际时间跨度来调整。time_span越大，模型能够区分更细粒度的时间间隔。

**Q: 使用TiSASRec是否需要时间信息？**
A: 是的，TiSASRec需要每个交互的时间戳信息。如果数据不包含时间信息，建议使用标准SASRec。

**Q: 模型训练速度如何？**
A: TiSASRec由于额外的时间矩阵计算，训练时间约为SASRec的1.2-1.5倍，但仍可在合理时间内完成训练。

---

如有问题或建议，欢迎创建Issue或提交Pull Request。
