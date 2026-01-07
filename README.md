update on 05/23/2025: thx to [Wentworth1028](https://github.com/Wentworth1028) and [Tiny-Snow](https://github.com/Tiny-Snow), we have LayerNorm update, for higher NDCG&HR, and here's the [doc](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)👍.

---

## mHC: 流形约束超连接（新增功能）

本仓库支持**mHC（Manifold-Constrained Hyper-Connections）**，这是对标准残差连接的扩展，通过将残差映射投影到双随机矩阵流形来增强训练稳定性和模型性能。

### 核心思想

传统残差连接：`x_{l+1} = x_l + F(x_l)`

mHC残差连接：`x_{l+1} = H_l^{res} × x_l + H_l^{post}^T × F(H_l^{pre} × x_l, W_l)`

**关键创新：**
1. **扩展残差流宽度**：将维度C扩展为n×C（n为扩展因子，默认4）
2. **三个可学习映射**：H_pre、H_post、H_res
3. **流形约束**：H_res通过Sinkhorn-Knopp算法投影到双随机矩阵流形
4. **身份映射保持**：双随机矩阵确保信号范数有界，梯度稳定

### 技术优势

| 特性 | 标准残差 | Hyper-Connections | mHC |
|------|---------|-------------------|-----|
| 残差流宽度 | C | n×C | n×C |
| 身份映射保持 | ✓ | ✗ | ✓ |
| 训练稳定性 | ✓ | ✗ | ✓ |
| 信息交换能力 | ✗ | ✓ | ✓ |

## 使用方法

使用统一训练脚本 `main.py`，支持所有模型组合：

```bash
# SASRec (基准)
python main.py --dataset=ml-1m --train_dir=sasrec_base

# SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --use_mhc

# TiSASRec
python main.py --dataset=ml-1m --train_dir=tisasrec --use_time

# TiSASRec + mHC
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc --use_time --use_mhc
```

### mHC参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_mhc` | False | 是否启用mHC |
| `--mhc_expansion_rate` | 4 | 残差流扩展因子n |
| `--mhc_init_gate` | 0.01 | 门控因子α初始值 |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn-Knopp迭代次数 |

### 参数量对比

| 模型 | 参数量 | 增加比例 |
|------|--------|----------|
| SASRec | 115,328 | - |
| SASRec+mHC | 140,012 | +21.4% |

### mHC模块架构

```
Input x: (batch, seq, C)
    ↓
x_expanded = repeat(1,1,n): (batch, seq, n, C)
    ↓
H_pre: (batch, seq, n) ← sigmoid激活
H_post: (batch, seq, n) ← sigmoid激活
H_res: (batch, seq, n, n) ← Sinkhorn双随机矩阵
    ↓
x_res = H_res × x_expanded: (batch, seq, n, C)
f_out = H_post × F(H_pre × x): (batch, seq, C)
    ↓
Output = sum_n(x_res) + f_out: (batch, seq, C)
```

### 引用信息

**mHC论文：**
```
@misc{deepseek2025mhcsurvey,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={DeepSeek-AI},
  year={2025}
}
```

---

## TiSASRec 时序感知序列推荐

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

## 数据准备

### 下载MovieLens 1M数据集

```bash
cd data
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

### 转换数据格式

运行转换脚本生成带时间戳的数据集：

```bash
cd ..
python convert_ml1m.py
```

转换后的数据格式（`data/ml-1m.txt`）：
```
UserID MovieID Timestamp
1 3186 978300019
1 1270 978300055
...
```

**注意**：如果使用不带时间戳的旧版数据（仅 `UserID MovieID`），模型将退化为标准SASRec。

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install torch numpy
```

### 训练SASRec（标准版本）

```bash
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

### 训练TiSASRec（时序感知版本）

```bash
python main_tisasrec.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_full \
    --use_time \
    --time_span=100 \
    --time_unit=hour \
    --device=cuda \
    --lr=0.005 \
    --l2_emb=0.0001 \
    --dropout_rate=0.3 \
    --num_epochs=1000 \
    --patience=50 \
    --batch_size=128 \
    --num_workers=6
```

**推荐配置（ml-1m数据集）：**
```bash
python main_tisasrec.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_full \
    --use_time \
    --time_span=100 \
    --time_unit=hour \
    --device=cuda
```

### 推理测试

```bash
python main_tisasrec.py \
    --device=cuda \
    --dataset=ml-1m \
    --train_dir=tisasrec_full \
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
| `--l2_emb` | 0.0001 | 嵌入层L2正则化系数 |
| `--maxlen` | 200 | 序列最大长度 |
| `--hidden_units` | 50 | 隐藏层维度 |
| `--num_blocks` | 2 | Transformer编码器块数量 |
| `--num_heads` | 1 | 多头注意力头数 |
| `--dropout_rate` | 0.2 | Dropout比率 |
| `--device` | cuda | 训练设备 |
| `--num_epochs` | 1000 | 训练轮数 |
| `--num_workers` | 3 | 数据加载线程数 |

### TiSASRec特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_time` | False | 是否启用时序感知机制 |
| `--time_span` | 100 | 时间间分离散化范围（将连续时间间隔映射到[1, time_span]） |
| `--time_unit` | hour | 时间单位（second/minute/hour/day） |
| `--patience` | 50 | 早停耐心值 |
| `--min_delta` | 0.001 | 验证指标提升最小阈值 |

### 参数调优建议

1. **`time_span`**：时间间分离散化范围
   - 数据集时间跨度大（跨月/年）：建议100-200
   - 数据集时间跨度小（跨天/周）：建议50-100

2. **`time_unit`**：时间单位选择
   - 交互频繁（分钟级）：使用`minute`
   - 交互一般（小时级）：使用`hour`（推荐）
   - 交互稀疏（天级）：使用`day`

3. **`batch_size`**：根据显存调整
   - 8GB显存：建议64
   - 16GB+显存：建议128

---

## 输出示例

```
============================================================================================
Training TiSASRec on dataset: ml-1m
Use Time Information: True
Time Span: 100, Time Unit: hour
============================================================================================
average sequence length: 165.32
Evaluating epoch:20, time: 125.6(s), valid (NDCG@10: 0.2654, HR@10: 0.4821), test (NDCG@10: 0.2587, HR@10: 0.4712)
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
│   ├── main.py              # 统一训练脚本（支持SASRec/TiSASRec + 可选mHC）
│   ├── model.py             # SASRec/TiSASRec模型定义
│   ├── model_mhc.py         # SASRec/TiSASRec + mHC模型定义
│   ├── utils.py             # 工具函数
│   └── convert_ml1m.py      # ML-1M数据转换脚本
├── data/                    # 数据目录
│   ├── ml-1m/               # MovieLens 1M原始数据
│   └── ml-1m.txt            # 转换后的数据（含时间戳）
├── docs/
│   ├── mHC_manifold_constrained_hyper_connections.md  # mHC论文
│   └── MHC_README.md        # mHC实现文档
├── latex/                   # 论文源码
└── README.md                # 本文档
```

---

## 常见问题

### mHC相关

**Q: mHC与标准残差连接相比有什么优势？**
A: mHC通过扩展残差流宽度和引入可学习的连接矩阵，增强了信息交换能力。同时，通过双随机矩阵约束保持身份映射属性，确保训练稳定性。

**Q: mHC会增加多少参数量？**
A: mHC约增加21%的参数量，主要来自三个线性映射层（nC→n, nC→n, nC→n²）。

**Q: 如何选择mhc_expansion_rate？**
A: 论文推荐n=4作为默认选择。更高的n值提供更强的表达能力，但也会增加计算和显存开销。

**Q: Sinkhorn-Knopp迭代次数如何影响效果？**
A: 迭代次数越多，矩阵越接近理想双随机矩阵。论文使用20次迭代，在精度和效率之间取得平衡。

### TiSASRec相关

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
