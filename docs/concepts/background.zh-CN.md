# 序列推荐背景知识

## 什么是序列推荐？

序列推荐（Sequential Recommendation）是一种推荐系统任务，其核心思想是：

**根据用户的历史行为序列，预测用户下一个可能交互的物品。**

举例说明：
- 用户在电商网站的浏览序列：[手机壳, 充电器, 手机膜, 蓝牙耳机]
- 模型需要预测：用户下一个可能会买什么？（可能是：手机支架、耳机包等）

## 为什么需要序列推荐？

传统的推荐系统（如协同过滤）假设用户的兴趣是静态的，忽略了：
1. 用户兴趣会随时间演变
2. 物品之间存在时序依赖关系（如买了手机后会买配件）
3. 用户行为具有上下文相关性

序列推荐通过建模用户行为的时间动态性，能够更准确地捕捉用户当前的需求。

## SASRec 模型概述

**SASRec = Self-Attentive Sequential Recommendation**

这是2018年由Kang和McAuley提出的模型，将Transformer架构引入序列推荐任务。

### 核心创新

1. **自注意力机制**：替代传统的RNN/LSTM，能够更好地捕捉序列中的长期依赖
2. **位置嵌入**：为序列中的每个位置添加位置信息
3. **多层堆叠**：通过堆叠多个注意力块，学习不同层次的序列模式

## 模型架构详解

### 1. 嵌入层（Embedding Layer）

```
输入：物品ID（如 [5, 3, 7, ...]）
输出：密集向量表示（如 [[0.1, 0.3, ...], [0.2, 0.1, ...], ...]）
```

**物品嵌入**：将每个物品ID映射为固定维度的向量

**位置嵌入**：记录序列中每个位置的信息（如"这是第1个、第2个..."）

### 2. 自注意力机制（Self-Attention）

这是Transformer的核心组件。简单理解：

```
对于序列中的每个位置，计算它与所有其他位置的关联强度
然后根据关联强度加权求和，得到新的表示
```

举例：
- 位置4（最近的行为）可能更关注位置3、位置2
- 位置1（最早的行为）可能只与少数位置有关联

### 3. 前馈网络（Feed Forward Network）

每个注意力层之后还有一个前馈网络：

```
输入 -> 线性变换 -> ReLU -> Dropout -> 线性变换 -> Dropout -> 输出
```

作用：增加非线性表达能力，让模型学习更复杂的模式。

### 4. LayerNorm 和残差连接

每个子层（注意力、前馈网络）都使用：
- **LayerNorm**：标准化，稳定训练
- **残差连接**：缓解梯度消失，帮助深层网络的训练

## 训练策略

### 对比学习

SASRec使用对比学习策略进行训练：

**正样本**：用户实际交互的下一个物品
**负样本**：用户从未交互过的随机物品

**训练目标**：让正样本的得分高于负样本

### 损失函数

使用二元交叉熵损失（BCE Loss）：

```
Loss = -log(sigmoid(正样本得分)) - log(1 - sigmoid(负样本得分))
```

直观理解：希望正样本得分高，负样本得分低。

## 评估指标

### NDCG（Normalized Discounted Cumulative Gain）

衡量推荐列表排序质量的指标。

- 真实物品排名越靠前，NDCG越高
- 范围：0到1，越高越好

### HR（Hit Rate）

命中率，衡量推荐列表是否包含目标物品。

- 如果真实物品在推荐列表的前10名中，HR@10 = 1
- 范围：0到1，越高越好

## TiSASRec 模型

**TiSASRec = Time Interval Aware Self-Attention**

在SASRec基础上引入时间间隔信息，让模型能够学习到：
- 越近的行为越相关
- 用户兴趣随时间的变化趋势
- 不同时间间隔对推荐的影响

### 注意力计算

```
标准注意力：
    A_ij = softmax(Q_i · K_j^T)

TiSASRec注意力：
    A_ij = softmax(Q_i · K_j^T 
                   + Q_i · abs_pos_K_i^T 
                   + time_matrix_K_j · Q_i)
```

## mHC（流形约束超连接）

### 为什么需要mHC？

传统残差连接：
```
x_{l+1} = x_l + F(x_l)
```

mHC残差连接：
```
x_{l+1} = H_res × x_l + H_post^T × F(H_pre × x_l)
```

### 核心思想

1. **扩展残差流宽度**：将维度从C扩展到n×C
2. **引入三个可学习映射**：H_pre, H_post, H_res
3. **流形约束**：使用Sinkhorn-Knopp算法将H_res投影到双随机矩阵流形
4. **保持身份映射属性**：确保信号传播的稳定性

### Sinkhorn-Knopp算法

```python
def sinkhorn_knopp(M, max_iter=20):
    """将矩阵投影到双随机矩阵流形"""
    n = M.shape[-1]
    M = torch.exp(M)
    for _ in range(max_iter):
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + 1e-12)  # 行归一化
        
        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + 1e-12)  # 列归一化
    
    return M  # 双随机矩阵
```

### mHC优势

- **训练稳定性**：双随机约束确保信号范数保持
- **可扩展性**：深层网络梯度稳定
- **性能提升**：通过多流信息交换改善表示学习

## 代码结构说明

### main.py

程序入口，负责：
1. 解析命令行参数
2. 加载和划分数据
3. 初始化模型
4. 训练循环
5. 模型评估和保存

### model.py

模型定义，包含：
- `PointWiseFeedForward`：前馈神经网络
- `SASRec`：完整的序列推荐模型
- `TiSASRec`：时序感知序列推荐模型

### model_mhc.py

mHC变体模型定义，包含：
- `mHCResidual`：流形约束超连接残差模块
- `SASRec_mHC`：带mHC的SASRec
- `TiSASRec_mHC`：带mHC的TiSASRec

### utils.py

工具函数，包含：
- `build_index`：构建用户-物品索引
- `WarpSampler`：多进程采样器
- `data_partition`：数据划分
- `evaluate`/`evaluate_valid`：模型评估
- `WarpSamplerWithTime`：时序感知采样器

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 128 | 批次大小，越大越稳定但越慢 |
| `--lr` | 0.001 | 学习率，太大不稳定，太小收敛慢 |
| `--maxlen` | 200 | 序列最大长度 |
| `--hidden_units` | 50 | 嵌入维度 |
| `--num_blocks` | 2 | Transformer Block数量 |
| `--num_heads` | 2 | 注意力头数量 |
| `--dropout_rate` | 0.2 | Dropout比率 |
| `--num_epochs` | 300 | 训练轮数 |
| `--mhc_expansion_rate` | 4 | mHC扩展因子 |
| `--mhc_sinkhorn_iter` | 20 | Sinkhorn迭代次数 |

## 运行命令示例

### 训练模型

```bash
cd python

# SASRec基准
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc

# SASRec + mHC
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

# TiSASRec
python main.py --dataset=ml-1m --train_dir=tisasrec

# TiSASRec + mHC
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc
```

### 仅推理

```bash
python main.py --device=cuda --dataset=ml-1m --train_dir=default \
    --state_dict_path=xxx.pth --inference_only=true --maxlen=200
```

### 多卡训练

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --dataset=ml-1m --train_dir=tisasrec_dist --batch_size=512 \
    --use_amp --multi_gpu
```

## 负采样详解

负采样是训练推荐模型的关键技术。

**为什么需要负采样？**
- 物品数量通常很大（数万到数百万）
- 不可能计算所有物品的损失
- 只需让模型"知道"某些物品不合适即可

**负采样策略：**
1. 从用户未交互的物品中随机选择
2. 确保负样本不被用户交互过
3. 一般每个正样本配多个负样本

## 因果注意力掩码

在序列预测任务中，需要防止信息泄露。

**问题**：预测第4个位置的物品时，不应该看到第5、6...个位置的信息

**解决方案**：使用下三角掩码

```
位置:   1   2   3   4
位置1:  ✓   ✗   ✗   ✗
位置2:  ✓   ✓   ✗   ✗
位置3:  ✓   ✓   ✓   ✗
位置4:  ✓   ✓   ✓   ✓
```

## 数据格式

训练数据文件格式（data/xxx.txt）：
```
用户ID 物品ID 时间戳(可选)
1 5 1234567890
1 7 1234567900
1 3 1234567910
2 10 1234567920
...
```

每行表示一次用户-物品交互，按时间顺序排列。

## 核心概念详解

### 1. 神经元（Neuron）

**定义**：神经网络中的基本计算单元，模拟人脑神经元的工作方式

**结构**：
```
输入: [x1, x2, x3, ...]
       ↓
权重: [w1, w2, w3, ...]
       ↓
加权求和: Σ(xi × wi)
       ↓
激活函数: f(Σ(xi × wi))
       ↓
输出
```

### 2. Dropout（随机丢弃）

**作用**：防止模型过拟合（过度记忆训练数据）

**操作**：训练时随机将某些神经元的输出设为0

**目的**：
- 防止模型过度依赖特定神经元
- 增强模型泛化能力
- 减少过拟合风险

### 3. LayerNorm（层归一化）

**操作**：对每个样本的每一层做标准化

**公式**：
```
output = (x - mean(x)) / std(x) × γ + β
```

**目的**：
- 稳定训练过程
- 加快收敛速度
- 减少对初始化的敏感度

### 4. 残差连接（Residual Connection）

**操作**：将输入直接加到输出上

**结构**：
```
输入 → [网络层] → 输出
  ↓              ↓
   →——相加——→
```

**目的**：
- 缓解梯度消失问题
- 让模型更容易学习"恒等映射"
- 允许更深的网络

### 5. Transformer Block

**定义**：Transformer架构的基本组成单元，包含自注意力层和前馈网络

**结构**：
```
输入
  ↓
LayerNorm
  ↓
Multi-Head Attention (自注意力)
  ↓
残差连接 (+)
  ↓
LayerNorm
  ↓
Feed Forward (前馈网络)
  ↓
残差连接 (+)
  ↓
输出
```

### 6. 自注意力机制（Self-Attention）

**核心思想**：序列中每个位置都可以"关注"所有其他位置

**计算步骤**：

**步骤1：生成Q/K/V向量**
```
输入: [h1, h2, h3] (每个是50维向量)
       ↓
   Q/K/V矩阵变换
       ↓
   Q = [q1, q2, q3]
   K = [k1, k2, k3]
   V = [v1, v2, v3]
```

**步骤2：计算注意力分数**
```
score_ij = q_i · k_j / √d (d是向量维度，如50)
```

**步骤3：Softmax归一化**
```
attention_ij = exp(score_ij) / Σexp(score_i)
```

**步骤4：加权求和**
```
new_h1 = attention_11×v1 + attention_12×v2 + attention_13×v3
```

### 7. 损失函数（Loss）

**作用**：衡量模型预测与真实目标的差距

**BCE Loss公式**：
```
Loss = -log(sigmoid(正样本得分)) - log(1 - sigmoid(负样本得分))
```

**直观理解**：
- 正样本（用户实际交互的物品）：希望得分越高越好
- 负样本（用户未交互的物品）：希望得分越低越好

## 为什么要这样设计？

| 设计 | 解决的问题 | 效果 |
|------|-----------|------|
| 自注意力 | RNN长序列信息丢失 | 直接建模任意距离依赖 |
| 多头注意力 | 单一注意力模式有限 | 捕捉多种关联类型 |
| 位置嵌入 | 序列顺序信息丢失 | 注入位置信息 |
| LayerNorm | 训练不稳定 | 稳定训练过程 |
| 残差连接 | 深层网络难训练 | 支持更深的网络 |
| Dropout | 过拟合 | 提高泛化能力 |
| 前馈网络 | 注意力层表达有限 | 增强非线性能力 |
| 多层堆叠 | 浅层模型能力有限 | 学习多层次特征 |
| mHC | 深层训练不稳定 | 保持身份映射属性 |

**整体思路**：
1. 用位置嵌入保留序列顺序
2. 用自注意力建模依赖关系
3. 用多层堆叠学习多层次模式
4. 用各种技术（Norm、残差、Dropout）保证训练稳定
5. 用mHC增强深层网络的稳定性

## 常见问题

### Q: 为什么模型能预测用户下一个想买的物品？

A: 模型学习到了：
1. 物品之间的共现模式（如买手机后会买配件）
2. 用户兴趣的演变趋势
3. 时序依赖关系

### Q: 位置嵌入为什么重要？

A: 如果没有位置信息，模型无法区分：
- [A, B] 和 [B, A] 序列
- 用户先买什么后买什么

### Q: 如何选择超参数？

A: 通常通过实验调参：
- 从默认值开始
- 根据验证集表现调整
- 注意过拟合问题

### Q: mHC有什么作用？

A: mHC通过流形约束：
- 提高训练稳定性
- 支持更深的网络
- 改善模型性能
- 代价是增加约10-15%的计算和内存开销

## 延伸学习

1. **Transformer原论文**：Attention Is All You Need
2. **SASRec原始论文**：Self-attentive sequential recommendation
3. **TiSASRec原始论文**：Time Interval Aware Self-Attention
4. **mHC原始论文**：Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025)
5. **其他序列推荐模型**：GRU4Rec、BERT4Rec

## 参考文献

```bibtex
@article{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}

@article{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Zeynep and McAuley, Julian},
  year={2020}
}

@misc{deepseek2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={DeepSeek-AI},
  year={2025}
}
```
