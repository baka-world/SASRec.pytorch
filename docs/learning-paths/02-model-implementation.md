# SASRec.pytorch 项目学习指南（二）：模型实现

> **目标读者**：已理解数据与模型设计，需要理解代码实现
> **学习目标**：理解代码结构、核心模块实现、训练流程

---

## 第一章：项目代码结构

### 1.1 目录概览

```
SASRec.pytorch/
├── python/                    # 核心代码
│   ├── main.py               # 训练入口（单卡/多卡）
│   ├── main_distributed.py   # 分布式训练入口
│   ├── model.py              # SASRec/TiSASRec实现
│   ├── model_mhc.py          # mHC变体实现
│   ├── utils.py              # 工具函数
│   ├── run_experiments_v2.py # 实验管理
│   └── test/                 # 单元测试
├── data/                      # 数据目录
└── docs/                      # 文档目录
```

### 1.2 核心文件关系

```
main.py (训练入口)
    │
    ├── 解析参数
    ├── 加载数据
    ├── 初始化模型 ← model.py / model_mhc.py
    ├── 训练循环
    └── 评估模型 ← utils.py (evaluate函数)
```

### 1.3 从命令行开始

```bash
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc
```

这行命令会：
1. 读取 `--dataset=ml-1m` → 从 `data/ml-1m.txt` 加载数据
2. 创建 `--train_dir=sasrec_base` → 结果保存到 `python/sasrec_base/`
3. `--no_time` → 不使用时序信息（用SASRec）
4. `--no_mhc` → 不使用mHC

---

## 第二章：PyTorch基础知识（快速回顾）

### 2.1 什么是Module？

PyTorch中，所有神经网络组件都继承自`torch.nn.Module`：

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # 全连接层
    
    def forward(self, x):
        return self.linear(x)  # 前向传播
```

**关键方法**：
- `__init__`：定义网络结构
- `forward`：定义数据如何流动

### 2.2 什么是Embedding？

**作用**：把整数ID转成向量

```python
# 创建嵌入层：1000个物品，每个转成50维向量
item_emb = torch.nn.Embedding(1000, 50)

# 使用：输入物品ID，输出向量
ids = torch.tensor([5, 3, 7])           # 3个物品
vectors = item_emb(ids)                  # (3, 50)
# 输出：3个50维向量，每个对应一个物品ID
```

### 2.3 什么是张量（Tensor）？

**本质**：多维数组

| 维度 | 名称 | 示例 |
|------|------|------|
| 0维 | 标量 | 5.0 |
| 1维 | 向量 | [1, 2, 3] |
| 2维 | 矩阵 | [[1,2],[3,4]] |
| 3维 | 3D数组 | 批量序列 |

**形状约定**：
- 序列数据：(batch_size, sequence_length, hidden_units)
- 批次128个，每个序列长200，每个位置是50维向量

---

## 第三章：SASRec模型实现

### 3.1 整体结构（model.py）

```python
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        # 1. 定义嵌入层
        self.item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        
        # 2. 定义Transformer块
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        for _ in range(args.num_blocks):
            # 创建多头注意力层
            self.attention_layers.append(MultiHeadAttention(...))
            # 创建前馈网络层
            self.forward_layers.append(PointWiseFeedForward(...))
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # 前向传播：计算正负样本得分
```

### 3.2 多头注意力实现

**核心代码**（简化的`MultiHeadAttention`）：

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_units // num_heads
        
        # 3组线性变换，得到Q, K, V
        self.q_w = torch.nn.Linear(hidden_units, hidden_units)
        self.k_w = torch.nn.Linear(hidden_units, hidden_units)
        self.v_w = torch.nn.Linear(hidden_units, hidden_units)
    
    def forward(self, queries, keys, values, attn_mask):
        # 1. 线性变换得到Q, K, V
        Q = self.q_w(queries)  # (batch, seq, C)
        K = self.k_w(keys)
        V = self.v_w(values)
        
        # 2. 分割成多个头
        Q = Q.view(batch, seq, num_heads, head_size).transpose(1, 2)
        K = K.view(...).transpose(1, 2)
        V = V.view(...).transpose(1, 2)
        
        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_size ** 0.5)
        
        # 4. 应用掩码（防止看到未来信息）
        scores = scores.masked_fill(attn_mask, -1e9)
        
        # 5. Softmax归一化
        weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # 6. 加权求和得到输出
        output = torch.matmul(weights, V)
        
        return output
```

**注意力计算图解**：

```
Query向量(Q) ──┬── 与所有Key向量计算相似度
               │      ↓
               │   Softmax
               │      ↓
               └───▶ 加权Value向量 ──▶ 输出
```

### 3.3 前馈网络实现

```python
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        # 两层卷积（等价于全连接层）
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, inputs):
        # 输入: (batch, seq, hidden_units)
        # 转置适应Conv1d格式
        x = inputs.transpose(-1, -2)  # (batch, hidden_units, seq)
        
        x = self.conv1(x)            # 第一次变换
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)            # 第二次变换
        x = self.dropout(x)
        
        # 转置回来
        return x.transpose(-1, -2)   # (batch, seq, hidden_units)
```

### 3.4 Transformer块组装

```python
for i in range(num_blocks):
    # ===== 多头注意力 =====
    Q = self.attention_layernorms[i](seqs)  # LayerNorm
    attention_output = self.attention_layers[i](Q, seqs, seqs, attn_mask)
    seqs = seqs + attention_output          # 残差连接
    
    # ===== 前馈网络 =====
    seqs = self.forward_layernorms[i](seqs)  # LayerNorm
    ffn_output = self.forward_layers[i](seqs)
    seqs = seqs + ffn_output                 # 残差连接
```

**结构图**：

```
输入x
  │
  ├─► LayerNorm ─► Multi-Head Attention ──┤
  │                                         │
  └─────────────────────────────────────────┘
                    │
                    ▼
                 残差相加
                    │
                    ▼
              LayerNorm ─► Feed Forward ──┐
                    │                      │
                    └──────────────────────┘
                                  │
                                  ▼
                               残差相加
                                  │
                                  ▼
                                输出
```

---

## 第四章：TiSASRec实现

### 4.1 时间感知注意力

TiSASRec和SASRec的主要区别在注意力计算：

```python
class TimeAwareMultiHeadAttention(torch.nn.Module):
    def forward(self, queries, keys, ..., time_matrix_K, time_matrix_V, ...):
        # 标准注意力分数
        attn_weights = Q_.matmul(K_.transpose(1, 2))
        
        # 加入位置编码的影响
        attn_weights += Q_.matmul(abs_pos_K_.transpose(1, 2))
        
        # 加入时间间隔的影响（核心创新！）
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        
        return outputs
```

### 4.2 时间矩阵编码

```python
# 时间矩阵：记录每对物品之间的时间间隔
time_matrix_K = self.time_matrix_K_emb(time_matrices)  # (batch, seq, seq, C)
time_matrix_V = self.time_matrix_V_emb(time_matrices)

# 使用时序感知采样器生成时间矩阵
sampler = WarpSamplerWithTime(user_train, user_timestamps, ...)
# 生成 (u, seq, pos, neg, time_matrix) 的批次
```

---

## 第五章：mHC实现

### 5.1 mHC残差模块

```python
class mHCResidual(torch.nn.Module):
    def __init__(self, hidden_units, expansion_rate=4, sinkhorn_iter=20):
        super().__init__()
        self.n = expansion_rate  # 扩展因子
        self.C = hidden_units
        self.nC = self.n * self.C
        
        # 三个投影层
        self.phi_pre = torch.nn.Linear(self.nC, self.n)   # H_pre
        self.phi_post = torch.nn.Linear(self.nC, self.n)  # H_post
        self.phi_res = torch.nn.Linear(self.nC, self.n * self.n)  # H_res
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, function_output):
        # 1. 扩展：x -> (batch, seq, n, C)
        x_expanded = x.repeat(1, 1, self.n).view(..., self.n, self.C)
        
        # 2. 计算三个H矩阵
        H_pre = self.sigmoid(self.phi_pre(x_norm) * self.alpha_pre)
        H_post = 2 * self.sigmoid(self.phi_post(x_norm) * self.alpha_post)
        H_res = sinkhorn_knopp(self.phi_res(x_norm) * self.alpha_res)
        
        # 3. 加权混合
        x_res = torch.einsum("bsnc,bsnm->bsmc", x_expanded, H_res).sum(dim=2)
        f_out = (H_post * func_expanded).sum(dim=2)
        
        return x_res + f_out
```

### 5.2 Sinkhorn-Knopp算法

```python
def sinkhorn_knopp(M, max_iter=20):
    """将矩阵投影到双随机矩阵流形"""
    n = M.shape[-1]
    M = torch.exp(M)  # 指数变换
    
    for _ in range(max_iter):
        # 行归一化
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + 1e-12)
        
        # 列归一化
        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + 1e-12)
    
    return M  # 双随机矩阵：每行每列和都为1
```

**直观理解**：
- 经过反复归一化，矩阵变成"公平分配"的形式
- 确保信息不会在某处过度集中

---

## 第六章：训练流程

### 6.1 训练循环（main.py）

```python
for epoch in range(num_epochs):
    # ===== 训练 =====
    for batch in sampler:
        # 前向传播
        pos_logits, neg_logits = model(user_ids, seqs, pos_items, neg_items)
        
        # 计算损失
        loss = bce_loss(pos_logits, 1) + bce_loss(neg_logits, 0)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # ===== 验证 =====
    if is_best:
        save_checkpoint()  # 保存最佳模型
```

### 6.2 损失函数

```python
# 二元交叉熵损失
bce_criterion = torch.nn.BCEWithLogitsLoss()

# 正样本损失：希望得分高
pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))

# 负样本损失：希望得分低
neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))

# 总损失
loss = pos_loss + neg_loss
```

### 6.3 优化器

```python
# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学习率调度
scheduler = torch.optim.lr_schedular.StepLR(
    optimizer, step_size=1000, gamma=0.95
)
```

---

## 第七章：分布式训练

### 7.1 DDP基本概念

**DataParallel vs DistributedDataParallel**：

| 方面 | DataParallel | DistributedDataParallel |
|------|--------------|------------------------|
| 通信 | 串行 | 并行 |
| 速度 | 慢 | 快 |
| 复杂度 | 简单 | 稍复杂 |
| 推荐场景 | 2卡 | 4卡+ |

### 7.2 启动分布式训练

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \      # 4张卡
    main.py \
    --dataset=ml-1m \
    --train_dir=tisasrec_dist \
    --batch_size=512 \
    --multi_gpu
```

### 7.3 DDP关键代码

```python
# 初始化
torch.distributed.init_process_group(backend='nccl')

# 模型包装
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# 数据分布式采样
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```

---

## 第八章：工具函数

### 8.1 数据分区

```python
def data_partition(dataset):
    """划分训练/验证/测试集"""
    # 训练集：每个用户的前N-2个物品
    # 验证集：第N-1个物品
    # 测试集：最后一个物品
    return user_train, user_valid, user_test
```

### 8.2 评估函数

```python
def evaluate(model, user_train, user_valid, user_test):
    """评估模型：计算NDCG和HR"""
    # 对每个用户，计算所有物品的得分
    # 排序，取top-K
    # 计算NDCG@10和HR@10
```

---

## 第九章：代码阅读指南

### 9.1 推荐阅读顺序

1. **model.py**：`SASRec`类 → `MultiHeadAttention`类 → `PointWiseFeedForward`类
2. **model_mhc.py**：`mHCResidual`类 → `TiSASRec`类
3. **main.py**：训练主循环
4. **utils.py**：数据处理和评估函数

### 9.2 调试技巧

```python
# 1. 查看张量形状
print(x.shape)  # (128, 200, 50)

# 2. 查看是否有NaN
print(torch.isnan(x).any())

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 9.3 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| CUDA out of memory | 显存不足 | 减小batch_size |
| NaN loss | 学习率太大 | 减小lr或启用AMP |
| 形状不匹配 | 张量维度错误 | 检查各层输出形状 |

---

## 总结

你现在应该理解：

✅ 项目代码的整体结构
✅ SASRec模型的核心实现（注意力、前馈网络）
✅ TiSASRec的时间感知注意力机制
✅ mHC残差连接的Sinkhorn-Knopp算法
✅ 训练流程和分布式训练
✅ 如何调试和排查错误

**下一步**：阅读配套的"模型测试与评估"文档，学习如何验证模型效果。
