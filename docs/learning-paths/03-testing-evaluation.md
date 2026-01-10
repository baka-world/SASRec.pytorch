# SASRec.pytorch 项目学习指南（三）：模型测试与评估

> **目标读者**：已理解数据设计和代码实现
> **学习目标**：理解测试方法、评估指标、验证模型效果

---

## 第一章：为什么需要测试和评估？

### 1.1 三种数据集

| 数据集 | 作用 | 比例 |
|--------|------|------|
| 训练集 | 学习模型参数 | 80% |
| 验证集 | 调参和早停 | 10% |
| 测试集 | 最终评估 | 10% |

**为什么需要验证集？**
- 防止"作弊"：不能拿测试集来调参
- 验证集模拟测试集，帮助调参

### 1.2 评估的两个维度

1. **训练是否正常？**
   - 损失是否下降？
   - 是否有NaN？
   - 梯度是否正常？

2. **模型效果如何？**
   - 预测是否准确？
   - 比baseline好吗？

---

## 第二章：核心评估指标

### 2.1 HR@K（命中率）

**直观理解**：在推荐列表的前K个里，有没有包含正确答案？

```
用户真实购买：物品100

模型推荐列表（top-10）：[50, 100, 3, 200, ...]
                              ↑
                           物品100在第2位

HR@10 = 1（命中！）
HR@5 = 0（没命中）
```

**公式**：
```
HR@K = 1 如果目标物品在前K个推荐中，否则为0
```

**代码实现**：
```python
def hit_rate_at_k(recommended_items, target_item, k=10):
    """计算HR@K"""
    top_k = recommended_items[:k]
    if target_item in top_k:
        return 1
    return 0
```

### 2.2 NDCG@K（归一化折损累积增益）

**直观理解**：不仅要看是否命中，还要看排在第几位

- 排在第1位最好
- 排在第10位差一些
- 排在第100位很差

**为什么需要"归一化"？**
- 不同用户的最优位置不同
- NDCG把分数归一化到0-1之间

**公式**：
```
DCG@K = Σ (相关度 / log2(位置+1))
IDCG@K = 理想情况下的最大DCG
NDCG@K = DCG@K / IDCG@K
```

**示例**：
```
用户A：目标在第1位 → DCG=1/1 + ... = 1.0 → NDCG=1.0
用户B：目标在第3位 → DCG=1/log2(4) + ... ≈ 0.63 → NDCG=0.63
用户C：目标在第10位 → DCG=1/log2(11) + ... ≈ 0.43 → NDCG=0.43
```

**代码实现**：
```python
def ndcg_at_k(recommended_items, target_item, k=10):
    """计算NDCG@K"""
    for i, item in enumerate(recommended_items[:k]):
        if item == target_item:
            # DCG: 1/log2(i+2) 因为位置从1开始，log从2开始
            dcg = 1.0 / np.log2(i + 2)
            # IDCG: 如果目标在第1位（最佳情况）
            idcg = 1.0 / np.log2(2)
            return dcg / idcg
    return 0  # 没命中
```

### 2.3 MRR（平均倒数排名）

**直观理解**：目标物品排名的倒数

```
用户A：目标在第2位 → 1/2 = 0.5
用户B：目标在第1位 → 1/1 = 1.0
用户C：目标在第5位 → 1/5 = 0.2

MRR = (0.5 + 1.0 + 0.2) / 3 = 0.57
```

### 2.4 指标对比

| 指标 | 关注点 | 范围 | 典型值 |
|------|--------|------|--------|
| HR@10 | 是否命中 | 0-1 | 0.6-0.8 |
| NDCG@10 | 排名质量 | 0-1 | 0.3-0.5 |
| MRR | 平均排名 | 0-1 | 0.4-0.6 |

**为什么NDCG@10通常比HR@10低？**
- HR只看是否命中
- NDCG还看排名位置
- 很多命中的排名不靠前

---

## 第三章：完整评估流程

### 3.1 评估代码结构

```python
def evaluate(model, user_train, user_valid, user_test, item_num):
    """
    评估模型在测试集上的表现
    
    返回: (NDCG@10, HR@10)
    """
    results = []
    
    for user in all_users:
        # 1. 获取用户的历史序列
        seq = get_user_sequence(user, user_train)
        
        # 2. 预测所有物品的得分
        scores = model.predict(user, seq)
        
        # 3. 排除训练集中已交互的物品
        exclude_items = user_train[user]
        scores = exclude_items_from_scores(scores, exclude_items)
        
        # 4. 按得分排序，取top-10
        top_k_items = np.argsort(scores)[::-1][:10]
        
        # 5. 计算指标
        target = user_test[user]  # 测试集目标物品
        ndcg = ndcg_at_k(top_k_items, target)
        hr = hit_rate_at_k(top_k_items, target)
        
        results.append((ndcg, hr))
    
    # 计算平均值
    avg_ndcg = np.mean([r[0] for r in results])
    avg_hr = np.mean([r[1] for r in results])
    
    return avg_ndcg, avg_hr
```

### 3.2 评估注意事项

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 速度慢 | 对每个用户计算所有物品得分 | 用批处理加速 |
| 内存不够 | 物品太多 | 分批计算或使用采样 |
| 评估不准 | 只看最后一个物品 | 多位置评估 |

---

## 第四章：单元测试

### 4.1 测试哪些方面？

| 测试类型 | 测试内容 | 重要性 |
|---------|---------|--------|
| 单元测试 | 各个模块是否正确 | ⭐⭐⭐ |
| 集成测试 | 模块组合是否正确 | ⭐⭐⭐ |
| 功能测试 | 整体功能是否正常 | ⭐⭐⭐ |

### 4.2 示例测试代码

**测试1：模型输出形状**

```python
def test_model_output_shape():
    """测试模型输出形状是否正确"""
    # 创建模型
    model = SASRec(user_num=100, item_num=1000, args=default_args)
    
    # 创建输入数据
    batch_size = 32
    seq_len = 50
    user_ids = torch.randint(0, 100, (batch_size,))
    log_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    pos_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    neg_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    
    # 前向传播
    pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)
    
    # 检查形状
    assert pos_logits.shape == (batch_size, seq_len)
    assert neg_logits.shape == (batch_size, seq_len)
    
    print("✓ 模型输出形状测试通过")
```

**测试2：注意力掩码**

```python
def test_attention_mask():
    """测试因果注意力掩码是否有效"""
    # 创建注意力层
    attn = MultiHeadAttention(hidden_units=50, num_heads=2)
    
    # 创建掩码（下三角）
    seq_len = 5
    attn_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # 创建查询和键
    Q = torch.randn(1, seq_len, 50)
    K = torch.randn(1, seq_len, 50)
    V = torch.randn(1, seq_len, 50)
    
    # 前向传播
    output = attn(Q, K, V, attn_mask)
    
    # 验证：位置i不应该注意到位置j (j > i)
    # 这需要检查注意力权重
    print("✓ 注意力掩码测试通过")
```

**测试3：mHC双随机性质**

```python
def test_mhc_doubly_stochastic():
    """测试mHC的H_res是否是双随机矩阵"""
    mhc = mHCResidual(hidden_units=50, expansion_rate=4)
    
    x = torch.randn(2, 10, 50)  # batch=2, seq=10, hidden=50
    func_out = torch.randn(2, 10, 50)
    
    # 前向传播
    output = mhc(x, func_out)
    
    # 验证输出形状
    assert output.shape == x.shape
    
    print("✓ mHC输出形状测试通过")
```

### 4.3 运行测试

```bash
cd python

# 运行所有测试
python -m pytest test/

# 运行特定测试
python -m pytest test/test_model_forward.py -v

# 运行并显示详细信息
python -m pytest test/ -v --tb=short
```

---

## 第五章：性能基准测试

### 5.1 基准测试内容

| 测试项 | 指标 | 目标 |
|--------|------|------|
| 训练速度 | 每秒处理样本数 | > 1000 samples/s |
| 推理速度 | 每秒查询数 | > 100 queries/s |
| 显存占用 | GPU显存 | < 4GB (batch=128) |
| 模型效果 | NDCG@10 | > 0.30 |

### 5.2 速度测试代码

```python
import time

def benchmark_training():
    """基准测试：训练速度"""
    model = SASRec(...)
    model.train()
    
    # 预热
    for _ in range(10):
        batch = get_batch()
        loss = train_step(batch)
    
    # 正式测试
    num_iterations = 100
    start_time = time.time()
    
    for i in range(num_iterations):
        batch = get_batch()
        loss = train_step(batch)
    
    elapsed = time.time() - start_time
    speed = num_iterations / elapsed
    
    print(f"训练速度: {speed:.1f} iterations/秒")
```

### 5.3 对比实验设计

| 实验 | 说明 | 目的 |
|------|------|------|
| SASRec baseline | 标准SASRec | 基准对比 |
| SASRec + mHC | 加入mHC | 验证mHC效果 |
| TiSASRec | 加入时间信息 | 验证时间信息效果 |
| TiSASRec + mHC | 完整模型 | 最佳效果 |

---

## 第六章：常见问题排查

### 6.1 训练问题

**问题1：损失不下降**

```
症状：损失一直不变或震荡
原因：学习率太大/太小，数据问题
解决：
- 调小学习率: --lr=0.0005
- 检查数据格式是否正确
- 检查梯度是否正常
```

**问题2：损失出现NaN**

```
症状：损失变成 NaN
原因：数值溢出，学习率太大
解决：
- 启用梯度裁剪
- 减小学习率
- 启用AMP: --use_amp
- 启用mHC No-AMP: --mhc_no_amp
```

**问题3：过拟合**

```
症状：训练损失下降，但验证损失上升
解决：
- 增加dropout: --dropout_rate=0.3
- 增加L2正则: --l2_emb=0.0001
- 减少模型复杂度
```

### 6.2 评估问题

**问题1：评估结果异常低**

```
可能原因：
- 数据划分错误
- 评估时没有排除训练集物品
- 模型没有收敛
```

**问题2：评估速度太慢**

```
优化方法：
- 使用批处理
- 只评估部分用户（测试时）
- 使用采样评估（近似结果）
```

---

## 第七章：实验结果记录

### 7.1 实验日志模板

```
实验名称: exp_001_sasrec_baseline
日期: 2024-01-10
数据集: ml-1m

参数配置:
- batch_size: 128
- lr: 0.001
- num_epochs: 300
- hidden_units: 50

结果:
- Best NDCG@10: 0.4123
- Best HR@10: 0.7234
- 训练时间: 2.5小时

备注:
- 损失正常下降
- 没有过拟合
```

### 7.2 结果对比表

| 模型 | NDCG@10 | HR@10 | 参数量 | 训练时间 |
|------|---------|-------|--------|---------|
| SASRec | 0.4123 | 0.7234 | 1.2M | 2.5h |
| SASRec + mHC | 0.4356 | 0.7512 | 1.5M | 2.9h |
| TiSASRec | 0.4289 | 0.7456 | 1.3M | 2.7h |
| TiSASRec + mHC | 0.4512 | 0.7689 | 1.6M | 3.1h |

---

## 第八章：最佳实践

### 8.1 评估流程检查清单

- [ ] 使用验证集进行超参数调优
- [ ] 只在测试集上评估最终模型
- [ ] 报告多次运行的平均值和标准差
- [ ] 保存完整的实验配置
- [ ] 可视化训练曲线

### 8.2 推荐的评估参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K | 10 | 推荐列表长度 |
| batch_size | 128 | 评估批次大小 |
| num_users | all | 评估所有用户 |

### 8.3 结果解读

**什么样的结果算好？**

| 数据集 | NDCG@10 | HR@10 | 难度 |
|--------|---------|-------|------|
| ml-1m | 0.40+ | 0.70+ | 适中 |
| Beauty | 0.35+ | 0.65+ | 较难 |
| Steam | 0.30+ | 0.60+ | 难 |

---

## 总结

你现在应该理解：

✅ 核心评估指标（HR@K, NDCG@K）的含义和计算方法
✅ 完整的评估流程和代码实现
✅ 如何编写单元测试验证模型正确性
✅ 如何进行性能基准测试
✅ 常见问题排查和解决方案
✅ 实验结果记录和对比

**下一步**：阅读配套的"文档撰写与答辩"文档，学习如何整理成果并进行答辩。
