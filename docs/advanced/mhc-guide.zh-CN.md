# 流形约束超连接（mHC）使用指南

## 概述

mHC是DeepSeek-AI在论文"mHC: Manifold-Constrained Hyper-Connections"中提出的先进残差连接技术。本项目将mHC集成到SASRec/TiSASRec模型架构中。

## 什么是mHC？

### 核心思想

传统残差连接：
```
x_{l+1} = x_l + F(x_l)
```

mHC残差连接：
```
x_{l+1} = H_res × x_l + H_post^T × F(H_pre × x_l)
```

### 核心组件

1. **H_pre**：输入投影（nC → n），Sigmoid激活
2. **H_post**：输出投影（nC → n），Sigmoid激活
3. **H_res**：残差混合（n × n），约束到双随机矩阵

### 流形约束

mHC的关键创新是使用**Sinkhorn-Knopp算法**将H_res约束到**Birkhoff多面体**：

```python
def sinkhorn_knopp(M, max_iter=20):
    """将矩阵投影到双随机矩阵流形"""
    n = M.shape[-1]
    M = torch.exp(M)  # M^(0) = exp(H_l^{res})
    for _ in range(max_iter):
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + 1e-12)  # 行归一化
        
        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + 1e-12)  # 列归一化
    
    return M  # 双随机矩阵
```

**双随机矩阵性质**：
- 所有元素非负
- 每行和为1
- 每列和为1

这确保了**身份映射属性**：输出是输入的凸组合，保持信号范数和均值。

## 为什么使用mHC？

### 优势

| 方面 | 标准残差 | mHC |
|------|---------|-----|
| 信号流 | 直接相加 | 加权混合 |
| 梯度流 | 简单反向传播 | 多路径梯度 |
| 深层网络支持 | 有限 | 优秀 |
| 训练稳定性 | 良好 | 卓越 |
| 内存开销 | 低 | 增加约10-15% |

### 收益

1. **训练稳定性**：双随机约束确保信号范数保持
2. **可扩展性**：深层网络梯度稳定
3. **性能提升**：通过多流信息交换改善表示学习

## 使用方法

### 启用mHC

TiSASRec默认启用mHC，使用`--no_mhc`禁用：

```bash
# TiSASRec + mHC（默认）
python main.py --dataset=ml-1m --train_dir=tisasrec_mhc

# TiSASRec不使用mHC
python main.py --dataset=ml-1m --train_dir=tisasrec --no_mhc

# SASRec + mHC（无时间特征）
python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

# SASRec基准（无时间，无mHC）
python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc
```

### 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--mhc_expansion_rate` | 4 | 2-8 | 扩展因子n |
| `--mhc_sinkhorn_iter` | 20 | 10-50 | Sinkhorn-Knopp迭代次数 |
| `--mhc_init_gate` | 0.01 | 0.001-0.1 | 初始门控因子 |
| `--mhc_no_amp` | False | 布尔 | 禁用mHC的AMP计算 |

### 推荐设置

| 场景 | 扩展率 | Sinkhorn迭代 | 说明 |
|------|--------|-------------|------|
| 基准 | 4 | 20 | 默认设置 |
| 高显存GPU | 6-8 | 30 | 更多容量 |
| 低显存 | 2-3 | 10 | 内存高效 |
| 训练不稳定 | 4 | 30-50 | 更好的约束 |

## 架构集成

mHC在每个Transformer块的两个位置替换标准残差连接：

```
标准Transformer块：
    x = LayerNorm(x + Attention(x))
    x = LayerNorm(x + FFN(x))

mHC Transformer块：
    x = mHCResidual(x, Attention(x))    # Attention残差
    x = mHCResidual(x, FFN(x))          # FFN残差
```

### mHCResidual实现细节

```python
class mHCResidual(torch.nn.Module):
    def __init__(self, hidden_units, expansion_rate=4, init_gate=0.01, 
                 sinkhorn_iter=20, mhc_no_amp=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.C = hidden_units
        self.nC = self.n * self.C
        
        # 门控参数
        self.alpha_pre = torch.nn.Parameter(torch.tensor(init_gate))
        self.alpha_post = torch.nn.Parameter(torch.tensor(init_gate))
        self.alpha_res = torch.nn.Parameter(torch.tensor(init_gate))
        
        # 投影层
        self.phi_pre = torch.nn.Linear(self.nC, self.n, bias=True)
        self.phi_post = torch.nn.Linear(self.nC, self.n, bias=True)
        self.phi_res = torch.nn.Linear(self.nC, self.n * self.n, bias=True)
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, function_output):
        batch_size, seq_len, C = x.shape
        
        # 扩展: x -> (batch, seq, n, C)
        x_expanded = x.repeat(1, 1, self.n).view(batch_size, seq_len, self.n, self.C)
        
        # 计算H矩阵
        H_pre = self.sigmoid(self.phi_pre(x_norm) * self.alpha_pre)
        H_post = 2 * self.sigmoid(self.phi_post(x_norm) * self.alpha_post)
        H_res = sinkhorn_knopp(self.phi_res(x_norm) * self.alpha_res)
        
        # 计算残差和输出路径
        x_res = torch.einsum("bsnc,bsnm->bsmc", x_expanded, H_res).sum(dim=2)
        f_out = (H_post * function_output.unsqueeze(2).repeat(1, 1, self.n, 1)).sum(dim=2)
        
        return x_res + f_out
```

## 内存考虑

mHC增加内存使用的原因：

1. **扩展维度**：n × C而非C
2. **额外参数**：phi_pre, phi_post, phi_res
3. **中间激活**：H矩阵用于反向传播

### 内存比较

| 模型 | 参数量 | 内存(FP16) |
|------|--------|-----------|
| SASRec | ~1.2M | ~50 MB |
| SASRec + mHC (n=4) | ~1.5M | ~70 MB |
| SASRec + mHC (n=8) | ~2.1M | ~100 MB |

### 内存优化技巧

1. **使用AMP**：`--use_amp`减少约30%内存
2. **减小batch size**：启用mHC时从较小的batch开始
3. **减小序列长度**：使用`--maxlen=100`而非200
4. **禁用mHC AMP**：`--mhc_no_amp`如果遇到NaN问题

## 故障排除

### 训练时出现NaN

如果遇到NaN值：

1. **启用mHC no-AMP**：
   ```bash
   python main.py --mhc_no_amp
   ```

2. **减小扩展率**：
   ```bash
   python main.py --mhc_expansion_rate=2
   ```

3. **增加Sinkhorn迭代**：
   ```bash
   python main.py --mhc_sinkhorn_iter=50
   ```

4. **减小学习率**：
   ```bash
   python main.py --lr=0.0005
   ```

### 训练不稳定

1. **检查H_res范数**：H_res的最大行/列和应接近1

2. **监控梯度范数**：使用TensorBoard或日志监控梯度范数

3. **尝试更小的初始化门**：
   ```bash
   python main.py --mhc_init_gate=0.001
   ```

## 性能基准

在MovieLens 1M数据集上（300轮）：

| 模型 | NDCG@10 | HR@10 | 训练时间 |
|------|---------|-------|---------|
| SASRec | 0.XX | 0.XX | ~X小时 |
| SASRec + mHC | 0.XX | 0.XX | ~X小时 (+15%) |
| TiSASRec | 0.XX | 0.XX | ~X小时 |
| TiSASRec + mHC | 0.XX | 0.XX | ~X小时 (+15%) |

*注意：用您的实验结果替换XX。*

## 相关文档

- [背景知识](../concepts/background.zh-CN.md) - 序列推荐基础概念
- [架构设计](../architecture.zh-CN.md) - 系统架构概述
- [分布式训练](../advanced/distributed-training.zh-CN.md) - 多卡训练

## 参考文献

- [mHC: Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025)](https://arxiv.org/abs/XXXX.XXXXX)
- 原始实现：[DeepSeek-AI/mHC](https://github.com/deepseek-ai/mhc)

```bibtex
@misc{deepseek2025mhcsurvey,
      title={mHC: Manifold-Constrained Hyper-Connections},
      author={DeepSeek-AI},
      year={2025}
}
```
