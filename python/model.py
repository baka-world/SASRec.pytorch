import numpy as np
import torch
import sys

FLOAT_MIN = -1000.0


class NumericallyStableMultiheadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, add_zero_attn=False):
        super(NumericallyStableMultiheadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_size = hidden_units // num_heads

        self.q_w = torch.nn.Linear(hidden_units, hidden_units)
        self.k_w = torch.nn.Linear(hidden_units, hidden_units)
        self.v_w = torch.nn.Linear(hidden_units, hidden_units)
        self.out_w = torch.nn.Linear(hidden_units, hidden_units)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.add_zero_attn = add_zero_attn

        torch.nn.init.xavier_uniform_(self.q_w.weight)
        torch.nn.init.xavier_uniform_(self.k_w.weight)
        torch.nn.init.xavier_uniform_(self.v_w.weight)
        torch.nn.init.xavier_uniform_(self.out_w.weight)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size = queries.shape[0]
        seq_len = queries.shape[1]

        Q = self.q_w(queries)
        K = self.k_w(keys)
        V = self.v_w(values)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        if torch.isnan(Q).any() or torch.isnan(K).any() or torch.isnan(V).any():
            print(f"DEBUG: NaN in Q, K, or V in attention layer")
            import sys

            sys.exit(1)

        attn_scores = torch.matmul(Q, K.transpose(2, 3)) / (self.head_size**0.5)

        if torch.isnan(attn_scores).any():
            print(f"DEBUG: NaN in attn_scores after matmul")
            import sys

            sys.exit(1)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = (
                    attn_mask.bool()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, self.num_heads, 1, 1)
                )
            elif attn_mask.dim() == 3:
                mask = attn_mask.bool().unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            elif attn_mask.dim() == 4:
                mask = attn_mask.bool().repeat(
                    max(1, batch_size // attn_mask.shape[0]), 1, 1, 1
                )
            else:
                mask = None
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, FLOAT_MIN)

        if torch.isnan(attn_scores).any():
            print(f"DEBUG: NaN in attn_scores after masked_fill")
            import sys

            sys.exit(1)

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        if torch.isnan(attn_weights).any():
            print(f"DEBUG: NaN in attn_weights after softmax")
            print(
                f"  attn_scores min: {attn_scores.min().item()}, max: {attn_scores.max().item()}"
            )
            import sys

            sys.exit(1)

        attn_weights = self.dropout(attn_weights)

        outputs = torch.matmul(attn_weights, V)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs.view(batch_size, seq_len, self.hidden_units)
        outputs = self.out_w(outputs)

        return outputs, attn_weights


class PointWiseFeedForward(torch.nn.Module):
    """
    点式前馈神经网络（Point-wise Feed Forward Network）

    这是Transformer架构中的重要组成部分，对每个位置的表示进行两次线性变换。
    结构：输入 -> 线性变换 -> ReLU激活 -> Dropout -> 线性变换 -> Dropout -> 输出

    作用：
    - 增加模型的非线性表达能力
    - 让每个位置的表示可以独立地进行非线性变换
    - 与自注意力机制配合，提升模型性能
    """

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        # 第一次卷积：将隐藏维度映射到相同维度
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        # 第二次卷积：将隐藏维度映射回原始维度
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        前向传播函数

        参数:
            inputs: 输入张量，形状为 (batch_size, seq_len, hidden_units)

        返回:
            outputs: 输出张量，形状为 (batch_size, seq_len, hidden_units)
        """
        # 转置以适应Conv1d的输入格式 (batch, channels, length)
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        # 转置回原始格式
        outputs = outputs.transpose(-1, -2)
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    """
    时序感知多头注意力机制（Time-Aware Multi-Head Attention）

    这是TiSASRec的核心创新，在标准自注意力的基础上融入时间间隔信息。

    注意力权重计算公式：
    A_ij = softmax( Q_i * K_j^T + Q_i * abs_pos_K_i^T + time_matrix_K_j * Q_i )

    组成部分：
    1. Q * K^T：标准内容相似度
    2. Q * abs_pos_K^T：绝对位置编码的影响
    3. time_matrix_K * Q：时间间隔的影响（核心创新点）

    作用：
    - 让模型学习到"越近的行为越相关"的模式
    - 捕捉用户兴趣随时间的演变
    - 增强序列推荐的准确性
    """

    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()

        # Q, K, V 的线性变换
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(
        self,
        queries,
        keys,
        time_mask,
        attn_mask,
        time_matrix_K,
        time_matrix_V,
        abs_pos_K,
        abs_pos_V,
    ):
        """
        前向传播

        参数:
            queries: 查询张量，形状为 (batch_size, seq_len, hidden_size)
            keys: 键张量，形状为 (batch_size, seq_len, hidden_size)
            time_mask: 时间掩码，形状为 (batch_size, seq_len)
            attn_mask: 注意力掩码，用于因果注意力
            time_matrix_K: 时间矩阵K，形状为 (batch_size, seq_len, seq_len, hidden_size)
            time_matrix_V: 时间矩阵V，形状为 (batch_size, seq_len, seq_len, hidden_size)
            abs_pos_K: 绝对位置编码K，形状为 (batch_size, seq_len, hidden_size)
            abs_pos_V: 绝对位置编码V，形状为 (batch_size, seq_len, hidden_size)

        返回:
            outputs: 输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # 线性变换得到 Q, K, V
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # 分割成多个头 (num_head * N, T, C / num_head)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # 分割时间矩阵
        time_matrix_K_ = torch.cat(
            torch.split(time_matrix_K, self.head_size, dim=3), dim=0
        )
        time_matrix_V_ = torch.cat(
            torch.split(time_matrix_V, self.head_size, dim=3), dim=0
        )
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # 计算注意力权重（核心创新：三部分相加）
        # 1. 内容相似度
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        # 2. 绝对位置编码的影响
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        # 3. 时间间隔的影响（核心创新点）
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # 缩放因子
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # 应用掩码
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        # 填充位置的权重设为极小值
        paddings = torch.ones(attn_weights.shape) * (-(2**32) + 1)
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        # Softmax归一化
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += (
            attn_weights.unsqueeze(2)
            .matmul(time_matrix_V_)
            .reshape(outputs.shape)
            .squeeze(2)
        )

        # 合并多个头 (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class TiSASRec(torch.nn.Module):
    """
    TiSASRec：Time Interval Aware Self-Attentive Sequential Recommendation

    这是论文 "Time Interval Aware Self-Attention for Sequential Recommendation"
    (Li et al., 2020) 的PyTorch实现。

    核心创新：
    在SASRec的基础上引入时间间隔信息，让模型能够学习到：
    - 越近的行为越相关
    - 用户兴趣随时间的变化趋势
    - 不同时间间隔对推荐的影响

    模型架构：
    1. 物品嵌入层：将物品ID映射为密集向量
    2. 位置嵌入层：给序列中的每个位置添加位置信息
    3. 时间矩阵嵌入层：学习时间间隔的向量表示（核心创新）
    4. 多层时序感知自注意力块：学习带时间信息的序列依赖
    5. 输出层：基于序列表示预测下一个物品
    """

    def __init__(self, user_num, item_num, time_span, args):
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.time_span = time_span
        self.dev = args.device
        self.norm_first = getattr(args, "norm_first", False)

        # 物品嵌入层
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 绝对位置嵌入（与SASRec相同）
        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        # 时间矩阵嵌入（核心创新：学习时间间隔的向量表示）
        self.time_matrix_K_emb = torch.nn.Embedding(
            args.time_span + 1, args.hidden_units
        )
        self.time_matrix_V_emb = torch.nn.Embedding(
            args.time_span + 1, args.hidden_units
        )

        # Dropout层
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Transformer块组件
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 构建多个时序感知Transformer编码器块
        for _ in range(args.num_blocks):
            # 自注意力前的LayerNorm
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # 时序感知多头注意力层（核心创新）
            new_attn_layer = TimeAwareMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate, args.device
            )
            self.attention_layers.append(new_attn_layer)

            # 前馈网络前的LayerNorm
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 点式前馈网络
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        """
        将用户行为日志序列转换为特征表示（包含时间信息）

        流程：
        1. 将物品ID转换为嵌入向量
        2. 添加绝对位置嵌入
        3. 计算并嵌入时间间隔矩阵
        4. 通过多个时序感知Transformer编码器块学习序列特征
        5. 最终输出序列的向量表示

        参数:
            user_ids: 用户ID列表
            log_seqs: 用户行为序列，形状为 (batch_size, seq_len)
            time_matrices: 时间间隔矩阵，形状为 (batch_size, seq_len, seq_len)

        返回:
            log_feats: 序列的特征表示，形状为 (batch_size, seq_len, hidden_units)
        """
        # 物品嵌入
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        seqs = self.item_emb_dropout(seqs)

        # 绝对位置嵌入
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        positions = torch.tensor(positions, dtype=torch.long, device=self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        # 时间矩阵嵌入（核心创新）
        time_matrices = torch.tensor(time_matrices, dtype=torch.long, device=self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # 掩码：标记padding位置（物品ID为0）
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # 因果注意力掩码：防止信息泄露
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        # 依次通过每个时序感知Transformer编码器块
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                # Pre-LN结构：先LayerNorm，再注意力，再残差连接
                Q = self.attention_layernorms[i](seqs)

                # 时序感知自注意力计算
                mha_outputs = self.attention_layers[i](
                    Q,
                    seqs,
                    timeline_mask,
                    attention_mask,
                    time_matrix_K,
                    time_matrix_V,
                    abs_pos_K,
                    abs_pos_V,
                )

                # 残差连接
                seqs = Q + mha_outputs

                # FFN部分
                seqs = self.forward_layernorms[i](seqs)
                seqs = self.forward_layers[i](seqs)
            else:
                # Post-LN结构：先注意力，再LayerNorm，再残差连接
                # 使用seqs作为Q，K，V（不经过LayerNorm）
                mha_outputs = self.attention_layers[i](
                    seqs,
                    seqs,
                    timeline_mask,
                    attention_mask,
                    time_matrix_K,
                    time_matrix_V,
                    abs_pos_K,
                    abs_pos_V,
                )

                # 残差连接后应用LayerNorm
                seqs = self.attention_layernorms[i](seqs + mha_outputs)

                # FFN部分
                ffn_output = self.forward_layers[i](seqs)
                seqs = self.forward_layernorms[i](seqs + ffn_output)

            seqs *= ~timeline_mask.unsqueeze(-1)

        # 最终LayerNorm归一化
        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs):
        """
        训练时的前向传播

        使用对比学习策略：
        - 正样本：用户实际交互的下一个物品
        - 负样本：用户未交互的随机物品

        目标：让正样本的得分高于负样本

        参数:
            user_ids: 用户ID列表
            log_seqs: 用户历史行为序列
            time_matrices: 时间间隔矩阵
            pos_seqs: 正样本序列（用户下一个实际交互的物品）
            neg_seqs: 负样本序列（随机采样的未交互物品）

        返回:
            pos_logits: 正样本的预测得分
            neg_logits: 负样本的预测得分
        """
        # 获取序列特征表示（包含时间信息）
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        # 获取正负样本的嵌入向量
        pos_embs = self.item_emb(
            torch.tensor(pos_seqs, dtype=torch.long, device=self.dev)
        )
        neg_embs = self.item_emb(
            torch.tensor(neg_seqs, dtype=torch.long, device=self.dev)
        )

        # 计算正负样本与序列表示的点积得分
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_matrices, item_indices):
        """
        推理时的预测函数

        对给定用户和候选物品集合，计算每个物品的推荐得分

        参数:
            user_ids: 用户ID列表
            log_seqs: 用户历史行为序列
            time_matrices: 时间间隔矩阵
            item_indices: 候选物品ID列表

        返回:
            logits: 每个候选物品的预测得分
        """
        # 获取序列特征表示（包含时间信息）
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        # 只使用序列最后一个位置的表示进行预测
        final_feat = log_feats[:, -1, :]

        # 获取候选物品的嵌入向量
        item_embs = self.item_emb(
            torch.tensor(item_indices, dtype=torch.long, device=self.dev)
        )

        # 计算候选物品与用户兴趣表示的点积得分
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class SASRec(torch.nn.Module):
    """
    SASRec：Self-Attentive Sequential Recommendation（自注意力序列推荐）

    这是论文 "Self-attentive sequential recommendation" (Kang & McAuley, 2018) 的PyTorch实现。

    核心思想：
    - 将用户的历史行为序列（如点击、购买记录）建模为有序的序列
    - 使用Transformer的自注意力机制学习序列中的长期依赖关系
    - 基于学习到的序列表示，预测用户下一个可能交互的物品

    模型架构：
    1. 物品嵌入层：将物品ID映射为密集向量
    2. 位置嵌入层：给序列中的每个位置添加位置信息
    3. 多层自注意力块：学习序列内部的依赖关系
    4. 输出层：基于序列表示预测下一个物品
    """

    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num  # 用户总数
        self.item_num = item_num  # 物品总数
        self.dev = args.device  # 计算设备
        self.norm_first = args.norm_first  # 是否先进行LayerNorm

        # 物品嵌入层：将物品ID映射为 hidden_units 维的向量
        # padding_idx=0 表示物品ID为0的位置嵌入向量为全0（用于padding）
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )

        # 位置嵌入层：记录序列中每个位置的相对位置信息
        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 定义多组LayerNorm和层，用于构建多个Transformer Block
        self.attention_layernorms = torch.nn.ModuleList()  # 自注意力前的LayerNorm
        self.attention_layers = torch.nn.ModuleList()  # 自注意力层
        self.forward_layernorms = torch.nn.ModuleList()  # 前馈网络前的LayerNorm
        self.forward_layers = torch.nn.ModuleList()  # 前馈网络层

        # 最终的LayerNorm，对整个序列输出进行归一化
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 构建 num_blocks 个Transformer编码器块
        for _ in range(args.num_blocks):
            # 自注意力部分的LayerNorm
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # 多头自注意力层
            new_attn_layer = NumericallyStableMultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            # 前馈网络部分的LayerNorm
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 点式前馈网络
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        将用户行为日志序列转换为特征表示

        流程：
        1. 将物品ID转换为嵌入向量
        2. 添加位置嵌入以捕捉序列顺序信息
        3. 通过多个Transformer编码器块学习序列特征
        4. 最终输出序列的向量表示

        参数:
            log_seqs: 用户行为序列，形状为 (batch_size, seq_len)，存储物品ID

        返回:
            log_feats: 序列的特征表示，形状为 (batch_size, seq_len, hidden_units)
        """
        # 物品ID转换为嵌入向量
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))
        # 缩放嵌入向量（与Transformer原论文一致）
        seqs *= self.item_emb.embedding_dim**0.5

        # 创建位置索引 [1, 2, 3, ..., seq_len]
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # 将padding位置的位置嵌入设为0
        poss = poss * (log_seqs.cpu().numpy() != 0)
        # 添加位置嵌入
        seqs += self.pos_emb(torch.tensor(poss, dtype=torch.long, device=self.dev))
        seqs = self.emb_dropout(seqs)

        # 创建因果注意力掩码，防止信息泄露
        # 只允许位置i关注位置i及之前的位置（符合序列预测的任务设定）
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        # 依次通过每个Transformer编码器块
        for i in range(len(self.attention_layers)):
            # 转置以适应MultiheadAttention的输入格式 (seq_len, batch, features)
            seqs = torch.transpose(seqs, 0, 1)

            if self.norm_first:
                # Pre-LN结构：先LayerNorm，再注意力，再残差连接
                x = self.attention_layernorms[i](seqs)
                attn_layer = self.attention_layers[i]
                x_attn = x.transpose(0, 1)
                mha_outputs_T, _ = attn_layer(
                    x_attn, x_attn, x_attn, attn_mask=attention_mask
                )
                mha_outputs = mha_outputs_T.transpose(0, 1)
                seqs = seqs + mha_outputs

                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # Post-LN结构：先注意力，再LayerNorm，再残差连接
                attn_layer = self.attention_layers[i]
                seqs_attn = seqs.transpose(0, 1)
                mha_outputs_T, _ = attn_layer(
                    seqs_attn, seqs_attn, seqs_attn, attn_mask=attention_mask
                )
                mha_outputs = mha_outputs_T.transpose(0, 1)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        # 最终LayerNorm归一化
        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        训练时的前向传播

        使用对比学习策略：
        - 正样本：用户实际交互的下一个物品
        - 负样本：用户未交互的随机物品

        目标：让正样本的得分高于负样本

        参数:
            user_ids: 用户ID列表（当前未使用）
            log_seqs: 用户历史行为序列
            pos_seqs: 正样本序列（用户下一个实际交互的物品）
            neg_seqs: 负样本序列（随机采样的未交互物品）

        返回:
            pos_logits: 正样本的预测得分
            neg_logits: 负样本的预测得分
        """
        # 获取序列特征表示
        log_feats = self.log2feats(log_seqs)

        # 获取正负样本的嵌入向量
        pos_embs = self.item_emb(
            torch.tensor(pos_seqs, dtype=torch.long, device=self.dev)
        )
        neg_embs = self.item_emb(
            torch.tensor(neg_seqs, dtype=torch.long, device=self.dev)
        )

        # 计算正负样本与序列表示的点积得分
        # 点积越大，表示该物品与用户兴趣越匹配
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        推理时的预测函数

        对给定用户和候选物品集合，计算每个物品的推荐得分

        参数:
            user_ids: 用户ID列表
            log_seqs: 用户历史行为序列
            item_indices: 候选物品ID列表

        返回:
            logits: 每个候选物品的预测得分
        """
        # 获取序列特征表示
        log_feats = self.log2feats(log_seqs)

        # 只使用序列最后一个位置的表示进行预测
        # 这个位置包含了用户所有历史行为的信息
        final_feat = log_feats[:, -1, :]

        # 获取候选物品的嵌入向量
        item_embs = self.item_emb(
            torch.tensor(item_indices, dtype=torch.long, device=self.dev)
        )

        # 计算候选物品与用户兴趣表示的点积得分
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
