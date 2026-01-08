import numpy as np
import torch
import sys

FLOAT_MIN = -sys.float_info.max


def sinkhorn_knopp(M, max_iter=20):
    """
    Sinkhorn-Knopp算法：将矩阵投影到双随机矩阵流形

    论文公式: M^(0) = exp(H_l^{res})

    双随机矩阵要求：
    1. 所有元素非负
    2. 每行和为1
    3. 每列和为1

    参数:
        M: 输入矩阵，形状为 (..., n, n)
        max_iter: 最大迭代次数

    返回:
        双随机矩阵，形状与输入相同
    """
    n = M.shape[-1]
    M = torch.clamp(M, min=-10.0, max=10.0)
    M = torch.exp(M)
    for _ in range(max_iter):
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + 1e-12)

        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + 1e-12)

    return M


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

        attn_scores = torch.matmul(Q, K.transpose(2, 3)) / (self.head_size**0.5)

        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.num_heads, seq_len, seq_len)
                )
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif mask.dim() == 4:
                mask = mask.expand(batch_size, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask, FLOAT_MIN)

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
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

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    """
    时序感知多头注意力机制（Time-Aware Multi-Head Attention）
    """

    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()

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
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(
            torch.split(time_matrix_K, self.head_size, dim=3), dim=0
        )
        time_matrix_V_ = torch.cat(
            torch.split(time_matrix_V, self.head_size, dim=3), dim=0
        )
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        paddings = torch.ones(attn_weights.shape) * (-(2**32) + 1)
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += (
            attn_weights.unsqueeze(2)
            .matmul(time_matrix_V_)
            .reshape(outputs.shape)
            .squeeze(2)
        )

        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class mHCResidual(torch.nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) 残差连接模块

    论文核心思想：
    1. 将残差流扩展为n-stream（扩展因子n）
    2. 使用三个映射矩阵：H_pre, H_post, H_res
    3. H_pre和H_post使用Sigmoid激活确保非负性
    4. H_res使用Sinkhorn-Knopp算法投影到双随机矩阵流形
    5. 这确保了身份映射属性，保持信号传播的稳定性

    公式：
    x_{l+1} = H_l^{res} × x_l + H_l^{post}^T × F(H_l^{pre} × x_l, W_l)

    正确实现：
    - 输入 x: (batch, seq, C)
    - 扩展到 n*C: x_expanded = x.repeat(1, 1, n) -> (batch, seq, n*C)
    - H_pre: (batch, seq, n) - 每个位置有n个权重
    - H_res: (batch, n, n) - n×n双随机矩阵
    - 输出: 压缩回 C 维度
    """

    def __init__(
        self,
        hidden_units,
        expansion_rate=4,
        init_gate=0.01,
        sinkhorn_iter=20,
        mhc_no_amp=False,
    ):
        """
        参数:
            hidden_units: 隐藏层维度 C
            expansion_rate: 扩展因子 n，默认为4
            init_gate: 门控因子α的初始值，默认为0.01
            sinkhorn_iter: Sinkhorn-Knopp算法的迭代次数，默认为20
            mhc_no_amp: 是否禁用mHC模块的AMP计算，默认为False
        """
        super(mHCResidual, self).__init__()

        self.hidden_units = hidden_units
        self.expansion_rate = expansion_rate
        self.sinkhorn_iter = sinkhorn_iter
        self.mhc_no_amp = mhc_no_amp

        self.n = expansion_rate
        self.C = hidden_units
        self.nC = self.n * self.C

        self.alpha_pre = torch.nn.Parameter(torch.tensor(init_gate))
        self.alpha_post = torch.nn.Parameter(torch.tensor(init_gate))
        self.alpha_res = torch.nn.Parameter(torch.tensor(init_gate))

        self.phi_pre = torch.nn.Linear(self.nC, self.n, bias=True)
        self.phi_post = torch.nn.Linear(self.nC, self.n, bias=True)
        self.phi_res = torch.nn.Linear(self.nC, self.n * self.n, bias=True)

        torch.nn.init.xavier_uniform_(self.phi_pre.weight)
        torch.nn.init.xavier_uniform_(self.phi_post.weight)
        torch.nn.init.xavier_uniform_(self.phi_res.weight)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, function_output):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_len, C)
            function_output: 残差函数F的输出，形状为 (batch_size, seq_len, C)

        返回:
            输出张量，形状为 (batch_size, seq_len, C)
        """
        if self.mhc_no_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward_impl(x, function_output)
        return self._forward_impl(x, function_output)

    def _forward_impl(self, x, function_output):
        batch_size, seq_len, C = x.shape

        if torch.isnan(x).any():
            print(f"DEBUG: NaN detected in mHCResidual input x")
            import sys

            sys.exit(1)

        x_expanded = x.repeat(1, 1, self.n)
        x_expanded = x_expanded.view(batch_size, seq_len, self.n, self.C)

        x_flat = x_expanded.reshape(batch_size, seq_len, self.nC)
        x_norm = x_flat / (
            torch.norm(x_flat, p=2, dim=-1, keepdim=True) / (self.nC**0.5) + 1e-12
        )

        if torch.isnan(x_norm).any():
            print(f"DEBUG: NaN detected in x_norm")
            import sys

            sys.exit(1)

        H_tilde_pre = self.alpha_pre * (
            x_norm @ self.phi_pre.weight.t() + self.phi_pre.bias
        )
        H_tilde_post = self.alpha_post * (
            x_norm @ self.phi_post.weight.t() + self.phi_post.bias
        )
        H_tilde_res_flat = self.alpha_res * (
            x_norm @ self.phi_res.weight.t() + self.phi_res.bias
        )

        if torch.isnan(H_tilde_pre).any() or torch.isnan(H_tilde_res_flat).any():
            print(f"DEBUG: NaN detected in H_tilde values")
            import sys

            sys.exit(1)

        H_pre = self.sigmoid(H_tilde_pre)
        H_post = 2 * self.sigmoid(H_tilde_post)
        H_res = sinkhorn_knopp(
            H_tilde_res_flat.view(-1, self.n, self.n), max_iter=self.sinkhorn_iter
        )

        if torch.isnan(H_res).any():
            print(f"DEBUG: NaN detected in H_res after sinkhorn_knopp")
            import sys

            sys.exit(1)

        H_pre = H_pre.view(batch_size, seq_len, self.n, 1)
        H_post = H_post.view(batch_size, seq_len, self.n, 1)
        H_res = H_res.view(batch_size, seq_len, self.n, self.n)

        func_expanded = function_output.unsqueeze(2).repeat(1, 1, self.n, 1)

        if torch.isnan(func_expanded).any():
            print(f"DEBUG: NaN in func_expanded")
            import sys

            sys.exit(1)

        x_in = (H_pre * x_expanded).sum(dim=2)
        f_out = (H_post * func_expanded).sum(dim=2)

        if torch.isnan(f_out).any():
            print(f"DEBUG: NaN in f_out after sum")
            import sys

            sys.exit(1)

        x_res = torch.einsum("bsnc,bsnm->bsmc", x_expanded, H_res)
        x_res = x_res.sum(dim=2)

        if torch.isnan(x_res).any():
            print(f"DEBUG: NaN in x_res after einsum")
            print(f"  x_expanded has NaN: {torch.isnan(x_expanded).any().item()}")
            print(f"  H_res has NaN: {torch.isnan(H_res).any().item()}")
            import sys

            sys.exit(1)

        output = x_res + f_out

        if torch.isnan(output).any():
            print(f"DEBUG: NaN in output before return")
            import sys

            sys.exit(1)

        return output

    def expand_input(self, x):
        if x.shape[-1] == self.nC:
            return x
        return x.repeat(1, 1, self.n)

    def compress_output(self, x):
        if x.shape[-1] == self.C:
            return x
        return x.mean(dim=-1, keepdim=True).expand(-1, -1, self.C)


class TiSASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, time_span, args):
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.time_span = time_span
        self.dev = args.device

        self.use_mhc = getattr(args, "use_mhc", False)
        self.mhc_expansion_rate = getattr(args, "mhc_expansion_rate", 4)
        self.mhc_no_amp = getattr(args, "mhc_no_amp", False)

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.time_matrix_K_emb = torch.nn.Embedding(
            args.time_span + 1, args.hidden_units
        )
        self.time_matrix_V_emb = torch.nn.Embedding(
            args.time_span + 1, args.hidden_units
        )

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        if self.use_mhc:
            self.mhc_attn = torch.nn.ModuleList()
            self.mhc_ffn = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate, args.device
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            if self.use_mhc:
                self.mhc_attn.append(
                    mHCResidual(
                        args.hidden_units,
                        expansion_rate=self.mhc_expansion_rate,
                        mhc_no_amp=self.mhc_no_amp,
                    )
                )
                self.mhc_ffn.append(
                    mHCResidual(
                        args.hidden_units,
                        expansion_rate=self.mhc_expansion_rate,
                        mhc_no_amp=self.mhc_no_amp,
                    )
                )

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrices = torch.clamp(time_matrices, min=0, max=self.time_span)
        time_matrix_K = self.time_matrix_K_emb(time_matrices).float()
        time_matrix_V = self.time_matrix_V_emb(time_matrices).float()
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)

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

            if self.use_mhc:
                seqs = self.mhc_attn[i](seqs, mha_outputs)
            else:
                seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            ffn_output = self.forward_layers[i](seqs)

            if self.use_mhc:
                seqs = self.mhc_ffn[i](seqs, ffn_output)
            else:
                seqs = seqs + ffn_output

            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs):
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_matrices, item_indices):
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.use_mhc = getattr(args, "use_mhc", False)
        self.mhc_expansion_rate = getattr(args, "mhc_expansion_rate", 4)
        self.mhc_no_amp = getattr(args, "mhc_no_amp", False)

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )

        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        if self.use_mhc:
            self.mhc_attn = torch.nn.ModuleList()
            self.mhc_ffn = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = NumericallyStableMultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            if self.use_mhc:
                self.mhc_attn.append(
                    mHCResidual(
                        args.hidden_units,
                        expansion_rate=self.mhc_expansion_rate,
                        mhc_no_amp=self.mhc_no_amp,
                    )
                )
                self.mhc_ffn.append(
                    mHCResidual(
                        args.hidden_units,
                        expansion_rate=self.mhc_expansion_rate,
                        mhc_no_amp=self.mhc_no_amp,
                    )
                )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        if torch.isnan(seqs).any():
            print(f"DEBUG: NaN in item_emb output")
            import sys

            sys.exit(1)

        seqs *= self.item_emb.embedding_dim**0.5

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss = poss * (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        if torch.isnan(seqs).any():
            print(f"DEBUG: NaN in seqs after emb_dropout")
            import sys

            sys.exit(1)

        for i in range(len(self.attention_layers)):
            batch_size, seq_len, hidden_dim = seqs.shape

            if self.norm_first:
                seqs_T = torch.transpose(seqs, 0, 1)
                x = self.attention_layernorms[i](seqs_T)

                attention_mask = ~torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev)
                )

                mha_outputs, _ = self.attention_layers[i](
                    x, x, x, attn_mask=attention_mask
                )

                if torch.isnan(mha_outputs).any():
                    print(f"DEBUG: NaN in mha_outputs at block {i}")
                    import sys

                    sys.exit(1)

                seqs_T = seqs_T + mha_outputs
                seqs_after_attn = torch.transpose(seqs_T, 0, 1)

                if torch.isnan(seqs_after_attn).any():
                    print(f"DEBUG: NaN in seqs_after_attn at block {i}")
                    import sys

                    sys.exit(1)

                if self.use_mhc:
                    seqs = self.mhc_attn[i](seqs, seqs_after_attn)
                    if torch.isnan(seqs).any():
                        print(f"DEBUG: NaN after mhc_attn[{i}]")
                        import sys

                        sys.exit(1)
                else:
                    seqs = seqs_after_attn

                seqs_T = torch.transpose(seqs, 0, 1)

                if torch.isnan(seqs_T).any():
                    print(f"DEBUG: NaN in seqs_T before FFN at block {i}")
                    import sys

                    sys.exit(1)

                ffn_input = self.forward_layernorms[i](seqs_T)
                if torch.isnan(ffn_input).any():
                    print(f"DEBUG: NaN in ffn_input (layernorm output) at block {i}")
                    import sys

                    sys.exit(1)

                ffn_output = self.forward_layers[i](ffn_input)

                if torch.isnan(ffn_output).any():
                    print(f"DEBUG: NaN in ffn_output at block {i}")
                    import sys

                    sys.exit(1)

                seqs_T = seqs_T + ffn_output
                seqs_after_ffn = torch.transpose(seqs_T, 0, 1)

                if self.use_mhc:
                    seqs = self.mhc_ffn[i](seqs, seqs_after_ffn)
                    if torch.isnan(seqs).any():
                        print(f"DEBUG: NaN after mhc_ffn[{i}]")
                        import sys

                        sys.exit(1)
                else:
                    seqs = seqs_after_ffn
            else:
                seqs_T = torch.transpose(seqs, 0, 1)

                if torch.isnan(seqs_T).any():
                    print(
                        f"DEBUG: NaN in seqs_T before attention at block {i} (else branch)"
                    )
                    import sys

                    sys.exit(1)

                attention_mask = ~torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev)
                )

                attn_layer = self.attention_layers[i]
                mha_outputs, attn_weights = attn_layer(
                    seqs_T, seqs_T, seqs_T, attn_mask=attention_mask
                )

                if torch.isnan(mha_outputs).any():
                    print(f"DEBUG: NaN in mha_outputs at block {i} (else branch)")
                    print(f"  seqs_T has NaN: {torch.isnan(seqs_T).any().item()}")
                    print(
                        f"  attn_weights has NaN: {torch.isnan(attn_weights).any().item()}"
                    )
                    import sys

                    sys.exit(1)

                seqs_T = self.attention_layernorms[i](seqs_T + mha_outputs)
                seqs_after_attn = torch.transpose(seqs_T, 0, 1)

                if torch.isnan(seqs_after_attn).any():
                    print(f"DEBUG: NaN in seqs_after_attn at block {i} (else branch)")
                    import sys

                    sys.exit(1)

                if self.use_mhc:
                    seqs = self.mhc_attn[i](seqs, seqs_after_attn)
                    if torch.isnan(seqs).any():
                        print(f"DEBUG: NaN after mhc_attn[{i}] (else branch)")
                        import sys

                        sys.exit(1)
                else:
                    seqs = seqs_after_attn

                seqs_T = torch.transpose(seqs, 0, 1)
                ffn_output = self.forward_layers[i](seqs_T)
                seqs_T = self.forward_layernorms[i](seqs_T + ffn_output)
                seqs_after_ffn = torch.transpose(seqs_T, 0, 1)

                if self.use_mhc:
                    seqs = self.mhc_ffn[i](seqs, seqs_after_ffn)
                    if torch.isnan(seqs).any():
                        print(f"DEBUG: NaN after mhc_ffn[{i}] (else branch)")
                        import sys

                        sys.exit(1)
                else:
                    seqs = seqs_after_ffn

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        if torch.isnan(log_feats).any():
            print(f"DEBUG: NaN detected in log_feats")
            print(f"  NaN count: {torch.isnan(log_feats).sum().item()}")
            import sys

            sys.exit(1)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
