"""
TiSASRec Training Script

训练时序感知序列推荐模型TiSASRec。

使用方法：
    python main_tisasrec.py --dataset=ml-1m --train_dir=tisasrec_default \
        --use_time --time_span=100 --device=cuda

主要参数：
    --use_time: 启用时序感知机制
    --time_span: 时间间分离散化范围
    --time_unit: 时间单位（second, minute, hour, day）
"""

import os
import sys
import time
import torch
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SASRec, TiSASRec
from utils import *
from dataset_hf import (
    load_hf_ml1m_dataset,
    process_dataset_for_sequential,
    compute_time_matrix,
)


def str2bool(s):
    """将字符串转换为布尔值，用于命令行参数解析"""
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


# 命令行参数解析器
parser = argparse.ArgumentParser(description="TiSASRec Training")

# 基础参数
parser.add_argument("--dataset", required=True, help="数据集名称")
parser.add_argument("--train_dir", required=True, help="训练结果保存的目录名")
parser.add_argument(
    "--batch_size", default=128, type=int, help="每个训练批次的样本数量"
)
parser.add_argument("--lr", default=0.001, type=float, help="学习率")
parser.add_argument("--maxlen", default=200, type=int, help="序列的最大长度")
parser.add_argument("--hidden_units", default=50, type=int, help="隐藏层维度")
parser.add_argument(
    "--num_blocks", default=2, type=int, help="Transformer编码器块的数量"
)
parser.add_argument("--num_epochs", default=1000, type=int, help="训练轮数")
parser.add_argument(
    "--num_heads", default=1, type=int, help="多头注意力机制中注意力头的数量"
)
parser.add_argument("--dropout_rate", default=0.2, type=float, help="Dropout比率")
parser.add_argument("--l2_emb", default=0.0, type=float, help="嵌入层的L2正则化系数")
parser.add_argument("--device", default="cuda", type=str, help="训练设备")
parser.add_argument(
    "--inference_only", default=False, type=str2bool, help="是否仅进行推理"
)
parser.add_argument(
    "--state_dict_path", default=None, type=str, help="预训练模型权重文件的路径"
)
parser.add_argument(
    "--norm_first",
    action="store_true",
    default=False,
    help="是否在每个Block中先进行LayerNorm",
)

# TiSASRec特有参数
parser.add_argument(
    "--use_time",
    action="store_true",
    default=False,
    help="是否使用时序感知机制（TiSASRec），否则使用标准SASRec",
)
parser.add_argument(
    "--time_span",
    default=100,
    type=int,
    help="时间间分离散化范围，将连续时间间隔映射到[1, time_span]",
)
parser.add_argument(
    "--time_unit",
    default="hour",
    type=str,
    choices=["second", "minute", "hour", "day"],
    help="时间单位，用于计算时间间隔",
)
parser.add_argument(
    "--use_hf",
    action="store_true",
    default=False,
    help="是否使用HuggingFace数据集（包含时间信息）",
)

args = parser.parse_args()

# 创建训练输出目录
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)

# 保存超参数配置
with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
    f.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )
f.close()


def prepare_timestamp_dict(user_history):
    """
    从用户历史中提取时间戳字典

    参数:
        user_history: {用户ID: [{'iid': 物品ID, 'timestamp': 时间戳, ...}, ...]}

    返回:
        timestamp_dict: {用户ID: [时间戳列表]}
    """
    timestamp_dict = {}
    for uid, history in user_history.items():
        timestamps = [item.get("timestamp", 0) for item in history]
        timestamp_dict[uid] = timestamps
    return timestamp_dict


if __name__ == "__main__":
    print("=" * 60)
    print(
        f"Training {'TiSASRec' if args.use_time else 'SASRec'} on dataset: {args.dataset}"
    )
    print(f"Use Time Information: {args.use_time}")
    print(f"Time Span: {args.time_span}, Time Unit: {args.time_unit}")
    print("=" * 60)

    if args.use_hf:
        print("Loading dataset from HuggingFace...")
        hf_dataset = load_hf_ml1m_dataset()
        user_history, itemnum, usernum = process_dataset_for_sequential(hf_dataset)

        # 分割训练/验证/测试集
        from dataset_hf import split_train_val_test

        user_train, user_valid, user_test = split_train_val_test(user_history)

        # 提取时间戳字典
        user_timestamps = prepare_timestamp_dict(user_history)

        # 计算时间间隔矩阵
        print(f"Computing time matrices with time_span={args.time_span}...")
        time_matrix_dict = compute_time_matrix(
            user_train, args.maxlen, args.time_span, args.time_unit
        )

        dataset = [user_train, user_valid, user_test, usernum, itemnum]
    else:
        # 使用原始数据格式（不包含时间信息）
        u2i_index, i2u_index = build_index(args.dataset)
        dataset = data_partition(args.dataset)
        user_train, user_valid, user_test, usernum, itemnum = dataset
        user_timestamps = {u: [0] * len(user_train[u]) for u in user_train}
        time_matrix_dict = None

    # 计算训练批次数
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    # 统计平均序列长度
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    # 打开日志文件
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    # 创建采样器
    if args.use_time and time_matrix_dict is not None:
        sampler = WarpSamplerWithTime(
            user_train,
            user_timestamps,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            time_span=args.time_span,
            n_workers=3,
            unit=args.time_unit,
        )
    else:
        sampler = WarpSampler(
            user_train,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=3,
        )

    # 初始化模型
    if args.use_time:
        model = TiSASRec(usernum, itemnum, args.time_span, args).to(args.device)
    else:
        model = SASRec(usernum, itemnum, args).to(args.device)

    # 参数初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # 初始化TiSASRec特有的嵌入层
    if args.use_time:
        # 将时间矩阵嵌入的padding位置设为0
        model.time_matrix_K_emb.weight.data[0, :] = 0
        model.time_matrix_V_emb.weight.data[0, :] = 0

    # 切换到训练模式
    model.train()

    # 从检查点恢复
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except Exception as e:
            print(f"Failed loading state_dicts: {e}")

    # 仅推理模式
    if args.inference_only:
        model.eval()
        if args.use_time:
            t_test = evaluate_tisasrec(model, dataset, args)
        else:
            t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    # 损失函数和优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 最佳指标
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    # 时间统计
    T = 0.0
    t0 = time.time()

    # 主训练循环
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        for step in range(num_batch):
            # 获取批次数据
            batch_data = sampler.next_batch()
            u, seq, pos, neg = (
                batch_data[0],
                batch_data[1],
                batch_data[2],
                batch_data[3],
            )
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            if args.use_time:
                time_mat = np.array(batch_data[4])
                pos_logits, neg_logits = model(u, seq, time_mat, pos, neg)
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            # 清空梯度
            adam_optimizer.zero_grad()

            # 计算损失
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param**2)

            # 反向传播
            loss.backward()
            adam_optimizer.step()

            if step % 100 == 0:
                print(
                    "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
                )

        # 评估
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")

            if args.use_time:
                t_test = evaluate_tisasrec(model, dataset, args)
                t_valid = evaluate_valid_tisasrec(model, dataset, args)
            else:
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)

            print(
                " epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            # 保存最佳模型
            if (
                t_valid[0] > best_val_ndcg
                or t_valid[1] > best_val_hr
                or t_test[0] > best_test_ndcg
                or t_test[1] > best_test_hr
            ):
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)

                folder = args.dataset + "_" + args.train_dir
                model_name = "TiSASRec" if args.use_time else "SASRec"
                fname = f"{model_name}.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth"
                torch.save(model.state_dict(), os.path.join(folder, fname))

            # 记录日志
            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        # 保存最终模型
        if epoch == args.num_epochs:
            folder = args.dataset + "_" + args.train_dir
            model_name = "TiSASRec" if args.use_time else "SASRec"
            fname = f"{model_name}.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth"
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done!")
