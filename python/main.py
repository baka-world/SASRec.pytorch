"""
SASRec Benchmark: Unified Training Script for All Model Variants

支持以下模式：
1. SASRec (基准)
2. SASRec + mHC
3. TiSASRec (use_time)
4. TiSASRec + mHC

使用方法：
    # SASRec基准
    python main_benchmark.py --dataset=ml-1m --train_dir=sasrec_base

    # SASRec + mHC
    python main_benchmark.py --dataset=ml-1m --train_dir=sasrec_mhc --use_mhc=True

    # TiSASRec
    python main_benchmark.py --dataset=ml-1m --train_dir=tisasrec --use_time

    # TiSASRec + mHC
    python main_benchmark.py --dataset=ml-1m --train_dir=tisasrec_mhc --use_time --use_mhc=True
"""

import os
import sys
import time
import torch
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SASRec, TiSASRec
from model_mhc import SASRec as SASRec_mHC, TiSASRec as TiSASRec_mHC
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser(description="SASRec Benchmark Training")

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

# 模型选择参数
parser.add_argument(
    "--use_time",
    action="store_true",
    default=False,
    help="启用TiSASRec（使用时序感知机制）",
)
parser.add_argument(
    "--use_mhc", action="store_true", default=False, help="启用mHC（流形约束超连接）"
)

# mHC参数
parser.add_argument("--mhc_expansion_rate", default=4, type=int, help="mHC扩展因子n")
parser.add_argument(
    "--mhc_init_gate", default=0.01, type=float, help="mHC门控因子α初始值"
)
parser.add_argument(
    "--mhc_sinkhorn_iter", default=20, type=int, help="mHC Sinkhorn-Knopp迭代次数"
)

# TiSASRec参数
parser.add_argument("--time_span", default=100, type=int, help="时间间分离散化范围")
parser.add_argument(
    "--time_unit",
    default="hour",
    type=str,
    choices=["second", "minute", "hour", "day"],
    help="时间单位",
)

args = parser.parse_args()


def get_model(usernum, itemnum, time_span):
    """根据参数选择合适的模型"""
    if args.use_time:
        if args.use_mhc:
            print(f"==> 使用 TiSASRec + mHC (expansion_rate={args.mhc_expansion_rate})")
            return TiSASRec_mHC(usernum, itemnum, time_span, args)
        else:
            print("==> 使用 TiSASRec")
            return TiSASRec(usernum, itemnum, time_span, args)
    else:
        if args.use_mhc:
            print(f"==> 使用 SASRec + mHC (expansion_rate={args.mhc_expansion_rate})")
            return SASRec_mHC(usernum, itemnum, args)
        else:
            print("==> 使用 SASRec (基准)")
            return SASRec(usernum, itemnum, args)


def get_sampler(user_train, usernum, itemnum, time_span):
    """根据参数选择合适的采样器"""
    if args.use_time:
        user_timestamps = get_user_timestamps(args.dataset)
        return WarpSamplerWithTime(
            user_train,
            user_timestamps,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            time_span=time_span,
            n_workers=3,
            unit=args.time_unit,
        )
    else:
        return WarpSampler(
            user_train,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=3,
        )


def get_user_timestamps(dataset_name):
    """加载用户时间戳数据"""
    data = np.loadtxt("data/%s.txt" % dataset_name, dtype=np.int32)
    user_train = data_partition(dataset_name)[0]
    timestamps = {}
    for u in user_train:
        user_items = data[data[:, 0] == u]
        timestamps[u] = (
            user_items[:, 2].tolist()
            if len(user_items[0]) > 2
            else [0] * len(user_items)
        )
    return timestamps


if __name__ == "__main__":
    if not os.path.isdir(args.dataset + "_" + args.train_dir):
        os.makedirs(args.dataset + "_" + args.train_dir)

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

    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = (len(user_train) - 1) // args.batch_size + 1

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    time_span = args.time_span if args.use_time else 0
    sampler = get_sampler(user_train, usernum, itemnum, time_span)

    model = get_model(usernum, itemnum, time_span).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    if hasattr(model, "pos_emb"):
        model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:
            print(
                "failed loading state_dicts, pls check file path:", args.state_dict_path
            )
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        if args.use_time:
            t_test = evaluate_tisasrec(model, dataset, args)
        else:
            t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        for step in range(num_batch):
            batch_data = sampler.next_batch()

            if args.use_time:
                u, seq, pos, neg, time_mat = batch_data
                u, seq, pos, neg, time_mat = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                    np.array(time_mat),
                )
                pos_logits, neg_logits = model(u, seq, time_mat, pos, neg)
            else:
                u, seq, pos, neg = batch_data
                u, seq, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )
                pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels, neg_labels = (
                torch.ones(pos_logits.shape, device=args.device),
                torch.zeros(neg_logits.shape, device=args.device),
            )

            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param**2)

            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

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
                fname = "model.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.time={}.mhc={}.pth"
                fname = fname.format(
                    epoch,
                    args.lr,
                    args.num_blocks,
                    args.num_heads,
                    args.hidden_units,
                    args.maxlen,
                    args.use_time,
                    args.use_mhc,
                )
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + "_" + args.train_dir
            fname = "model.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.time={}.mhc={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
                args.use_time,
                args.use_mhc,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
