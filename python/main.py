"""
SASRec Benchmark: Unified Training Script for All Model Variants

支持以下模式：
1. SASRec (基准)
2. SASRec + mHC
3. TiSASRec (默认，使用时序感知机制)
4. TiSASRec + mHC

使用方法：
    # SASRec基准
    python main.py --dataset=ml-1m --train_dir=sasrec_base --no_time --no_mhc

    # SASRec + mHC
    python main.py --dataset=ml-1m --train_dir=sasrec_mhc --no_time

    # TiSASRec (默认)
    python main.py --dataset=ml-1m --train_dir=tisasrec

    # TiSASRec + mHC
    python main.py --dataset=ml-1m --train_dir=tisasrec_mhc
"""

import os
import sys
import time
import math
import gc
import signal
import torch
import argparse
from torch.cuda import amp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SASRec, TiSASRec
from model_mhc import SASRec as SASRec_mHC, TiSASRec as TiSASRec_mHC
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def get_lr(step, args):
    """计算当前学习率，支持warmup和衰减"""
    if args.warmup_steps > 0 and step < args.warmup_steps:
        # Warmup阶段：线性增加学习率
        return args.lr * (step + 1) / args.warmup_steps
    else:
        # 衰减阶段：按步长衰减
        decay_steps = step - args.warmup_steps
        decay_factor = args.lr_decay_rate ** (decay_steps / args.lr_decay_step)
        return args.lr * decay_factor


parser = argparse.ArgumentParser(description="SASRec Benchmark Training")

# 基础参数
parser.add_argument("--dataset", required=True, help="数据集名称")
parser.add_argument("--train_dir", required=True, help="训练结果保存的目录名")
parser.add_argument(
    "--batch_size", default=128, type=int, help="每个训练批次的样本数量"
)
parser.add_argument("--lr", default=0.001, type=float, help="学习率")
parser.add_argument(
    "--lr_decay_step", default=1000, type=int, help="学习率衰减步长（按epoch）"
)
parser.add_argument("--lr_decay_rate", default=0.95, type=float, help="学习率衰减率")
parser.add_argument(
    "--warmup_steps", default=100, type=int, help="Warmup步数（0表示不使用warmup）"
)
parser.add_argument("--maxlen", default=200, type=int, help="序列的最大长度")
parser.add_argument("--hidden_units", default=256, type=int, help="隐藏层维度")
parser.add_argument(
    "--num_blocks", default=3, type=int, help="Transformer编码器块的数量"
)
parser.add_argument("--num_epochs", default=1000, type=int, help="训练轮数")
parser.add_argument(
    "--num_heads", default=2, type=int, help="多头注意力机制中注意力头的数量"
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
    "--no_time",
    action="store_true",
    default=False,
    help="禁用TiSASRec时序感知机制，使用标准SASRec",
)
parser.add_argument(
    "--use_time",
    action="store_true",
    default=False,
    help="启用TiSASRec时序感知机制（默认行为，可忽略）",
)
parser.add_argument(
    "--no_mhc",
    action="store_true",
    default=False,
    help="禁用mHC模块，使用标准模型",
)

# mHC参数
parser.add_argument("--mhc_expansion_rate", default=4, type=int, help="mHC扩展因子n")
parser.add_argument(
    "--mhc_init_gate", default=0.01, type=float, help="mHC门控因子α初始值"
)
parser.add_argument(
    "--mhc_sinkhorn_iter", default=20, type=int, help="mHC Sinkhorn-Knopp迭代次数"
)

parser.add_argument(
    "--mhc_no_amp",
    action="store_true",
    default=False,
    help="禁用mHC模块的AMP计算（解决NaN问题）。当启用--use_mhc时，"
    "mHC模块会使用FP32精度计算，避免自动混合精度导致的数值溢出问题。",
)

# 训练优化参数
parser.add_argument(
    "--use_amp",
    action="store_true",
    default=True,
    help="启用自动混合精度训练（节省显存）",
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
    if args.use_time or not args.no_time:
        if not args.no_mhc:
            print(f"==> 使用 TiSASRec + mHC (expansion_rate={args.mhc_expansion_rate})")
            return TiSASRec_mHC(usernum, itemnum, time_span, args)
        else:
            print("==> 使用 TiSASRec")
            return TiSASRec(usernum, itemnum, time_span, args)
    else:
        if not args.no_mhc:
            print(f"==> 使用 SASRec + mHC (expansion_rate={args.mhc_expansion_rate})")
            return SASRec_mHC(usernum, itemnum, args)
        else:
            print("==> 使用 SASRec (基准)")
            return SASRec(usernum, itemnum, args)


def get_sampler(user_train, usernum, itemnum, time_span):
    """根据参数选择合适的采样器"""
    if args.use_time or not args.no_time:
        user_timestamps = get_user_timestamps(args.dataset, user_train)
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


def get_user_timestamps(dataset_name, user_train=None):
    """加载用户时间戳数据"""
    data = np.loadtxt("data/%s.txt" % dataset_name, dtype=np.int32)
    if user_train is None:
        user_train = data_partition(dataset_name)[0]
    timestamps = {}
    for u in user_train:
        user_items = data[data[:, 0] == u]
        timestamps[u] = (
            user_items[:, 2].tolist()
            if len(user_items) > 0 and user_items.shape[1] > 2
            else [0] * len(user_items)
        )
    return timestamps


if __name__ == "__main__":
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    sampler = None

    def cleanup(signum, frame):
        print("\n清理中...")
        if sampler is not None:
            try:
                sampler.close()
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        print("完成，退出。")
        exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

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

    time_span = args.time_span if args.use_time or not args.no_time else 0
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
        if args.use_time or not args.no_time:
            t_test = evaluate_tisasrec(model, dataset, args)
        else:
            t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    T = 0.0
    t0 = time.time()

    total_step = 0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        for step in range(num_batch):
            batch_data = sampler.next_batch()

            if args.use_time or not args.no_time:
                u, seq, pos, neg, time_mat = batch_data
                u, seq, pos, neg, time_mat = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                    np.array(time_mat),
                )
                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    pos_logits, neg_logits = model(u, seq, time_mat, pos, neg)

                if torch.isnan(pos_logits).any() or torch.isnan(neg_logits).any():
                    print(f"DEBUG: NaN detected in logits at iteration {step}")
                    print(
                        f"  pos_logits has NaN: {torch.isnan(pos_logits).sum().item()}"
                    )
                    print(
                        f"  neg_logits has NaN: {torch.isnan(neg_logits).sum().item()}"
                    )
                    print(
                        f"  log_seqs dtype: {type(seq)}, min: {seq.min()}, max: {seq.max()}"
                    )
                    import sys

                    sys.exit(1)
            else:
                u, seq, pos, neg = batch_data
                u, seq, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )
                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    pos_logits, neg_logits = model(u, seq, pos, neg)

                if torch.isnan(pos_logits).any() or torch.isnan(neg_logits).any():
                    print(f"DEBUG: NaN detected in logits at iteration {step}")
                    print(
                        f"  pos_logits has NaN: {torch.isnan(pos_logits).sum().item()}"
                    )
                    print(
                        f"  neg_logits has NaN: {torch.isnan(neg_logits).sum().item()}"
                    )
                    print(
                        f"  log_seqs dtype: {type(seq)}, min: {seq.min()}, max: {seq.max()}"
                    )
                    import sys

                    sys.exit(1)

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

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(adam_optimizer)
                scaler.update()
            else:
                loss.backward()
                adam_optimizer.step()

            current_lr = get_lr(total_step, args)
            for param_group in adam_optimizer.param_groups:
                param_group["lr"] = current_lr

            total_step += 1
            print(
                "loss in epoch {} iteration {}: {:.4f} lr: {:.6f}".format(
                    epoch, step, loss.item(), current_lr
                )
            )

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")

            if args.use_time or not args.no_time:
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
                    args.use_time or not args.no_time,
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
                args.use_time or not args.no_time,
                args.use_mhc,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
