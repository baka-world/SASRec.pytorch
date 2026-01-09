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

    # 多卡训练 (4卡)
    python -m torch.distributed.launch --nproc_per_node=4 main.py --dataset=ml-1m --train_dir=tisasrec_dist --multi_gpu

    # 多卡训练 (指定GPU数量)
    python -m torch.distributed.launch --nproc_per_node=2 main.py --dataset=ml-1m --train_dir=tisasrec_dist --multi_gpu
"""

import os
import sys
import time
import math
import gc
import signal
import torch
import torch.distributed as dist
import torch.nn as nn
import argparse
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SASRec, TiSASRec
from model_mhc import SASRec as SASRec_mHC, TiSASRec as TiSASRec_mHC
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def get_lr(step, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    else:
        if hasattr(args, 'lr_decay_epoch') and args.lr_decay_epoch > 0:
            epoch = step // args.steps_per_epoch
            decay_epochs = max(0, epoch - args.warmup_steps // args.steps_per_epoch)
            decay_factor = args.lr_decay_rate ** (decay_epochs / args.lr_decay_epoch)
            return args.lr * decay_factor
        else:
            decay_steps = step - args.warmup_steps
            decay_factor = args.lr_decay_rate ** (decay_steps / args.lr_decay_step)
            return args.lr * decay_factor


def is_main_process():
    return not hasattr(args, "local_rank") or args.local_rank == 0


def setup_distributed(args):
    if args.multi_gpu:
        if "SLURM_PROCID" in os.environ:
            args.local_rank = int(os.environ["SLURM_PROCID"])
            args.world_size = int(os.environ["SLURM_NTASKS"])
            args.rank = int(os.environ["SLURM_PROCID"])
        else:
            if not hasattr(args, "local_rank"):
                args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            args.rank = int(os.environ.get("RANK", 0))
            args.world_size = int(
                os.environ.get("WORLD_SIZE", torch.cuda.device_count())
            )

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )

        print(f"[GPU {args.rank}] 分布式训练初始化完成 | 世界大小: {args.world_size}")

        args.batch_size = args.batch_size // args.world_size
        args.num_workers = max(1, (args.num_workers // args.world_size))


def cleanup_distributed():
    if args.multi_gpu:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(description="SASRec Benchmark Training")

parser.add_argument("--dataset", required=True, help="数据集名称")
parser.add_argument("--train_dir", required=True, help="训练结果保存的目录名")
parser.add_argument(
    "--batch_size", default=128, type=int, help="每个训练批次的样本数量"
)
parser.add_argument("--lr", default=0.001, type=float, help="初始学习率")
parser.add_argument(
    "--lr_decay_step", default=1000, type=int, help="学习率衰减步长（按step）"
)
parser.add_argument(
    "--lr_decay_epoch", default=20, type=int, help="学习率衰减epoch间隔"
)
parser.add_argument("--lr_decay_rate", default=0.98, type=float, help="学习率衰减率")
parser.add_argument(
    "--warmup_steps", default=200, type=int, help="Warmup步数（0表示不使用warmup）"
)
parser.add_argument("--maxlen", default=200, type=int, help="序列的最大长度")
parser.add_argument("--hidden_units", default=50, type=int, help="隐藏层维度")
parser.add_argument(
    "--num_blocks", default=2, type=int, help="Transformer编码器块的数量"
)
parser.add_argument("--num_epochs", default=300, type=int, help="训练轮数")
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
parser.add_argument("--num_workers", default=3, type=int, help="数据加载的线程数")

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
    help="禁用mHC模块的AMP计算",
)

parser.add_argument(
    "--use_amp",
    action="store_true",
    default=False,
    help="启用自动混合精度训练（节省显存）",
)

parser.add_argument("--time_span", default=100, type=int, help="时间间分离散化范围")
parser.add_argument(
    "--time_unit",
    default="hour",
    type=str,
    choices=["second", "minute", "hour", "day"],
    help="时间单位",
)

parser.add_argument(
    "--multi_gpu",
    action="store_true",
    default=False,
    help="启用多卡分布式训练",
)
parser.add_argument(
    "--backend",
    type=str,
    default="nccl",
    choices=["nccl", "gloo"],
    help="分布式训练通信后端",
)
parser.add_argument(
    "--master_port",
    type=int,
    default=29500,
    help="主节点端口号",
)

args = parser.parse_args()


def get_model(usernum, itemnum, time_span):
    if args.use_time or not args.no_time:
        if not args.no_mhc:
            return TiSASRec_mHC(usernum, itemnum, time_span, args)
        else:
            return TiSASRec(usernum, itemnum, time_span, args)
    else:
        if not args.no_mhc:
            return SASRec_mHC(usernum, itemnum, args)
        else:
            return SASRec(usernum, itemnum, args)


def get_sampler(user_train, usernum, itemnum, time_span):
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
            n_workers=args.num_workers,
            unit=args.time_unit,
        )
    else:
        return WarpSampler(
            user_train,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=args.num_workers,
        )


def get_user_timestamps(dataset_name, user_train=None):
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

    setup_distributed(args)

    sampler = None

    def cleanup(signum, frame):
        if is_main_process():
            print("\n清理中...")
        if sampler is not None:
            try:
                sampler.close()
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_distributed()
        if is_main_process():
            print("完成，退出。")
        exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    output_dir = f"{args.dataset}_{args.train_dir}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if is_main_process():
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
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
    args.steps_per_epoch = num_batch

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    if is_main_process():
        print("average sequence length: %.2f" % (cc / len(user_train)))

    if is_main_process():
        f = open(os.path.join(output_dir, "log.txt"), "w")
        f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    time_span = args.time_span if args.use_time or not args.no_time else 0
    sampler = get_sampler(user_train, usernum, itemnum, time_span)

    device = torch.device(args.device)
    if args.multi_gpu:
        device = torch.device(args.local_rank)
    model = get_model(usernum, itemnum, time_span).to(device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    if hasattr(model, "pos_emb"):
        model.pos_emb.weight.data[0, :] = 0
    if hasattr(model, "item_emb"):
        model.item_emb.weight.data[0, :] = 0

    model.train()

    if args.multi_gpu:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=device))
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:
            if is_main_process():
                print(
                    "failed loading state_dicts, pls check file path:",
                    args.state_dict_path,
                )

    if args.inference_only:
        model.eval()
        if args.use_time or not args.no_time:
            t_test = evaluate_tisasrec(model, dataset, args)
        else:
            t_test = evaluate(model, dataset, args)
        if is_main_process():
            print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = amp.GradScaler("cuda") if use_amp else None

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    T = 0.0
    t0 = time.time()

    total_step = 0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

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
                with amp.autocast(enabled=use_amp):
                    pos_logits, neg_logits = model(u, seq, time_mat, pos, neg)
            else:
                u, seq, pos, neg = batch_data
                u, seq, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )
                with amp.autocast(enabled=use_amp):
                    pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels, neg_labels = (
                torch.ones(pos_logits.shape, device=device),
                torch.zeros(neg_logits.shape, device=device),
            )

            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in (
                model.module.item_emb.parameters()
                if hasattr(model, "module")
                else model.item_emb.parameters()
            ):
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

            epoch_loss_sum += loss.item()
            epoch_loss_count += 1

        if is_main_process():
            avg_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0
            print(f"Epoch {epoch}: Loss={avg_loss:.4f} LR={current_lr:.6f}")

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1

            if args.use_time or not args.no_time:
                t_test = evaluate_tisasrec(model, dataset, args)
                t_valid = evaluate_valid_tisasrec(model, dataset, args)
            else:
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)

            if is_main_process():
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
                folder = output_dir
                fname = "model.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.time={}.mhc={}.pth"
                fname = fname.format(
                    epoch,
                    args.lr,
                    args.num_blocks,
                    args.num_heads,
                    args.hidden_units,
                    args.maxlen,
                    args.use_time or not args.no_time,
                    not args.no_mhc,
                )
                torch.save(model.state_dict(), os.path.join(folder, fname))

            if is_main_process():
                f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
                f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = output_dir
            fname = "model.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.time={}.mhc={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
                args.use_time or not args.no_time,
                not args.no_mhc,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))

    if is_main_process():
        f.close()
    sampler.close()
    cleanup_distributed()
    if is_main_process():
        print("Done")
