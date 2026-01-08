#!/usr/bin/env python
"""
SASRec Distributed Training Script

使用方法：
    # 4卡训练
    python -m torch.distributed.launch --nproc_per_node=4 main_distributed.py --dataset=ml-1m --train_dir=tisasrec_dist

    # 2卡训练
    python -m torch.distributed.launch --nproc_per_node=2 main_distributed.py --dataset=ml-1m --train_dir=tisasrec_dist

    # 自定义GPU数量和参数
    python -m torch.distributed.launch --nproc_per_node=4 main_distributed.py \
        --dataset=ml-1m \
        --train_dir=tisasrec_dist \
        --batch_size=256 \
        --num_epochs=200 \
        --lr=0.001
"""

import os
import sys
import time
import math
import gc
import signal
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
    if s.lower() in {"false", "true"}:
        return s.lower() == "true"
    raise ValueError("Not a valid boolean string")


def get_lr(step, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
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
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            args.rank = int(os.environ.get("RANK", 0))

            env_world_size = os.environ.get("WORLD_SIZE")
            if env_world_size is not None:
                args.world_size = int(env_world_size)
                gpu_count = torch.cuda.device_count()
                if args.world_size > gpu_count * 10:
                    if is_main_process():
                        print(
                            f"[Warning] WORLD_SIZE={args.world_size} 异常，使用GPU数量={gpu_count}"
                        )
                    args.world_size = gpu_count
            else:
                args.world_size = torch.cuda.device_count()

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )

        if is_main_process():
            print(
                f"分布式训练初始化完成 | GPU数量: {args.world_size} | 主节点端口: {args.master_port}"
            )

        if is_main_process():
            print(
                f"[Debug] Before: batch_size={args.batch_size}, world_size={args.world_size}"
            )

        args.batch_size = max(1, args.batch_size // args.world_size)
        args.num_workers = max(1, args.num_workers // args.world_size)

        if is_main_process():
            print(
                f"[Debug] After: batch_size={args.batch_size}, num_workers={args.num_workers}"
            )


def cleanup_distributed():
    if args.multi_gpu:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(description="SASRec Distributed Training")

parser.add_argument("--dataset", required=True, help="数据集名称")
parser.add_argument("--train_dir", required=True, help="训练结果保存的目录名")
parser.add_argument("--batch_size", default=128, type=int, help="总batch大小")
parser.add_argument("--lr", default=0.001, type=float, help="学习率")
parser.add_argument("--lr_decay_step", default=1000, type=int, help="学习率衰减步长")
parser.add_argument("--lr_decay_rate", default=0.95, type=float, help="学习率衰减率")
parser.add_argument("--warmup_steps", default=100, type=int, help="Warmup步数")
parser.add_argument("--maxlen", default=200, type=int, help="序列的最大长度")
parser.add_argument("--hidden_units", default=50, type=int, help="隐藏层维度")
parser.add_argument("--num_blocks", default=2, type=int, help="Transformer编码器块数量")
parser.add_argument("--num_epochs", default=1000, type=int, help="训练轮数")
parser.add_argument("--num_heads", default=2, type=int, help="多头注意力头数")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="Dropout比率")
parser.add_argument("--l2_emb", default=0.0, type=float, help="嵌入层L2正则化")
parser.add_argument("--inference_only", default=False, type=str2bool, help="仅推理")
parser.add_argument("--state_dict_path", default=None, type=str, help="预训练模型路径")
parser.add_argument(
    "--norm_first", action="store_true", default=False, help="Pre-LN结构"
)
parser.add_argument("--num_workers", default=3, type=int, help="数据加载线程数")
parser.add_argument(
    "--no_time", action="store_true", default=False, help="禁用时序感知"
)
parser.add_argument(
    "--use_time", action="store_true", default=False, help="启用时序感知"
)
parser.add_argument("--no_mhc", action="store_true", default=False, help="禁用mHC模块")
parser.add_argument("--mhc_expansion_rate", default=4, type=int, help="mHC扩展因子")
parser.add_argument("--mhc_init_gate", default=0.01, type=float, help="mHC门控因子")
parser.add_argument(
    "--mhc_sinkhorn_iter", default=20, type=int, help="Sinkhorn迭代次数"
)
parser.add_argument(
    "--mhc_no_amp", action="store_true", default=False, help="mHC禁用AMP"
)
parser.add_argument("--use_amp", action="store_true", default=True, help="启用AMP")
parser.add_argument(
    "--gradient_accumulation_steps", default=1, type=int, help="梯度累积步数"
)
parser.add_argument("--time_span", default=100, type=int, help="时间间分离散化范围")
parser.add_argument(
    "--time_unit", default="hour", type=str, choices=["second", "minute", "hour", "day"]
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Local rank for distributed training"
)
parser.add_argument(
    "--multi_gpu", action="store_true", default=False, help="启用多卡训练"
)
parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
parser.add_argument("--master_port", type=int, default=29500, help="主节点端口")

args = parser.parse_args()


def get_model(usernum, itemnum, time_span):
    use_time_model = args.use_time and not args.no_time
    if use_time_model:
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
    use_time_model = args.use_time and not args.no_time
    if use_time_model:
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
    log_file = None

    def cleanup(signum, frame):
        cleanup_distributed()
        if is_main_process():
            print("\n清理中...")
        if sampler is not None:
            try:
                sampler.close()
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_process() and log_file is not None:
            log_file.close()
        if is_main_process():
            print("完成，退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    output_dir = f"{args.dataset}_{args.train_dir}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if is_main_process():
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            f.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    if is_main_process():
        print(
            f"[Debug] Total batch_size: {args.batch_size}, num_batch per epoch: {num_batch}"
        )

    cc = sum(len(user_train[u]) for u in user_train)
    if is_main_process():
        print(
            f"[Debug] batch_size={args.batch_size}, world_size={args.world_size}, num_batch={num_batch}"
        )
    if is_main_process():
        print(f"Average sequence length: {cc / len(user_train):.2f}")

    if is_main_process():
        log_file = open(os.path.join(output_dir, "log.txt"), "w")
        log_file.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    time_span = args.time_span if args.use_time and not args.no_time else 0
    sampler = get_sampler(user_train, usernum, itemnum, time_span)

    args.device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model = get_model(usernum, itemnum, time_span).to(args.local_rank)

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
            broadcast_buffers=False,  # 禁用以提升性能
        )

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            state_dict = torch.load(
                args.state_dict_path, map_location=torch.device(args.local_rank)
            )
            model.load_state_dict(state_dict)
            if is_main_process():
                print(f"成功加载预训练模型: {args.state_dict_path}")
        except Exception as e:
            if is_main_process():
                print(f"加载预训练模型失败: {e}")

    if args.inference_only:
        model.eval()
        use_time_model = args.use_time and not args.no_time
        if use_time_model:
            t_test = evaluate_tisasrec(model, dataset, args)
        else:
            t_test = evaluate(model, dataset, args)
        if is_main_process():
            print(f"测试结果 (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})")

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    use_amp = args.use_amp

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    T = 0.0
    t0 = time.time()
    total_step = 0

    # 测试前向传播是否正常
    if is_main_process():
        print("[Debug] Testing forward pass...")
    torch.distributed.barrier()
    try:
        test_u = torch.LongTensor([1]).to(args.local_rank)
        test_seq = torch.zeros((1, args.maxlen), dtype=torch.long).to(args.local_rank)
        test_pos = torch.zeros((1, args.maxlen), dtype=torch.long).to(args.local_rank)
        test_neg = torch.zeros((1, args.maxlen), dtype=torch.long).to(args.local_rank)
        use_time_model = args.use_time and not args.no_time
        if use_time_model:
            test_time_mat = torch.zeros(
                (1, args.maxlen, args.maxlen), dtype=torch.long
            ).to(args.local_rank)
            with torch.no_grad():
                _, _ = model(test_u, test_seq, test_time_mat, test_pos, test_neg)
        else:
            with torch.no_grad():
                _, _ = model(test_u, test_seq, test_pos, test_neg)
        torch.distributed.barrier()
        if is_main_process():
            print("[Debug] Forward pass test successful!")
    except Exception as e:
        if is_main_process():
            print(f"[Debug] Forward pass test failed: {e}")
        raise

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break

        accumulated_loss = None
        accumulation_step = 0

        for step in range(num_batch):
            batch_data = sampler.next_batch()

            use_time_model = args.use_time and not args.no_time
            if use_time_model:
                u, seq, pos, neg, time_mat = batch_data
                u = torch.LongTensor(np.array(u)).to(args.local_rank)
                seq = torch.LongTensor(np.array(seq)).to(args.local_rank)
                pos = torch.LongTensor(np.array(pos)).to(args.local_rank)
                neg = torch.LongTensor(np.array(neg)).to(args.local_rank)
                time_mat = torch.LongTensor(np.array(time_mat)).to(args.local_rank)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pos_logits, neg_logits = model(u, seq, time_mat, pos, neg)
            else:
                u, seq, pos, neg = batch_data
                u = torch.LongTensor(np.array(u)).to(args.local_rank)
                seq = torch.LongTensor(np.array(seq)).to(args.local_rank)
                pos = torch.LongTensor(np.array(pos)).to(args.local_rank)
                neg = torch.LongTensor(np.array(neg)).to(args.local_rank)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels = torch.ones(pos_logits.shape, device=args.local_rank)
            neg_labels = torch.zeros(neg_logits.shape, device=args.local_rank)

            indices = torch.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.module.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param**2)

            loss = loss / args.gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss = (
                loss.item() * args.gradient_accumulation_steps
                if accumulated_loss is None
                else accumulated_loss + loss.item() * args.gradient_accumulation_steps
            )
            accumulation_step += 1

            if accumulation_step % args.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(adam_optimizer)
                    scaler.update()
                else:
                    adam_optimizer.step()

                adam_optimizer.zero_grad()
                total_step += 1

                current_lr = get_lr(total_step, args)
                for param_group in adam_optimizer.param_groups:
                    param_group["lr"] = current_lr

                if is_main_process():
                    print(
                        f"Epoch {epoch} Step {step}: Loss={accumulated_loss / args.gradient_accumulation_steps:.4f} LR={current_lr:.6f}"
                    )

                accumulated_loss = None

        if accumulation_step % args.gradient_accumulation_steps != 0:
            if use_amp:
                scaler.step(adam_optimizer)
                scaler.update()
            else:
                adam_optimizer.step()
            adam_optimizer.zero_grad()
            total_step += 1

        if epoch % 20 == 0 or epoch == args.num_epochs:
            model.eval()
            t1 = time.time() - t0
            T += t1

            use_time_model = args.use_time and not args.no_time
            if use_time_model:
                t_test = evaluate_tisasrec(model, dataset, args)
                t_valid = evaluate_valid_tisasrec(model, dataset, args)
            else:
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)

            if is_main_process():
                print(
                    f"Epoch {epoch}: Time={T:.1f}s, Valid(NDCG={t_valid[0]:.4f}, HR={t_valid[1]:.4f}), "
                    f"Test(NDCG={t_test[0]:.4f}, HR={t_test[1]:.4f})"
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

                if is_main_process():
                    fname = f"model.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.pth"
                    torch.save(model.state_dict(), os.path.join(output_dir, fname))

            if is_main_process():
                log_file.write(f"{epoch} {t_valid} {t_test}\n")
                log_file.flush()
            t0 = time.time()
            model.train()

    if is_main_process():
        log_file.close()
    sampler.close()
    cleanup_distributed()

    if is_main_process():
        print("训练完成!")
        print(f"最佳验证指标: NDCG={best_val_ndcg:.4f}, HR={best_val_hr:.4f}")
        print(f"最佳测试指标: NDCG={best_test_ndcg:.4f}, HR={best_test_hr:.4f}")
