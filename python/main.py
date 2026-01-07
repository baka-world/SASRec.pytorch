import os
import time
import torch
import argparse

from model import SASRec
from utils import *


def str2bool(s):
    """将字符串转换为布尔值，用于命令行参数解析"""
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


# 命令行参数解析器，配置训练的各种超参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="数据集名称，用于加载对应数据文件")
parser.add_argument("--train_dir", required=True, help="训练结果保存的目录名")
parser.add_argument(
    "--batch_size", default=128, type=int, help="每个训练批次的样本数量"
)
parser.add_argument(
    "--lr", default=0.001, type=float, help="学习率，控制参数更新的步长"
)
parser.add_argument(
    "--maxlen", default=200, type=int, help="序列的最大长度，超过此长度的序列会被截断"
)
parser.add_argument(
    "--hidden_units", default=50, type=int, help="隐藏层维度，决定模型内部表示的复杂度"
)
parser.add_argument(
    "--num_blocks", default=2, type=int, help="Transformer编码器块的数量"
)
parser.add_argument(
    "--num_epochs",
    default=1000,
    type=int,
    help="训练轮数，一个epoch表示遍历整个训练集一次",
)
parser.add_argument(
    "--num_heads", default=1, type=int, help="多头注意力机制中注意力头的数量"
)
parser.add_argument(
    "--dropout_rate", default=0.2, type=float, help="Dropout比率，用于防止过拟合"
)
parser.add_argument("--l2_emb", default=0.0, type=float, help="嵌入层的L2正则化系数")
parser.add_argument(
    "--device", default="cuda", type=str, help="训练设备，cuda表示GPU，cpu表示CPU"
)
parser.add_argument(
    "--inference_only", default=False, type=str2bool, help="是否仅进行推理（不训练）"
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

args = parser.parse_args()

# 创建训练输出目录，格式为: 数据集名_训练目录名
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)

# 保存本次运行的超参数配置到文件，方便后续复现实验
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

if __name__ == "__main__":
    # 构建用户-物品交互索引，用于快速查找
    # u2i_index: 用户 -> 交互过的物品列表
    # i2u_index: 物品 -> 交互过的用户列表
    u2i_index, i2u_index = build_index(args.dataset)

    # 全局数据集划分：训练集、验证集、测试集、用户数、物品数
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # 计算训练批次数：向上取整，确保所有样本都被处理
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    # 统计平均序列长度
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    # 打开日志文件，记录每个epoch的评估指标
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    # 创建采样器，用于生成训练批次的正负样本
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )

    # 初始化SASRec模型并移动到指定设备
    model = SASRec(usernum, itemnum, args).to(args.device)

    # 使用Xavier均匀分布初始化模型参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # 忽略初始化失败的层（如某些特殊的参数）

    # 将位置嵌入和物品嵌入的padding位置设为0
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # 切换模型到训练模式
    model.train()

    # 从检查点恢复训练时使用的起始epoch
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:
            print("failed loading state_dicts, pls check file path: ", end="")
            print(args.state_dict_path)
            print(
                "pdb enabled for your quick check, pls type exit() if you do not need it"
            )
            import pdb

            pdb.set_trace()

    # 仅推理模式：评估模型在测试集上的表现
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    # 使用BCEWithLogitsLoss作为损失函数，结合了Sigmoid和BCE
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    # 使用Adam优化器更新模型参数
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 记录最佳评估指标
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0

    # 时间统计
    T = 0.0
    t0 = time.time()

    # 主训练循环
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break
        # 遍历每个batch进行训练
        for step in range(num_batch):
            # 从采样器获取一个批次的训练数据
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            # 前向传播：计算正样本和负样本的logits
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = (
                torch.ones(pos_logits.shape, device=args.device),
                torch.zeros(neg_logits.shape, device=args.device),
            )

            # 清空梯度，准备反向传播
            adam_optimizer.zero_grad()

            # 过滤掉padding位置（pos != 0），只计算有效位置的损失
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 添加L2正则化项，防止过拟合
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param**2)

            # 反向传播更新参数
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        # 每20个epoch评估一次模型性能
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            # 保存表现最好的模型权重
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
                fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
                fname = fname.format(
                    epoch,
                    args.lr,
                    args.num_blocks,
                    args.num_heads,
                    args.hidden_units,
                    args.maxlen,
                )
                torch.save(model.state_dict(), os.path.join(folder, fname))

            # 记录评估结果到日志文件
            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        # 训练结束时保存最终模型
        if epoch == args.num_epochs:
            folder = args.dataset + "_" + args.train_dir
            fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
