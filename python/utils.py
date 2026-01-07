import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def build_index(dataset_name):
    """
    构建用户-物品交互索引

    从原始数据文件加载用户-物品交互记录，构建双向索引。
    这个索引用于快速查找：
    - 某个用户交互过哪些物品
    - 某个物品被哪些用户交互过

    参数:
        dataset_name: 数据集名称，对应 data/目录下的 txt 文件

    返回:
        u2i_index: 列表，u2i_index[u] 返回用户u交互过的物品列表
        i2u_index: 列表，i2u_index[i] 返回交互过物品i的用户列表
    """
    # 加载用户-物品交互矩阵，每行格式: 用户ID 物品ID
    ui_mat = np.loadtxt("data/%s.txt" % dataset_name, dtype=np.int32)

    # 获取用户和物品的最大ID
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    # 初始化双向索引
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    # 填充索引
    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index


def random_neq(l, r, s):
    """
    随机采样一个不在集合s中的整数

    用于负采样：从用户未交互过的物品中随机选择一个

    参数:
        l, r: 采样范围 [l, r)
        s: 需要排除的集合

    返回:
        一个不在s中的随机整数
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(
    user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED
):
    """
    采样函数：生成训练批次数据

    这个函数在独立进程中运行，不断生成训练样本放入队列。

    采样策略（以用户u为例）：
    1. 从用户的历史序列中，构建训练样本
    2. 对于序列中的每个位置i：
       - seq[i]: 位置i的物品
       - pos[i]: seq[i]之后用户实际交互的下一个物品（正样本）
       - neg[i]: 用户从未交互过的随机物品（负样本）

    参数:
        user_train: 训练数据，{用户ID: [物品序列]}
        usernum: 用户总数
        itemnum: 物品总数
        batch_size: 每个批次的样本数
        maxlen: 序列最大长度
        result_queue: 进程间通信队列，用于传递采样结果
        SEED: 随机种子，保证可复现
    """

    def sample(uid):
        # 确保用户有足够的历史记录（至少2个）
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        # 初始化序列
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        # 从后向前构建序列
        nxt = user_train[uid][-1]  # 用户交互的最后一个物品
        idx = maxlen - 1  # 从序列末尾开始填充

        # 已交互物品集合，用于负采样时排除
        ts = set(user_train[uid])

        # 逆序遍历用户历史（从最近到最远）
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i  # 当前物品
            pos[idx] = nxt  # 下一个物品作为正样本
            neg[idx] = random_neq(1, itemnum + 1, ts)  # 随机负样本
            nxt = i  # 更新下一个物品
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        # 打乱用户顺序
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        # 将批次数据放入队列
        result_queue.put(tuple(zip(*one_batch)))


class WarpSampler(object):
    """
    多进程采样器

    使用多个工作进程并行生成训练批次，提高数据加载效率。

    使用方式：
    - 初始化时启动多个进程，每个进程运行sample_function
    - 调用next_batch()从队列获取一个批次的数据
    - 使用完毕后调用close()终止所有进程
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)  # 队列最大容量
        self.processors = []

        # 启动多个采样进程
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(0, 2000000000),  # 随机种子
                    ),
                )
            )
            self.processors[-1].daemon = True  # 设为守护进程，随主进程退出
            self.processors[-1].start()

    def next_batch(self):
        """从队列获取一个批次的训练数据"""
        return self.result_queue.get()

    def close(self):
        """终止所有采样进程"""
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    """
    数据划分

    将原始交互数据划分为训练集、验证集和测试集。

    支持两种格式：
    1. UserID MovieID (无时间戳)
    2. UserID MovieID Timestamp (带时间戳，会按时间排序)

    划分策略：
    - 对于每个用户，将其交互序列按时间顺序排列
    - 前 N-2 个交互作为训练数据
    - 倒数第2个交互作为验证集
    - 倒数第1个交互作为测试集
    - 如果用户交互少于4次，则全部作为训练数据

    参数:
        fname: 数据集名称

    返回:
        [user_train, user_valid, user_test, usernum, itemnum]
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)  # {用户ID: [(timestamp, item_id)] 或 [item_id]}
    user_train = {}
    user_valid = {}
    user_test = {}
    has_timestamp = False

    # 加载数据
    f = open("data/%s.txt" % fname, "r")
    first_line = f.readline()
    f.close()

    parts = first_line.rstrip().split(" ")
    if len(parts) >= 3:
        has_timestamp = True
        print("Detected timestamp format: UserID MovieID Timestamp")

    f = open("data/%s.txt" % fname, "r")
    for line in f:
        parts = line.rstrip().split(" ")
        u = int(parts[0])
        i = int(parts[1])

        if has_timestamp and len(parts) >= 3:
            ts = int(parts[2])
            User[u].append((ts, i))
        else:
            User[u].append(i)

        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
    f.close()

    # 按时间戳排序（如果有的话）
    if has_timestamp:
        for u in User:
            User[u].sort(key=lambda x: x[0])
            # 提取排序后的物品ID
            User[u] = [item for _, item in User[u]]

    # 划分数据
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:
            # 交互太少，全部作为训练数据
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # 正常划分
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


def load_timestamps(fname):
    """
    加载时间戳数据

    返回:
        user_timestamps: {用户ID: [时间戳列表]}
    """
    user_timestamps = defaultdict(list)

    f = open("data/%s.txt" % fname, "r")
    first_line = f.readline()
    f.close()

    parts = first_line.rstrip().split(" ")
    if len(parts) < 3:
        return None

    f = open("data/%s.txt" % fname, "r")
    for line in f:
        parts = line.rstrip().split(" ")
        if len(parts) >= 3:
            u = int(parts[0])
            ts = int(parts[2])
            user_timestamps[u].append(ts)
    f.close()

    # 按时间戳排序
    for u in user_timestamps:
        user_timestamps[u].sort()

    return user_timestamps


def evaluate(model, dataset, args):
    """
    在测试集上评估模型性能

    评估指标：
    - NDCG@10: 归一化折损累积增益，衡量推荐列表排序质量
    - HR@10: 命中率，衡量推荐列表中是否包含真实物品

    评估策略：
    - 对于每个用户，用其历史行为预测下一个物品
    - 从所有物品中采样99个负样本，与真实物品组成100个候选
    - 计算模型对候选物品的得分，排序后计算指标

    参数:
        model: 训练好的SASRec模型
        dataset: [train, valid, test, usernum, itemnum]
        args: 配置参数

    返回:
        (NDCG@10, HR@10)
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # 为了加速，如果用户数超过10000，随机采样10000个用户评估
    if usernum > 10000:
        users = random.sample(list(train.keys()), min(10000, len(train)))
    else:
        users = list(train.keys())

    for u in users:
        # 跳过没有训练数据或测试数据的用户
        if u not in train or len(train[u]) < 1 or u not in test or len(test[u]) < 1:
            continue

        # 构建用户历史序列（仅使用训练集，不包含验证集，避免数据泄露）
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        train_items = train[u]
        if train_items and isinstance(train_items[0], dict):
            for i in range(len(train_items) - 1, -1, -1):
                seq[idx] = train_items[i]["iid"]
                idx -= 1
                if idx == -1:
                    break
        else:
            for i in reversed(train_items):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

        # 构建候选物品列表：1个真实物品 + 99个负样本
        if train_items and isinstance(train_items[0], dict):
            rated = {item["iid"] for item in train_items}
        else:
            rated = set(train_items)
        rated.add(0)  # 排除padding
        # 提取测试集物品ID
        if test[u] and isinstance(test[u][0], dict):
            test_item = test[u][0]["iid"]
        else:
            test_item = test[u][0]
        item_idx = [test_item]  # 真实物品
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # 预测：获取模型对候选物品的得分
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        # 计算排名
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        # 计算NDCG和HR（只考虑前10名）
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    """
    在验证集上评估模型性能

    与evaluate类似，但在验证集上进行评估
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(list(train.keys()), min(10000, len(train)))
    else:
        users = list(train.keys())
    for u in users:
        if u not in train or len(train[u]) < 1 or u not in valid or len(valid[u]) < 1:
            continue

        # 构建用户历史序列（仅使用训练集，不包含验证集）
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        train_items = train[u]
        if train_items and isinstance(train_items[0], dict):
            for i in range(len(train_items) - 1, -1, -1):
                seq[idx] = train_items[i]["iid"]
                idx -= 1
                if idx == -1:
                    break
        else:
            for i in reversed(train_items):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

        if train_items and isinstance(train_items[0], dict):
            rated = {item["iid"] for item in train_items}
        else:
            rated = set(train_items)
        rated.add(0)
        # 提取验证集物品ID
        if valid[u] and isinstance(valid[u][0], dict):
            valid_item = valid[u][0]["iid"]
        else:
            valid_item = valid[u][0]
        item_idx = [valid_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def discretize_time_interval(time_diff, time_span, unit="hour"):
    """
    将时间间隔离散化到指定范围

    参数:
        time_diff: 时间间隔（秒）
        time_span: 最大离散化值
        unit: 时间单位 ('second', 'minute', 'hour', 'day')

    返回:
        离散化后的时间间隔值
    """
    unit_seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
    divisor = unit_seconds.get(unit, 3600)

    # 线性离散化，然后钳制到 [0, time_span]
    # 注意：允许返回 0 用于填充
    discretized = int(time_diff // divisor) + 1
    discretized = max(0, min(time_span, discretized))
    return discretized


def compute_time_matrix_for_user(history, maxlen, time_span, unit="hour"):
    """
    为单个用户计算时间间隔矩阵

    对于序列中每个位置i，计算与之后所有位置j的时间间隔

    参数:
        history: 用户交互历史列表，每个元素是包含'iid'和'timestamp'的字典
        maxlen: 序列最大长度
        time_span: 时间间分离散化范围
        unit: 时间单位

    返回:
        time_matrix: 时间间隔矩阵，形状为 (maxlen, maxlen)
    """
    seq_len = min(len(history), maxlen)
    time_matrix = np.zeros((maxlen, maxlen), dtype=np.int32)

    # 提取时间戳
    timestamps = [item["timestamp"] for item in history[:seq_len]]

    # 计算时间间隔矩阵
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            # 时间间隔 = later_time - earlier_time
            time_diff = timestamps[j] - timestamps[i]
            time_matrix[i, j] = discretize_time_interval(time_diff, time_span, unit)

    return time_matrix


def sample_function_with_time(
    user_train,
    user_timestamps,
    usernum,
    itemnum,
    batch_size,
    maxlen,
    time_span,
    result_queue,
    SEED,
    unit="hour",
):
    """
    采样函数：生成包含时间信息的训练批次数据

    这个函数在独立进程中运行，不断生成训练样本放入队列。

    采样策略（以用户u为例）：
    1. 从用户的历史序列中，构建训练样本
    2. 对于序列中的每个位置i：
       - seq[i]: 位置i的物品
       - pos[i]: seq[i]之后用户实际交互的下一个物品（正样本）
       - neg[i]: 用户从未交互过的随机物品（负样本）
       - time_mat[i]: 与之后位置的时间间隔矩阵
    3. 构建时间间隔矩阵

    参数:
        user_train: 训练数据，{用户ID: [物品列表]}
        user_timestamps: 时间戳数据，{用户ID: [时间戳列表]}
        usernum: 用户总数
        itemnum: 物品总数
        batch_size: 每个批次的样本数
        maxlen: 序列最大长度
        time_span: 时间间分离散化范围
        result_queue: 进程间通信队列，用于传递采样结果
        SEED: 随机种子，保证可复现
        unit: 时间单位
    """

    def sample(uid):
        uid = int(uid)
        # 确保用户有足够的历史记录（至少2个）且用户ID存在于训练数据中
        while uid not in user_train or len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)
            uid = int(uid)

        # 初始化序列
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        time_mat = np.zeros([maxlen, maxlen], dtype=np.int32)

        # 构建完整的历史记录（包含时间戳）
        # 支持两种格式：1) [item_id, ...] 2) [{'iid': item_id, 'timestamp': ...}, ...]
        user_items = user_train[uid]
        if user_items and isinstance(user_items[0], dict):
            # 已经是dict格式，直接使用
            history = [item.copy() for item in user_items]
        else:
            # 转换为dict格式
            history = []
            for i, item_id in enumerate(user_items):
                history.append(
                    {
                        "iid": item_id,
                        "timestamp": user_timestamps[uid][i]
                        if uid in user_timestamps
                        else 0,
                    }
                )

        # 计算时间间隔矩阵
        time_mat = compute_time_matrix_for_user(history, maxlen, time_span, unit)

        # 从后向前构建序列
        nxt = history[-1]["iid"]
        idx = maxlen - 1

        # 已交互物品集合，用于负采样时排除
        # 支持两种格式：1) [item_id, ...] 2) [{'iid': item_id, ...}, ...]
        user_items = user_train[uid]
        if user_items and isinstance(user_items[0], dict):
            ts = {item["iid"] for item in user_items}
        else:
            ts = set(user_items)

        # 逆序遍历用户历史（从最近到最远）
        for i in range(len(history) - 2, -1, -1):
            seq[idx] = history[i]["iid"]
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = history[i]["iid"]
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg, time_mat)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        # 打乱用户顺序
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        # 将批次数据放入队列
        result_queue.put(tuple(zip(*one_batch)))


class WarpSamplerWithTime(object):
    """
    支持时间信息的多进程采样器

    使用多个工作进程并行生成训练批次，提高数据加载效率。
    同时生成时间间隔矩阵用于TiSASRec模型。

    使用方式：
    - 初始化时启动多个进程，每个进程运行sample_function_with_time
    - 调用next_batch()从队列获取一个批次的数据（包含时间矩阵）
    - 使用完毕后调用close()终止所有进程
    """

    def __init__(
        self,
        User,
        user_timestamps,
        usernum,
        itemnum,
        batch_size=64,
        maxlen=10,
        time_span=100,
        n_workers=1,
        unit="hour",
    ):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        # 启动多个采样进程
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function_with_time,
                    args=(
                        User,
                        user_timestamps,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        time_span,
                        self.result_queue,
                        np.random.randint(0, 2000000000),
                        unit,
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        """从队列获取一个批次的训练数据（包含时间矩阵）"""
        return self.result_queue.get()

    def close(self):
        """终止所有采样进程"""
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate_tisasrec(model, dataset, args):
    """
    在测试集上评估TiSASRec模型性能

    评估指标：
    - NDCG@10: 归一化折损累积增益，衡量推荐列表排序质量
    - HR@10: 命中率，衡量推荐列表中是否包含真实物品

    评估策略：
    - 对于每个用户，用其历史行为预测下一个物品
    - 从所有物品中采样99个负样本，与真实物品组成100个候选
    - 计算模型对候选物品的得分，排序后计算指标

    参数:
        model: 训练好的TiSASRec模型
        dataset: [train, valid, test, usernum, itemnum]
        args: 配置参数

    返回:
        (NDCG@10, HR@10)
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # 为每个用户预计算时间间隔矩阵
    time_matrix_dict = {}
    for u in train.keys():
        if len(train[u]) >= 1:
            # 支持两种格式：1) [item_id, ...] 2) [{'iid': item_id, ...}, ...]
            if train[u] and isinstance(train[u][0], dict):
                history = [item.copy() for item in train[u]]
            else:
                history = [{"iid": item, "timestamp": 0} for item in train[u]]
            time_matrix_dict[u] = compute_time_matrix_for_user(
                history, args.maxlen, args.time_span
            )

    if usernum > 10000:
        users = random.sample(list(train.keys()), min(10000, len(train)))
    else:
        users = list(train.keys())

    for u in users:
        if u not in train or len(train[u]) < 1 or u not in test or len(test[u]) < 1:
            continue

        # 构建用户历史序列（仅使用训练集，不包含验证集）
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        train_items = train[u]
        if train_items and isinstance(train_items[0], dict):
            for i in range(len(train_items) - 1, -1, -1):
                seq[idx] = train_items[i]["iid"]
                idx -= 1
                if idx == -1:
                    break
        else:
            for i in reversed(train_items):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

        # 获取时间间隔矩阵
        time_mat = time_matrix_dict.get(
            u, np.zeros((args.maxlen, args.maxlen), dtype=np.int32)
        )

        # 构建候选物品列表
        if train_items and isinstance(train_items[0], dict):
            rated = {item["iid"] for item in train_items}
        else:
            rated = set(train_items)
        rated.add(0)
        # 提取测试集物品ID
        if test[u] and isinstance(test[u][0], dict):
            test_item = test[u][0]["iid"]
        else:
            test_item = test[u][0]
        item_idx = [test_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # 预测
        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [time_mat], item_idx]]
        )
        predictions = predictions[0]

        # 计算排名
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        # 计算NDCG和HR
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid_tisasrec(model, dataset, args):
    """
    在验证集上评估TiSASRec模型性能

    与evaluate_tisasrec类似，但在验证集上进行评估
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    # 为每个用户预计算时间间隔矩阵
    time_matrix_dict = {}
    for u in train.keys():
        if len(train[u]) >= 1:
            # 支持两种格式：1) [item_id, ...] 2) [{'iid': item_id, ...}, ...]
            if train[u] and isinstance(train[u][0], dict):
                history = [item.copy() for item in train[u]]
            else:
                history = [{"iid": item, "timestamp": 0} for item in train[u]]
            time_matrix_dict[u] = compute_time_matrix_for_user(
                history, args.maxlen, args.time_span
            )

    if usernum > 10000:
        users = random.sample(list(train.keys()), min(10000, len(train)))
    else:
        users = list(train.keys())

    for u in users:
        if u not in train or len(train[u]) < 1 or u not in valid or len(valid[u]) < 1:
            continue

        # 构建用户历史序列（仅使用训练集，不包含验证集）
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        train_items = train[u]
        if train_items and isinstance(train_items[0], dict):
            for i in range(len(train_items) - 1, -1, -1):
                seq[idx] = train_items[i]["iid"]
                idx -= 1
                if idx == -1:
                    break
        else:
            for i in reversed(train_items):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

        # 获取时间间隔矩阵
        time_mat = time_matrix_dict.get(
            u, np.zeros((args.maxlen, args.maxlen), dtype=np.int32)
        )

        # 支持两种格式：1) [item_id, ...] 2) [{'iid': item_id, ...}, ...]
        train_items = train[u]
        if train_items and isinstance(train_items[0], dict):
            rated = {item["iid"] for item in train_items}
        else:
            rated = set(train_items)
        rated.add(0)
        # 提取验证集物品ID
        if valid[u] and isinstance(valid[u][0], dict):
            valid_item = valid[u][0]["iid"]
        else:
            valid_item = valid[u][0]
        item_idx = [valid_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], [time_mat], item_idx]]
        )
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
