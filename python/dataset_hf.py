"""
HuggingFace ML-1M Dataset Loader with Temporal Information

从HuggingFace加载ml-1m数据集，提取时间信息用于时序推荐。
"""

import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 延迟导入datasets库
try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not installed. Run: pip install datasets")


def load_hf_ml1m_dataset(dataset_name="cep-ter/ML-1M"):
    """
    从HuggingFace加载ml-1m数据集

    返回包含所有分割的字典，每个样本包含：
    - uid: 用户ID
    - iid: 物品ID
    - timestamp: Unix时间戳
    - time: 一天中的小时（0-23）
    - rating: 评分
    - genres: 电影类型
    """
    global load_dataset
    if not DATASETS_AVAILABLE:
        from datasets import load_dataset as _load_dataset

        load_dataset = _load_dataset

    print(f"Loading dataset from HuggingFace: {dataset_name}")
    start_time = time.time()

    dataset = load_dataset(dataset_name)

    print(f"Dataset loaded in {time.time() - start_time:.2f}s")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")

    return dataset


def process_dataset_for_sequential(data_dict: Any) -> Tuple[Any, int, int]:
    """
    将HuggingFace数据集转换为适合序列推荐的格式

    返回:
        user_history: {用户ID: [(物品ID, 时间戳, 小时)]的列表，按时间排序}
        item_count: 物品总数
        user_count: 用户总数
    """
    # 合并所有分割
    all_data = []
    for split_name, split_data in data_dict.items():
        for item in split_data:
            all_data.append(
                {
                    "uid": item["uid"],
                    "iid": item["iid"],
                    "timestamp": item["timestamp"],
                    "hour": item["time"],
                    "rating": item.get("rating", 1),
                    "split": split_name,
                }
            )

    # 按用户分组
    user_history = defaultdict(list)
    for item in all_data:
        user_history[item["uid"]].append(
            {
                "iid": item["iid"],
                "timestamp": item["timestamp"],
                "hour": item["hour"],
                "rating": item["rating"],
                "split": item["split"],
            }
        )

    # 按时间戳排序每个用户的交互历史
    for uid in user_history:
        user_history[uid].sort(key=lambda x: x["timestamp"])

    # 统计用户和物品数量
    all_items = set()
    all_users = set()
    for uid, history in user_history.items():
        all_users.add(uid)
        for item in history:
            all_items.add(item["iid"])

    item_count = max(all_items) if all_items else 0
    user_count = max(all_users) if all_users else 0

    print(f"Total users: {user_count}")
    print(f"Total items: {item_count}")
    print(f"Total interactions: {len(all_data)}")

    return user_history, item_count, user_count


def split_train_val_test(
    user_history: Dict,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_interactions: int = 4,
) -> Tuple[Dict, Dict, Dict]:
    """
    将用户历史分割为训练集、验证集和测试集

    策略：对于每个用户
    - 最后1个作为测试集
    - 倒数第2个作为验证集
    - 其余作为训练集

    如果用户交互少于min_interactions，全部作为训练集
    """
    user_train = {}
    user_valid = {}
    user_test = {}

    for uid, history in user_history.items():
        n_interactions = len(history)

        if n_interactions < min_interactions:
            # 交互太少，全部作为训练数据
            user_train[uid] = history
            user_valid[uid] = []
            user_test[uid] = []
        else:
            # 正常划分
            # 保留完整的交互历史用于构建序列
            user_train[uid] = history[:-2]
            user_valid[uid] = [history[-2]]
            user_test[uid] = [history[-1]]

    return user_train, user_valid, user_test


def compute_time_matrix(
    user_history: Dict, maxlen: int, time_span: int, unit: str = "second"
) -> Dict:
    """
    为每个用户计算时间间隔矩阵

    对于序列中每个位置i，计算与之后所有位置j的时间间隔

    Args:
        user_history: 用户历史字典
        maxlen: 序列最大长度
        time_span: 时间间分离散化范围
        unit: 时间单位 ('second', 'minute', 'hour', 'day')

    Returns:
        time_matrix_dict: {用户ID: 时间间隔矩阵}
    """
    unit_seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
    divisor = unit_seconds.get(unit, 1)

    time_matrix_dict = {}

    for uid, history in user_history.items():
        seq_len = min(len(history), maxlen)
        time_matrix = np.zeros((maxlen, maxlen), dtype=np.int32)

        # 提取时间戳（只取序列长度的部分）
        timestamps = [item["timestamp"] for item in history[:seq_len]]

        # 计算时间间隔矩阵
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # 时间间隔 = later_time - earlier_time
                time_diff = timestamps[j] - timestamps[i]

                # 离散化到 [1, time_span]
                discretized = max(1, min(time_span, int(time_diff // divisor) + 1))
                time_matrix[i, j] = discretized

        time_matrix_dict[uid] = time_matrix

    return time_matrix_dict


def prepare_hf_dataset(args: Any) -> Tuple[Any, Any, Any, Any, int, int]:
    """
    准备HuggingFace数据集用于训练

    Returns:
        user_train: 训练集用户历史
        user_valid: 验证集用户历史
        user_test: 测试集用户历史
        time_matrix_dict: 用户时间间隔矩阵
        user_count: 用户数量
        item_count: 物品数量
    """
    # 加载数据集
    dataset = load_hf_ml1m_dataset()

    # 处理为序列格式
    user_history, item_count, user_count = process_dataset_for_sequential(dataset)

    # 分割训练/验证/测试集
    user_train, user_valid, user_test = split_train_val_test(
        user_history, val_ratio=0.1, test_ratio=0.1
    )

    # 计算时间间隔矩阵
    print(f"Computing time matrices with time_span={args.time_span}...")
    time_matrix_dict = compute_time_matrix(
        user_train,
        args.maxlen,
        args.time_span,
        unit="hour",  # 使用小时为单位
    )

    return user_train, user_valid, user_test, time_matrix_dict, user_count, item_count


def extract_hour_features(user_history: Dict, maxlen: int) -> Dict:
    """
    提取每个交互的小时信息作为额外特征

    Returns:
        hour_dict: {用户ID: [小时列表]}
    """
    hour_dict = {}

    for uid, history in user_history.items():
        hours = [item["hour"] for item in history[:maxlen]]
        # 填充到maxlen长度
        if len(hours) < maxlen:
            hours = hours + [0] * (maxlen - len(hours))
        hour_dict[uid] = np.array(hours, dtype=np.int32)

    return hour_dict


if __name__ == "__main__":
    # 测试数据加载
    from pprint import pprint

    try:
        dataset = load_hf_ml1m_dataset()
        user_history, item_count, user_count = process_dataset_for_sequential(dataset)

        # 打印示例用户
        sample_uid = list(user_history.keys())[0]
        print(f"\nSample user {sample_uid}:")
        pprint(user_history[sample_uid][:5])

        # 测试时间矩阵计算
        class Args:
            maxlen = 200
            time_span = 100

        time_matrix_dict = compute_time_matrix(
            user_history, Args.maxlen, Args.time_span
        )
        print(f"\nTime matrix shape: {time_matrix_dict[sample_uid].shape}")
        print(f"Sample time matrix (first 5x5):")
        print(time_matrix_dict[sample_uid][:5, :5])
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install datasets library: pip install datasets")
