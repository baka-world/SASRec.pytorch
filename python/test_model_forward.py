#!/usr/bin/env python3
"""
单元测试：验证模型前向传播和参数传递正确性
"""

import sys
import os

# 添加 python 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from model_mhc import SASRec, TiSASRec


def test_sasrec_forward():
    """测试 SASRec 模型前向传播"""
    print("[Test] SASRec 前向传播测试...")

    # 参数设置
    usernum = 100
    itemnum = 1000
    maxlen = 50
    hidden_units = 50
    num_heads = 2
    num_blocks = 2
    dropout_rate = 0.2

    # 创建参数对象
    class Args:
        def __init__(self):
            self.hidden_units = hidden_units
            self.maxlen = maxlen
            self.dropout_rate = dropout_rate
            self.num_heads = num_heads
            self.num_blocks = num_blocks
            self.norm_first = False
            self.l2_emb = 0.0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()

    # 创建模型
    model = SASRec(usernum, itemnum, args)
    model.to(args.device)
    model.eval()

    # 测试输入
    batch_size = 4
    user_ids = torch.randint(1, usernum, (batch_size,)).to(args.device)
    log_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)
    pos_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)
    neg_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)

    # 前向传播
    with torch.no_grad():
        pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)

    # 验证输出形状
    assert pos_logits.shape == (batch_size, maxlen), (
        f"pos_logits 形状错误: {pos_logits.shape}"
    )
    assert neg_logits.shape == (batch_size, maxlen), (
        f"neg_logits 形状错误: {neg_logits.shape}"
    )

    # 验证输出值范围合理
    assert not torch.isnan(pos_logits).any(), "pos_logits 包含 NaN"
    assert not torch.isnan(neg_logits).any(), "neg_logits 包含 NaN"

    print("[Test] SASRec 前向传播测试通过!")
    return True


def test_tisasrec_forward():
    """测试 TiSASRec 模型前向传播"""
    print("[Test] TiSASRec 前向传播测试...")

    # 参数设置
    usernum = 100
    itemnum = 1000
    maxlen = 50
    hidden_units = 50
    num_heads = 2
    num_blocks = 2
    dropout_rate = 0.2
    time_span = 100

    # 创建参数对象
    class Args:
        def __init__(self):
            self.hidden_units = hidden_units
            self.maxlen = maxlen
            self.dropout_rate = dropout_rate
            self.num_heads = num_heads
            self.num_blocks = num_blocks
            self.norm_first = False
            self.l2_emb = 0.0
            self.time_span = time_span
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()

    # 创建模型
    model = TiSASRec(usernum, itemnum, time_span, args)
    model.to(args.device)
    model.eval()

    # 测试输入
    batch_size = 4
    user_ids = torch.randint(1, usernum, (batch_size,)).to(args.device)
    log_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)
    time_matrices = torch.randint(0, time_span, (batch_size, maxlen, maxlen)).to(
        args.device
    )
    pos_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)
    neg_seqs = torch.randint(0, itemnum, (batch_size, maxlen)).to(args.device)

    # 前向传播
    with torch.no_grad():
        pos_logits, neg_logits = model(
            user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs
        )

    # 验证输出形状
    assert pos_logits.shape == (batch_size, maxlen), (
        f"pos_logits 形状错误: {pos_logits.shape}"
    )
    assert neg_logits.shape == (batch_size, maxlen), (
        f"neg_logits 形状错误: {neg_logits.shape}"
    )

    # 验证输出值范围合理
    assert not torch.isnan(pos_logits).any(), "pos_logits 包含 NaN"
    assert not torch.isnan(neg_logits).any(), "neg_logits 包含 NaN"

    print("[Test] TiSASRec 前向传播测试通过!")
    return True


def test_tensor_device_consistency():
    """测试 tensor device 一致性"""
    print("[Test] Tensor Device 一致性测试...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 numpy 数组
    numpy_array = np.random.randint(0, 100, (10, 50))

    # 测试 torch.as_tensor 创建的 tensor 是否在正确 device 上
    tensor = torch.as_tensor(numpy_array, device=device)
    assert str(tensor.device).startswith(str(device)), (
        f"Tensor device {tensor.device} 不匹配期望的 {device}"
    )

    # 测试 embedding lookup
    item_emb = torch.nn.Embedding(1000, 50)
    item_emb.to(device)

    result = item_emb(tensor)
    assert str(result.device).startswith(str(device)), (
        f"Embedding result device {result.device} 不匹配期望的 {device}"
    )

    print("[Test] Tensor Device 一致性测试通过!")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行单元测试...")
    print("=" * 50)

    tests = [
        ("SASRec 前向传播测试", test_sasrec_forward),
        ("TiSASRec 前向传播测试", test_tisasrec_forward),
        ("Tensor Device 一致性测试", test_tensor_device_consistency),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print()
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
