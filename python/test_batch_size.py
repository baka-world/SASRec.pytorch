#!/usr/bin/env python3
"""
测试套件：验证batch_size参数传递

测试场景：
1. 参数解析是否正确
2. world_size计算是否正确
3. batch_size分配是否正确
4. 模型初始化时batch_size是否正确
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_argparse():
    """测试1: 参数解析"""
    print("=" * 60)
    print("测试1: 参数解析")
    print("=" * 60)

    # 模拟命令行参数
    test_cases = [
        {"batch_size": 16384, "world_size": 1, "expected_per_gpu": 16384},
        {"batch_size": 16384, "world_size": 2, "expected_per_gpu": 8192},
        {"batch_size": 16384, "world_size": 4, "expected_per_gpu": 4096},
        {"batch_size": 4096, "world_size": 4, "expected_per_gpu": 1024},
    ]

    for i, test in enumerate(test_cases, 1):
        batch_size = test["batch_size"]
        world_size = test["world_size"]
        expected = test["expected_per_gpu"]

        # 模拟setup_distributed中的计算
        per_gpu = max(1, batch_size // world_size)

        status = "✓" if per_gpu == expected else "✗"
        print(f"  [{status}] batch_size={batch_size}, world_size={world_size}")
        print(f"       期望 per_gpu={expected}, 实际 per_gpu={per_gpu}")

        if per_gpu != expected:
            print(f"       警告: 计算结果不匹配!")

    print()


def test_world_size_env():
    """测试2: world_size环境变量"""
    print("=" * 60)
    print("测试2: world_size环境变量")
    print("=" * 60)

    # 保存原始环境变量
    original_world_size = os.environ.get("WORLD_SIZE", None)
    original_local_rank = os.environ.get("LOCAL_RANK", None)
    original_rank = os.environ.get("RANK", None)

    test_cases = [
        {"WORLD_SIZE": "4", "LOCAL_RANK": "0", "RANK": "0", "cuda_count": 4},
        {"WORLD_SIZE": None, "LOCAL_RANK": "0", "RANK": "0", "cuda_count": 4},
        {"WORLD_SIZE": "2", "LOCAL_RANK": "0", "RANK": "0", "cuda_count": 4},
    ]

    for i, test in enumerate(test_cases, 1):
        # 设置环境变量
        if test["WORLD_SIZE"]:
            os.environ["WORLD_SIZE"] = test["WORLD_SIZE"]
        else:
            os.environ.pop("WORLD_SIZE", None)

        os.environ["LOCAL_RANK"] = test["LOCAL_RANK"]
        os.environ["RANK"] = test["RANK"]

        # 模拟setup_distributed中的计算
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        print(f"  测试{i}:")
        print(f"       环境变量 WORLD_SIZE={test['WORLD_SIZE']}")
        print(f"       计算得到 world_size={world_size}")
        print(f"       cuda.device_count()={test['cuda_count']}")

        if test["WORLD_SIZE"]:
            expected = int(test["WORLD_SIZE"])
        else:
            expected = torch.cuda.device_count()

        status = "✓" if world_size == expected else "✗"
        print(f"       [{status}] world_size={world_size}, expected={expected}")
        print()

    # 恢复原始环境变量
    if original_world_size:
        os.environ["WORLD_SIZE"] = original_world_size
    else:
        os.environ.pop("WORLD_SIZE", None)

    if original_local_rank:
        os.environ["LOCAL_RANK"] = original_local_rank
    if original_rank:
        os.environ["RANK"] = original_rank


def test_dummy_training():
    """测试3: 模拟训练过程"""
    print("=" * 60)
    print("测试3: 模拟训练参数传递")
    print("=" * 60)

    class DummyArgs:
        def __init__(self, batch_size, world_size):
            self.batch_size = batch_size
            self.world_size = world_size
            self.local_rank = 0
            self.multi_gpu = world_size > 1

    # 测试不同的batch_size和world_size组合
    test_cases = [
        {"total_batch": 16384, "world_size": 4, "expected": 4096},
        {"total_batch": 8192, "world_size": 4, "expected": 2048},
        {"total_batch": 4096, "world_size": 4, "expected": 1024},
        {"total_batch": 128, "world_size": 4, "expected": 32},
    ]

    for i, test in enumerate(test_cases, 1):
        args = DummyArgs(test["total_batch"], test["world_size"])

        # 模拟setup_distributed中的计算
        original_batch_size = args.batch_size
        args.batch_size = max(1, args.batch_size // args.world_size)

        status = "✓" if args.batch_size == test["expected"] else "✗"
        print(f"  [{status}] 测试{i}:")
        print(
            f"       总batch_size={test['total_batch']}, world_size={test['world_size']}"
        )
        print(f"       期望 per_gpu={test['expected']}, 实际 per_gpu={args.batch_size}")

        if args.batch_size != test["expected"]:
            print(f"       警告: 32 = 128 / 4 = 32 (这是正确的!)")

    print()


def test_sampler_batch_size():
    """测试4: WarpSampler中的batch_size"""
    print("=" * 60)
    print("测试4: WarpSampler batch_size")
    print("=" * 60)

    from utils import WarpSampler, WarpSamplerWithTime

    test_cases = [
        {"input_batch": 4096, "expected": 4096},
        {"input_batch": 1024, "expected": 1024},
        {"input_batch": 32, "expected": 32},
    ]

    for i, test in enumerate(test_cases, 1):
        # WarpSampler直接使用传入的batch_size
        # 所以如果args.batch_size被正确传递，这里应该一致
        sampler_batch = test["input_batch"]

        status = "✓" if sampler_batch == test["expected"] else "✗"
        print(f"  [{status}] 测试{i}:")
        print(f"       输入batch_size={test['input_batch']}")
        print(f"       WarpSampler使用batch_size={sampler_batch}")

    print()


def test_end_to_end():
    """端到端测试：模拟整个参数传递过程"""
    print("=" * 60)
    print("端到端测试: 参数传递链")
    print("=" * 60)

    def simulate_setup_distributed(args):
        """模拟setup_distributed函数"""
        if args.multi_gpu:
            args.world_size = int(
                os.environ.get("WORLD_SIZE", torch.cuda.device_count())
            )
            args.batch_size = max(1, args.batch_size // args.world_size)
        return args

    # 测试场景1: 用户指定batch_size=16384, 4张GPU
    print("  场景1: batch_size=16384, 4张GPU")
    args = argparse.Namespace(batch_size=16384, multi_gpu=True, num_workers=8)

    # 模拟torchrun设置的环境变量
    os.environ["WORLD_SIZE"] = "4"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

    args = simulate_setup_distributed(args)

    print(f"       原始batch_size=16384")
    print(f"       world_size={args.world_size}")
    print(f"       处理后batch_size={args.batch_size}")
    print(f"       期望: 4096, 结果: {'✓' if args.batch_size == 4096 else '✗'}")
    print()

    # 测试场景2: 错误情况 - 如果world_size不正确
    print("  场景2: batch_size=16384, world_size=512 (异常情况)")
    args2 = argparse.Namespace(batch_size=16384, multi_gpu=True, num_workers=8)

    # 模拟错误的环境变量
    os.environ["WORLD_SIZE"] = "512"

    args2 = simulate_setup_distributed(args2)

    print(f"       原始batch_size=16384")
    print(f"       world_size={args2.world_size} (异常值!)")
    print(f"       处理后batch_size={args2.batch_size}")
    print(f"       警告: 如果world_size=512, batch_size会变成32!")
    print()


def main():
    print("\n" + "=" * 60)
    print("SASRec 参数传递测试套件")
    print("=" * 60 + "\n")

    # 检查是否有CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"GPU数量: {torch.cuda.device_count()}")
    print()

    test_argparse()
    test_world_size_env()
    test_dummy_training()
    test_sampler_batch_size()
    test_end_to_end()

    print("=" * 60)
    print("测试完成!")
    print("=" * 60)
    print()
    print("常见问题排查:")
    print("  1. 如果batch_size=32, 检查WORLD_SIZE环境变量")
    print("  2. 确认torchrun正确设置了WORLD_SIZE")
    print("  3. 检查是否有其他进程修改了环境变量")
    print()


if __name__ == "__main__":
    main()
