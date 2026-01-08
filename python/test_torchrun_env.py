#!/usr/bin/env python3
"""
实际环境测试：检查torchrun是否正确设置环境变量
"""

import os
import subprocess
import sys


def test_torchrun_env():
    """测试torchrun是否正确设置环境变量"""
    print("=" * 60)
    print("测试: torchrun环境变量设置")
    print("=" * 60)

    # 创建测试脚本
    test_script = """
import os
import torch
import torch.distributed as dist

# 检查环境变量
print("=" * 40)
print("环境变量检查")
print("=" * 40)

env_vars = ["WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]
for var in env_vars:
    value = os.environ.get(var, "NOT SET")
    print(f"{var}: {value}")

# 检查torch.distributed
if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    
    print()
    print("=" * 40)
    print("分布式参数")
    print("=" * 40)
    print(f"world_size: {world_size}")
    print(f"local_rank: {local_rank}")
    print(f"rank: {rank}")
    
    # 模拟batch_size计算
    batch_size = 16384
    per_gpu_batch = max(1, batch_size // world_size)
    print()
    print("=" * 40)
    print("batch_size计算")
    print("=" * 40)
    print(f"原始batch_size: {batch_size}")
    print(f"world_size: {world_size}")
    print(f"每卡batch_size: {per_gpu_batch}")
    
    if per_gpu_batch == 32:
        print()
        print("警告: batch_size=32 说明 world_size=512!")
        print("这意味着torchrun没有正确设置WORLD_SIZE")
else:
    print("WORLD_SIZE未设置 - torchrun可能未正确初始化")
"""

    # 写临时测试文件
    with open("/tmp/test_torchrun_env.py", "w") as f:
        f.write(test_script)

    # 获取GPU数量
    gpu_count = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.count("\n")
        or 1
    )

    print(f"检测到 {gpu_count} 张GPU")
    print()

    # 使用torchrun运行测试
    print("运行 torchrun 测试...")
    print()

    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node",
            str(gpu_count),
            "--master_port",
            "29501",
            "/tmp/test_torchrun_env.py",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    print(result.stdout)
    if result.stderr:
        # 过滤掉常见的警告
        stderr_lines = [
            l
            for l in result.stderr.split("\n")
            if "UserWarning" not in l and "warnings.warn" not in l
        ]
        if stderr_lines:
            print("STDERR:", "\n".join(stderr_lines))


def check_current_env():
    """检查当前shell环境"""
    print("=" * 60)
    print("当前Shell环境检查")
    print("=" * 60)

    env_vars = [
        "WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
    ]
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

    print()


def main():
    print("\n" + "=" * 60)
    print("torchrun 环境变量验证测试")
    print("=" * 60 + "\n")

    check_current_env()
    test_torchrun_env()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print()
    print("如果看到 'batch_size=32'，说明WORLD_SIZE被设置为512")
    print("这通常是torchrun初始化失败或环境变量被污染导致的")
    print()


if __name__ == "__main__":
    main()
