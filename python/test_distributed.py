#!/usr/bin/env python
"""
快速验证多卡训练设置是否正确
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_distributed_setup(rank, world_size):
    print(f"[GPU {rank}] 测试分布式设置...")

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    x = torch.randn(100, 100).cuda(rank)
    y = torch.randn(100, 100).cuda(rank)
    z = x + y

    dist.all_reduce(z, op=dist.ReduceOp.SUM)

    print(f"[GPU {rank}] NCCL通信测试通过!")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 张GPU")

    if world_size < 2:
        print("单GPU环境，使用简单验证...")
        x = torch.randn(10, 10).cuda()
        print("CUDA张量创建成功")
        print("多卡功能需要至少2张GPU")
        return

    print("启动多卡通信测试...")
    mp.spawn(test_distributed_setup, args=(world_size,), nprocs=world_size, join=True)
    print("所有测试通过!")


if __name__ == "__main__":
    main()
