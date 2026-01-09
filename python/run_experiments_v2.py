#!/usr/bin/env python3
"""
SASRec/TiSASRec 实验管理器
- 队列式实验调度
- 显存监控与自动等待
- 美化输出
- 结果保存
"""

import os
import sys
import time
import json
import subprocess
import threading
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import shutil


# 颜色定义
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Status(Enum):
    PENDING = "⏳ 等待中"
    RUNNING = "🔄 运行中"
    COMPLETED = "✅ 完成"
    FAILED = "❌ 失败"
    CANCELLED = "🚫 取消"
    WAITING_GPU = "⏸️ 等待GPU"


@dataclass
class Experiment:
    """实验配置"""

    name: str
    gpu: int
    cmd: str
    log_file: str
    status: Status = Status.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    ndcg10: Optional[float] = None
    hr10: Optional[float] = None
    output_dir: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gpu": self.gpu,
            "status": self.status.value,
            "start_time": datetime.fromtimestamp(self.start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if self.start_time
            else None,
            "end_time": datetime.fromtimestamp(self.end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if self.end_time
            else None,
            "duration": f"{self.end_time - self.start_time:.1f}s"
            if self.end_time and self.start_time
            else None,
            "ndcg10": self.ndcg10,
            "hr10": self.hr10,
            "output_dir": self.output_dir,
            "error": self.error,
        }


class ExperimentManager:
    """实验管理器"""

    def __init__(self, work_dir: str = "experiments"):
        self.work_dir = work_dir
        self.experiments: List[Experiment] = []
        self.running: Dict[int, Experiment] = {}  # gpu_id -> experiment
        self.results_file = os.path.join(work_dir, "results.json")
        os.makedirs(work_dir, exist_ok=True)

    def add_experiment(
        self, name: str, gpu: int, cmd: str, output_dir: Optional[str] = None
    ) -> Experiment:
        """添加实验"""
        log_file = os.path.join(self.work_dir, f"log_{name}.log")
        if output_dir is None:
            output_dir = f"ml-1m_{name}"
        exp = Experiment(
            name=name, gpu=gpu, cmd=cmd, log_file=log_file, output_dir=output_dir
        )
        self.experiments.append(exp)
        return exp

    def get_gpu_memory(self, gpu_id: int) -> Optional[float]:
        """获取GPU显存使用量(GB)"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            if gpu_id < len(lines):
                return float(lines[gpu_id]) / 1024  # 转换为GB
        except:
            pass
        return None

    def auto_assign_gpu(self, exp: Experiment) -> int:
        """自动分配GPU（选择最空闲的GPU）

        Returns:
            分配的GPU编号
        """
        available_gpus = []
        for gpu_id in range(4):
            if gpu_id not in self.running:
                mem = self.get_gpu_memory(gpu_id)
                available_gpus.append((gpu_id, mem))

        # 按显存从小到大排序，选择最空闲的
        available_gpus.sort(key=lambda x: x[1] if x[1] else float("inf"))
        return available_gpus[0][0] if available_gpus else 0

    def get_available_gpu(self, min_memory: float = 4.0) -> Optional[int]:
        """获取可用GPU"""
        for gpu_id in range(4):
            if gpu_id in self.running:
                continue
            mem = self.get_gpu_memory(gpu_id)
            if mem is not None and mem < 32 - min_memory:
                return gpu_id
        return None

    def is_gpu_free(self, gpu_id: int) -> bool:
        """检查GPU是否空闲"""
        return gpu_id not in self.running

    def start_experiment(self, exp: Experiment):
        """启动实验

        如果exp.gpu == -1，则自动分配最空闲的GPU
        """
        # 自动分配GPU
        if exp.gpu == -1:
            exp.gpu = self.auto_assign_gpu(exp)
            print(
                f"{Colors.CYAN}自动分配GPU: {exp.name} -> cuda:{exp.gpu}{Colors.ENDC}"
            )

        exp.status = Status.RUNNING
        exp.start_time = time.time()

        # 创建日志文件
        with open(exp.log_file, "w") as f:
            f.write(f"实验: {exp.name}\n")
            f.write(f"GPU: {exp.gpu}\n")
            f.write(f"命令: {exp.cmd}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

        # 启动进程
        full_cmd = f"python main.py --device=cuda:{exp.gpu} {exp.cmd}"
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # 后台线程监控输出
        def monitor_output():
            for line in process.stdout:
                with open(exp.log_file, "a") as f:
                    f.write(line)
            process.wait()

            exp.end_time = time.time()
            if process.returncode == 0:
                exp.status = Status.COMPLETED
                # 解析结果
                self.parse_results(exp)
            else:
                exp.status = Status.FAILED
                exp.error = f"返回码: {process.returncode}"

            # 保存结果
            self.save_results()

            # 从运行列表移除
            if exp.gpu in self.running:
                del self.running[exp.gpu]

        thread = threading.Thread(target=monitor_output, daemon=True)
        thread.start()
        self.running[exp.gpu] = exp

    def parse_results(self, exp: Experiment):
        """解析实验结果"""
        try:
            with open(exp.log_file, "r") as f:
                content = f.read()
                import re

                # 精确匹配 test 结果（避免匹配到 valid 结果）
                # 格式: "test (NDCG@10: 0.XXXX, HR@10: 0.XXXX)" 或 "..., test (NDCG@10: 0.XXXX, HR@10: 0.XXXX)"
                test_match = re.search(
                    r"test\s*\(NDCG@10:\s*([\d.]+),\s*HR@10:\s*([\d.]+)\)", content
                )

                if test_match:
                    exp.ndcg10 = float(test_match.group(1))
                    exp.hr10 = float(test_match.group(2))
        except Exception as e:
            pass

    def save_results(self):
        """保存结果到JSON"""
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiments": [exp.to_dict() for exp in self.experiments],
        }
        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def run(self, max_concurrent: int = 4):
        """运行所有实验"""
        self.clear_screen()
        self.print_header()

        while True:
            # 启动新实验
            started = False
            for exp in self.experiments:
                if exp.status == Status.PENDING and exp.gpu in self.running:
                    continue

                if exp.status == Status.PENDING and self.is_gpu_free(exp.gpu):
                    self.start_experiment(exp)
                    started = True
                    self.print_status()
                    break

            if not started:
                # 检查是否所有实验都完成
                if all(
                    exp.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]
                    for exp in self.experiments
                ):
                    self.print_final_results()
                    break

                # 显示等待状态
                pending = [e for e in self.experiments if e.status == Status.PENDING]
                if pending:
                    print(
                        f"\n{Colors.YELLOW}等待GPU可用... ({len(pending)}个实验等待中){Colors.ENDC}"
                    )
                    time.sleep(10)
                    self.print_status()

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║         SASRec/TiSASRec 实验管理器                           ║")
        print("║         Experiment Manager for SASRec/TiSASRec               ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  实验数量: {len(self.experiments):<44}║")
        print(f"║  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<41}║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")

    def print_status(self):
        """打印当前状态"""
        self.clear_screen()
        self.print_header()

        # GPU状态
        print(f"{Colors.CYAN}{Colors.BOLD}GPU 状态:{Colors.ENDC}")
        print(
            "┌──────┬─────────────────────────────┬─────────────┬──────────────────────────┐"
        )
        print(
            "│ GPU  │ 实验                         │ 状态        │ 显存     进度           │"
        )
        print(
            "├──────┼─────────────────────────────┼─────────────┼──────────────────────────┤"
        )

        for gpu_id in range(4):
            if gpu_id in self.running:
                exp = self.running[gpu_id]
                mem = self.get_gpu_memory(gpu_id)
                mem_str = f"{mem:.1f}GB" if mem else "?"
                status = exp.status.value
                duration = (
                    f"{time.time() - exp.start_time:.0f}s" if exp.start_time else ""
                )
                name = exp.name[:27] + "..." if len(exp.name) > 30 else exp.name
                print(
                    f"│ {gpu_id}   │ {name:<29} │ {status:<11} │ {mem_str:<9} {duration:<8} │"
                )
            else:
                print(f"│ {gpu_id}   │ {'空闲':<29} │ {'🟢 可用':<11} │ {'-':<21} │")

        print(
            "└──────┴─────────────────────────────┴─────────────┴──────────────────────────┘"
        )

        # 等待中的实验
        pending = [e for e in self.experiments if e.status == Status.PENDING]
        if pending:
            print(f"\n{Colors.YELLOW}等待中的实验 ({len(pending)}个):{Colors.ENDC}")
            for exp in pending[:5]:
                print(f"  • {exp.name} (GPU {exp.gpu})")
            if len(pending) > 5:
                print(f"  ... 还有 {len(pending) - 5} 个")

        # 已完成的实验
        completed = [e for e in self.experiments if e.status == Status.COMPLETED]
        if completed:
            print(
                f"\n{Colors.GREEN}已完成 ({len(completed)}/{len(self.experiments)}):{Colors.ENDC}"
            )
            best_ndcg = max((e.ndcg10 for e in completed if e.ndcg10), default=0)
            for exp in completed:
                ndcg_str = f"NDCG@{exp.ndcg10:.4f}" if exp.ndcg10 else "NDCG:?   "
                hr_str = f"HR@{exp.hr10:.4f}" if exp.hr10 else "HR:?   "
                print(f"  ✓ {exp.name:<30} {ndcg_str} {hr_str}")

    def print_final_results(self):
        """打印最终结果"""
        self.clear_screen()
        self.print_header()

        print(
            f"{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗"
        )
        print("║                      实验结果汇总                            ║")
        print(
            f"╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}\n"
        )

        # 按NDCG排序
        sorted_exps = sorted(
            [e for e in self.experiments if e.status == Status.COMPLETED],
            key=lambda x: x.ndcg10 or 0,
            reverse=True,
        )

        print(
            f"{Colors.CYAN}{'排名':<4} {'实验名称':<35} {'NDCG@10':<12} {'HR@10':<12} {'耗时':<10}{Colors.ENDC}"
        )
        print("─" * 80)

        for i, exp in enumerate(sorted_exps, 1):
            ndcg = f"{exp.ndcg10:.4f}" if exp.ndcg10 else "N/A"
            hr = f"{exp.hr10:.4f}" if exp.hr10 else "N/A"
            duration = (
                f"{exp.end_time - exp.start_time:.0f}s"
                if exp.end_time and exp.start_time
                else "N/A"
            )
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{medal} {i:<3} {exp.name:<35} {ndcg:<12} {hr:<12} {duration:<10}")

        print("\n" + "=" * 80)
        print(
            f"{Colors.GREEN}最佳配置: {sorted_exps[0].name if sorted_exps else 'N/A'}{Colors.ENDC}"
        )
        print(f"最佳 NDCG@10: {sorted_exps[0].ndcg10 if sorted_exps else 'N/A'}")
        print("=" * 80)

        # 保存最终报告
        self.save_final_report(sorted_exps)

    def save_final_report(self, sorted_exps: List[Experiment]):
        """保存最终报告"""
        report_file = os.path.join(self.work_dir, "final_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("SASRec/TiSASRec 实验报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验总数: {len(self.experiments)}\n")
            f.write(
                f"成功数量: {len([e for e in self.experiments if e.status == Status.COMPLETED])}\n\n"
            )

            f.write("排名  实验名称                      NDCG@10   HR@10      耗时\n")
            f.write("-" * 80 + "\n")
            for i, exp in enumerate(sorted_exps, 1):
                ndcg = f"{exp.ndcg10:.4f}" if exp.ndcg10 else "N/A    "
                hr = f"{exp.hr10:.4f}" if exp.hr10 else "N/A    "
                duration = (
                    f"{exp.end_time - exp.start_time:.0f}s"
                    if exp.end_time and exp.start_time
                    else "N/A"
                )
                f.write(f"{i:<5} {exp.name:<30} {ndcg}   {hr}   {duration}\n")

            f.write("\n最佳配置:\n")
            if sorted_exps:
                f.write(f"  名称: {sorted_exps[0].name}\n")
                f.write(f"  NDCG@10: {sorted_exps[0].ndcg10}\n")
                f.write(f"  HR@10: {sorted_exps[0].hr10}\n")

        print(f"\n报告已保存: {report_file}")


# 实验配置定义
def get_experiments() -> List[tuple]:
    """定义所有实验"""
    experiments = []

    # ===== 对比实验 (E1-E4) =====
    experiments.append(
        (
            "exp_e1_sasrec",
            -1,
            "--dataset=ml-1m --train_dir=exp_e1_sasrec --no_time --no_mhc "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --dropout_rate 0.2 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "exp_e2_sasrec_mhc",
            -1,
            "--dataset=ml-1m --train_dir=exp_e2_sasrec_mhc --no_time "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --dropout_rate 0.2 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "exp_e3_tisasrec",
            -1,
            "--dataset=ml-1m --train_dir=exp_e3_tisasrec --no_mhc "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --dropout_rate 0.2 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "exp_e4_tisasrec_mhc",
            -1,
            "--dataset=ml-1m --train_dir=exp_e4_tisasrec_mhc "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --dropout_rate 0.2 --num_epochs 300",
        )
    )

    # ===== 调参实验 (T1-T12) =====
    # GPU 1: batch & hidden & n
    experiments.append(
        (
            "tune_t1_batch512",
            -1,
            "--dataset=ml-1m --train_dir=tune_t1_batch512 --batch_size 512 "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --mhc_expansion_rate 4 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t2_h150_n4",
            -1,
            "--dataset=ml-1m --train_dir=tune_t2_h150_n4 --hidden_units 150 "
            "--maxlen 50 --lr 0.01 --mhc_expansion_rate 4 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t3_h150_batch512",
            -1,
            "--dataset=ml-1m --train_dir=tune_t3_h150_batch512 --hidden_units 150 --batch_size 512 "
            "--maxlen 50 --lr 0.01 --mhc_expansion_rate 4 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t4_n8",
            -1,
            "--dataset=ml-1m --train_dir=tune_t4_n8 --mhc_expansion_rate 8 "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --num_epochs 300",
        )
    )

    # GPU 2: n & maxlen
    experiments.append(
        (
            "tune_t5_n12",
            -1,
            "--dataset=ml-1m --train_dir=tune_t5_n12 --mhc_expansion_rate 12 "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t6_maxlen100",
            -1,
            "--dataset=ml-1m --train_dir=tune_t6_maxlen100 --maxlen 100 "
            "--hidden_units 100 --lr 0.01 --mhc_expansion_rate 4 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t7_max100_n8",
            -1,
            "--dataset=ml-1m --train_dir=tune_t7_max100_n8 --maxlen 100 --mhc_expansion_rate 8 "
            "--hidden_units 100 --lr 0.01 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t8_h150_n8_max100",
            -1,
            "--dataset=ml-1m --train_dir=tune_t8_h150_n8_max100 --hidden_units 150 --maxlen 100 --mhc_expansion_rate 8 "
            "--lr 0.01 --num_epochs 300",
        )
    )

    # GPU 3: 极限探索
    experiments.append(
        (
            "tune_t9_h200_n8",
            -1,
            "--dataset=ml-1m --train_dir=tune_t9_h200_n8 --hidden_units 200 --mhc_expansion_rate 8 "
            "--maxlen 50 --lr 0.01 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t10_batch1024",
            -1,
            "--dataset=ml-1m --train_dir=tune_t10_batch1024 --batch_size 1024 "
            "--hidden_units 100 --maxlen 50 --lr 0.01 --mhc_expansion_rate 4 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t11_h200_n12",
            -1,
            "--dataset=ml-1m --train_dir=tune_t11_h200_n12 --hidden_units 200 --mhc_expansion_rate 12 "
            "--maxlen 50 --lr 0.01 --num_epochs 300",
        )
    )

    experiments.append(
        (
            "tune_t12_best_guess",
            -1,
            "--dataset=ml-1m --train_dir=tune_t12_best_guess --hidden_units 150 --maxlen 100 --mhc_expansion_rate 8 "
            "--lr 0.01 --batch_size 256 --num_epochs 300",
        )
    )

    return experiments


def main():
    parser = argparse.ArgumentParser(description="SASRec/TiSASRec 实验管理器")
    parser.add_argument("--work-dir", default="experiments", help="工作目录")
    parser.add_argument("--max-concurrent", type=int, default=4, help="最大并行数")
    args = parser.parse_args()

    # 创建管理器
    manager = ExperimentManager(work_dir=args.work_dir)

    # 添加实验
    print("加载实验配置...")
    experiments = get_experiments()
    for name, gpu, cmd in experiments:
        manager.add_experiment(name, gpu, cmd)
        print(f"  ✓ {name} (GPU {gpu})")

    print(f"\n共 {len(experiments)} 个实验")
    print("开始运行...\n")

    # 运行
    manager.run(max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
