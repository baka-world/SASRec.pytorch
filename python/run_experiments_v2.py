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
import signal
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from rich import print as rprint
from enum import Enum
import shutil


# 动态检测 GPU 数量
def get_gpu_count() -> int:
    """检测可用 GPU 数量"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        return max(len(lines), 1)
    except:
        return 1


NUM_GPUS = get_gpu_count()
print(f"检测到 {NUM_GPUS} 个 GPU")


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
    pid: Optional[int] = None  # 进程 PID
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
        """获取GPU显存使用量(MiB)，返回None表示检测失败"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            if gpu_id < len(lines):
                line = lines[gpu_id].strip()
                value = line.split()[0] if line else None
                return float(value) if value else None
            return None
        except:
            return None

    def auto_assign_gpu(self, exp: Experiment) -> int:
        """自动分配GPU（尝试所有GPU，找到显存足够的）

        Returns:
            分配的GPU编号，如果都不可用返回-1
        """
        # 尝试所有GPU，找到显存足够的
        candidates = []
        for gpu_id in range(NUM_GPUS):
            mem = self.get_gpu_memory(gpu_id)
            candidates.append((gpu_id, mem))

        # 按显存从小到大排序
        candidates.sort(key=lambda x: x[1] if x[1] else float("inf"))

        # 返回显存最少的GPU（允许运行新实验）
        for gpu_id, mem in candidates:
            if mem < 30000:  # 显存 < 30GB
                return gpu_id

        return -1  # 所有GPU都满

    def get_available_gpu(self, min_memory: float = 4.0) -> Optional[int]:
        """获取可用GPU"""
        for gpu_id in range(NUM_GPUS):
            if gpu_id in self.running:
                continue
            mem = self.get_gpu_memory(gpu_id)
            if mem is not None and mem < 32 - min_memory:
                return gpu_id
        return None

    def is_gpu_free(self, gpu_id: int) -> bool:
        """检查GPU显存是否足够"""
        mem = self.get_gpu_memory(gpu_id)
        return mem < 30000  # 30GB

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

        # 清理旧的输出目录（如果存在）
        output_dir = exp.output_dir
        if output_dir and os.path.isdir(output_dir):
            print(f"{Colors.YELLOW}清理旧输出目录: {output_dir}{Colors.ENDC}")
            shutil.rmtree(output_dir)

        # 创建日志文件
        with open(exp.log_file, "w") as f:
            f.write(f"实验: {exp.name} ")
            f.write(f"GPU: {exp.gpu} ")
            f.write(f"命令: {exp.cmd} ")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
            f.write("=" * 60 + "\n\n")

        # 启动进程 - 使用 start_new_session 创建新进程组，方便清理
        full_cmd = f"python main.py --device=cuda:{exp.gpu} {exp.cmd}"
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # 创建新进程组
        )

        exp.pid = process.pid  # 记录 PID
        print(f"{Colors.CYAN}启动实验: {exp.name} (PID: {exp.pid}){Colors.ENDC}")

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
            if exp.gpu in self.running and exp in self.running[exp.gpu]:
                self.running[exp.gpu].remove(exp)
                if not self.running[exp.gpu]:
                    del self.running[exp.gpu]

        thread = threading.Thread(target=monitor_output, daemon=True)
        thread.start()
        # 允许多个实验在同一GPU，使用列表存储
        if exp.gpu not in self.running:
            self.running[exp.gpu] = []
        self.running[exp.gpu].append(exp)

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

    def is_experiment_ready_for_next(self, exp: Experiment) -> bool:
        """检查实验是否已经输出有效信息（loss/lr/epoch），可以开始下一个任务"""
        if exp.status != Status.RUNNING:
            return False

        log_file = exp.log_file
        if not log_file or not os.path.exists(log_file):
            return False

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    return False

                content = "".join(lines)
                content_lower = content.lower()

                has_loss = "loss" in content_lower
                has_lr = "lr=" in content_lower
                has_epoch = "epoch" in content_lower
                has_early_stop = "early stop" in content_lower
                has_done = "best" in content_lower and "model" in content_lower

                if has_done or has_early_stop:
                    return True

                result = has_loss or has_lr or has_epoch
                if not result:
                    print(
                        f"  DEBUG {exp.name}: loss={has_loss}, lr={has_lr}, epoch={has_epoch}, content_preview={content[:50]}"
                    )
                return result
        except Exception as e:
            print(f"  DEBUG: is_ready error: {e}")
            return False

    def get_experiments_on_gpu(self, gpu_id: int) -> List[Experiment]:
        """获取指定 GPU 上所有运行中的实验（通过 PID 检查进程状态）"""
        result = []
        for exp in self.running.get(gpu_id, []):
            if exp.status != Status.RUNNING:
                continue
            # 通过 PID 检查进程是否真的在运行
            if exp.pid is not None:
                try:
                    os.kill(exp.pid, 0)  # 信号 0 只检查进程是否存在
                except ProcessLookupError:
                    # 进程不存在，标记为失败
                    exp.status = Status.FAILED
                    exp.error = "进程意外终止"
                    continue
            result.append(exp)
        return result

    def is_process_alive(self, exp: Experiment) -> bool:
        """检查实验进程是否还在运行"""
        if exp.pid is None:
            return False
        try:
            os.kill(exp.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def run(self):
        """运行所有实验

        资源管理策略：
        1. 首次分配：每个空闲 GPU 分配一个任务
        2. 后续分配：等该 GPU 上的任务开始输出 loss/lr/epoch 后再分配新任务
        """
        import signal

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.clear_screen()
        self.print_header()

        gpu_ready_for_next = {gpu_id: True for gpu_id in range(NUM_GPUS)}

        while True:
            started_any = False

            # 查找等待中的实验
            pending_exps = [e for e in self.experiments if e.status == Status.PENDING]

            if not pending_exps:
                # 检查是否所有实验都完成
                if all(
                    exp.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]
                    for exp in self.experiments
                ):
                    self.print_final_results()
                    break
                time.sleep(2)
                self.print_status()
                continue

            # 为每个 GPU 尝试分配任务
            for gpu_id in range(NUM_GPUS):
                running_exps = self.get_experiments_on_gpu(gpu_id)

                if running_exps:
                    # 有任务在运行，检查是否准备好接收新任务
                    earliest_exp = min(running_exps, key=lambda e: e.start_time or 0)
                    if self.is_experiment_ready_for_next(earliest_exp):
                        # 该 GPU 可以开始下一个任务
                        gpu_ready_for_next[gpu_id] = True
                    else:
                        gpu_ready_for_next[gpu_id] = False
                        continue
                else:
                    # 首次分配：检查显存是否足够且没有运行任务
                    mem = self.get_gpu_memory(gpu_id)
                    if mem is None or mem >= 30000:
                        continue
                    gpu_ready_for_next[gpu_id] = True

                if not gpu_ready_for_next.get(gpu_id, True):
                    continue

                # 如果 GPU 上已有任务在运行且还没准备好，不分配新任务
                if len(running_exps) >= 1 and not gpu_ready_for_next.get(gpu_id, False):
                    continue

                # 分配新任务给这个 GPU
                for exp in pending_exps:
                    if exp.status != Status.PENDING:
                        continue
                    if exp.gpu != -1 and exp.gpu != gpu_id:
                        continue

                    if exp.gpu == -1:
                        exp.gpu = gpu_id

                    self.start_experiment(exp)
                    gpu_ready_for_next[gpu_id] = False  # 等待这个任务开始
                    started_any = True
                    self.print_status()
                    break

            if not started_any:
                pending = [e for e in self.experiments if e.status == Status.PENDING]
                if pending:
                    print(
                        f"{Colors.YELLOW}等待中... ({len(pending)}个实验){Colors.ENDC}"
                    )
                time.sleep(2)
                self.print_status()

    def kill_process_group(self, pid: int) -> bool:
        """杀死进程及其所有子进程"""
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return False

    def cleanup_all(self):
        """停止所有实验并清理GPU显存"""
        print(f"{Colors.YELLOW}正在停止所有实验...{Colors.ENDC}")

        for gpu_id, exps in list(self.running.items()):
            for exp in exps:
                if exp:
                    print(f"  停止实验: {exp.name} (GPU {gpu_id})")
                    exp.status = Status.CANCELLED
                    exp.end_time = time.time()

        self.running.clear()

        print(f"{Colors.CYAN}清理残留进程...{Colors.ENDC}")

        killed_pids = set()
        for _ in range(3):
            try:
                result = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.split("\n"):
                    if "python main.py" in line and "grep" not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                if pid > 0 and pid not in killed_pids:
                                    killed_pids.add(pid)
                                    if self.kill_process_group(pid):
                                        print(f"  已终止进程组 PID: {pid}")
                                    else:
                                        subprocess.run(
                                            ["kill", "-9", str(pid)],
                                            capture_output=True,
                                        )
                                        print(f"  已终止 PID: {pid}")
                            except:
                                pass
            except Exception as e:
                print(f"  清理失败: {e}")
                break
            time.sleep(1)

        print(f"{Colors.CYAN}GPU 状态:{Colors.ENDC}")
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "MiB" in line or "Tesla" in line or "GPU" in line:
                    print(f"  {line}")
        except:
            pass

        print(f"{Colors.GREEN}已停止所有实验{Colors.ENDC}")

    def signal_handler(self, signum, frame):
        """信号处理：Ctrl+C 优雅退出"""
        print(f"{Colors.RED}收到终止信号，正在清理...{Colors.ENDC}")
        self.cleanup_all()
        self.save_results()
        print(f"{Colors.YELLOW}实验结果已保存到: {self.results_file}{Colors.ENDC}")
        sys.exit(0)

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

    def get_latest_output(self, exp: Experiment) -> str:
        """获取实验的最新输出行"""
        if not exp.log_file or not os.path.exists(exp.log_file):
            return ""
        try:
            with open(exp.log_file, "r") as f:
                lines = f.readlines()
                for line in reversed(lines[-20:]):
                    line = line.strip()
                    if (
                        line
                        and not line.startswith("实验:")
                        and not line.startswith("GPU:")
                        and not line.startswith("命令:")
                        and not line.startswith("开始时间:")
                        and not line.startswith("=")
                    ):
                        return line[:100] + ("..." if len(line) > 100 else "")
                return ""
        except:
            return ""

    def print_status(self):
        """打印当前状态（使用rich美化输出）"""
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich.box import ROUNDED

        self.clear_screen()

        # 标题
        rprint(
            f"[bold magenta]╔══════════════════════════════════════════════════════════════╗[/]"
        )
        rprint(
            f"[bold magenta]║[/]  [bold white]SASRec/TiSASRec 实验管理器[/]                           [bold magenta]║[/]"
        )
        rprint(
            f"[bold magenta]║[/]  [dim]Experiment Manager for SASRec/TiSASRec[/]                  [bold magenta]║[/]"
        )
        rprint(
            f"[bold magenta]╠══════════════════════════════════════════════════════════════╣[/]"
        )
        rprint(
            f"[bold magenta]║[/]  实验数量: [cyan]{len(self.experiments)}[/]                                              [bold magenta]║[/]"
        )
        rprint(
            f"[bold magenta]║[/]  开始时间: [cyan]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]                         [bold magenta]║[/]"
        )
        rprint(
            f"[bold magenta]╚══════════════════════════════════════════════════════════════╝[/]"
        )

        # 实验状态表格
        table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("GPU", width=5, justify="center")
        table.add_column("实验", width=28)
        table.add_column("最新输出", width=48)
        table.add_column("显存", width=10, justify="right")
        table.add_column("运行时", width=10, justify="right")

        for gpu_id in range(NUM_GPUS):
            exps = self.running.get(gpu_id, [])
            mem = self.get_gpu_memory(gpu_id)
            mem_str = f"{mem / 1024:.1f}GB"

            if exps:
                for i, exp in enumerate(exps):
                    name = exp.name[:26] + "..." if len(exp.name) > 26 else exp.name
                    latest = self.get_latest_output(exp) or "-"
                    latest = latest[:46] + ("..." if len(latest) > 46 else "")
                    duration = (
                        f"{time.time() - exp.start_time:.0f}s"
                        if exp.start_time
                        else "-"
                    )

                    gpu_prefix = str(gpu_id) if i == 0 else " "
                    table.add_row(gpu_prefix, name, latest, mem_str, duration)
            else:
                table.add_row(str(gpu_id), "[dim]空闲[/]", "-", mem_str, "-")

        rprint("")
        rprint("[bold cyan]运行中的实验:[/]")
        rprint(table)

        # GPU显存摘要
        rprint("")
        rprint("[bold cyan]显存使用:[/]", end="")
        for gpu_id in range(NUM_GPUS):
            mem = self.get_gpu_memory(gpu_id)
            exps = self.running.get(gpu_id, [])
            count = len(exps)
            if count > 0:
                rprint(
                    f"  [green]GPU{gpu_id}:[/] {mem / 1024:.1f}GB ([yellow]{count}实验[/])",
                    end="",
                )
            else:
                rprint(f"  GPU{gpu_id}: {mem / 1024:.1f}GB", end="")
        rprint("")

        # 进度统计
        completed = [e for e in self.experiments if e.status == Status.COMPLETED]
        failed = [e for e in self.experiments if e.status == Status.FAILED]
        pending = [e for e in self.experiments if e.status == Status.PENDING]
        total = len(self.experiments)

        rprint("")
        rprint(f"[bold]进度:[/] [green]{completed}[/]/{total}完成", end="")
        if failed:
            rprint(f"  [red]X {len(failed)}失败[/]", end="")
        if pending:
            rprint(f"  [yellow]O {len(pending)}等待[/]", end="")

        if completed:
            best = max((e.ndcg10 for e in completed if e.ndcg10), default=0)
            if best > 0:
                rprint(f"  [bold]Best NDCG:[/] [green]{best:.4f}[/]")
        rprint("")

        # 等待中的实验
        if pending:
            names = [e.name for e in pending[:12]]
            rprint(f"")
            rprint(
                f"[bold yellow]等待 ({len(pending)}个):[/] "
                + ", ".join(names)
                + ("..." if len(pending) > 12 else "")
            )

    def print_final_results(self):
        """打印最终结果"""
        self.clear_screen()
        self.print_header()

        print(
            f"{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗"
        )
        print("║                      实验结果汇总                            ║")
        print(
            f"╚══════════════════════════════════════════════════════════════╝{Colors.ENDC} "
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
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
            f.write(f"实验总数: {len(self.experiments)} ")
            f.write(
                f"成功数量: {len([e for e in self.experiments if e.status == Status.COMPLETED])}\n "
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
                f.write(f"{i:<5} {exp.name:<30} {ndcg}   {hr}   {duration} ")

            f.write("\n最佳配置:\n")
            if sorted_exps:
                f.write(f"  名称: {sorted_exps[0].name} ")
                f.write(f"  NDCG@10: {sorted_exps[0].ndcg10} ")
                f.write(f"  HR@10: {sorted_exps[0].hr10} ")

        print(f"报告已保存: {report_file}")


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

    print(f"共 {len(experiments)} 个实验")
    print("开始运行...\n")

    # 运行
    manager.run()


if __name__ == "__main__":
    main()
