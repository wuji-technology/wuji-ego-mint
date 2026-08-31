"""
后台采样 GPU 利用率 / 显存 / CPU，结束时生成时序图。

用法（上下文管理器）:
    from script_utils.resource_monitor import ResourceMonitor
    with ResourceMonitor(output_dir) as mon:
        ... 你的任务 ...
    # 自动保存 output_dir/resource_usage.png

用法（手动）:
    mon = ResourceMonitor(output_dir, interval=2.0)
    mon.start()
    ... 你的任务 ...
    mon.stop()
"""
import subprocess
import threading
import time
from pathlib import Path


def _configure_matplotlib_fonts(matplotlib):
    matplotlib.rcParams['axes.unicode_minus'] = False
    candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans SC',
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS',
    ]
    current = list(matplotlib.rcParams.get('font.sans-serif', []))
    merged = []
    for name in candidates + current:
        if name and name not in merged:
            merged.append(name)
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = merged


class ResourceMonitor:
    def __init__(self, output_dir, interval: float = 0.5, gpu_ids: list[int] | None = None):
        """
        output_dir : 图片保存目录
        interval   : 采样间隔（秒），默认 2s
        gpu_ids    : 要监控的 GPU 编号列表；None = 自动检测所有 GPU
        """
        self.output_dir = Path(output_dir)
        self.interval   = interval
        self.gpu_ids    = gpu_ids

        self._timestamps: list[float] = []
        self._gpu_util:   dict[int, list[float]] = {}   # gpu_id → [%]
        self._gpu_mem:    dict[int, list[float]] = {}   # gpu_id → [GB]
        self._gpu_mem_total: dict[int, float]    = {}   # gpu_id → total GB
        self._cpu_pct:    list[float] = []

        self._stop_evt = threading.Event()
        self._thread:  threading.Thread | None = None
        self._t0: float = 0.0

    # ── 采样 ──────────────────────────────────────────────────────────────────

    def _detect_gpus(self) -> list[int]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip()]
        except Exception:
            return []

    def _sample_gpu(self) -> dict[int, tuple[float, float]]:
        """返回 {gpu_id: (util_pct, mem_used_gb)}"""
        ids = self.gpu_ids or list(self._gpu_util.keys())
        if not ids:
            return {}
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return {}
        result = {}
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                gid  = int(parts[0])
                util = float(parts[1])
                used = float(parts[2]) / 1024   # MB → GB
                total = float(parts[3]) / 1024
                if gid in ids:
                    result[gid] = (util, used)
                    self._gpu_mem_total[gid] = total
            except ValueError:
                pass
        return result

    def _sample_cpu(self) -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    def _loop(self):
        try:
            import psutil
            psutil.cpu_percent(interval=None)   # 第一次调用预热（返回值无意义）
        except Exception:
            pass

        while not self._stop_evt.wait(self.interval):
            t  = time.time() - self._t0
            gd = self._sample_gpu()
            cp = self._sample_cpu()

            self._timestamps.append(t)
            for gid in self._gpu_util:
                util, mem = gd.get(gid, (0.0, 0.0))
                self._gpu_util[gid].append(util)
                self._gpu_mem[gid].append(mem)
            self._cpu_pct.append(cp)

    # ── 控制 ──────────────────────────────────────────────────────────────────

    def start(self):
        ids = self.gpu_ids if self.gpu_ids is not None else self._detect_gpus()
        for gid in ids:
            self._gpu_util[gid] = []
            self._gpu_mem[gid]  = []

        self._t0 = time.time()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ResourceMonitor")
        self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
        self._save_plot()

    # ── 绘图 ──────────────────────────────────────────────────────────────────

    def _save_plot(self):
        if not self._timestamps:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            _configure_matplotlib_fonts(matplotlib)
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            print("[ResourceMonitor] matplotlib 未安装，跳过绘图")
            return

        ts   = self._timestamps
        n_gpu = len(self._gpu_util)
        n_rows = 2 + (1 if n_gpu > 0 else 0)   # GPU util / GPU mem / CPU

        # x 轴单位
        if ts[-1] > 3600:
            xs = [t / 3600 for t in ts];  xlabel = "时间 (小时)"
        elif ts[-1] > 60:
            xs = [t / 60   for t in ts];  xlabel = "时间 (分钟)"
        else:
            xs = ts;                       xlabel = "时间 (秒)"

        colors = plt.cm.tab10.colors
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows),
                                 sharex=True, tight_layout=True)
        if n_rows == 1:
            axes = [axes]

        ax_idx = 0

        # ── GPU 利用率 ────────────────────────────────────────────────────────
        if n_gpu > 0:
            ax = axes[ax_idx]; ax_idx += 1
            for i, (gid, vals) in enumerate(self._gpu_util.items()):
                ax.plot(xs, vals, label=f"GPU {gid}", color=colors[i % 10], linewidth=1.2)
            ax.set_ylabel("GPU 利用率 (%)")
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title("GPU 利用率")

        # ── GPU 显存 ──────────────────────────────────────────────────────────
        if n_gpu > 0:
            ax = axes[ax_idx]; ax_idx += 1
            for i, (gid, vals) in enumerate(self._gpu_mem.items()):
                total = self._gpu_mem_total.get(gid, 0)
                label = f"GPU {gid}" + (f" (/{total:.0f}GB)" if total else "")
                ax.plot(xs, vals, label=label, color=colors[i % 10], linewidth=1.2)
                if total:
                    ax.axhline(total, color=colors[i % 10], linestyle="--",
                               linewidth=0.7, alpha=0.5)
            ax.set_ylabel("显存占用 (GB)")
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title("GPU 显存")

        # ── CPU ───────────────────────────────────────────────────────────────
        ax = axes[ax_idx]
        ax.plot(xs, self._cpu_pct, color="steelblue", linewidth=1.2, label="CPU")
        ax.set_ylabel("CPU 使用率 (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_title("CPU 使用率")

        duration = ts[-1]
        if duration > 3600:
            dur_str = f"{duration/3600:.1f}h"
        elif duration > 60:
            dur_str = f"{duration/60:.1f}min"
        else:
            dur_str = f"{duration:.0f}s"
        fig.suptitle(f"资源占用  (采样间隔 {self.interval}s，总时长 {dur_str})",
                     fontsize=11, y=1.01)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "resource_usage.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[monitor] 资源图 → {out}")

    # ── 上下文管理器 ──────────────────────────────────────────────────────────

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
