
import json
import os
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import psutil


_MIB = 1024 * 1024



_WORKER_RE = re.compile(r'ray::(\w+Worker)\.run_loop')


def _scan_worker_pids() -> dict[int, str]:
    """Internal helper."""
    out: dict[int, str] = {}
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = p.info['cmdline']
            if not cmd:
                continue
            title = cmd[0] if len(cmd) == 1 else ' '.join(cmd)
            m = _WORKER_RE.search(title)
            if m:
                out[p.info['pid']] = m.group(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return out


def _read_worker_rss(pid_labels: dict[int, str]) -> dict[str, int]:
    """Internal helper."""
    agg: dict[str, int] = {}
    dead: list[int] = []
    for pid, label in pid_labels.items():
        try:
            rss = psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            dead.append(pid)
            continue
        agg[label] = agg.get(label, 0) + (rss >> 20)
    for pid in dead:
        pid_labels.pop(pid, None)
    return agg


def _disp_w(s: str) -> int:
    """Internal helper."""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _pad(s: str, width: int, align: str = '<') -> str:
    """Internal helper."""
    n = width - _disp_w(s)
    if n <= 0:
        return s
    if align == '>':
        return ' ' * n + s
    if align == '^':
        left = n // 2
        return ' ' * left + s + ' ' * (n - left)
    return s + ' ' * n

try:
    import pynvml as _nvml
    _nvml.nvmlInit()
    _N_GPU: int = _nvml.nvmlDeviceGetCount()
    _HAS_GPU = _N_GPU > 0
except Exception:
    _HAS_GPU = False
    _N_GPU   = 0


def _sample_gpu() -> list[dict]:
    out = []
    for i in range(_N_GPU):
        h    = _nvml.nvmlDeviceGetHandleByIndex(i)
        util = _nvml.nvmlDeviceGetUtilizationRates(h)
        mem  = _nvml.nvmlDeviceGetMemoryInfo(h)
        temp = _nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU)
        out.append({
            'id':        i,
            'util':      util.gpu,
            'mem_used':  mem.used  >> 20,
            'mem_total': mem.total >> 20,
            'temp':      temp,
        })
    return out


def _sample_mem() -> dict:
    """Internal helper."""
    try:
        vm = psutil.virtual_memory()
        return {
            'used_mib':  vm.used  >> 20,
            'total_mib': vm.total >> 20,
            'percent':   round(vm.percent, 1),
        }
    except Exception:
        return {'used_mib': 0, 'total_mib': 0, 'percent': 0.0}


def _disk_io_counters():
    try:
        return psutil.disk_io_counters(nowrap=True)
    except TypeError:
        return psutil.disk_io_counters()
    except Exception:
        return None


def _net_io_counters():
    try:
        return psutil.net_io_counters(nowrap=True)
    except TypeError:
        return psutil.net_io_counters()
    except Exception:
        return None


def _counter_delta(now, prev, name: str) -> int:
    return max(0, int(getattr(now, name, 0) or 0) - int(getattr(prev, name, 0) or 0))


def _sample_disk_io(now, prev, dt: float) -> dict:
    if now is None or prev is None:
        return {
            'read_mib_s': 0.0,
            'write_mib_s': 0.0,
            'read_iops': 0.0,
            'write_iops': 0.0,
            'busy_pct': 0.0,
        }

    read_bytes = _counter_delta(now, prev, 'read_bytes')
    write_bytes = _counter_delta(now, prev, 'write_bytes')
    read_count = _counter_delta(now, prev, 'read_count')
    write_count = _counter_delta(now, prev, 'write_count')
    busy_ms = _counter_delta(now, prev, 'busy_time')

    return {
        'read_mib_s': round(read_bytes / dt / _MIB, 2),
        'write_mib_s': round(write_bytes / dt / _MIB, 2),
        'read_iops': round(read_count / dt, 1),
        'write_iops': round(write_count / dt, 1),

        'busy_pct': round(busy_ms / (dt * 10.0), 1),
    }


def _sample_net_io(now, prev, dt: float) -> dict:
    if now is None or prev is None:
        return {'recv_mib_s': 0.0, 'sent_mib_s': 0.0}

    recv_bytes = _counter_delta(now, prev, 'bytes_recv')
    sent_bytes = _counter_delta(now, prev, 'bytes_sent')
    return {
        'recv_mib_s': round(recv_bytes / dt / _MIB, 2),
        'sent_mib_s': round(sent_bytes / dt / _MIB, 2),
    }




_DRAW_TASK_BARS = os.environ.get('MINT_MONITOR_TASK_BARS', '0').lower() not in (
    '0', '', 'false', 'off', 'no')


_TASK_STYLE: dict[str, tuple[str, str]] = {
    'GeoCalib':   ('#C3FAE8', '#0CA678'),
    'MoGe':       ('#FFE8CC', '#E8590C'),
    'MoGe-GeoIdle': ('#FFF4E6', '#F08C00'),
    'MoGe-HaWoRIdle': ('#FFF4E6', '#E67700'),
    'HaWoR':      ('#E5DBFF', '#7048E8'),
    'SLAM':       ('#FFE3E3', '#C92A2A'),
    'SLAM-Idle':  ('#FFC9C9', '#E03131'),

    'MoGe-Infer': ('#FFF4E6', '#F76707'),
    'MoGe-Flush': ('#FFD8A8', '#D9480F'),

    'HaWoR-S1':   ('#F3F0FF', '#9775FA'),
    'HaWoR-S2':   ('#D0BFFF', '#5F3DC4'),

    'SLAM-Align': ('#FFF0F6', '#D6336C'),
    'SLAM-Init':  ('#F3D9FA', '#862E9C'),
    'SLAM-Track': ('#FFC9C9', '#C92A2A'),
    'SLAM-BA':    ('#FFA8A8', '#A61E1E'),
    'SLAM-Final': ('#DEE2E6', '#495057'),
}


def _read_jsonl(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _plot(jsonl_path: Path, events: list[dict] | None = None, quiet: bool = False) -> None:
    """Internal helper."""
    _render(_read_jsonl(jsonl_path), jsonl_path.parent / 'usage.png', events, quiet)


def _render(
    rows: list[dict],
    out_path: Path,
    events: list[dict] | None = None,
    quiet: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        if not quiet:
            print('[pipeline]')
        return

    if not rows:
        return

    ts   = [r['t'] for r in rows]
    cpu  = [r['cpu'] for r in rows]
    mem  = [r.get('mem', {}) for r in rows]
    disk = [r.get('disk_io', {}) for r in rows]
    net  = [r.get('net_io', {}) for r in rows]
    n_gpu = len(rows[0]['gpus'])


    worker_labels: list[str] = []
    for r in rows:
        for k in (r.get('workers') or {}):
            if k not in worker_labels:
                worker_labels.append(k)
    has_workers = bool(worker_labels)

    def fmt_t(t: float) -> str:
        return f'{t/60:.1f}m' if t > 120 else f'{t:.0f}s'

    x_labels = [fmt_t(t) for t in ts]
    step = max(1, len(ts) // 10)
    xtick_pos = list(range(0, len(ts), step))


    n_rows = 4 + n_gpu * 2 + (1 if has_workers else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    GPU_COLORS = ['#4C8EFF', '#FF6B6B', '#51CF66', '#FFD43B',
                  '#F783AC', '#74C0FC', '#63E6BE', '#FFA94D']


    ax = axes[0]
    ax.plot(ts, cpu, color='#339AF0', linewidth=1.2, label='CPU avg')
    ax.fill_between(ts, cpu, alpha=0.15, color='#339AF0')
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title('Resource Usage', fontsize=11, fontweight='bold')




    ax = axes[1]
    _GIB = 1024.0
    mem_used  = [m.get('used_mib', 0) / _GIB for m in mem]
    mem_total = next((m.get('total_mib', 0) / _GIB for m in mem if m.get('total_mib')), 0.0)
    mem_pct   = [m.get('percent', 0.0) for m in mem]
    lo, hi = min(mem_used), max(mem_used)
    pad    = max((hi - lo) * 0.10, 0.5)
    y0, y1 = max(0.0, lo - pad), hi + pad
    total_lbl = f'  (total {mem_total:.0f} GiB)' if mem_total > 0 else ''
    ax.plot(ts, mem_used, color='#7950F2', linewidth=1.2,
            label=f'Used GiB{total_lbl}')
    ax.fill_between(ts, mem_used, y0, alpha=0.15, color='#7950F2')
    ax.set_ylim(y0, y1)

    if 0 < mem_total <= y1:
        ax.axhline(mem_total, color='#7950F2', linewidth=0.6, linestyle=':')
    ax.set_ylabel('RAM\n(GiB)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=8)

    ax_mempct = ax.twinx()
    ax_mempct.plot(ts, mem_pct, color='#868E96', linewidth=0.0, alpha=0.0)
    ax_mempct.set_ylabel('Used (%)')
    ax_mempct.set_ylim(100.0 * y0 / mem_total if mem_total > 0 else 0,
                       100.0 * y1 / mem_total if mem_total > 0 else 100)
    ax_mempct.tick_params(axis='y', labelsize=7)


    ax = axes[2]
    disk_read = [d.get('read_mib_s', d.get('read_mb_s', 0.0) / 1.048576) for d in disk]
    disk_write = [d.get('write_mib_s', d.get('write_mb_s', 0.0) / 1.048576) for d in disk]
    disk_busy = [d.get('busy_pct', 0.0) for d in disk]
    ax.plot(ts, disk_read, color='#2F9E44', linewidth=1.2, label='Read MiB/s')
    ax.plot(ts, disk_write, color='#F08C00', linewidth=1.2, label='Write MiB/s')
    ax.set_ylabel('Disk I/O\n(MiB/s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax_busy = ax.twinx()
    ax_busy.plot(ts, disk_busy, color='#868E96', linewidth=0.9, alpha=0.7, label='Busy %')
    ax_busy.set_ylabel('Busy (%)')
    ax_busy.set_ylim(bottom=0)
    ax_busy.tick_params(axis='y', labelsize=7)


    ax = axes[3]
    net_recv = [n.get('recv_mib_s', n.get('recv_mb_s', 0.0) / 1.048576) for n in net]
    net_sent = [n.get('sent_mib_s', n.get('sent_mb_s', 0.0) / 1.048576) for n in net]
    ax.plot(ts, net_recv, color='#15AABF', linewidth=1.2, label='Recv MiB/s')
    ax.plot(ts, net_sent, color='#BE4BDB', linewidth=1.2, label='Sent MiB/s')
    ax.set_ylabel('Network\n(MiB/s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)

    if n_gpu > 0:
        gpu_util  = [[r['gpus'][i]['util']    for r in rows] for i in range(n_gpu)]
        gpu_mem   = [[r['gpus'][i]['mem_used'] for r in rows] for i in range(n_gpu)]
        gpu_total = [rows[0]['gpus'][i]['mem_total'] for i in range(n_gpu)]

        for i in range(n_gpu):
            c = GPU_COLORS[i % len(GPU_COLORS)]


            ax_util = axes[4 + i * 2]
            ax_util.plot(ts, gpu_util[i], color=c, linewidth=1.2)
            ax_util.fill_between(ts, gpu_util[i], alpha=0.12, color=c)
            ax_util.set_ylabel(f'GPU {i}\nUtil (%)')
            ax_util.set_ylim(0, 105)
            ax_util.yaxis.set_major_locator(ticker.MultipleLocator(25))
            ax_util.grid(True, linestyle='--', alpha=0.4)


            ax_mem = axes[5 + i * 2]
            ax_mem.plot(ts, gpu_mem[i], color=c, linewidth=1.2)
            ax_mem.fill_between(ts, gpu_mem[i], alpha=0.12, color=c)
            ax_mem.axhline(gpu_total[i], color=c, linewidth=0.6, linestyle=':',
                           label=f'Total {gpu_total[i]} MiB')
            ax_mem.set_ylabel(f'GPU {i}\nMem (MiB)')
            ax_mem.legend(loc='upper right', fontsize=7)
            ax_mem.grid(True, linestyle='--', alpha=0.4)





            if events and _DRAW_TASK_BARS:
                gpu_evs = [e for e in events if e['gpu_id'] == i]
                total_dur = (ts[-1] - ts[0]) if len(ts) > 1 else 1.0

                task_row: dict[str, int] = {}
                task_count: dict[str, int] = {}
                for ev in gpu_evs:
                    task_row.setdefault(ev['task'], len(task_row))
                    task_count[ev['task']] = task_count.get(ev['task'], 0) + 1

                n_rows = max(1, len(task_row))
                row_step = min(0.14, 0.90 / max(n_rows, 4))


                labeled: set[str] = set()

                for ev in gpu_evs:
                    t_s, t_e = ev['t_start'], ev['t_end']
                    task     = ev['task']
                    bg_c, tc = _TASK_STYLE.get(task, ('#EEEEEE', '#333333'))
                    k     = task_row[task]
                    y_top = 0.97 - k * row_step
                    dur   = max(0.0, t_e - t_s)

                    show_time_labels = task_count[task] <= 2
                    show_center_label = (
                        task not in labeled
                        or (total_dur > 0 and dur / total_dur > 0.03)
                    )

                    for ax_ann in (ax_util, ax_mem):
                        trans = ax_ann.get_xaxis_transform()

                        ax_ann.axvspan(t_s, t_e, alpha=0.18, color=bg_c, zorder=0)

                        ax_ann.axvline(t_s, color=tc, lw=0.8, ls='--', alpha=0.65, zorder=2)
                        ax_ann.axvline(t_e, color=tc, lw=0.8, ls='--', alpha=0.65, zorder=2)
                        if show_time_labels:
                            ax_ann.text(t_s, y_top - 0.07,
                                        f'[pipeline]  {fmt_t(t_s)}.', ha='left', va='top',
                                        fontsize=5.5, color=tc, transform=trans, zorder=5)
                            ax_ann.text(t_e, y_top - 0.07,
                                        f'[pipeline]  {fmt_t(t_e)}.', ha='right', va='top',
                                        fontsize=5.5, color=tc, transform=trans, zorder=5)
                        if show_center_label:
                            ax_ann.text((t_s + t_e) / 2, y_top, task,
                                        ha='center', va='top', fontsize=6.5,
                                        color=tc, fontweight='bold',
                                        transform=trans, zorder=5,
                                        bbox=dict(boxstyle='round,pad=0.18',
                                                  fc='white', ec=tc, alpha=0.85, lw=0.5))
                    labeled.add(task)




    if has_workers:
        ax_w = axes[4 + n_gpu * 2]
        _GIB = 1024.0
        WORKER_COLOR = {
            'GeoCalibWorker': '#0CA678',
            'MoGeWorker':     '#E8590C',
            'HaWorWorker':    '#7048E8',
            'SlamWorker':     '#C92A2A',
        }
        _FALLBACK = ['#1971C2', '#F08C00', '#2F9E44', '#AE3EC9', '#0C8599']
        all_vals: list[float] = []
        for j, lbl in enumerate(worker_labels):
            ys = []
            for r in rows:
                v = (r.get('workers') or {}).get(lbl)
                ys.append(v / _GIB if v is not None else float('nan'))
            all_vals += [v for v in ys if v == v]
            last = next((v for v in reversed(ys) if v == v), None)
            peak = max((v for v in ys if v == v), default=None)
            c = WORKER_COLOR.get(lbl, _FALLBACK[j % len(_FALLBACK)])
            tag = lbl[:-6] if lbl.endswith('Worker') else lbl
            label = tag
            if last is not None:
                label += f'  {last:.1f} GiB (peak {peak:.1f})'
            ax_w.plot(ts, ys, color=c, linewidth=1.3, label=label)
        ax_w.set_ylabel('Worker RSS\n(GiB)')
        ax_w.grid(True, linestyle='--', alpha=0.4)
        ax_w.legend(loc='upper left', fontsize=7,
                    ncol=max(1, (len(worker_labels) + 1) // 2))
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad = max((hi - lo) * 0.10, 0.2)
            ax_w.set_ylim(max(0.0, lo - pad), hi + pad)


    xtick_vals   = [ts[i]      for i in xtick_pos]
    xtick_labels = [x_labels[i] for i in xtick_pos]
    for ax in axes:
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=30, ha='right')
        ax.tick_params(axis='x', labelbottom=True)
    axes[-1].set_xlabel('Time')

    fig.tight_layout()
    out = out_path


    tmp = out.with_suffix('.png.tmp')
    fig.savefig(tmp, dpi=150, bbox_inches='tight', format='png')
    plt.close(fig)
    tmp.replace(out)
    if not quiet:
        print(f'[pipeline]  {out}.')


def _print_gpu_summary(jsonl_path: Path, events: list[dict]) -> None:
    """Internal helper."""
    rows: list[dict] = []
    try:
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get('carry'):
                    continue
                rows.append(rec)
    except Exception:
        return
    if not rows:
        return

    n_gpu     = len(rows[0]['gpus'])
    ts        = [r['t'] for r in rows]
    total_dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0



    if n_gpu == 0:
        print('[pipeline]')
        return

    def _stats(gpu_id: int, t0: float, t1: float):
        if gpu_id < 0 or gpu_id >= n_gpu:
            return None
        utils = [r['gpus'][gpu_id]['util'] for r in rows if t0 <= r['t'] <= t1]
        if not utils:
            return None
        return (
            sum(utils) / len(utils),
            max(utils),
            100.0 * sum(1 for u in utils if u == 0) / len(utils),
        )

    def _ft(s: float) -> str:
        return f'{s/60:.1f}m' if s >= 120 else f'{s:.1f}s'

    W = 64
    line_sep = '[pipeline]' * W
    print(f'[pipeline]  {"═" * W}.')
    print(_pad('[pipeline]', W, '^'))
    print(f'[pipeline]  {"═" * W}.')

    cols = [
        ('GPU',    3,  '>'),
        ('[pipeline]',   11, '<'),
        ('[pipeline]',   7,  '>'),
        ('[pipeline]',   6,  '>'),
        ('[pipeline]',   6,  '>'),
        ('[pipeline]', 7,  '>'),
    ]
    header = '  ' + '  '.join(_pad(name, w, a) for name, w, a in cols)
    print(header)
    print(line_sep)

    def _fmt_row(gid_s: str, task: str, dur_s: str,
                 mean_s: str, peak_s: str, idle_s: str) -> str:
        vals = [gid_s, task, dur_s, mean_s, peak_s, idle_s]
        return '  ' + '  '.join(
            _pad(v, w, a) for v, (_, w, a) in zip(vals, cols)
        )


    if events and _DRAW_TASK_BARS:
        for ev in sorted(events, key=lambda e: (e['gpu_id'], e['t_start'])):
            gid  = ev['gpu_id']
            task = ev['task']
            dur  = ev['t_end'] - ev['t_start']
            st   = _stats(gid, ev['t_start'], ev['t_end'])
            if st is None:
                continue
            mean, peak, idle = st
            print(_fmt_row(str(gid), task, _ft(dur),
                           f'{mean:.1f}%', f'{peak:.0f}%', f'{idle:.1f}%'))
        print(line_sep)


    for gid in range(n_gpu):
        st = _stats(gid, ts[0], ts[-1])
        if st is None:
            continue
        mean, peak, idle = st
        print(_fmt_row(str(gid), '[pipeline]', _ft(total_dur),
                       f'{mean:.1f}%', f'{peak:.0f}%', f'{idle:.1f}%'))

    print(f'[pipeline]  {"═" * W}.')


class ResourceMonitor:


    PLOT_INTERVAL = 30.0




    SAMPLE_SCHEDULE: tuple[tuple[float | None, float], ...] = (
        (600.0,  0.05),
        (1800.0, 0.25),
        (None,   1.0),
    )



    MAX_PLOT_POINTS = 6000

    def __init__(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        self._path     = save_dir / 'stats.jsonl'
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._loop, daemon=True, name='res-monitor')
        self._plot_th  = threading.Thread(target=self._plot_loop, daemon=True, name='res-monitor-plot')
        self._t0       = time.time()
        self._events:  list[dict] = []
        self._ev_lock  = threading.Lock()
        self._stopped  = False
        self._stop_lock = threading.Lock()
        psutil.cpu_percent(percpu=True)
        self._last_t = None
        self._last_disk = None
        self._last_net = None



        self._worker_pids: dict[int, str] = {}
        self._last_worker_scan = 0.0



        self._series: list[dict] = []
        self._series_lock = threading.Lock()
        self._sample_n = 0
        self._keep_every = 1

    def _sample_interval(self, elapsed: float) -> float:
        """Internal helper."""
        for upper, interval in self.SAMPLE_SCHEDULE:
            if upper is None or elapsed < upper:
                return interval
        return self.SAMPLE_SCHEDULE[-1][1]

    def _append_plot_series(self, record: dict) -> None:
        self._sample_n += 1
        if self._sample_n % self._keep_every != 0:
            return
        with self._series_lock:
            self._series.append(record)
            if len(self._series) > self.MAX_PLOT_POINTS:
                self._series = self._series[::2]
                self._keep_every *= 2

    def log_event(
        self,
        gpu_ids: 'int | list[int]',
        task:    str,
        t_start: float,
        t_end:   float,
    ) -> None:
        """Internal helper."""
        ids = [gpu_ids] if isinstance(gpu_ids, int) else list(gpu_ids)
        with self._ev_lock:
            for gid in ids:
                self._events.append(
                    {'gpu_id': gid, 'task': task, 't_start': t_start, 't_end': t_end}
                )

    def start(self) -> None:
        self._thread.start()
        self._plot_th.start()
        print(f'[pipeline]  {self._path.parent}; {self.PLOT_INTERVAL:.0f}.')

    def stop(self, *, final_plot: bool = True) -> None:

        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop_evt.set()
        self._thread.join(timeout=5)
        self._plot_th.join(timeout=5)
        with self._ev_lock:
            events = list(self._events)
        _print_gpu_summary(self._path, events)
        if not final_plot:
            print('[pipeline]')
            return
        print(f'[pipeline]')
        try:
            _plot(self._path, events or None)
        except Exception as e:
            print(f'[pipeline]  {e}.')

    def _carry_prev_tail(self) -> list[dict]:
        if os.environ.get('MINT_MONITOR_CARRY_PREV', '1') == '0':
            return []
        cur_dir  = self._path.parent
        logs_dir = cur_dir.parent
        if not logs_dir.is_dir():
            return []

        prevs = [d for d in logs_dir.iterdir()
                 if d.is_dir() and d.name < cur_dir.name and (d / 'stats.jsonl').exists()]
        if not prevs:
            return []
        prev = max(prevs, key=lambda d: d.name)
        prev_start = datetime.strptime(prev.name, '%Y%m%d_%H%M%S').timestamp()
        try:
            tail_s = float(os.environ.get('MINT_MONITOR_CARRY_TAIL_S', '120'))
        except ValueError:
            tail_s = 120.0
        try:
            max_gap = float(os.environ.get('MINT_MONITOR_CARRY_MAX_GAP_S', '3600'))
        except ValueError:
            max_gap = 3600.0

        samples: list[dict] = []
        with (prev / 'stats.jsonl').open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get('carry') or 't' not in rec:
                    continue
                samples.append(rec)
        if not samples:
            return []
        last_t = samples[-1]['t']
        keep = [s for s in samples if s['t'] >= last_t - tail_s]
        if not keep:
            return []

        cur_n = len(_sample_gpu()) if _HAS_GPU else 0
        if len(keep[0].get('gpus') or []) != cur_n:
            return []
        gap = self._t0 - (prev_start + last_t)
        if gap < 0 or gap > max_gap:
            return []

        out: list[dict] = []
        for s in keep:
            r = dict(s)
            r['t'] = round((prev_start + s['t']) - self._t0, 1)
            r['carry'] = 1
            out.append(r)
        nan = float('nan')
        out.append({
            't': round(-gap / 2, 1),
            'cpu': nan,
            'mem': {'used_mib': nan, 'total_mib': 0, 'percent': nan},
            'disk_io': {'read_mib_s': nan, 'write_mib_s': nan, 'busy_pct': nan},
            'net_io': {'recv_mib_s': nan, 'sent_mib_s': nan},
            'gpus': [{'util': nan, 'mem_used': nan,
                      'mem_total': keep[0]['gpus'][i].get('mem_total', 0)}
                     for i in range(cur_n)],
            'workers': {},
            'carry': 1, 'gap': 1,
        })
        print(f'[pipeline]  {len(keep)}; {gap:.0f}.'
              f'[pipeline]  {prev.name}.', flush=True)
        return out

    def _loop(self) -> None:
        try:
            carry = self._carry_prev_tail()
        except Exception as e:
            carry = []
            print(f'[pipeline]  {e}.', flush=True)
        with self._path.open('w') as f:

            for rec in carry:
                f.write(json.dumps(rec) + '\n')
            if carry:
                f.flush()
                with self._series_lock:
                    self._series[:0] = carry
            while not self._stop_evt.is_set():
                now = time.time()
                disk_now = _disk_io_counters()
                net_now = _net_io_counters()
                dt = max(1e-6, now - self._last_t) if self._last_t is not None else 0.0
                per_cpu = psutil.cpu_percent(percpu=True)

                if now - self._last_worker_scan > 5.0:
                    self._worker_pids.update(_scan_worker_pids())
                    self._last_worker_scan = now


                record  = {
                    't':    round(now - self._t0, 1),
                    'cpu':  round(sum(per_cpu) / len(per_cpu), 1),
                    'mem': _sample_mem(),
                    'workers': _read_worker_rss(self._worker_pids),
                    'disk_io': _sample_disk_io(disk_now, self._last_disk, dt),
                    'net_io': _sample_net_io(net_now, self._last_net, dt),
                    'gpus': _sample_gpu() if _HAS_GPU else [],
                }
                self._last_t = now
                self._last_disk = disk_now
                self._last_net = net_now
                f.write(json.dumps(record) + '\n')
                f.flush()
                self._append_plot_series(record)
                self._stop_evt.wait(self._sample_interval(now - self._t0))

    def _plot_loop(self) -> None:
        out = self._path.parent / 'usage.png'
        while not self._stop_evt.wait(self.PLOT_INTERVAL):
            try:
                with self._series_lock:
                    rows = list(self._series)
                with self._ev_lock:
                    events = list(self._events)
                if rows:
                    _render(rows, out, events or None, quiet=True)
            except Exception as e:
                print(f'[pipeline]  {e}.')


def start(save_dir: 'Path | str') -> ResourceMonitor:
    m = ResourceMonitor(Path(save_dir))
    m.start()
    return m


def stop(monitor: ResourceMonitor, *, final_plot: bool = True) -> None:
    monitor.stop(final_plot=final_plot)


def _cli() -> None:
    import sys
    if len(sys.argv) < 2:
        print('[pipeline]')
        sys.exit(2)
    target = Path(sys.argv[1])
    if target.is_dir():
        target = target / 'stats.jsonl'
    if not target.exists():
        print(f'[pipeline]  {target}.')
        sys.exit(1)
    _plot(target)


if __name__ == '__main__':
    _cli()
