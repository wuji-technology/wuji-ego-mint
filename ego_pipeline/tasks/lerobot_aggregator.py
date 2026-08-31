
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_SCRIPT = PROJECT_ROOT / 'tools' / 'aggregate_lerobot_v3.py'


class IncrementalAggregator:
    def __init__(
        self,
        output_root: Path,
        video_size_mb: int = 500,
        data_size_mb:  int = 100,
        extra_args:   list[str] | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self._root   = Path(output_root)
        self._target = self._root / 'lerobot_v3'



        self._log_dir = Path(log_dir) if log_dir is not None else self._root
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._video_size_mb = video_size_mb
        self._data_size_mb  = data_size_mb
        self._extra_args    = list(extra_args or [])
        self._pending = threading.Event()
        self._stop    = threading.Event()
        self._last_returncode: int | None = None
        self._thread  = threading.Thread(
            target=self._loop, name='lerobot-aggregator', daemon=True)
        self._thread.start()

    @property
    def last_returncode(self) -> int | None:
        """Internal helper."""
        return self._last_returncode

    def notify(self, parent_work: Path | str | None = None) -> None:
        """Internal helper."""
        if self._stop.is_set():
            return
        self._pending.set()

    def shutdown(self, wait: bool = True) -> None:
        """Internal helper."""
        self._stop.set()
        self._pending.set()
        if wait:
            self._thread.join()

    def _loop(self) -> None:
        while True:
            self._pending.wait()

            self._pending.clear()
            if self._stop.is_set() and not self._has_any_input():
                return

            self._run_once(validate=self._stop.is_set())
            if self._stop.is_set() and not self._pending.is_set():
                return

    def _has_any_input(self) -> bool:
        try:
            return any(self._root.rglob('meta/info.json'))
        except OSError:
            return False

    def _run_once(self, validate: bool = False) -> None:
        if not self._has_any_input():
            print('[pipeline]')
            return
        if not AGGREGATE_SCRIPT.exists():
            print(f'[pipeline]  {AGGREGATE_SCRIPT}.')
            return
        cmd = [
            sys.executable, str(AGGREGATE_SCRIPT),
            '--input',  str(self._root),
            '--output', str(self._target),
            '--append',
            '--video_files_size_mb', str(self._video_size_mb),
            '--data_files_size_mb',  str(self._data_size_mb),
            *self._extra_args,
        ]
        if not validate:
            cmd.append('--skip_validate')
        log_path = self._log_dir / 'aggregate_lerobot.log'
        try:
            with open(log_path, 'ab') as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            print(f'[pipeline]  {exc}.')
            return
        print(f'[pipeline]  {proc.pid}; {self._target}.')
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        self._last_returncode = rc
        if rc == 0:
            print(f'[pipeline]  {self._target}.')
        else:
            print(f'[pipeline]  {rc}; {log_path}.')
