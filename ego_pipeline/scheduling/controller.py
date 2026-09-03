"""Runtime scheduler for the Ray video pipeline."""

from __future__ import annotations

from collections import deque
from typing import Callable
import queue
import time

import ray

from tasks import (
    _cpu_finalize_and_post,
    extract_all_frames_step,
    extract_short_frames_step,
    link_clip_frames_step,
    preprocess_meta,
)

from .config import SchedulerConfig
from .pools import (
    promote_released_worker_to_slam,
    WorkerPool,
    release_workers,
    start_worker_pool,
)
from .queues import PipelineQueues, create_pipeline_queues, inject_sentinels
from .resources import build_schedule_plan, print_schedule_plan

from actors._utils import MallocTrimmer


class RuntimeScheduler:
    """Owns pipeline-level scheduling while actors/tasks own actual work."""

    def __init__(
        self,
        config: SchedulerConfig,
        monitor=None,
        on_video_done: Callable[[dict], None] | None = None,
        on_stage_done: Callable[[dict], None] | None = None,
    ):
        self.config = config
        self.monitor = monitor


        self.on_video_done = on_video_done
        self.on_stage_done = on_stage_done
        self.plan = build_schedule_plan(config.n_gpus)
        self.queues: PipelineQueues | None = None
        self.pool: WorkerPool | None = None
        self.t_geo_start: float | None = None
        self.t_geo_done: float | None = None
        self.t_moge_done: float | None = None
        self.t_haw_done: float | None = None
        self.t_slam_done: float | None = None
        self._open_videos = 0
        self._open_clip_credit = 0
        self._scene_open_credits: dict[str, int] = {}
        self._finished_scenes: set[str] = set()
        self._geo_sentinels_sent = False
        self._upstream_sentinels_sent = False
        self._slam_sentinels_sent = False


        self._drained_results: list[dict] = []
        self._drained_count: int = 0





        self._driver_trimmer = MallocTrimmer('driver')


        self._parent_total: dict[str, int] = {}
        self._parent_done: dict[str, int] = {}
        self._done_videos = 0
        self._total_clips_seen = 0

    def run(self, videos: list[str]) -> list[dict]:
        n = len(videos)
        sep = '=' * 60
        print_schedule_plan(self.plan, n, self.config.output_dir)

        self.queues = create_pipeline_queues()
        self.pool = start_worker_pool(self.plan, self.queues, self.config)
        t_wall = time.perf_counter()

        post_refs = self._run_sliding_window(videos)
        self._finish_gpu_stages()
        self._log_stage_events()
        all_results = self._collect_post_results(post_refs, n)

        elapsed = time.perf_counter() - t_wall
        self._print_summary(all_results, elapsed, sep)
        return all_results

    def _run_sliding_window(self, videos: list[str]) -> list:
        assert self.queues is not None
        assert self.pool is not None

        pending_videos = deque(videos)
        ready_meta = deque()
        meta_refs: list = []
        submitted_videos = 0
        total_items = 0
        done_items = 0
        post_refs: list = []
        post_submitted: set[str] = set()
        per_item_events: list = []
        last_health_check = 0.0
        last_wait_log = 0.0
        last_join_close_log = 0.0
        geo_remaining: list = []
        upstream_remaining: list = []
        ref_kind: dict = {}
        moge_n_done = 0
        haw_done_times: list = []

        def _known_items(pre: dict) -> int:
            if pre.get('is_long') and not pre.get('error'):
                return max(1, len(pre.get('clips', [])))
            return 1

        def _can_preload_meta() -> bool:
            if not pending_videos:
                return False
            reserved_videos = self._open_videos + len(ready_meta) + len(meta_refs)
            if reserved_videos >= self.config.max_open_videos:
                return False
            return True

        def _preload_meta_until_window() -> None:
            nonlocal submitted_videos
            while _can_preload_meta():
                video = pending_videos.popleft()
                meta_refs.append(preprocess_meta.remote(
                    video, self.config.output_dir, self.config.input_root))
                submitted_videos += 1

        def _can_open_pre(pre: dict) -> bool:
            if self._open_videos >= self.config.max_open_videos:
                return False
            n_items = _known_items(pre)
            if self._open_clip_credit == 0:
                return True
            return (
                self._open_clip_credit + n_items
                <= self.config.max_open_clip_credit
            )

        def _feed_ready_meta() -> None:
            nonlocal total_items
            while ready_meta and _can_open_pre(ready_meta[0]):
                pre = ready_meta.popleft()
                n_items = self._feed_preprocessed(pre)
                total_items += n_items
                self._open_videos += 1
                self._open_clip_credit += n_items
                self._scene_open_credits[pre['scene']] = n_items
                self._parent_total[pre['scene']] = n_items
                self._total_clips_seen += n_items
                print(
                    f'[Window] open {pre["scene"]} clips={n_items} '
                    f'open_videos={self._open_videos} '
                    f'open_clip_credit={self._open_clip_credit}'
                )

        def _try_get_post_item():
            try:
                return self.queues.q_post.get(block=False)
            except queue.Empty:
                return None
            except Exception as exc:
                if 'Empty' in exc.__class__.__name__:
                    return None
                raise

        def _drain_stage_events() -> None:
            self._drain_stage_events()

        def _submit_post_immediately(item: dict) -> None:


            scene = item.get('scene') if isinstance(item, dict) else None
            if not scene or scene in post_submitted:
                return
            post_submitted.add(scene)
            post_refs.append(_cpu_finalize_and_post.remote(item))
            print(f'[pipeline]  {scene}.')

        def _all_meta_fed() -> bool:
            return not pending_videos and not meta_refs and not ready_meta

        def _start_geo_finish_if_ready() -> None:
            nonlocal geo_remaining
            if self._geo_sentinels_sent or not _all_meta_fed():
                return
            inject_sentinels(self.queues.q_pre, self.plan.n_geo)
            self._geo_sentinels_sent = True
            geo_remaining = list(self.pool.geo_refs)
            print('[pipeline]')

        def _close_slam_queue() -> None:
            nonlocal last_join_close_log
            if self._slam_sentinels_sent:
                return
            emitted = ray.get(self.pool.join_store.flush_errors.remote(
                self.queues.q_slam))
            if emitted:
                print(f'[Pipeline] join_store flush error items={emitted}')
            pending_join = ray.get(self.pool.join_store.pending_count.remote())
            if pending_join:
                now = time.perf_counter()
                if now - last_join_close_log > 10.0:
                    print(f'[pipeline]'
                          f'join_pending={pending_join}')
                    last_join_close_log = now
                return
            inject_sentinels(self.queues.q_slam, len(self.pool.slam_refs))
            self._slam_sentinels_sent = True
            print(f'[pipeline]'
                  f'SLAM worker={len(self.pool.slam_refs)}')

        def _poll_finish_stages() -> None:
            nonlocal geo_remaining, upstream_remaining, ref_kind
            nonlocal moge_n_done, haw_done_times

            if self._geo_sentinels_sent and geo_remaining:
                done, geo_remaining = ray.wait(
                    geo_remaining, num_returns=len(geo_remaining), timeout=0)
                for ref in done:
                    ray.get(ref)
                if not geo_remaining and self.t_geo_done is None:
                    self.t_geo_done = time.time()
                    self._promote_upstream_to_tail_slam(
                        'Geo', self.pool.geo_ws[:self.plan.n_geo])
                    inject_sentinels(self.queues.q_moge, self.plan.n_moge)
                    inject_sentinels(self.queues.q_hawor, self.plan.n_hawor)
                    self._upstream_sentinels_sent = True
                    ref_kind = {ref: 'moge' for ref in self.pool.moge_refs}
                    ref_kind.update({ref: 'haw' for ref in self.pool.haw_refs})
                    upstream_remaining = list(ref_kind)
                    print('[pipeline]'
                          '[pipeline]')

            if self._upstream_sentinels_sent and upstream_remaining:
                done, upstream_remaining = ray.wait(
                    upstream_remaining,
                    num_returns=len(upstream_remaining),
                    timeout=0,
                )
                for ref in done:
                    kind = ref_kind[ref]
                    result = ray.get(ref)
                    if kind == 'moge':
                        moge_n_done += 1
                        if (moge_n_done == self.plan.n_moge
                                and self.t_moge_done is None):
                            self.t_moge_done = time.time()
                            self._promote_upstream_to_tail_slam(
                                'MoGe', self.pool.moge_ws[:self.plan.n_moge])
                            print('[pipeline]')
                    else:
                        haw_done_times.append(
                            result if isinstance(result, (int, float))
                            else time.time()
                        )
                        if (len(haw_done_times) == self.plan.n_hawor
                                and self.t_haw_done is None):
                            self.t_haw_done = max(haw_done_times)
                            self._promote_upstream_to_tail_slam(
                                'HaWoR', self.pool.haw_ws[:self.plan.n_hawor])
                            print('[pipeline]')

            if (self._upstream_sentinels_sent
                    and not upstream_remaining
                    and self.t_moge_done is not None
                    and self.t_haw_done is not None):
                _close_slam_queue()

        _preload_meta_until_window()

        while (
            done_items < total_items
            or pending_videos
            or meta_refs
            or ready_meta
            or not self._slam_sentinels_sent
        ):
            now = time.perf_counter()
            if now - last_health_check > 2.0:
                self._raise_if_worker_exited_early()
                last_health_check = now
            if now - last_wait_log > 30.0 and total_items:
                pending_join = self._join_pending_count()
                print(
                    f'[Window] waiting done_items={done_items}/{total_items} '
                    f'open_videos={self._open_videos} '
                    f'open_clip_credit={self._open_clip_credit} '
                    f'join_pending={pending_join}'
                )
                last_wait_log = now

            if meta_refs:
                done, meta_refs = ray.wait(meta_refs, num_returns=1, timeout=0.1)
                for ref in done:
                    pre = ray.get(ref)
                    ready_meta.append(pre)
                    if submitted_videos % 10 == 0 or submitted_videos == len(videos):
                        print(
                            f'[Window] meta_ready={len(ready_meta)} '
                            f'submitted={submitted_videos}/{len(videos)} '
                            f'open_videos={self._open_videos} '
                            f'open_clip_credit={self._open_clip_credit} '
                        )

            _feed_ready_meta()
            _drain_stage_events()
            _start_geo_finish_if_ready()
            _poll_finish_stages()
            self._drain_ready_post_refs(post_refs, len(videos))
            _drain_stage_events()

            if done_items < total_items:
                item = _try_get_post_item()
                if isinstance(item, dict):
                    slam_evts = item.pop('slam_events', None) or []
                    per_item_events.extend(slam_evts)
                    if len(per_item_events) >= 500:
                        self._log_per_item_events(per_item_events)
                        per_item_events.clear()
                    self._release_open_credit(item)
                    _submit_post_immediately(item)
                if item is not None:
                    done_items += 1
                    if done_items % 10 == 0 or done_items == total_items:
                        print(
                            f'[Window] done_items={done_items}/{total_items} '
                            f'open_videos={self._open_videos} '
                            f'open_clip_credit={self._open_clip_credit}'
                        )

            if (
                self._open_videos < self.config.low_open_videos
                or self._open_clip_credit < self.config.low_open_clip_credit
                or (not meta_refs and not ready_meta and self._open_videos == 0)
            ):
                _preload_meta_until_window()
                _feed_ready_meta()
                _drain_stage_events()
                _start_geo_finish_if_ready()
                _poll_finish_stages()

            if not meta_refs and done_items >= total_items and pending_videos:
                _preload_meta_until_window()
                _feed_ready_meta()
                _drain_stage_events()
                _start_geo_finish_if_ready()
                _poll_finish_stages()

            if (
                not meta_refs and not ready_meta and done_items >= total_items
                and not pending_videos and self._slam_sentinels_sent
            ):
                break

            time.sleep(0.01)

        if per_item_events:
            self._log_per_item_events(per_item_events)
            per_item_events.clear()
        return post_refs

    def _account_post_result(self, result: dict, total_videos: int) -> str:
        parent = result.get('parent_scene') or result.get('scene')
        if parent:
            self._parent_done[parent] = self._parent_done.get(parent, 0) + 1
            expected = self._parent_total.get(parent)
            if expected is not None and self._parent_done[parent] == expected:
                self._done_videos += 1
        done_clips = self._drained_count
        total_clips = max(self._total_clips_seen, done_clips)
        return (f'[{done_clips:>3}/{total_clips} clip | '
                f'[pipeline]  {self._done_videos:>2}; {total_videos}.')

    def _drain_stage_events(self) -> None:
        assert self.queues is not None
        if self.on_stage_done is None:
            return
        while True:
            try:
                event = self.queues.q_stage.get(block=False)
            except queue.Empty:
                return
            except Exception as exc:
                if 'Empty' in exc.__class__.__name__:
                    return
                raise
            if not isinstance(event, dict):
                continue
            try:
                self.on_stage_done(event)
            except Exception as e:
                scene = event.get('scene', '?')
                stage = event.get('stage', '?')
                print(f'[pipeline]  {scene}; {stage}; {e}.')

    def _join_pending_count(self) -> int | str:
        assert self.pool is not None
        try:
            return ray.get(self.pool.join_store.pending_count.remote(), timeout=1.0)
        except Exception:
            return '?'

    def _promote_upstream_to_tail_slam(self, source: str, workers: list) -> None:
        assert self.queues is not None
        assert self.pool is not None
        if not workers or self._slam_sentinels_sent:
            return
        if self.plan.upstream_frac is not None:
            release_workers(workers)
            print(f'[pipeline]  {source}.')
            return
        for w in workers:
            promote_released_worker_to_slam(
                self.pool, self.queues, self.config, w, source)
        print(f'[pipeline]  {source}.')

    def _raise_if_worker_exited_early(self) -> None:
        assert self.pool is not None
        refs = []
        if not self._geo_sentinels_sent:
            refs.extend(self.pool.geo_refs)
        if not self._upstream_sentinels_sent:
            refs.extend(self.pool.moge_refs)
            refs.extend(self.pool.haw_refs)
        if not self._slam_sentinels_sent:
            refs.extend(self.pool.slam_refs)
        if not refs:
            return
        done, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if not done:
            return
        try:
            ray.get(done[0])
        except Exception as exc:
            raise RuntimeError('[pipeline]') from exc
        raise RuntimeError('[pipeline]')

    def _feed_preprocessed(self, pre: dict) -> int:
        assert self.queues is not None

        if not pre.get('error'):
            if pre.get('is_long'):
                all_ref = extract_all_frames_step.remote(
                    pre['video'], pre['work_dir'],
                )
                for clip in pre['clips']:
                    clip['frames_ref'] = link_clip_frames_step.remote(
                        all_ref, pre['video'], clip, pre['work_dir'],
                        self.config.frame_stride,
                    )
            else:
                pre['frames_ref'] = extract_short_frames_step.remote(
                    pre['video'], pre['image_dir'], self.config.frame_stride,
                )

        resume_clip_stages = self.config.resume_clip_stages or {}
        pre['resume_clip_stages'] = resume_clip_stages.get(pre['scene'], {})


        pre['label_dir'] = self.config.label_dir

        if self.t_geo_start is None:
            self.t_geo_start = time.time()
        self.queues.q_pre.put(pre)

        if pre.get('is_long') and not pre.get('error'):
            return len(pre.get('clips', []))
        return 1

    def _release_open_credit(self, item: dict) -> None:
        parent = item.get('_parent') if isinstance(item, dict) else None
        scene = parent['scene'] if parent else item.get('scene')
        if not scene:
            return
        remaining = self._scene_open_credits.get(scene)
        if remaining is None:
            return
        remaining -= 1
        self._open_clip_credit = max(0, self._open_clip_credit - 1)
        if remaining <= 0:
            self._scene_open_credits.pop(scene, None)
            if scene not in self._finished_scenes:
                self._finished_scenes.add(scene)
                self._open_videos = max(0, self._open_videos - 1)
        else:
            self._scene_open_credits[scene] = remaining

    def _finish_gpu_stages(self) -> None:
        assert self.queues is not None
        assert self.pool is not None

        if not self._geo_sentinels_sent:
            inject_sentinels(self.queues.q_pre, self.plan.n_geo)
            self._geo_sentinels_sent = True
        if self.t_geo_done is None:
            ray.get(self.pool.geo_refs)
            self.t_geo_done = time.time()
            self._promote_upstream_to_tail_slam(
                'Geo', self.pool.geo_ws[:self.plan.n_geo])

        if not self._upstream_sentinels_sent:
            inject_sentinels(self.queues.q_moge, self.plan.n_moge)
            inject_sentinels(self.queues.q_hawor, self.plan.n_hawor)
            self._upstream_sentinels_sent = True

        if self.t_moge_done is None or self.t_haw_done is None:
            ref_kind: dict = {ref: 'moge' for ref in self.pool.moge_refs}
            ref_kind.update({ref: 'haw' for ref in self.pool.haw_refs})
            remaining = list(ref_kind)
            moge_n_done = 0
            t_haw_done_list: list = []

            while remaining:
                done, remaining = ray.wait(remaining, num_returns=1)
                ref = done[0]
                kind = ref_kind[ref]
                result = ray.get(ref)
                if kind == 'moge':
                    moge_n_done += 1
                    if (moge_n_done == self.plan.n_moge
                            and self.t_moge_done is None):
                        self.t_moge_done = time.time()
                        self._promote_upstream_to_tail_slam(
                            'MoGe', self.pool.moge_ws[:self.plan.n_moge])
                else:
                    t_haw_done_list.append(result)

            if self.t_haw_done is None:
                self.t_haw_done = (
                    max(t_haw_done_list) if t_haw_done_list
                    else self.t_moge_done
                )
                self._promote_upstream_to_tail_slam(
                    'HaWoR', self.pool.haw_ws[:self.plan.n_hawor])

        if not self._slam_sentinels_sent:
            while True:
                ray.get(self.pool.join_store.flush_errors.remote(self.queues.q_slam))
                pending_join = ray.get(self.pool.join_store.pending_count.remote())
                if not pending_join:
                    break
                print(f'[pipeline]'
                      f'join_pending={pending_join}')
                time.sleep(1.0)
            inject_sentinels(self.queues.q_slam, len(self.pool.slam_refs))
            self._slam_sentinels_sent = True

        ray.get(self.pool.slam_refs)
        self.t_slam_done = time.time()
        self._drain_stage_events()
        release_workers(self.pool.slam_ws)

    def _log_stage_events(self) -> None:
        if self.monitor is None:
            return
        assert self.t_geo_done is not None
        assert self.t_moge_done is not None
        assert self.t_haw_done is not None
        assert self.t_slam_done is not None

        t0 = self.monitor._t0
        t_geo_start_rel = (self.t_geo_start or self.t_geo_done) - t0
        self.monitor.log_event(
            self.plan.geo_gpu_ids, 'GeoCalib', t_geo_start_rel, self.t_geo_done - t0)
        self.monitor.log_event(
            self.plan.moge_gpu_ids, 'MoGe', t_geo_start_rel, self.t_moge_done - t0)
        self.monitor.log_event(
            self.plan.haw_gpu_ids, 'HaWoR', t_geo_start_rel, self.t_haw_done - t0)
        self.monitor.log_event(
            list(range(self.plan.n_gpus)), 'SLAM', self.t_geo_done - t0,
            self.t_slam_done - t0)

    def _log_per_item_events(self, per_item_events: list) -> None:
        if self.monitor is None or not per_item_events:
            return
        t0 = self.monitor._t0
        for ev in per_item_events:
            gid = ev.get('gpu_id', -1)
            if gid < 0:
                continue
            self.monitor.log_event(
                gid, ev['task'], ev['t_start'] - t0, ev['t_end'] - t0)

    def _drain_ready_post_refs(self, post_refs: list, video_count: int) -> None:
        if not post_refs:
            return
        done, _ = ray.wait(post_refs, num_returns=len(post_refs), timeout=0)
        if not done:
            return
        for ref in done:
            post_refs.remove(ref)
            result = ray.get(ref)
            self._drained_results.append(result)
            self._drained_count += 1
            self._driver_trimmer.tick()
            scene = result.get('scene', '?')
            tag = self._account_post_result(result, video_count)
            if result.get('error'):
                err_tail = (result['error'].splitlines() or [''])[-1][:60]
                print(f'[pipeline]  {tag}; {scene:<30}; {err_tail}.')
            else:
                print(f'[pipeline]  {tag}; {scene:<30}.'
                      f'[pipeline]  {result.get("cleaned", "")}.')
            if self.on_video_done is not None:
                try:
                    self.on_video_done(result)
                except Exception as e:
                    print(f'[pipeline]  {scene}; {e}.')
            self._drain_stage_events()

    def _collect_post_results(self, post_refs: list, video_count: int) -> list[dict]:

        all_results: list[dict] = list(self._drained_results)
        remaining_post = list(post_refs)
        done_n = len(all_results)

        while remaining_post:
            done, remaining_post = ray.wait(remaining_post, num_returns=1)
            result = ray.get(done[0])
            done_n += 1
            all_results.append(result)
            self._drained_results.append(result)
            self._drained_count += 1
            self._driver_trimmer.tick()
            scene = result.get('scene', '?')
            tag = self._account_post_result(result, video_count)
            if result.get('error'):
                print(f'[pipeline]  {tag}; {scene:<30}.'
                      f'{result["error"].splitlines()[-1][:60]}')
            else:
                print(f'[pipeline]  {tag}; {scene:<30}.'
                      f'[pipeline]  {result.get("cleaned", "")}.')
            if self.on_video_done is not None:
                try:
                    self.on_video_done(result)
                except Exception as e:
                    print(f'[pipeline]  {scene}; {e}.')
            self._drain_stage_events()
        return all_results

    @staticmethod
    def _print_summary(all_results: list[dict], elapsed: float, sep: str) -> None:
        ok = [r for r in all_results if not r.get('error')]
        fail = [r for r in all_results if r.get('error')]

        print(f'\n{sep}')
        print(f'[pipeline]  {len(ok)}; {len(fail)}; {elapsed:.1f}.')
        if fail:
            print('[pipeline]')
            print(fail[0].get('error', ''))
            if len(fail) > 1:
                for r in fail[1:]:
                    print(f'  {r.get("scene","?")}: '
                          f'{str(r["error"]).splitlines()[-1][:100]}')
        print(sep)
