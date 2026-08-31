"""Internal helper."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import ray


def _clip_identity(item: dict) -> tuple[str, int]:
    """Return (parent_scene, clip_idx) used by manifest callbacks."""
    parent = item.get('_parent') if isinstance(item, dict) else None
    if parent:
        return parent.get('scene') or item.get('scene', ''), int(parent.get('clip_idx', 0))
    return item.get('scene', ''), int(item.get('clip_idx', 0))


@ray.remote
def _cpu_finalize_and_post(item: dict) -> dict:
    from steps.cpu.save_ops import save_raw_result, clean_and_save
    from steps.cpu.result_sidecars import write_lerobot3_sidecars

    scene    = item['scene']
    work_dir = Path(item['work_dir'])
    parent_scene, clip_idx = _clip_identity(item)

    base = {
        'video':        item['video'],
        'scene':        scene,
        'parent_scene': parent_scene,
        'clip_idx':     clip_idx,
        'work_dir':     str(work_dir),
        'timings':      [],
        't_pre':        item.get('t_pre', 0.0),
        't_gpu':        item.get('t_geo', 0.0) + item.get('t_moge', 0.0)
                        + item.get('t_slam', 0.0),
    }

    if item.get('error'):
        return {**base, 'pth': None, 'cleaned': None, 'error': item['error']}

    try:
        pred_result  = item['pred_result']
        result_dir   = work_dir / 'result'
        t = time.perf_counter()
        pth_path,     _ = save_raw_result(pred_result, result_dir, scene)
        cleaned_path, _ = clean_and_save(pred_result, result_dir, scene,
                                         cam_c2w=item.get('cam_c2w'))
        sidecar = write_lerobot3_sidecars(
            pred_result=pred_result,
            result_dir=result_dir,
            scene_name=scene,
            source_video=item['video'],
            work_dir=work_dir,
            cam_c2w=item.get('cam_c2w'),
            K=item.get('K'),
            slam_hw=item.get('slam_hw'),
            fps=item.get('fps'),
            raw_pth_path=pth_path,
            cleaned_pth_path=cleaned_path,
            label_dir=item.get('label_dir'),
        )
        elapsed = time.perf_counter() - t
        print(f'[pipeline]  {scene}; {elapsed:.1f}; {cleaned_path}.')
        return {**base, 'pth': pth_path, 'cleaned': cleaned_path,
                'lerobot3_manifest': sidecar.get('result_dir') and str(result_dir / 'lerobot3_manifest.json'),
                't_post': elapsed, 'total_s': base['t_gpu'] + elapsed, 'error': None}
    except Exception:
        err = traceback.format_exc()
        print(f'[pipeline]  {scene}; {err.splitlines()[-1]}.')
        return {**base, 'pth': None, 'cleaned': None, 'error': err}


