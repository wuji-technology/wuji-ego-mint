from __future__ import annotations

import time
from pathlib import Path


def prefetch_megasam_alignment(
    image_dir: str,
    scene_name: str,
    output_dir: str | Path,
    focal_px: float,
    depth_base_dir: str | None = None,
) -> dict:
    from ego_pipeline.backends.megasam import compute_depth_alignment

    output_dir = Path(output_dir)
    scene_name = Path(scene_name).name
    if depth_base_dir is not None:
        _depth_base = Path(depth_base_dir)
    else:
        _depth_base = output_dir / 'depth'

    mono_outdir   = str(_depth_base / 'Depth-Anything' / scene_name)
    metric_outdir = str(_depth_base / 'UniDepth')

    return compute_depth_alignment(image_dir, mono_outdir, metric_outdir, scene_name, focal_px)


def run_megasam_step(
    image_dir: str,
    scene_name: str,
    output_dir: str | Path,
    focal_px: float,
    depth_base_dir: str,
    ba_steps1: int = 10,
    ba_steps2: int = 20,
    ba_steps3: int = 15,
    _precomputed: dict | None = None,
    _sub_events:  list | None = None,
    return_dense: bool = True,
) -> tuple[object, object, tuple[int, int] | None, float]:
    from ego_pipeline.backends.megasam import run as megasam_run

    output_dir = Path(output_dir)
    scene_name = Path(scene_name).name
    out_path   = str(output_dir.resolve() / f'{scene_name}.npz')

    existing = Path(out_path)
    if existing.exists():
        import numpy as np
        with np.load(existing) as data:
            if 'cam_c2w' not in data:
                raise RuntimeError(f'[pipeline]  {existing}.')
            cam_c2w = data['cam_c2w'].copy()
            K = data['K'].copy() if 'K' in data else None
            if 'slam_hw' in data.files:
                hw = data['slam_hw']
                slam_hw = (int(hw[0]), int(hw[1]))
            elif 'images' in data.files:
                slam_hw = (int(data['images'].shape[1]), int(data['images'].shape[2]))
            elif 'depths' in data.files:
                slam_hw = (int(data['depths'].shape[1]), int(data['depths'].shape[2]))
            else:
                slam_hw = None
        fx = float(K[0, 0]) if K is not None else float(focal_px)
        fy = float(K[1, 1]) if K is not None else float(focal_px)
        print(f'[pipeline]  {scene_name}; {cam_c2w.shape}; {fx:.1f}; {fy:.1f}.')
        return cam_c2w, K, slam_hw, 0.0

    t = time.perf_counter()
    result = megasam_run(
        image_dir=str(image_dir),
        scene_name=scene_name,
        output_path=out_path,
        depth_base_dir=depth_base_dir,
        focal_px=focal_px,
        ba_steps1=ba_steps1, ba_steps2=ba_steps2, ba_steps3=ba_steps3,
        _precomputed=_precomputed,
        _sub_events=_sub_events,
        return_dense=return_dense,
    )

    cam_c2w = result['cam_c2w'].copy()
    K = result['K']
    if 'slam_hw' in result:
        slam_hw = tuple(int(value) for value in result['slam_hw'])
    elif 'images' in result:
        slam_hw = (int(result['images'].shape[1]), int(result['images'].shape[2]))
    elif 'depths' in result:
        slam_hw = (int(result['depths'].shape[1]), int(result['depths'].shape[2]))
    else:
        slam_hw = None
    print(f'[megasam] {scene_name}  shape={cam_c2w.shape}  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}')
    del result

    elapsed = time.perf_counter() - t


    return cam_c2w, K, slam_hw, elapsed
