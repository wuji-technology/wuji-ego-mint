from __future__ import annotations

import time
from pathlib import Path


def save_raw_result(
    pred_result: list,
    result_dir: str | Path,
    scene_name: str,
) -> tuple[str, float]:
    import joblib

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    pth_path = str(result_dir / f'{Path(scene_name).name}.pth')

    t = time.perf_counter()
    joblib.dump(pred_result, pth_path)
    elapsed = time.perf_counter() - t

    pred_trans, _, _, _, pred_valid = pred_result
    T = pred_trans.shape[1]
    print(f'[pipeline]  {scene_name}; {T}; {pth_path}.')
    for i, name in enumerate(['[pipeline]', '[pipeline]']):
        vr = float(pred_valid[i].mean()) * 100
        x  = pred_trans[i, :, 0]
        print(f'[pipeline]  {name}; {vr:.1f}; {float(x.min()):.3f}; {float(x.max()):.3f}.')

    return pth_path, elapsed


def clean_and_save(
    pred_result: list,
    result_dir: str | Path,
    scene_name: str,
    cam_c2w=None,
) -> tuple[str, float]:
    import joblib
    from ego_pipeline.data_cleaning.final_clean import final_clean

    result_dir   = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = str(result_dir / f'{Path(scene_name).name}_cleaned.pth')

    t = time.perf_counter()
    cleaned = final_clean(pred_result, cam_c2w=cam_c2w)
    joblib.dump(cleaned, cleaned_path)
    elapsed = time.perf_counter() - t

    print(f'[pipeline]  {scene_name}; {cleaned_path}; {elapsed:.2f}.')
    return cleaned_path, elapsed
