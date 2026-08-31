

def run(
    gpu_id: int,
    vitra_dir: str,
    video_path: str,
    img_focal: float,
    image_dir,
    result_queue,
    log_path=None,
) -> None:
    import os
    import sys


    if log_path is not None:
        _log_f = open(log_path, 'w', buffering=1)
        os.dup2(_log_f.fileno(), 1)   # stdout
        os.dup2(_log_f.fileno(), 2)   # stderr
        sys.stdout = _log_f
        sys.stderr = _log_f

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, vitra_dir)

    try:
        import time as _tm
        import torch
        from ego_pipeline.backends.hawor_no_filler import load_hawor_model, run_stage12


        _t0_subprocess = _tm.perf_counter()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[hawor12_worker] GPU{gpu_id}  device={device}', flush=True)


        _t_load = _tm.perf_counter()
        dt_spawn_init = _t_load - _t0_subprocess
        model = load_hawor_model(device=device)
        dt_model_load = _tm.perf_counter() - _t_load


        _t_infer = _tm.perf_counter()
        detect_meta, fc_map, _ = run_stage12(model, video_path, img_focal, image_dir)
        dt_infer = _tm.perf_counter() - _t_infer

        del model
        result_queue.put(('ok', detect_meta, fc_map, dt_spawn_init, dt_model_load, dt_infer))
        print(f'[hawor12_worker] done  spawn_init={dt_spawn_init:.1f}s  '
              f'model_load={dt_model_load:.1f}s  infer={dt_infer:.1f}s', flush=True)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        result_queue.put(('error', tb))
