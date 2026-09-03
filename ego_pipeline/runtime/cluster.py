
import os
from pathlib import Path

import ray


def init(num_gpus: int | None = None, log: bool = False, **kwargs) -> int:
    tmp_dir = os.environ.get('MINT_RAY_TMP') or '/tmp/ray'
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    init_kwargs: dict = {
        'ignore_reinit_error': True,
        'log_to_driver':       log,
        '_temp_dir':           tmp_dir,
    }
    if num_gpus is not None:
        init_kwargs['num_gpus'] = num_gpus
    init_kwargs.update(kwargs)

    ray.init(**init_kwargs)

    n = gpu_count()
    if n == 0:
        print('[pipeline]')
    else:
        print(f'[pipeline]  {n}.')
    return n


def shutdown() -> None:
    """Internal helper."""
    ray.shutdown()
    print('[pipeline]')


def gpu_count() -> int:
    """Internal helper."""
    return int(ray.cluster_resources().get('GPU', 0))
