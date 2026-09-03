"""CUDA cleanup helpers for Ray GPU actors."""

from __future__ import annotations


def configure_memory_cap(label: str, gpu: str = '?') -> None:
    """Cap PyTorch's per-process allocator so one module cannot exceed 24GB visible VRAM."""
    import os

    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        alloc_gb = float(os.environ.get('MINT_GPU_ALLOC_GB', '22'))
    except ValueError:
        return
    if alloc_gb <= 0:
        return

    try:
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        fraction = min(1.0, alloc_gb * (1024 ** 3) / float(total))
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f'[{label} GPU{gpu}] PyTorch allocator cap={alloc_gb:.1f}GB')
    except Exception:
        pass


def release_cuda_cache(label: str, gpu: str = '?', *, log: bool = False) -> None:
    """Return unused CUDA allocator blocks to the driver as aggressively as possible."""
    import gc

    gc.collect()
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    if log:
        print(f'[pipeline]  {label}; {gpu}.')


def move_to_cpu(obj) -> None:
    """Best-effort move for PyTorch modules and wrappers with a nested .model."""
    if obj is None:
        return
    try:
        if hasattr(obj, 'cpu'):
            obj.cpu()
            return
    except Exception:
        pass
    try:
        if hasattr(obj, 'to'):
            obj.to('cpu')
            return
    except Exception:
        pass

    nested = getattr(obj, 'model', None)
    if nested is None:
        return
    try:
        if hasattr(nested, 'cpu'):
            nested.cpu()
            return
    except Exception:
        pass
    try:
        if hasattr(nested, 'to'):
            nested.to('cpu')
    except Exception:
        pass
