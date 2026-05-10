"""
Device management and GPU memory monitoring utilities.
"""

import warnings

import torch


def _cuda_runtime_usable() -> bool:
    """
    Return True only if CUDA is available and basic kernels execute correctly.
    This catches driver / architecture mismatches (e.g., "no kernel image").
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(8, 8, device="cuda")
        y = torch.mm(x, x)
        _ = float(y[0, 0].item())
        return True
    except Exception as exc:
        warnings.warn(
            "CUDA is visible but not usable with the current PyTorch build. "
            "Falling back to CPU. Install a CUDA build that supports this GPU "
            f"to use acceleration. Root error: {exc}"
        )
        return False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    Parameters
    ----------
    device_str : str
        One of "auto", "cuda", "cpu", or "cuda:N".

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if _cuda_runtime_usable() else "cpu")
    if device_str.startswith("cuda"):
        if _cuda_runtime_usable():
            return torch.device(device_str)
        warnings.warn(
            f"Requested device '{device_str}' but CUDA is not usable. "
            "Using CPU instead."
        )
        return torch.device("cpu")
    return torch.device(device_str)


def log_gpu_usage() -> dict:
    """
    Return a dict with GPU utilization info. Safe to call even without CUDA.

    Returns
    -------
    dict with keys: device_name, total_mb, allocated_mb, reserved_mb, free_mb
    """
    if not torch.cuda.is_available():
        return {"device_name": "cpu", "total_mb": 0, "allocated_mb": 0,
                "reserved_mb": 0, "free_mb": 0}

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**2
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2

    return {
        "device_name": props.name,
        "total_mb": round(total, 1),
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(total - allocated, 1),
    }


def gpu_memory_mb() -> float:
    """Return currently allocated GPU memory in MB. Returns 0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / 1024**2


def print_device_info(device: torch.device) -> None:
    """Print device information to stdout."""
    print(f"  Device: {device}")
    if device.type == "cuda":
        info = log_gpu_usage()
        print(f"  GPU: {info['device_name']}")
        print(f"  VRAM: {info['total_mb']:.0f} MB total, "
              f"{info['allocated_mb']:.0f} MB allocated, "
              f"{info['free_mb']:.0f} MB free")
    else:
        print("  Running on CPU (no CUDA available)")
