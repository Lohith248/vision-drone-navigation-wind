#!/usr/bin/env python3
import torch


def main():
    print(f"torch: {torch.__version__}")
    if not torch.cuda.is_available():
        print("cuda: not available")
        return 0

    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}")
    print(f"vram_gb: {props.total_memory / 1e9:.2f}")

    try:
        x = torch.randn(16, 16, device="cuda")
        y = torch.mm(x, x)
        _ = float(y[0, 0].item())
        print("cuda_runtime: usable")
    except Exception as exc:
        print("cuda_runtime: NOT usable")
        print(f"reason: {exc}")
        print("action: install a newer PyTorch CUDA build compatible with this GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
