"""
TensorBoard logger wrapper.
"""
from typing import Dict, Optional
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TBLogger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            print("  [Warning] TensorBoard not available, skipping TB logging")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
