"""
CSV metrics logger for offline analysis.
"""
import csv
import os
from typing import Dict, List, Optional


class CSVLogger:
    """Append-mode CSV logger that auto-detects headers."""

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None

    def log(self, metrics: Dict[str, float]) -> None:
        """Log a row of metrics. Headers are determined by the first call."""
        if self._file is None:
            self._fieldnames = list(metrics.keys())
            file_exists = os.path.isfile(self.path)
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames,
                                           extrasaction="ignore")
            if not file_exists:
                self._writer.writeheader()

        # Handle new keys by extending fieldnames
        for key in metrics:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
                # Reopen with new fieldnames
                self._file.close()
                self._file = open(self.path, "a", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames,
                                               extrasaction="ignore")

        self._writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                               for k, v in metrics.items()})
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
