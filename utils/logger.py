"""
logger.py
---------
Lightweight CSV logger for tracking training metrics across epochs.
Kaggle-friendly (no WandB dependency).
"""

import csv
import os
from pathlib import Path
from typing import Dict


class CSVLogger:
    """
    Logs epoch-level metrics to a CSV file.

    Args:
        log_dir  : Directory to save the CSV
        filename : CSV filename (default: 'train_log.csv')

    Usage:
        logger = CSVLogger(log_dir='./logs')
        logger.log(epoch=1, train_loss=0.45, val_srcc=0.82, val_plcc=0.83)
    """

    def __init__(self, log_dir: str = "./logs", filename: str = "train_log.csv"):
        self.log_path = Path(log_dir) / filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def log(self, **kwargs):
        """
        Log one row. Keys become column headers on first call.

        Args:
            **kwargs : Any key-value pairs (epoch, loss, srcc, etc.)
        """
        write_header = not self._header_written or not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kwargs.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in kwargs.items()})

    def print_row(self, epoch: int, metrics: Dict):
        """Pretty-print a metrics dict to stdout."""
        parts = [f"Epoch {epoch:03d}"]
        for k, v in metrics.items():
            parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print(" | ".join(parts))