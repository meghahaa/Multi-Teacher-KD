"""
metrics.py
----------
Evaluation metrics for IQA:
  - SRCC (Spearman Rank Correlation Coefficient)
  - PLCC (Pearson Linear Correlation Coefficient)

Both computed from accumulated prediction/target lists across batches.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_srcc(preds: list, targets: list) -> float:
    """
    Spearman Rank Correlation Coefficient.

    Args:
        preds   : List of predicted scores (floats)
        targets : List of ground-truth MOS scores (floats)

    Returns:
        SRCC in [-1, 1]
    """
    preds = np.array(preds, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    srcc, _ = spearmanr(preds, targets)
    return float(srcc)


def compute_plcc(preds: list, targets: list) -> float:
    """
    Pearson Linear Correlation Coefficient.

    Args:
        preds   : List of predicted scores (floats)
        targets : List of ground-truth MOS scores (floats)

    Returns:
        PLCC in [-1, 1]
    """
    preds = np.array(preds, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    plcc, _ = pearsonr(preds, targets)
    return float(plcc)


class MetricAccumulator:
    """
    Accumulates predictions and targets across batches for epoch-level metric computation.

    Usage:
        acc = MetricAccumulator()
        for batch in loader:
            ...
            acc.update(preds, targets)
        srcc, plcc = acc.compute()
        acc.reset()
    """

    def __init__(self):
        self.preds: list = []
        self.targets: list = []

    def update(self, preds, targets):
        """
        Args:
            preds   : torch.Tensor (B,) or list
            targets : torch.Tensor (B,) or list
        """
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().tolist()
        if hasattr(targets, "detach"):
            targets = targets.detach().cpu().tolist()
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self) -> dict:
        """Returns dict with 'srcc' and 'plcc'."""
        if len(self.preds) < 2:
            return {"srcc": 0.0, "plcc": 0.0}
        return {
            "srcc": compute_srcc(self.preds, self.targets),
            "plcc": compute_plcc(self.preds, self.targets),
        }

    def reset(self):
        self.preds = []
        self.targets = []