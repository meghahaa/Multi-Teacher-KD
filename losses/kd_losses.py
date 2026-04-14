"""
kd_losses.py
------------
Knowledge distillation losses between student and teacher scores.

Includes:
  - Standard KD: L1 or L2 between student and (possibly weighted) teacher target
  - Confidence-weighted KD: uses gating weights to form a soft teacher target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard KD Loss
# ---------------------------------------------------------------------------

class StandardKDLoss(nn.Module):
    """
    L1 or L2 loss between student predictions and a single teacher target.

    Useful as a baseline or when confidence-based gating is disabled.

    Args:
        mode : 'l1' (MAE) or 'l2' (MSE)
    """

    def __init__(self, mode: str = "l1"):
        super().__init__()
        if mode == "l1":
            self.loss_fn = nn.L1Loss()
        elif mode == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown KD mode '{mode}'. Use 'l1' or 'l2'.")
        self.mode = mode

    def forward(
        self,
        student_score: torch.Tensor,
        teacher_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_score : (B,)
            teacher_score : (B,) — single teacher or pre-averaged target

        Returns:
            scalar loss
        """
        return self.loss_fn(student_score, teacher_score)


# ---------------------------------------------------------------------------
# Confidence-Weighted Multi-Teacher KD Loss
# ---------------------------------------------------------------------------

class ConfidenceWeightedKDLoss(nn.Module):
    """
    Distillation loss using gating-module-derived weights to form
    a soft composite teacher target from multiple teachers.

    The target is:  T_soft = sum_i(w_i * T_i)
    The loss is:    L = loss_fn(student, T_soft)

    Args:
        mode : 'l1' or 'l2'
    """

    def __init__(self, mode: str = "l1"):
        super().__init__()
        if mode == "l1":
            self.loss_fn = nn.L1Loss()
        elif mode == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown KD mode '{mode}'. Use 'l1' or 'l2'.")

    def forward(
        self,
        student_score: torch.Tensor,
        teacher_scores: torch.Tensor,
        gating_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_score  : (B,)
            teacher_scores : (B, num_teachers)
            gating_weights : (B, num_teachers) — softmax-normalized

        Returns:
            scalar loss
        """
        # Weighted teacher target: (B,)
        soft_target = (gating_weights * teacher_scores).sum(dim=-1)
        return self.loss_fn(student_score, soft_target.detach())


# ---------------------------------------------------------------------------
# Ground Truth Loss (supervised signal)
# ---------------------------------------------------------------------------

class GTLoss(nn.Module):
    """
    Standard supervised loss between student predictions and ground-truth MOS.

    Args:
        mode : 'l1' or 'l2'
    """

    def __init__(self, mode: str = "l1"):
        super().__init__()
        self.loss_fn = nn.L1Loss() if mode == "l1" else nn.MSELoss()

    def forward(
        self,
        student_score: torch.Tensor,
        gt_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_score : (B,)
            gt_score      : (B,) — ground-truth MOS (normalized to [0,1])

        Returns:
            scalar loss
        """
        return self.loss_fn(student_score, gt_score)


# ---------------------------------------------------------------------------
# Combined KD Loss (orchestrator)
# ---------------------------------------------------------------------------

class KDLossComposer(nn.Module):
    """
    Composes GT loss + KD losses into a single weighted total.

    Controlled entirely by config — each component can be toggled.

    Args:
        cfg : full config dict
    """

    def __init__(self, cfg: dict):
        super().__init__()
        loss_cfg = cfg.get("losses", {})
        kd_mode = loss_cfg.get("kd_mode", "l1")

        self.lambda_gt = loss_cfg.get("lambda_gt", 1.0)
        self.lambda_kd = loss_cfg.get("lambda_kd", 1.0)

        self.use_gt = loss_cfg.get("use_gt_loss", True)
        self.use_standard_kd = loss_cfg.get("use_standard_kd", True)
        self.use_confidence_kd = loss_cfg.get("use_confidence_kd", True)

        self.gt_loss = GTLoss(mode=kd_mode)
        self.standard_kd = StandardKDLoss(mode=kd_mode)
        self.confidence_kd = ConfidenceWeightedKDLoss(mode=kd_mode)

    def forward(
        self,
        student_score: torch.Tensor,
        gt_score: torch.Tensor,
        teacher_scores: torch.Tensor,           # (B, num_teachers)
        gating_weights: torch.Tensor,           # (B, num_teachers)
    ) -> dict:
        """
        Returns:
            dict with keys: 'total', 'gt', 'kd_standard', 'kd_confidence'
        """
        losses = {}
        total = torch.zeros(1, device=student_score.device)

        # Ground-truth supervised loss
        if self.use_gt:
            losses["gt"] = self.gt_loss(student_score, gt_score)
            total = total + self.lambda_gt * losses["gt"]
        else:
            losses["gt"] = torch.zeros(1, device=student_score.device)

        # Standard KD: student vs. mean teacher target (simple baseline)
        if self.use_standard_kd:
            mean_teacher = teacher_scores.mean(dim=-1)  # (B,)
            losses["kd_standard"] = self.standard_kd(student_score, mean_teacher.detach())
            total = total + self.lambda_kd * losses["kd_standard"]
        else:
            losses["kd_standard"] = torch.zeros(1, device=student_score.device)

        # Confidence-weighted KD: student vs. gated teacher target
        if self.use_confidence_kd:
            losses["kd_confidence"] = self.confidence_kd(
                student_score, teacher_scores, gating_weights
            )
            total = total + self.lambda_kd * losses["kd_confidence"]
        else:
            losses["kd_confidence"] = torch.zeros(1, device=student_score.device)

        losses["total"] = total
        return losses