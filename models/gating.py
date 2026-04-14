"""
gating.py
---------
Gating module that outputs per-teacher weights from student embeddings.

Two modes:
  1. 'mlp'         : Pure learned gating via small MLP on student embedding
  2. 'disagreement': Combines MLP gating with teacher disagreement signal
                     (high disagreement → trust the more confident teacher more)

The output is always a softmax-normalized weight vector over teachers.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPGating(nn.Module):
    """
    Simple MLP gating network.

    Input : student embedding (B, embed_dim)
    Output: teacher weights   (B, num_teachers) — softmax normalized
    """

    def __init__(self, embed_dim: int, num_teachers: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_teachers),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding : (B, embed_dim)

        Returns:
            weights : (B, num_teachers) — sum to 1 along dim=-1
        """
        logits = self.net(embedding)          # (B, num_teachers)
        return F.softmax(logits, dim=-1)


class DisagreementGating(nn.Module):
    """
    Confidence-based gating using teacher disagreement as an uncertainty proxy.

    Strategy:
    - Compute base weights via MLP
    - Compute disagreement = |score_A - score_B| for each pair
    - When disagreement is HIGH: shift weight toward the teacher whose
      prediction is closer to the student's current prediction
    - When disagreement is LOW: trust the MLP weights

    This is computed as a soft blend controlled by a learned temperature.

    Args:
        embed_dim    : Dimension of student embedding
        num_teachers : Number of teacher models
        hidden_dim   : Hidden size of the MLP gating network
    """

    def __init__(
        self,
        embed_dim: int,
        num_teachers: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_teachers = num_teachers

        # Base MLP gating
        self.mlp = MLPGating(embed_dim, num_teachers, hidden_dim)

        # Learnable blend temperature: how much to trust disagreement signal
        # Initialized to 0 → starts as pure MLP gating
        self.blend_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        embedding: torch.Tensor,
        teacher_scores: torch.Tensor,
        student_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embedding      : (B, embed_dim)   — student embedding
            teacher_scores : (B, num_teachers) — stacked teacher predictions
            student_score  : (B,) optional    — student's current prediction

        Returns:
            weights : (B, num_teachers) — softmax-normalized
        """
        # Base MLP weights
        mlp_weights = self.mlp(embedding)    # (B, num_teachers)

        if student_score is None or self.num_teachers < 2:
            return mlp_weights

        # --- Disagreement signal ---
        # disagreement per sample: scalar in [0, inf)
        # For 2 teachers: |t1 - t2|. Generalizes to mean pairwise gap for N teachers.
        t = teacher_scores                   # (B, num_teachers)
        mean_score = t.mean(dim=-1, keepdim=True)          # (B, 1)
        disagreement = (t - mean_score).abs().mean(dim=-1) # (B,) — deviation from mean

        # Normalized disagreement per teacher: how far each teacher is from mean
        # Inverted so the teacher closer to mean gets MORE weight under high disagreement
        dist_from_mean = (t - mean_score).abs()            # (B, num_teachers)

        # If student score is available, use proximity to student instead
        if student_score is not None:
            student_score_exp = student_score.unsqueeze(-1)  # (B, 1)
            dist_from_mean = (t - student_score_exp).abs()   # (B, num_teachers)

        # Invert distances → closer teacher gets higher raw weight
        proximity = 1.0 / (dist_from_mean + 1e-6)         # (B, num_teachers)
        proximity_weights = F.softmax(proximity, dim=-1)   # (B, num_teachers)

        # Blend: alpha controlled by learned temp (sigmoid → [0,1])
        # and scaled by normalized disagreement
        alpha = torch.sigmoid(self.blend_temp)             # scalar
        disagreement_norm = disagreement.unsqueeze(-1)     # (B, 1)

        # High disagreement → more disagreement-based weighting
        blend = alpha * disagreement_norm.clamp(0, 1)
        weights = (1.0 - blend) * mlp_weights + blend * proximity_weights

        # Re-normalize (blend may not be exactly sum-to-1 due to clamping)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights


def build_gating(cfg: dict, embed_dim: int, num_teachers: int) -> nn.Module:
    """
    Build gating module from config.

    Expected config keys:
        gating.mode        : 'mlp' | 'disagreement'
        gating.hidden_dim  : int (default 64)
    """
    gating_cfg = cfg.get("gating", {})
    mode = gating_cfg.get("mode", "disagreement")
    hidden_dim = gating_cfg.get("hidden_dim", 64)

    if mode == "mlp":
        return MLPGating(embed_dim, num_teachers, hidden_dim)
    elif mode == "disagreement":
        return DisagreementGating(embed_dim, num_teachers, hidden_dim)
    else:
        raise ValueError(f"Unknown gating mode: '{mode}'. Choose 'mlp' or 'disagreement'.")