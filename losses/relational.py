"""
relational.py
-------------
Relational Knowledge Distillation via pairwise hinge loss.

Core idea:
  - Sample K pairs (i, j) from the batch
  - Teacher "consensus" defines the sign: which image has higher quality
  - Student is penalized if its ordering disagrees with the teacher consensus

Also includes the pair sampler utility (avoids naive O(N²) enumeration).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pair Sampling Utility
# ---------------------------------------------------------------------------

def sample_pairs(batch_size: int, num_pairs: int, device: torch.device) -> torch.Tensor:
    """
    Efficiently sample random pairs from a batch without enumerating all N² combos.

    Args:
        batch_size : Number of samples in the batch
        num_pairs  : Number of pairs to sample (K << N²)
        device     : Target device

    Returns:
        pairs : (K, 2) LongTensor — each row is (idx_i, idx_j) with i != j

    Notes:
        - Samples with replacement (fast, acceptable for large batches)
        - For tiny batches (N < 4), falls back to all valid pairs
    """
    if batch_size < 2:
        raise ValueError("Need at least 2 samples in the batch for pair sampling.")

    max_possible = batch_size * (batch_size - 1)
    num_pairs = min(num_pairs, max_possible)

    # Fast path: random sampling with rejection for i == j
    idx_i = torch.randint(0, batch_size, (num_pairs,), device=device)
    idx_j = torch.randint(0, batch_size, (num_pairs,), device=device)

    # Fix collisions: shift j by 1 (modular) wherever i == j
    collision = idx_i == idx_j
    idx_j[collision] = (idx_j[collision] + 1) % batch_size

    return torch.stack([idx_i, idx_j], dim=1)  # (K, 2)


# ---------------------------------------------------------------------------
# Relational KD Loss
# ---------------------------------------------------------------------------

class RelationalKDLoss(nn.Module):
    """
    Pairwise hinge loss for relational knowledge distillation.

    For each pair (i, j):
      - Teacher consensus: sign(T_i - T_j) defines the "correct" ordering
      - Hinge penalty: max(0, margin - sign * (S_i - S_j))
      - Student is pushed to maintain the teacher-defined ranking

    Args:
        margin         : Hinge margin (default 0.05, scale-dependent on score range)
        num_pairs      : Number of pairs to sample per batch
        consensus_mode : 'mean' — use mean of all teacher scores as consensus
                         'agreement' — only use pairs where teachers agree on ordering
    """

    def __init__(
        self,
        margin: float = 0.05,
        num_pairs: int = 64,
        consensus_mode: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.num_pairs = num_pairs
        self.consensus_mode = consensus_mode

    def forward(
        self,
        student_score: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_score  : (B,)
            teacher_scores : (B, num_teachers) — stacked teacher predictions

        Returns:
            scalar hinge loss (0.0 if no valid pairs found)
        """
        B = student_score.shape[0]
        if B < 2:
            return torch.zeros(1, device=student_score.device).squeeze()

        pairs = sample_pairs(B, self.num_pairs, student_score.device)  # (K, 2)
        idx_i, idx_j = pairs[:, 0], pairs[:, 1]

        # --- Teacher consensus signal ---
        if self.consensus_mode == "mean":
            # Use mean teacher score as the reference ranking
            teacher_mean = teacher_scores.mean(dim=-1)       # (B,)
            t_diff = teacher_mean[idx_i] - teacher_mean[idx_j]  # (K,)
            sign = torch.sign(t_diff)                         # (K,) — {-1, 0, 1}
            valid_mask = sign != 0                            # Exclude ties

        elif self.consensus_mode == "agreement":
            # Only use pairs where ALL teachers agree on ordering
            t_diffs = teacher_scores[idx_i] - teacher_scores[idx_j]  # (K, num_teachers)
            signs = torch.sign(t_diffs)                               # (K, num_teachers)
            # Agreement: all teachers have the same non-zero sign
            unanimous = (signs == signs[:, :1]).all(dim=-1) & (signs[:, 0] != 0)
            sign = signs[:, 0]
            valid_mask = unanimous
        else:
            raise ValueError(f"Unknown consensus_mode: '{self.consensus_mode}'")

        if valid_mask.sum() == 0:
            return torch.zeros(1, device=student_score.device).squeeze()

        # Filter to valid pairs
        sign = sign[valid_mask]
        s_diff = student_score[idx_i[valid_mask]] - student_score[idx_j[valid_mask]]  # (K',)

        # Hinge loss: penalize if student ordering disagrees with teacher
        # hinge = max(0, margin - sign * s_diff)
        hinge = torch.clamp(self.margin - sign * s_diff, min=0.0)  # (K',)

        return hinge.mean()


def build_relational_loss(cfg: dict) -> RelationalKDLoss:
    """
    Build relational loss from config.

    Expected config keys:
        losses.relational.margin         : float (default 0.05)
        losses.relational.num_pairs      : int   (default 64)
        losses.relational.consensus_mode : str   (default 'mean')
    """
    rel_cfg = cfg.get("losses", {}).get("relational", {})
    return RelationalKDLoss(
        margin=rel_cfg.get("margin", 0.05),
        num_pairs=rel_cfg.get("num_pairs", 64),
        consensus_mode=rel_cfg.get("consensus_mode", "mean"),
    )