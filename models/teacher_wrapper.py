"""
teacher_wrapper.py
------------------
Black-box wrapper for pretrained teacher models (e.g. MANIQA).

- Runs in eval mode with no_grad
- Optionally caches outputs to disk keyed by image hash
- Supports any model that returns a scalar score per image
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class TeacherWrapper(nn.Module):
    """
    Wraps any pretrained IQA model as a black-box teacher.

    Args:
        model       : A pretrained nn.Module that maps (B, C, H, W) -> (B,) or (B, 1)
        name        : Identifier string (e.g. 'synthetic', 'authentic')
        cache_dir   : If set, teacher outputs are cached to disk as .pt files
    """

    def __init__(
        self,
        model: nn.Module,
        name: str = "teacher",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.name = name
        self.cache_dir = Path(cache_dir) / name if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Put teacher in eval mode immediately and keep it there
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _batch_hash(self, x: torch.Tensor) -> str:
        """Compute a deterministic hash for a batch of images."""
        # Use the raw bytes of the tensor for hashing
        raw = x.detach().cpu().numpy().tobytes()
        return hashlib.md5(raw).hexdigest()

    def _cache_path(self, batch_hash: str) -> Path:
        return self.cache_dir / f"{batch_hash}.pt"

    def _load_cache(self, batch_hash: str) -> Optional[torch.Tensor]:
        path = self._cache_path(batch_hash)
        if path.exists():
            return torch.load(path, map_location="cpu")
        return None

    def _save_cache(self, batch_hash: str, scores: torch.Tensor) -> None:
        torch.save(scores.detach().cpu(), self._cache_path(batch_hash))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) input batch

        Returns:
            scores : (B,) quality scores
        """
        # --- Try cache first ---
        if self.cache_dir is not None:
            batch_hash = self._batch_hash(x)
            cached = self._load_cache(batch_hash)
            if cached is not None:
                return cached.to(x.device)

        # --- Run teacher inference ---
        with torch.no_grad():
            scores = self.model(x)

        # Normalize shape to (B,)
        scores = scores.squeeze(-1) if scores.dim() == 2 else scores

        # --- Save to cache ---
        if self.cache_dir is not None:
            self._save_cache(batch_hash, scores)

        return scores


def build_teachers(
    model_dict: dict,
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Convenience factory to wrap multiple teacher models.

    Args:
        model_dict : {'synthetic': nn.Module, 'authentic': nn.Module, ...}
        cache_dir  : Root directory for caching (subdirs per teacher)

    Returns:
        dict of TeacherWrapper instances keyed by teacher name

    Example:
        teachers = build_teachers(
            {'synthetic': synth_model, 'authentic': auth_model},
            cache_dir='./teacher_cache'
        )
    """
    return {
        name: TeacherWrapper(model, name=name, cache_dir=cache_dir)
        for name, model in model_dict.items()
    }