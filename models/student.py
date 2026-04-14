"""
student.py
----------
Lightweight MobileNetV2-based student model for NR-IQA.

~3M parameters. Returns a scalar quality score per image.
Optionally returns intermediate feature embeddings for future
feature-level KD extensions.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import MobileNetV2, mobilenet_v2


class IQAStudent(nn.Module):
    """
    MobileNetV2 backbone with a lightweight IQA regression head.

    Args:
        pretrained      : Load ImageNet weights for the backbone
        return_features : If True, forward() also returns the embedding vector
        dropout         : Dropout rate before the regression head

    Output shapes:
        score   : (B,)         — quality score in [0, 1] range (sigmoid applied)
        features: (B, 256)     — embedding (only if return_features=True)
    """

    def __init__(
        self,
        pretrained: bool = True,
        return_features: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.return_features = return_features

        # --- Backbone: MobileNetV2 feature extractor (strip classifier) ---
        backbone: MobileNetV2 = mobilenet_v2(pretrained=pretrained)
        # features output: (B, 1280, H/32, W/32)
        self.backbone = backbone.features

        # --- Pooling ---
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 1280, 1, 1)

        # --- Embedding projection ---
        # Compresses 1280 → 256 for a lighter head + useful for gating
        self.embed = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # --- Regression head ---
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Normalize score to [0, 1]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x : (B, C, H, W)

        Returns:
            score       : (B,)
            features    : (B, 256)  — only if self.return_features is True
        """
        # Backbone feature extraction
        feat_map = self.backbone(x)           # (B, 1280, H', W')
        pooled = self.pool(feat_map)          # (B, 1280, 1, 1)
        flat = pooled.flatten(1)              # (B, 1280)

        # Embedding
        embedding = self.embed(flat)          # (B, 256)

        # Regression
        score = self.head(embedding).squeeze(-1)  # (B,)

        if self.return_features:
            return score, embedding
        return score


def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_student(cfg: dict) -> IQAStudent:
    """
    Build student from config dict.

    Expected config keys:
        student.pretrained      (bool, default True)
        student.return_features (bool, default True)
        student.dropout         (float, default 0.2)
    """
    student_cfg = cfg.get("student", {})
    model = IQAStudent(
        pretrained=student_cfg.get("pretrained", True),
        return_features=student_cfg.get("return_features", True),
        dropout=student_cfg.get("dropout", 0.2),
    )
    n_params = count_parameters(model)
    print(f"[Student] Trainable parameters: {n_params:,} (~{n_params/1e6:.2f}M)")
    return model