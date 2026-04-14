"""
utils/dataset.py
----------------
Dataset utilities for KADID-10k and KonIQ-10k.

MOS normalisation:
  KADID  : dmos in [1, 5]   → (score - 1) / 4    → [0, 1]
  KonIQ  : MOS  in [1, 100] → (score - 1) / 99   → [0, 1]

Usage:
    from utils.dataset import load_kadid, load_koniq, IQADataset, build_dataloaders
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Record = Tuple[str, float]   # (absolute_image_path, mos_normalised_0_to_1)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(image_size: int, train: bool = True) -> transforms.Compose:
    """
    Standard IQA augmentation / eval pipeline.

    Train: random crop + flip + colour jitter
    Val  : centre resize only
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])


# ---------------------------------------------------------------------------
# Generic Dataset
# ---------------------------------------------------------------------------

class IQADataset(Dataset):
    """
    Generic IQA dataset.

    Args:
        records   : List of (image_path, mos_score) where score is in [0, 1]
        transform : torchvision transform to apply to each image

    Returns per item:
        image     : (3, H, W) float tensor
        mos_score : scalar float tensor in [0, 1]
    """

    def __init__(self, records: List[Record], transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, score = self.records[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(score, dtype=torch.float32)


# ---------------------------------------------------------------------------
# KADID-10k
# ---------------------------------------------------------------------------

def load_kadid(img_dir: str, csv_path: str) -> List[Record]:
    """
    Load KADID-10k records.

    Expected CSV columns: img_name (or similar), dmos (or mos).
    Column names are detected flexibly — exact casing does not matter.

    dmos range  : 1–5, higher = better quality
    Normalised  : (dmos - 1) / 4  →  [0, 1]

    Args:
        img_dir  : Directory containing the distorted images (*.png)
        csv_path : Path to dmos.csv

    Returns:
        List of (absolute_path, normalised_mos) tuples
    """
    df = pd.read_csv(csv_path)

    name_col  = next(c for c in df.columns
                     if "name" in c.lower() or "img" in c.lower())
    score_col = next(c for c in df.columns
                     if "dmos" in c.lower() or "mos" in c.lower())

    records: List[Record] = []
    missing = 0
    for _, row in df.iterrows():
        p = Path(img_dir) / str(row[name_col])
        if not p.exists():
            missing += 1
            continue
        mos = float(np.clip((float(row[score_col]) - 1.0) / 4.0, 0.0, 1.0))
        records.append((str(p), mos))

    print(f"[KADID]  Loaded {len(records)} images"
          + (f"  ({missing} missing)" if missing else "")
          + f"  MOS ∈ [{min(r[1] for r in records):.3f},"
            f" {max(r[1] for r in records):.3f}]")
    return records


# ---------------------------------------------------------------------------
# KonIQ-10k
# ---------------------------------------------------------------------------

def load_koniq(img_dir: str, csv_path: str) -> List[Record]:
    """
    Load KonIQ-10k records.

    Expected CSV columns: image_name (or similar), MOS (or score).
    Column names are detected flexibly.

    MOS range   : 1–100, higher = better quality
    Normalised  : (MOS - 1) / 99  →  [0, 1]

    Args:
        img_dir  : Directory containing images (*.jpg)
        csv_path : Path to koniq10k_scores.csv

    Returns:
        List of (absolute_path, normalised_mos) tuples
    """
    df = pd.read_csv(csv_path)

    name_col  = next(c for c in df.columns
                     if "name" in c.lower() or "image" in c.lower())
    score_col = next(c for c in df.columns
                     if "mos" in c.lower() or "score" in c.lower())

    records: List[Record] = []
    missing = 0
    for _, row in df.iterrows():
        p = Path(img_dir) / str(row[name_col])
        if not p.exists():
            missing += 1
            continue
        mos = float(np.clip((float(row[score_col]) - 1.0) / 99.0, 0.0, 1.0))
        records.append((str(p), mos))

    print(f"[KonIQ]  Loaded {len(records)} images"
          + (f"  ({missing} missing)" if missing else "")
          + f"  MOS ∈ [{min(r[1] for r in records):.3f},"
            f" {max(r[1] for r in records):.3f}]")
    return records


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------

def split_records(
    records: List[Record],
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Record], List[Record]]:
    """
    Deterministic random split.

    Each dataset (KADID, KonIQ) is split independently before merging
    so both datasets are represented in the validation set.

    Args:
        records  : Full list of records
        val_frac : Fraction to hold out for validation
        seed     : RNG seed for reproducibility

    Returns:
        (train_records, val_records)
    """
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Load both datasets, merge, split, and return DataLoaders.

    Expected cfg keys:
        kadid_img_dir, kadid_csv
        koniq_img_dir, koniq_csv
        val_split, image_size, batch_size, num_workers

    Returns:
        (train_loader, val_loader)
    """
    kadid_records = load_kadid(cfg["kadid_img_dir"], cfg["kadid_csv"])
    koniq_records = load_koniq(cfg["koniq_img_dir"], cfg["koniq_csv"])

    seed = cfg.get("seed", 42)
    k_train, k_val = split_records(kadid_records, cfg["val_split"], seed)
    q_train, q_val = split_records(koniq_records, cfg["val_split"], seed)

    train_records = k_train + q_train
    val_records   = k_val   + q_val
    random.Random(seed).shuffle(train_records)

    print(f"[Data]   Train={len(train_records)}  Val={len(val_records)}")

    sz = cfg["image_size"]
    train_loader = DataLoader(
        IQADataset(train_records, get_transforms(sz, train=True)),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        IQADataset(val_records, get_transforms(sz, train=False)),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader