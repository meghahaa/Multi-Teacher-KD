"""
utils/dataset.py
----------------
Dataset utilities for KADID-10k and KonIQ-10k.

MOS normalisation:
  KADID  : dmos in [1, 5]   -> (score - 1) / 4   -> [0, 1]
  KonIQ  : MOS  in [1, 100] -> (score - 1) / 99  -> [0, 1]

Split strategy:
  Each dataset is split independently into train / val / test before
  merging, so both datasets are always represented in every split.

Usage:
    from utils.dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
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

    Train : random crop + flip + colour jitter
    Val / Test : centre resize only (no augmentation — deterministic)
    """
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])


# ---------------------------------------------------------------------------
# Generic IQA Dataset
# ---------------------------------------------------------------------------

class IQADataset(Dataset):
    """
    Generic IQA dataset.

    Args:
        records   : List of (image_path, mos_score) where score is in [0, 1]
        transform : torchvision transform applied to each image

    Returns per item:
        image     : (3, H, W) float tensor
        mos_score : scalar float32 tensor in [0, 1]
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
# KADID-10k loader
# ---------------------------------------------------------------------------

def load_kadid(img_dir: str, csv_path: str) -> List[Record]:
    """
    Load KADID-10k records.

    CSV columns detected flexibly (case-insensitive):
      image name : any column containing 'name' or 'img'
      score      : any column containing 'dmos' or 'mos'

    dmos range : 1-5, higher = better quality
    Normalised : (dmos - 1) / 4  ->  [0, 1]

    Args:
        img_dir  : Directory containing distorted images (*.png)
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

    print(
        f"  [KADID]  {len(records):>5} images loaded"
        + (f"  ({missing} paths missing)" if missing else "")
        + f"  MOS in [{min(r[1] for r in records):.3f},"
          f" {max(r[1] for r in records):.3f}]"
    )
    return records


# ---------------------------------------------------------------------------
# KonIQ-10k loader
# ---------------------------------------------------------------------------

def load_koniq(img_dir: str, csv_path: str) -> List[Record]:
    """
    Load KonIQ-10k records.

    CSV columns detected flexibly (case-insensitive):
      image name : any column containing 'name' or 'image'
      score      : any column containing 'mos' or 'score'

    MOS range  : 1-100, higher = better quality
    Normalised : (MOS - 1) / 99  ->  [0, 1]

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

    print(
        f"  [KonIQ]  {len(records):>5} images loaded"
        + (f"  ({missing} paths missing)" if missing else "")
        + f"  MOS in [{min(r[1] for r in records):.3f},"
          f" {max(r[1] for r in records):.3f}]"
    )
    return records


# ---------------------------------------------------------------------------
# 3-way split
# ---------------------------------------------------------------------------

def split_records_3way(
    records: List[Record],
    val_frac: float  = 0.1,
    test_frac: float = 0.1,
    seed: int        = 42,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """
    Deterministic 3-way split into train / val / test.

    Called independently per dataset (KADID, KonIQ) so both are
    represented in every split after merging.

    Args:
        records   : Full record list for one dataset
        val_frac  : Fraction held out for validation  (default 0.1)
        test_frac : Fraction held out for test        (default 0.1)
        seed      : RNG seed for reproducibility

    Returns:
        (train_records, val_records, test_records)

    Raises:
        ValueError : if val_frac + test_frac >= 1.0
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0"
        )

    rng      = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n        = len(shuffled)
    n_val    = max(1, int(n * val_frac))
    n_test   = max(1, int(n * test_frac))
    n_train  = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Dataset too small ({n} records) for the requested split fractions."
        )

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load KADID + KonIQ, apply 3-way split, and return DataLoaders.

    Each dataset is split independently so both appear in every split,
    then the splits are merged and shuffled (train only).

    Expected cfg keys:
        kadid_img_dir, kadid_csv
        koniq_img_dir, koniq_csv
        val_split    (float, default 0.1)
        test_split   (float, default 0.1)
        image_size, batch_size, num_workers
        seed         (int,   default 42)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("Loading datasets...")
    kadid_records = load_kadid(cfg["kadid_img_dir"], cfg["kadid_csv"])
    koniq_records = load_koniq(cfg["koniq_img_dir"], cfg["koniq_csv"])

    seed      = cfg.get("seed", 42)
    val_frac  = cfg.get("val_split",  0.1)
    test_frac = cfg.get("test_split", 0.1)

    k_train, k_val, k_test = split_records_3way(kadid_records, val_frac, test_frac, seed)
    q_train, q_val, q_test = split_records_3way(koniq_records, val_frac, test_frac, seed)

    train_records = k_train + q_train
    val_records   = k_val   + q_val
    test_records  = k_test  + q_test

    # Shuffle train after merging (val/test are kept in split order — deterministic)
    random.Random(seed).shuffle(train_records)

    print(
        f"  [Split]  Train={len(train_records)}"
        f"  Val={len(val_records)}"
        f"  Test={len(test_records)}"
    )

    sz           = cfg["image_size"]
    batch_size   = cfg["batch_size"]
    num_workers  = cfg["num_workers"]

    train_loader = DataLoader(
        IQADataset(train_records, get_transforms(sz, train=True)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # Avoid single-sample batches that break BatchNorm
    )
    val_loader = DataLoader(
        IQADataset(val_records, get_transforms(sz, train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        IQADataset(test_records, get_transforms(sz, train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader