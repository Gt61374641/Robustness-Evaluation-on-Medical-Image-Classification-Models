"""ISIC 2020 skin lesion dataset loader.

ISIC 2020 challenge data — skin lesion melanoma classification (2 classes:
benign vs malignant). Loaded from train.csv + train/image/ directory.

Expected directory structure:
    data/isic2020/train.csv             (columns: isic_id, target, ...)
    data/isic2020/train/image/*.jpg     (the actual images)

There is no official val/test split for the public ISIC 2020 training set,
so we do a deterministic 80/10/10 split (seeded) over the full table.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

from .transforms import get_transforms


class ISIC2020Dataset(Dataset):
    """ISIC 2020 dataset loaded from CSV metadata + image directory."""

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        transform=None,
        image_column: str = "isic_id",
        label_column: str = "target",
    ):
        self.frame = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image_name = str(row[self.image_column])
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_name = f"{image_name}.jpg"
        img = Image.open(self.image_dir / image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row[self.label_column])
        return img, label


def _split_indices(length: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 80/10/10 split."""
    train_len = int(0.8 * length)
    val_len = int(0.1 * length)
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len : train_len + val_len]
    test_indices = indices[train_len + val_len :]
    return train_indices, val_indices, test_indices


def get_isic2020_loaders(cfg: dict):
    """Get ISIC 2020 train/val/test dataloaders.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders, plus 'num_classes' and 'class_names'.
    """
    img_size = cfg["data"].get("img_size", 224)
    batch_size = cfg["data"].get("batch_size", 32)
    num_workers = cfg["data"].get("num_workers", 4)
    seed = cfg.get("seed", 42)

    data_dir = Path(cfg["data"]["data_dir"]) / "isic2020"
    csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train" / "image"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ISIC 2020 metadata not found at {csv_path}. "
            "Run `python scripts/download_data.py --dataset isic2020` first."
        )
    if not image_dir.exists():
        raise FileNotFoundError(
            f"ISIC 2020 image directory not found at {image_dir}. "
            "Make sure the JPG images are extracted under data/isic2020/train/image/."
        )

    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    train_full = ISIC2020Dataset(csv_path, image_dir, transform=train_transform)
    eval_full = ISIC2020Dataset(csv_path, image_dir, transform=eval_transform)

    train_idx, val_idx, test_idx = _split_indices(len(eval_full), seed)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "num_classes": 2,
        "class_names": ["benign", "malignant"],
    }
