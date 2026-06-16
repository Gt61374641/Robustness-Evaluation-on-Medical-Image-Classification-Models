"""Chest X-ray pneumonia dataset loader.

Loads the Kaggle Pediatric Chest X-ray (Pneumonia) dataset (Kermany et al.),
2 classes (NORMAL / PNEUMONIA).

Expected directory structure (val/ is optional):
    data/chest_xray_pneumonia/train/NORMAL/      data/chest_xray_pneumonia/train/PNEUMONIA/
    data/chest_xray_pneumonia/val/NORMAL/        data/chest_xray_pneumonia/val/PNEUMONIA/
    data/chest_xray_pneumonia/test/NORMAL/       data/chest_xray_pneumonia/test/PNEUMONIA/

If val/ is missing (the Kaggle release ships only 16 val images and is often
omitted), we carve a deterministic 10% slice from train as a substitute.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

from .transforms import get_transforms


class ImageFolderDataset(Dataset):
    """Simple image folder dataset for Kaggle-style class-subdir directory structure.

    Shared by chest_xray_pneumonia and oct2017 loaders.
    """

    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = root_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _has_class_subdirs(path: Path) -> bool:
    """Return True if `path` exists and contains at least one non-empty class subdir."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and any(child.iterdir()):
            return True
    return False


def get_chest_xray_loaders(cfg: dict):
    """Get chest X-ray pneumonia dataloaders (Kaggle Kermany)."""
    img_size = cfg["data"].get("img_size", 224)
    batch_size = cfg["data"].get("batch_size", 32)
    num_workers = cfg["data"].get("num_workers", 4)
    seed = cfg.get("seed", 42)

    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    data_dir = Path(cfg["data"]["data_dir"]) / "chest_xray_pneumonia"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Chest X-ray data not found at {data_dir}. "
            "Run `python scripts/download_data.py --dataset chest_xray_pneumonia` first."
        )

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    if not _has_class_subdirs(train_dir):
        raise FileNotFoundError(f"Expected class subdirectories under {train_dir}")
    if not _has_class_subdirs(test_dir):
        raise FileNotFoundError(f"Expected class subdirectories under {test_dir}")

    test_ds = ImageFolderDataset(test_dir, eval_transform)

    if _has_class_subdirs(val_dir):
        train_ds = ImageFolderDataset(train_dir, train_transform)
        val_ds = ImageFolderDataset(val_dir, eval_transform)
        class_names = train_ds.class_names
    else:
        # No usable val/ folder — split 10% off train (deterministic).
        train_full_aug = ImageFolderDataset(train_dir, train_transform)
        train_full_eval = ImageFolderDataset(train_dir, eval_transform)
        n = len(train_full_aug)
        val_size = max(1, int(0.1 * n))
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        train_ds = Subset(train_full_aug, train_idx)
        val_ds = Subset(train_full_eval, val_idx)
        class_names = train_full_aug.class_names

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "num_classes": 2,
        "class_names": class_names,
    }
