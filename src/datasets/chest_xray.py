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

    def __init__(self, root_dir: Path, transform=None, class_names=None):
        """class_names: optional whitelist of class subdir names. When given,
        only those subdirs are used (ignores stray/duplicate dirs such as the
        nested ``cell_images/`` in the Kaggle malaria archive)."""
        self.transform = transform
        self.samples = []
        if class_names is not None:
            self.class_names = sorted(
                name for name in class_names if (root_dir / name).is_dir()
            )
        else:
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


class _SampleListDataset(Dataset):
    """Dataset over an explicit ``[(path, label), ...]`` list with one transform.

    Used by the pooled-random-split path, where samples are gathered from
    multiple source dirs (train/ + test/) and re-partitioned by index.
    """

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _stratified_split(samples, seed, val_frac, test_frac):
    """Deterministic per-class (stratified) split into train/val/test index lists."""
    by_label = {}
    for i, (_, label) in enumerate(samples):
        by_label.setdefault(label, []).append(i)

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_label):
        idxs = by_label[label]
        order = torch.randperm(len(idxs), generator=generator).tolist()
        shuffled = [idxs[k] for k in order]
        n = len(shuffled)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_idx += shuffled[:n_test]
        val_idx += shuffled[n_test:n_test + n_val]
        train_idx += shuffled[n_test + n_val:]
    return train_idx, val_idx, test_idx


def _stratified_val_carve(samples, seed, val_frac):
    """Carve a deterministic, class-STRATIFIED val split off a training sample list.

    Returns (train_idx, val_idx) and prints the per-class val counts so the split
    is auditable. Medical sets are class-imbalanced, where a plain random carve can
    skew val class proportions (and thus the val-balanced-acc checkpoint selection);
    stratifying keeps each class's share stable.
    """
    train_idx, val_idx, _ = _stratified_split(samples, seed, val_frac=val_frac, test_frac=0.0)
    counts = {}
    for i in val_idx:
        counts[samples[i][1]] = counts.get(samples[i][1], 0) + 1
    print(f"[val carve] stratified val: {len(val_idx)} samples, "
          f"per-class counts {dict(sorted(counts.items()))}")
    return train_idx, val_idx


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

    # --- Pooled random re-split (verification only) ---------------------------
    # Pools train/ + test/, shuffles, and re-partitions stratified by class so
    # that test is IN-distribution with train. This mirrors the Rodriguez et al.
    # paper's "randomly shuffled and split" protocol and is used to confirm that
    # the low official-split accuracy is a train->test distribution shift, NOT a
    # pipeline bug. NOT for the main study (it leaks distribution across splits).
    split_mode = cfg["data"].get("split", "official")
    if split_mode == "pooled_random":
        val_frac = float(cfg["data"].get("val_frac", 0.1))
        test_frac = float(cfg["data"].get("test_frac", 0.2))
        train_pool = ImageFolderDataset(train_dir, None)
        class_names = train_pool.class_names
        test_pool = ImageFolderDataset(test_dir, None, class_names=class_names)
        samples = list(train_pool.samples) + list(test_pool.samples)
        train_idx, val_idx, test_idx = _stratified_split(
            samples, seed, val_frac, test_frac
        )
        train_aug = _SampleListDataset(samples, train_transform)
        full_eval = _SampleListDataset(samples, eval_transform)
        return {
            "train": DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "val": DataLoader(Subset(full_eval, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(Subset(full_eval, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            "num_classes": 2,
            "class_names": class_names,
        }
    elif split_mode != "official":
        raise ValueError(f"Unknown data.split mode: {split_mode!r} (use 'official' or 'pooled_random')")

    test_ds = ImageFolderDataset(test_dir, eval_transform)

    if _has_class_subdirs(val_dir):
        train_ds = ImageFolderDataset(train_dir, train_transform)
        val_ds = ImageFolderDataset(val_dir, eval_transform)
        class_names = train_ds.class_names
    else:
        # No usable val/ folder — carve a deterministic STRATIFIED 10% off train.
        val_frac = float(cfg["data"].get("val_frac", 0.1))
        train_full_aug = ImageFolderDataset(train_dir, train_transform)
        train_full_eval = ImageFolderDataset(train_dir, eval_transform)
        train_idx, val_idx = _stratified_val_carve(train_full_aug.samples, seed, val_frac)
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
