"""OCT2017 retinal dataset loader.

Loads the Kaggle Kermany OCT2017 dataset, 4 classes (CNV, DME, DRUSEN, NORMAL).
The official release ships only train/ and test/ — we carve a val split from
train (deterministic 90/10).

Expected directory structure:
    data/oct2017/train/CNV/  .../DME/  .../DRUSEN/  .../NORMAL/
    data/oct2017/test/CNV/   .../DME/  .../DRUSEN/  .../NORMAL/
"""

from pathlib import Path

from torch.utils.data import DataLoader, Subset

from .chest_xray import (
    ImageFolderDataset,
    _stratified_val_carve,
    _stratified_subsample_indices,
)
from .transforms import get_transforms


def _find_oct_root(data_root: Path):
    """Locate the directory that directly contains train/ and test/.

    Tolerates the kermany2018 archive's quirks: an extra nested level
    (``oct2017/OCT2017/...``), a folder name with a TRAILING SPACE
    (``"OCT2017 "``), and junk ``__MACOSX`` dirs. Returns the matching Path or
    None. Searches the root and up to two levels of subdirectories.
    """
    seen = []
    for c in [data_root, *sorted(data_root.glob("*"))]:
        if not c.is_dir() or "__MACOSX" in c.name:
            continue
        seen.append(c)
        seen.extend(d for d in sorted(c.glob("*")) if d.is_dir() and "__MACOSX" not in d.name)
    for c in seen:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    return None


def get_retinal_loaders(cfg: dict):
    """Get OCT2017 dataloaders."""
    img_size = cfg["data"].get("img_size", 224)
    batch_size = cfg["data"].get("batch_size", 32)
    num_workers = cfg["data"].get("num_workers", 4)
    seed = cfg.get("seed", 42)

    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    data_root = Path(cfg["data"]["data_dir"]) / "oct2017"
    # The kermany2018 archive extracts inconsistently: sometimes data/oct2017/train,
    # sometimes a nested data/oct2017/OCT2017/train, and notoriously a folder named
    # with a TRAILING SPACE ("OCT2017 "). Auto-detect the dir holding train/+test/.
    data_dir = _find_oct_root(data_root)
    if data_dir is None:
        raise FileNotFoundError(
            f"OCT2017 train/+test/ not found under {data_root}. "
            "Run `python scripts/download_data.py --dataset oct2017` first."
        )

    # Build train (with augmentation) and a parallel eval-transform copy of train
    # so we can carve a deterministic val subset that uses eval transforms.
    train_full_aug = ImageFolderDataset(data_dir / "train", train_transform)
    train_full_eval = ImageFolderDataset(data_dir / "train", eval_transform)
    test_ds = ImageFolderDataset(data_dir / "test", eval_transform)

    val_frac = float(cfg["data"].get("val_frac", 0.1))
    train_idx, val_idx = _stratified_val_carve(train_full_aug.samples, seed, val_frac)

    # Optional stratified train subsampling: OCT's ~83k training set is highly
    # redundant, so a fixed fraction (same subset for every model, seed-based)
    # cuts training time ~4-5x while keeping the comparison fair. Val/test stay
    # full. Set data.train_subsample_frac < 1.0 to enable (default 1.0 = full).
    subsample_frac = float(cfg["data"].get("train_subsample_frac", 1.0))
    if subsample_frac < 1.0:
        n_before = len(train_idx)
        train_idx = _stratified_subsample_indices(
            train_full_aug.samples, train_idx, subsample_frac, seed
        )
        counts = {}
        for i in train_idx:
            counts[train_full_aug.samples[i][1]] = counts.get(train_full_aug.samples[i][1], 0) + 1
        print(f"[train subsample] frac={subsample_frac}: {n_before} -> {len(train_idx)} "
              f"train samples, per-class {dict(sorted(counts.items()))}")

    train_ds = Subset(train_full_aug, train_idx)
    val_ds = Subset(train_full_eval, val_idx)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "num_classes": 4,
        "class_names": train_full_aug.class_names,
    }
