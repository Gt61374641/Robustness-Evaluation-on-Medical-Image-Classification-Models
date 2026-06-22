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


def get_retinal_loaders(cfg: dict):
    """Get OCT2017 dataloaders."""
    img_size = cfg["data"].get("img_size", 224)
    batch_size = cfg["data"].get("batch_size", 32)
    num_workers = cfg["data"].get("num_workers", 4)
    seed = cfg.get("seed", 42)

    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    data_root = Path(cfg["data"]["data_dir"]) / "oct2017"
    # Kaggle's Kermany2018 archive extracts as data/oct2017/OCT2017/{train,test};
    # accept either that nested layout or a manually-flattened data/oct2017/{train,test}.
    if (data_root / "OCT2017" / "train").is_dir():
        data_dir = data_root / "OCT2017"
    elif (data_root / "train").is_dir():
        data_dir = data_root
    else:
        raise FileNotFoundError(
            f"OCT2017 data not found at {data_root}. Expected either "
            f"{data_root}/train/ or {data_root}/OCT2017/train/. "
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
