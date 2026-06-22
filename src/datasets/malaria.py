"""NIH Malaria Cell Images dataset loader (replaces ISIC 2020).

Rajaraman et al. (NLM/NIH) thin-blood-smear cell images, 2 classes
(Parasitized / Uninfected), ~27,558 images, roughly 50/50 balanced.

Expected directory structure (Kaggle `iarunava/cell-images-for-detecting-malaria`):
    data/malaria/cell_images/Parasitized/*.png
    data/malaria/cell_images/Uninfected/*.png
(a flattened `data/malaria/{Parasitized,Uninfected}` layout is also accepted)

IMPORTANT — patient-level split. Each blood smear yields many cell crops from
the SAME patient. A random per-image split would leak a patient across
train/test and inflate results. We therefore parse a patient/source code from
the file name and split by GROUP so a patient appears in exactly one split.

File names encode the cell-source code, e.g.
    C100P61ThinF_IMG_20150918_144104_cell_162.png   -> group "C100P61"
    C1_thinF_IMG_20150604_104722_cell_9.png         -> group "C1"
The leading ``C<digits>(P<digits>)?`` token identifies the source slide/patient.
Files that do not match fall back to the token before ``_IMG_`` (and finally the
stem), and the count of fallbacks is logged.
"""

import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .chest_xray import ImageFolderDataset
from .transforms import get_transforms

# Leading cell-source / patient code, e.g. "C100P61" or "C1".
_PATIENT_RE = re.compile(r"^(C\d+(?:P\d+)?)", re.IGNORECASE)


def _patient_group(img_path: Path) -> str:
    """Derive a patient/source group key from a malaria cell-image file name."""
    name = img_path.name
    m = _PATIENT_RE.match(name)
    if m:
        return m.group(1).upper()
    # Fallbacks: token before _IMG_, else the full stem (treated as its own group).
    if "_IMG_" in name:
        return name.split("_IMG_", 1)[0]
    return img_path.stem


def _split_groups(groups, seed, val_frac=0.10, test_frac=0.20):
    """Deterministically partition unique groups into train/val/test index lists.

    Splitting is done at the GROUP level (patient-level), then expanded to the
    sample indices belonging to each selected group.
    """
    unique = sorted(set(groups))
    perm = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed)).tolist()
    n = len(unique)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_groups = {unique[i] for i in perm[:n_test]}
    val_groups = {unique[i] for i in perm[n_test:n_test + n_val]}

    train_idx, val_idx, test_idx = [], [], []
    for i, g in enumerate(groups):
        if g in test_groups:
            test_idx.append(i)
        elif g in val_groups:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def get_malaria_loaders(cfg: dict):
    """Get NIH Malaria cell-image dataloaders with a patient-level split."""
    img_size = cfg["data"].get("img_size", 224)
    batch_size = cfg["data"].get("batch_size", 32)
    num_workers = cfg["data"].get("num_workers", 4)
    seed = cfg.get("seed", 42)

    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    root = Path(cfg["data"]["data_dir"]) / "malaria"
    img_root = root / "cell_images" if (root / "cell_images").is_dir() else root
    if not (img_root / "Parasitized").is_dir():
        raise FileNotFoundError(
            f"Malaria data not found at {img_root} (expected Parasitized/ and Uninfected/). "
            "Run `python scripts/download_data.py --dataset malaria` first."
        )

    # Whitelist the two real classes — the Kaggle archive nests a duplicate
    # cell_images/ dir that would otherwise be picked up as a spurious class.
    classes = ["Parasitized", "Uninfected"]
    full_aug = ImageFolderDataset(img_root, train_transform, class_names=classes)
    full_eval = ImageFolderDataset(img_root, eval_transform, class_names=classes)
    # Guarantee identical sample indexing between the two views so Subset indices
    # refer to the same images regardless of glob ordering.
    full_eval.samples = full_aug.samples

    groups = [_patient_group(path) for path, _ in full_aug.samples]
    n_fallback = sum(1 for g, (path, _) in zip(groups, full_aug.samples)
                     if not _PATIENT_RE.match(path.name))
    if n_fallback:
        print(f"[malaria] {n_fallback}/{len(groups)} files lacked a C..P.. code; "
              "used fallback grouping (token before _IMG_ / stem).")
    print(f"[malaria] {len(set(groups))} patient groups over {len(groups)} images.")

    train_idx, val_idx, test_idx = _split_groups(groups, seed)

    return {
        "train": DataLoader(Subset(full_aug, train_idx), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(Subset(full_eval, val_idx), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(Subset(full_eval, test_idx), batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True),
        "num_classes": 2,
        "class_names": full_aug.class_names,
    }
