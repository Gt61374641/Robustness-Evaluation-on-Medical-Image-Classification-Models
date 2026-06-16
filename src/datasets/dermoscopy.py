"""Deprecated module — superseded by isic2020.py.

This file used to host DermaMNIST / HAM10000 loaders. The active dermoscopy
dataset is now ISIC 2020; see src/datasets/isic2020.py.

Kept only as a stub so any stale `from src.datasets.dermoscopy import ...`
import surfaces a clear error rather than a silent fallback.
"""

raise ImportError(
    "src.datasets.dermoscopy is deprecated. "
    "Use src.datasets.isic2020 (registered as 'isic2020' in DATASET_REGISTRY)."
)
