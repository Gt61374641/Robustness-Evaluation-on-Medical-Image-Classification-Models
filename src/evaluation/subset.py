"""Fixed, stratified attack-evaluation subset.

For a fair complexity comparison every model must be attacked on the *same*
test samples. Slicing the first N items off a DataLoader is biased (class order,
folder order). This module instead draws a deterministic, class-stratified subset
of the collected test set and caches the chosen indices to disk so that all
models, attacks and epsilon values reuse exactly the same images.

The indices are positions into the test array collected in DataLoader order
(``shuffle=False``), which is deterministic for a given dataset + seed.
"""

import json
from pathlib import Path

import numpy as np


def stratified_indices(labels: np.ndarray, n_samples: int, seed: int) -> list:
    """Deterministically pick ``n_samples`` indices preserving class proportions.

    Args:
        labels: 1-D array of integer class labels for the full test set.
        n_samples: Target subset size (clamped to len(labels)).
        seed: RNG seed for reproducible selection.

    Returns:
        Sorted list of selected indices.
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    n_samples = min(n_samples, n_total)
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(labels, return_counts=True)
    # Largest-remainder allocation so per-class quotas sum exactly to n_samples.
    exact = counts / n_total * n_samples
    base = np.floor(exact).astype(int)
    remainder = n_samples - int(base.sum())
    order = np.argsort(-(exact - base))  # classes with largest fractional part first
    for i in range(remainder):
        base[order[i % len(order)]] += 1

    selected = []
    for cls, quota in zip(classes, base):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[:quota].tolist())

    return sorted(int(i) for i in selected)


def get_attack_subset(labels: np.ndarray, n_samples: int, seed: int, cache_path) -> list:
    """Return cached stratified subset indices, computing and caching if needed.

    The cache is invalidated when the total test size, requested size or seed
    changes, so a different split or sample budget recomputes safely.
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (cached.get("n_total") == n_total
                    and cached.get("n_samples") == min(n_samples, n_total)
                    and cached.get("seed") == seed):
                return cached["indices"]
        except (json.JSONDecodeError, KeyError):
            pass  # fall through and recompute

    indices = stratified_indices(labels, n_samples, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "n_total": n_total,
        "n_samples": min(n_samples, n_total),
        "seed": seed,
        "indices": indices,
    }, indent=2))
    return indices
