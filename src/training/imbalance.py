"""Utilities for class-imbalanced medical image training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def _label_from_dataset(dataset, index: int) -> int:
    """Read a label without loading image pixels when the dataset exposes metadata."""
    if isinstance(dataset, Subset):
        return _label_from_dataset(dataset.dataset, dataset.indices[index])

    if hasattr(dataset, "labels"):
        return int(dataset.labels[index])

    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])

    if hasattr(dataset, "samples"):
        return int(dataset.samples[index][1])

    if hasattr(dataset, "frame") and hasattr(dataset, "label_column"):
        return int(dataset.frame.iloc[index][dataset.label_column])

    sample = dataset[index]
    if isinstance(sample, Sequence) and len(sample) >= 2:
        return int(sample[1])

    raise TypeError(f"Cannot extract label from dataset type {type(dataset).__name__}")


def get_dataset_labels(dataset) -> torch.Tensor:
    """Return all labels for a dataset or subset as a long tensor."""
    return torch.tensor([_label_from_dataset(dataset, idx) for idx in range(len(dataset))], dtype=torch.long)


def compute_class_counts(dataset, num_classes: int) -> torch.Tensor:
    """Count labels in a dataset, preserving zero-count classes."""
    labels = get_dataset_labels(dataset)
    return torch.bincount(labels, minlength=num_classes).to(torch.float32)


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Compute balanced cross-entropy weights: total / (num_classes * count)."""
    counts = compute_class_counts(dataset, num_classes)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot compute class weights because classes are missing: {missing}")
    total = counts.sum()
    return total / (num_classes * counts)


def build_balanced_sampler(dataset, num_classes: int, seed: int = 42) -> WeightedRandomSampler:
    """Build a sampler that draws minority-class examples more often."""
    labels = get_dataset_labels(dataset)
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot build balanced sampler because classes are missing: {missing}")

    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def replace_loader_with_balanced_sampler(loader: DataLoader, num_classes: int, seed: int = 42) -> DataLoader:
    """Return a train loader with the same dataset and a balanced sampler."""
    sampler = build_balanced_sampler(loader.dataset, num_classes, seed=seed)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last,
    )
