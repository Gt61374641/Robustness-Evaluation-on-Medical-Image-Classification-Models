import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.imbalance import (
    build_balanced_sampler,
    compute_class_counts,
    compute_class_weights,
)


class TinyLabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.zeros(1), self.labels[idx]


def test_compute_class_counts_handles_subset_labels():
    dataset = TinyLabelDataset([0, 0, 0, 1, 1])
    subset = Subset(dataset, [0, 2, 3])

    counts = compute_class_counts(subset, num_classes=2)

    assert counts.tolist() == [2, 1]


def test_compute_class_weights_uses_balanced_formula():
    dataset = TinyLabelDataset([0, 0, 0, 1])

    weights = compute_class_weights(dataset, num_classes=2)

    assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])


def test_build_balanced_sampler_assigns_higher_weight_to_minority_class():
    dataset = TinyLabelDataset([0, 0, 0, 1])

    sampler = build_balanced_sampler(dataset, num_classes=2)

    sample_weights = sampler.weights.tolist()
    assert len(sample_weights) == 4
    assert sample_weights[3] > sample_weights[0]
    assert sampler.replacement is True
    assert sampler.num_samples == len(dataset)
