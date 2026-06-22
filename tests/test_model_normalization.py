import torch
import torch.nn as nn
import pytest

from src.models.model_factory import NormalizedModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward_features(self, x):
        return x.mean(dim=(2, 3))

    def forward_head(self, x, pre_logits=False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def test_normalized_model_applies_preprocessing_to_logits_and_features():
    backbone = TinyBackbone()
    model = NormalizedModel(backbone, mean=(0.5,) * 3, std=(0.25,) * 3)
    pixels = torch.ones(2, 3, 4, 4) * 0.75

    expected_features = torch.ones(2, 3)
    assert torch.allclose(model.forward_features(pixels), expected_features)
    assert torch.allclose(model(pixels), backbone.forward_head(expected_features))


def test_normalized_model_loads_legacy_unwrapped_state_dict():
    source = TinyBackbone()
    target = NormalizedModel(TinyBackbone(), mean=(0,) * 3, std=(1,) * 3)

    with pytest.warns(UserWarning, match="legacy checkpoint"):
        target.load_state_dict(source.state_dict())

    assert torch.equal(target.backbone.head.weight, source.head.weight)
    assert target.normalization_enabled is False
