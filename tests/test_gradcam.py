import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.gradcam import GradCAM


class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        feats = self.features(x).flatten(1)
        return self.classifier(feats)


def test_gradcam_returns_normalized_heatmap_matching_input_spatial_size():
    model = SmallConvNet()
    image = torch.rand(1, 3, 8, 8)

    cam = GradCAM(model, target_layer=model.features[0])
    heatmap = cam(image, class_idx=1)
    cam.close()

    assert tuple(heatmap.shape) == (1, 1, 8, 8)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


class ChannelsLastConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        feats = self.conv(x).permute(0, 2, 3, 1)
        feats = self.norm(feats)
        pooled = feats.mean(dim=(1, 2))
        return self.classifier(pooled)


def test_gradcam_supports_channels_last_vision_transformer_layers():
    model = ChannelsLastConvNet()
    image = torch.rand(1, 3, 8, 8)

    cam = GradCAM(model, target_layer=model.norm)
    heatmap = cam(image, class_idx=1)
    cam.close()

    assert tuple(heatmap.shape) == (1, 1, 8, 8)
