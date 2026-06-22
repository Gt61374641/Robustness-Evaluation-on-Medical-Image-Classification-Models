import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.gradcam import _save_panel


def test_gradcam_panel_saves_publication_svg_and_png(tmp_path):
    clean = torch.rand(3, 16, 16)
    adv = torch.rand(3, 16, 16)
    clean_cam = torch.rand(1, 1, 16, 16)
    adv_cam = torch.rand(1, 1, 16, 16)

    outputs = _save_panel(tmp_path / "sample_000", clean, adv, clean_cam, adv_cam)

    assert outputs["svg"].exists()
    assert outputs["png"].exists()
    assert outputs["svg"].read_text(encoding="utf-8").find("<text") != -1
