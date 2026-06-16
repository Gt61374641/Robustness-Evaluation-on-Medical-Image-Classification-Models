import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_style import DEFAULT_FORMATS, apply_publication_style, finalize_figure


def test_publication_style_keeps_svg_text_editable():
    apply_publication_style()

    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["svg.fonttype"] == "none"
    assert plt.rcParams["pdf.fonttype"] == 42
    assert "Arial" in plt.rcParams["font.sans-serif"]


def test_finalize_figure_writes_publication_bundle():
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_dir = Path(__file__).with_name("_plot_style_test_outputs")
    out_dir.mkdir(exist_ok=True)
    outputs = finalize_figure(fig, out_dir / "figure")

    assert tuple(path.suffix.lstrip(".") for path in outputs) == DEFAULT_FORMATS
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0
        path.unlink()
    out_dir.rmdir()
