"""Tile the per-sample Grad-CAM panels into one summary montage per dataset.

The per-sample panels (``figures/gradcam/<dataset>/<model>/sample_XXX.png``) are
produced on the GPU box by ``generate_gradcam_figures.py`` (needs torch + the
checkpoints). This script only *reads* those PNGs and lays them out as a grid
(rows = model, cols = samples) so the attention shift can be compared across
model complexity and standard-vs-AT in a single figure. It needs no torch, so it
runs on the local (figure-only) machine.

Usage:
    python scripts/generate_gradcam_summary.py                       # all datasets found
    python scripts/generate_gradcam_summary.py --dataset malaria oct2017
    python scripts/generate_gradcam_summary.py --per-model 4
Output: figures/gradcam/<dataset>_gradcam_summary.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prettier row labels; anything not listed falls back to the raw dir name.
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "resnet50_at": "ResNet-50 (AT)",
    "resnet152_at": "ResNet-152 (AT)",
}


def _model_sort_key(name: str):
    """Order plain models by depth, then their AT variants after them."""
    is_at = name.endswith("_at")
    base = name[:-3] if is_at else name
    digits = "".join(c for c in base if c.isdigit())
    depth = int(digits) if digits else 0
    return (is_at, depth, name)


def discover_models(dataset_dir: Path) -> list[Path]:
    models = [d for d in dataset_dir.iterdir() if d.is_dir() and any(d.glob("sample_*.png"))]
    return sorted(models, key=lambda d: _model_sort_key(d.name))


def build_summary(dataset_dir: Path, per_model: int, output_dir: Path) -> Path | None:
    model_dirs = discover_models(dataset_dir)
    if not model_dirs:
        print(f"  [skip] no model panels under {dataset_dir}")
        return None

    dataset = dataset_dir.name
    nrows = len(model_dirs)
    ncols = per_model
    # Panels are ~1134x1249 (aspect ~0.91 w/h); size the grid to keep them legible.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.85 * nrows),
        squeeze=False,
    )

    for r, mdir in enumerate(model_dirs):
        samples = sorted(mdir.glob("sample_*.png"))[:ncols]
        label = MODEL_LABELS.get(mdir.name, mdir.name)
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(samples):
                ax.imshow(Image.open(samples[c]))
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=90,
                              labelpad=8)
            if r == 0:
                ax.set_title(f"sample {c}", fontsize=9)

    fig.suptitle(
        f"Grad-CAM — {dataset}: clean vs adversarial attention across models\n"
        f"(each cell: clean img / clean CAM  •  adv img / adv CAM)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{dataset}_gradcam_summary.png"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{dataset}_gradcam_summary.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}  ({nrows} models x {ncols} samples)")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile Grad-CAM panels into per-dataset summary montages")
    parser.add_argument("--gradcam-dir", type=Path, default=PROJECT_ROOT / "figures" / "gradcam")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures" / "gradcam")
    parser.add_argument("--dataset", nargs="*", default=None,
                        help="Dataset name(s) to montage; default: every dataset dir found.")
    parser.add_argument("--per-model", type=int, default=4,
                        help="How many samples (columns) per model row.")
    args = parser.parse_args()

    if args.dataset:
        dataset_dirs = [args.gradcam_dir / d for d in args.dataset]
    else:
        dataset_dirs = sorted(d for d in args.gradcam_dir.iterdir() if d.is_dir())

    print(f"Grad-CAM summary montages -> {args.output_dir}")
    for ddir in dataset_dirs:
        if not ddir.is_dir():
            print(f"  [skip] {ddir} not found")
            continue
        build_summary(ddir, args.per_model, args.output_dir)


if __name__ == "__main__":
    main()
