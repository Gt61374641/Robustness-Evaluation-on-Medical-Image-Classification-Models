"""H2 AT-ladder figure (full 5-model complexity ladder, 3 datasets).

Two-panel Nature-style figure defending the (revised) H2 claim:
  Among models that adversarial training converges on, PGD@8/255 robustness rises
  with capacity (R50 < R152); but AT convergence itself is NOT monotonic in
  capacity -- R34 and several R101/R152 points collapse to trivial single-class
  classifiers, most often on the hardest (OCT, 4-class) task.

  (a) trend: PGD@8 full robust accuracy vs model capacity (params, log x),
      one line per dataset; collapsed points drawn as hollow markers with an x.
  (b) grouped bars: same metric, per model x dataset; collapsed bars hatched.

Reads the shared data file written by the extraction step:
  figures/data/at_ladder_h2.json

Usage:
    python scripts/generate_at_ladder_figure.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── MANDATORY: editable SVG text (nature-figure skill) ──────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One restrained palette: three dataset families + a red "collapse" signal.
DATASET_COLORS = {
    "chest_xray_pneumonia": "#0F4D92",  # blue_main
    "malaria": "#42949E",               # teal
    "oct2017": "#9A4D8E",               # violet
}
COLLAPSE_RED = "#B64342"

DISPLAY = {
    "resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
    "resnet101": "ResNet-101", "resnet152": "ResNet-152",
}


def load_data():
    p = PROJECT_ROOT / "figures" / "data" / "at_ladder_h2.json"
    data = json.load(open(p))
    # index rows[ds][model] -> row
    idx = {}
    for r in data["rows"]:
        idx.setdefault(r["dataset"], {})[r["model"]] = r
    return data, idx


def panel_trend(ax, data, idx):
    ladder = data["ladder"]
    params = [data["params_m"][m] for m in ladder]
    for ds in data["datasets"]:
        color = DATASET_COLORS[ds]
        y = [idx[ds][m]["robust8"] for m in ladder]
        col = [idx[ds][m]["collapsed"] for m in ladder]
        disp = idx[ds][ladder[0]]["dataset_display"]
        # continuous line across all points (dashed where it passes through collapse)
        ax.plot(params, y, color=color, lw=1.6, zorder=2, label=disp)
        # trained points: filled circle; collapsed points: hollow + x overlay
        for xp, yp, c in zip(params, y, col):
            if c:
                ax.plot(xp, yp, marker="o", ms=6, mfc="white", mec=color,
                        mew=1.3, zorder=3)
                ax.plot(xp, yp, marker="x", ms=5, color=COLLAPSE_RED,
                        mew=1.6, zorder=4)
            else:
                ax.plot(xp, yp, marker="o", ms=6, mfc=color, mec=color, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels([f"{p:.0f}" for p in params])
    ax.minorticks_off()
    ax.set_xlabel("Model capacity (parameters, M)")
    ax.set_ylabel("PGD-8/255 robust accuracy (full)")
    ax.set_ylim(-0.03, 1.0)
    ax.axhspan(-0.03, 0.02, color=COLLAPSE_RED, alpha=0.06, zorder=0)
    ax.legend(loc="upper left", fontsize=6.5)


def panel_bars(ax, data, idx):
    ladder = data["ladder"]
    datasets = data["datasets"]
    n_groups = len(datasets)
    w = 0.8 / n_groups
    x = np.arange(len(ladder))
    for i, ds in enumerate(datasets):
        color = DATASET_COLORS[ds]
        y = [idx[ds][m]["robust8"] for m in ladder]
        col = [idx[ds][m]["collapsed"] for m in ladder]
        offset = (i - (n_groups - 1) / 2) * w
        for xp, yp, c in zip(x, y, col):
            # trained -> solid fill; collapsed -> pale hollow bar + red x at base
            ax.bar(xp + offset, yp, width=w,
                   color=color if not c else "white",
                   edgecolor=color, linewidth=0.8,
                   alpha=1.0 if not c else 0.9, zorder=2)
            if c:
                ax.plot(xp + offset, 0.022, marker="x", ms=4.5,
                        color=COLLAPSE_RED, mew=1.5, zorder=5,
                        clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in ladder], rotation=30, ha="right")
    ax.set_ylabel("PGD-8/255 robust accuracy (full)")
    ax.set_ylim(0, 1.0)
    # Manual, clean legend: solid dataset swatches + collapsed marker
    handles = [Line2D([0], [0], marker="s", linestyle="none",
                      mfc=DATASET_COLORS[ds], mec=DATASET_COLORS[ds], markersize=7,
                      label=idx[ds][ladder[0]]["dataset_display"])
               for ds in datasets]
    handles.append(Line2D([0], [0], marker="x", linestyle="none",
                          color=COLLAPSE_RED, mew=1.5, markersize=6,
                          label="collapsed (trivial classifier)"))
    ax.legend(handles=handles, loc="upper left", fontsize=6.2)


def add_panel_label(ax, label):
    ax.text(-0.09, 1.03, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="bottom")


def main():
    data, idx = load_data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panel_trend(ax1, data, idx)
    panel_bars(ax2, data, idx)
    add_panel_label(ax1, "a")
    add_panel_label(ax2, "b")
    fig.tight_layout(pad=1.2)

    out_dir = PROJECT_ROOT / "figures" / "at_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "H2_at_ladder_py"
    for fmt in ("svg", "pdf", "png"):
        fig.savefig(f"{base}.{fmt}", dpi=600 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved {base}.svg / .pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
