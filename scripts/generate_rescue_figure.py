"""PGD-AT rescue / optimisation-stability figure.

Companion to the H2 AT ladder. Where the ladder (scripts/generate_at_ladder_figure.py)
reports the ORIGINAL unified PGD-AT protocol and marks collapsed points, this figure
isolates the follow-up question: *are those collapses intrinsic, or an artefact of an
under-stabilised optimiser?* It contrasts each collapsed point's original PGD-AT run
against a stronger-stabilisation "rescue" run (eps_warmup=8 / lr_warmup=5 / longer
schedule / LR halved / grad-clip), both under the same PGD-50+5restart strong eval.

The story the figure must carry:
  - OCT ResNet-152 collapses under the original protocol (robust@8 ~= 2.5%) but the
    rescue protocol fully recovers it (clean 97.8%, robust@8 89.2%) -> the collapse
    was optimisation, not capacity/task.
  - Chest ResNet-18 stays collapsed even under rescue -> collapses are heterogeneous;
    this one is recovered instead by switching method (TRADES/MART, see record 5.3c).

Two panels (Clean, PGD-8/255 robust), grouped bars original vs rescue per point,
collapsed bars drawn hollow with a red x at the base (same visual grammar as the
ladder figure). Reads the shared data file:
  figures/data/at_rescue.json   (written by scripts/extract_figure_data.py)

Usage:
    python scripts/generate_rescue_figure.py
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

ORIG_COLOR = "#9AA6B2"    # muted grey-blue: the under-stabilised original run
RESCUE_COLOR = "#0F4D92"  # blue_main: the stronger-stabilisation rescue run
COLLAPSE_RED = "#B64342"

PANELS = [
    ("clean", "Clean accuracy"),
    ("robust8", "PGD-8/255 robust accuracy (full)"),
]


def load_rows():
    p = PROJECT_ROOT / "figures" / "data" / "at_rescue.json"
    data = json.load(open(p))
    rows = data["rows"]
    # stable, story-first order: recovered point first, then the still-collapsed one
    rows = sorted(rows, key=lambda r: (r["rescue_collapsed"], r["dataset"]))
    return data, rows


def _label(r):
    disp_model = {"resnet18": "ResNet-18", "resnet34": "ResNet-34",
                  "resnet50": "ResNet-50", "resnet101": "ResNet-101",
                  "resnet152": "ResNet-152"}.get(r["model"], r["model"])
    return f"{r['dataset_display']}\n{disp_model}"


def _bar(ax, xc, val, collapsed, color, width):
    """One bar; collapsed -> pale hollow bar + red x at the base."""
    ax.bar(xc, val, width=width,
           color=color if not collapsed else "white",
           edgecolor=color, linewidth=0.9,
           alpha=1.0 if not collapsed else 0.9, zorder=2)
    if collapsed:
        ax.plot(xc, 0.02, marker="x", ms=5, color=COLLAPSE_RED,
                mew=1.6, zorder=5, clip_on=False)


def panel(ax, rows, key, ylabel):
    x = np.arange(len(rows))
    w = 0.36
    for i, r in enumerate(rows):
        _bar(ax, i - w / 2, r[f"orig_{key}"], r["orig_collapsed"], ORIG_COLOR, w)
        _bar(ax, i + w / 2, r[f"rescue_{key}"], r["rescue_collapsed"], RESCUE_COLOR, w)
        # annotate the recovery delta on the robust panel for recovered points
        if key == "robust8" and not r["rescue_collapsed"] and r["orig_collapsed"]:
            top = r[f"rescue_{key}"]
            ax.annotate(f"+{(top - r['orig_robust8']) * 100:.0f} pts",
                        xy=(i + w / 2, top), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.2, color=RESCUE_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(r) for r in rows], fontsize=6.4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def add_panel_label(ax, label):
    ax.text(-0.14, 1.03, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="bottom")


def main():
    data, rows = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    for ax, (key, ylabel) in zip(axes, PANELS):
        panel(ax, rows, key, ylabel)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")

    # shared legend below the panels
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=7,
               mfc=ORIG_COLOR, mec=ORIG_COLOR, label="Original PGD-AT"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=7,
               mfc=RESCUE_COLOR, mec=RESCUE_COLOR, label="PGD-AT rescue (stronger stabilisation)"),
        Line2D([0], [0], marker="x", linestyle="none", markersize=6,
               color=COLLAPSE_RED, mew=1.6, label="collapsed (trivial classifier)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)

    out_dir = PROJECT_ROOT / "figures" / "at_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "H2b_rescue_stability_py"
    for fmt in ("svg", "pdf", "png"):
        fig.savefig(f"{base}.{fmt}", dpi=600 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved {base}.svg / .pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
