"""Main manuscript figures (Python backend), nature-figure style.

Rebuilds the hero figures from the shared data in figures/data/ (written by
scripts/extract_figure_data.py) so every panel uses the complete extension-batch
data and a single restrained palette with explicit collapse marking.

Figures (each -> figures/main/<name>.{svg,pdf,png}):
  H1_pgd_across_datasets   PGD robust acc vs eps, 5-model ladder, 3 datasets
  H1_attack_budget         FGSM vs PGD vs eps (rows) x datasets (cols)
  H1_complexity_ushape     robust acc @0.1/255 vs model capacity (the U-shape)
  defense_methods          Standard/PGD-AT/TRADES/MART, chest R18/R50/R152
  attack_methods           CW/DeepFool L2 + AutoAttack/Square robust@8, 7 models

The H2 AT ladder is produced separately by scripts/generate_at_ladder_figure.py.

Run:  python scripts/generate_main_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── MANDATORY editable-SVG rules (nature-figure skill) ──────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data"
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

# Sequential blue->violet ladder for the 5 ResNets (capacity = darker/warmer).
LADDER_COLORS = {
    "resnet18": "#9FC4E8", "resnet34": "#5B8FD6", "resnet50": "#0F4D92",
    "resnet101": "#6A3D9A", "resnet152": "#9A4D8E",
}
DATASET_COLORS = {"Chest X-ray": "#0F4D92", "Malaria": "#42949E", "OCT": "#9A4D8E"}
METHOD_COLORS = {
    "Standard": "#CFCECE", "PGD-AT": "#0F4D92", "TRADES": "#42949E", "MART": "#9A4D8E",
}
COLLAPSE_RED = "#B64342"


def _load(name):
    return json.load(open(DATA / name))


def _save(fig, name):
    for fmt in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{fmt}", dpi=600 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/main/{name}.svg/.pdf/.png")


def _panel_label(ax, label, x=-0.13, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="bottom")


# ── H1: PGD robustness vs eps, 5-model ladder, 3 datasets ───────────────────
def fig_h1_pgd():
    data = _load("h1_pgd_curves.json")
    ladder, disp = data["ladder"], data["display"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, dd in zip(axes, data["datasets"]):
        for m in ladder:
            series = dd["models"][m]["PGD"]
            series = [p for p in series if p["eps"] <= 0.3]
            if not series:
                continue
            xs = [p["eps"] for p in series]
            mean = np.array([p["mean"] for p in series]) * 100
            std = np.array([p["std"] for p in series]) * 100
            ax.plot(xs, mean, "-o", color=LADDER_COLORS[m], ms=2.5, lw=1.2,
                    label=disp[m])
            ax.fill_between(xs, mean - std, mean + std, color=LADDER_COLORS[m],
                            alpha=0.13, lw=0)
        ax.set_xscale("log")
        ax.set_title(dd["display"], fontsize=8, fontweight="bold")
        ax.set_xlabel(r"$\epsilon$ (/255, log)")
        ax.set_ylim(-3, 100)
    axes[0].set_ylabel("PGD robust accuracy (%)")
    axes[-1].legend(fontsize=5.6, loc="upper right", handlelength=1.2)
    _panel_label(axes[0], "a", x=-0.22)
    fig.tight_layout(pad=0.8)
    _save(fig, "H1_pgd_across_datasets")


# ── H1: FGSM vs PGD vs eps (rows) x datasets (cols) ─────────────────────────
def fig_h1_attack_budget():
    data = _load("h1_pgd_curves.json")
    ladder, disp = data["ladder"], data["display"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for r, atk in enumerate(("FGSM", "PGD")):
        for c, dd in enumerate(data["datasets"]):
            ax = axes[r][c]
            for m in ladder:
                series = dd["models"][m][atk]
                if not series:
                    continue
                xs = [p["eps"] for p in series]
                mean = np.array([p["mean"] for p in series]) * 100
                std = np.array([p["std"] for p in series]) * 100
                ax.plot(xs, mean, "-o", color=LADDER_COLORS[m], ms=2, lw=1.1,
                        label=disp[m])
                ax.fill_between(xs, mean - std, mean + std,
                                color=LADDER_COLORS[m], alpha=0.12, lw=0)
            ax.set_xscale("log")
            ax.set_ylim(-3, 100)
            if r == 0:
                ax.set_title(dd["display"], fontsize=8, fontweight="bold")
            if r == 1:
                ax.set_xlabel(r"$\epsilon$ (/255, log)")
            if c == 0:
                ax.set_ylabel(f"{atk}\nrobust accuracy (%)")
    axes[0][-1].legend(fontsize=5.6, loc="upper right", handlelength=1.2)
    _panel_label(axes[0][0], "a", x=-0.28)
    _panel_label(axes[1][0], "b", x=-0.28)
    fig.tight_layout(pad=0.8)
    _save(fig, "H1_attack_budget")


# ── H1: U-shape, robust acc @0.1/255 vs capacity ────────────────────────────
def fig_h1_ushape():
    data = _load("h1_complexity_fixedeps.json")
    ladder, disp, params = data["ladder"], data["display"], data["params_m"]
    xpos = np.arange(len(ladder))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    for ax, dd in zip(axes, data["datasets"]):
        ys = np.array([(dd["models"][m]["mean"] or np.nan) * 100 for m in ladder])
        es = np.array([(dd["models"][m]["std"] or 0) * 100 for m in ladder])
        ax.plot(xpos, ys, "-", color="#767676", lw=1.4, zorder=1)
        for i, m in enumerate(ladder):
            # ends robust (green), middle fragile (red) -- the U-shape cue
            c = "#2E7D32" if i in (0, 4) else (COLLAPSE_RED if i in (2, 3) else "#767676")
            ax.errorbar(xpos[i], ys[i], yerr=es[i], fmt="o", ms=6, color=c,
                        ecolor=c, elinewidth=0.8, capsize=2, zorder=2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([disp[m].replace("ResNet-", "R") for m in ladder])
        ax.set_title(dd["display"], fontsize=8, fontweight="bold")
        ax.set_xlabel("model capacity →")
        top = np.nanmax(ys) if np.isfinite(ys).any() else 1
        ax.set_ylim(0, max(top * 1.35, 1))
    axes[0].set_ylabel(r"Robust accuracy (%) @ $\epsilon$=0.1/255")
    handles = [Line2D([0], [0], marker="o", ls="none", mfc="#2E7D32", mec="#2E7D32",
                      ms=6, label="ends robust"),
               Line2D([0], [0], marker="o", ls="none", mfc=COLLAPSE_RED,
                      mec=COLLAPSE_RED, ms=6, label="middle fragile")]
    axes[-1].legend(handles=handles, fontsize=5.6, loc="upper center")
    _panel_label(axes[0], "a", x=-0.22)
    fig.tight_layout(pad=0.8)
    _save(fig, "H1_complexity_ushape")


# ── Defense: Standard/PGD-AT/TRADES/MART on chest R18/R50/R152 ──────────────
def fig_defense_methods():
    data = _load("defense_methods.json")
    models, methods = data["models"], data["methods"]
    disp = data["display_names"]
    idx = {(r["model"], r["method"]): r for r in data["rows"]}
    x = np.arange(len(models))
    w = 0.8 / len(methods)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    def grouped(ax, field, ylabel, mark_collapse):
        for i, meth in enumerate(methods):
            vals = [(idx[(m, meth)][field] or 0) * 100 for m in models]
            col = [idx[(m, meth)]["collapsed"] for m in models]
            offset = (i - (len(methods) - 1) / 2) * w
            for j, (xp, yv, c) in enumerate(zip(x, vals, col)):
                ax.bar(xp + offset, yv, width=w,
                       color=METHOD_COLORS[meth] if not (c and mark_collapse) else "white",
                       edgecolor=METHOD_COLORS[meth], linewidth=0.7,
                       label=meth if j == 0 else None, zorder=2)
                if c and mark_collapse:
                    ax.plot(xp + offset, 2.2, marker="x", ms=4, color=COLLAPSE_RED,
                            mew=1.4, zorder=5, clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels([disp[m].replace("ResNet-", "R") for m in models])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)

    grouped(ax1, "rob8", "Robust accuracy (%) @ 8/255", True)
    grouped(ax2, "clean", "Clean accuracy (%)", False)
    handles = [Line2D([0], [0], marker="s", ls="none", mfc=METHOD_COLORS[m],
                      mec=METHOD_COLORS[m], ms=6, label=m) for m in methods]
    handles.append(Line2D([0], [0], marker="x", ls="none", color=COLLAPSE_RED,
                          mew=1.4, ms=6, label="collapsed"))
    ax1.legend(handles=handles, fontsize=5.8, loc="upper left", ncol=1)
    _panel_label(ax1, "a")
    _panel_label(ax2, "b")
    fig.tight_layout(pad=1.0)
    _save(fig, "defense_methods")


# ── Attacks: CW/DeepFool L2 (a) + AutoAttack/Square robust@8 (b), 7 models ──
def fig_attack_methods():
    data = _load("attack_methods.json")
    models = data["models"]
    disp = data["display_names"]
    idx = {r["model"]: r for r in data["rows"]}
    models = [m for m in models if m in idx]
    x = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) minimal-perturbation L2 to fool: CW vs DeepFool (higher = more robust)
    w = 0.4
    for i, (field, lab, color) in enumerate(
            [("CW_l2", "CW", "#0F4D92"), ("DeepFool_l2", "DeepFool", "#42949E")]):
        vals = [idx[m][field] or 0 for m in models]
        ax1.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor="black",
                linewidth=0.4, label=lab, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([disp[m] for m in models], rotation=35, ha="right")
    ax1.set_ylabel(r"Mean $L_2$ perturbation to fool")
    ax1.legend(fontsize=6, loc="upper left")

    # (b) bounded strong attacks @8/255: AutoAttack (white-box) vs Square (black-box)
    for i, (field, lab, color) in enumerate(
            [("AutoAttack8", "AutoAttack", "#B64342"), ("Square8", "Square", "#E28E2C")]):
        vals = [(idx[m][field] or 0) * 100 for m in models]
        ax2.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor="black",
                linewidth=0.4, label=lab, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([disp[m] for m in models], rotation=35, ha="right")
    ax2.set_ylabel("Conditional robust accuracy (%) @ 8/255")
    ymax = max(20, max((idx[m]["Square8"] or 0) * 100 for m in models) * 1.3)
    ax2.set_ylim(0, ymax)
    ax2.annotate("AutoAttack drives every model to ≈0",
                 xy=(0.5, 0.90), xycoords="axes fraction", ha="center",
                 fontsize=6, color="#B64342", fontstyle="italic")
    ax2.legend(fontsize=6, loc="upper right")

    _panel_label(ax1, "a", x=-0.16)
    _panel_label(ax2, "b", x=-0.16)
    fig.tight_layout(pad=1.0)
    _save(fig, "attack_methods")


def main():
    fig_h1_pgd()
    fig_h1_attack_budget()
    fig_h1_ushape()
    fig_defense_methods()
    fig_attack_methods()


if __name__ == "__main__":
    sys.exit(main())
