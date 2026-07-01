"""Combined cross-dataset figures (one image, datasets side-by-side).

- H1: PGD full-robust-accuracy vs eps, one panel per dataset (chest/malaria/oct),
  cropped to the discriminating fine regime (<=0.3/255) so the non-monotonic
  complexity ordering is visible across all three datasets in one figure.
- H2: AT clean vs robust@8/255 grouped bars, one panel per dataset.

Usage:  python scripts/generate_combined_figures.py
Output: figures/combined/H1_pgd_across_datasets.png  /  H2_at_across_datasets.png
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "combined"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
LABELS = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
          "resnet101": "ResNet-101", "resnet152": "ResNet-152"}
COLORS = {m: c for m, c in zip(MODELS, plt.cm.viridis(np.linspace(0.0, 0.9, len(MODELS))))}

DATASETS = [
    ("chest_xray_pneumonia", "Chest X-ray", ["seed42", "seed43", "seed44"]),
    ("malaria", "Malaria", ["seed42"]),
    ("oct2017", "OCT2017", ["seed42"]),
]
EPS_MAX_255 = 0.3   # crop to the discriminating fine regime (excludes large-eps collapse artifact)


def _eps255(key):
    m = re.search(r"eps=([0-9.]+)", key)
    return float(m.group(1)) * 255 if m else None


def pgd_points(dataset, model, seed):
    """{eps_in_255: full_robust_acc} from the fine + main PGD sweeps."""
    pts = {}
    for sec in ("fine", "main"):
        f = ROOT / "results" / dataset / model / "robustness" / seed / f"robustness_attacks_{sec}_max1024.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        for k, v in r.items():
            if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
                e = _eps255(k)
                if e is not None:
                    pts[round(e, 4)] = v["robust_accuracy"]["full_robust_accuracy"]
    return pts


def attack_points(dataset, model, seed, attack):
    """{eps_in_255: full_robust_acc} for a given attack ('FGSM'/'PGD').
    For PGD, drop the large-eps class-collapse artifact points (where the attack
    just pushes everything to one class -> spuriously high full robust acc)."""
    pts = {}
    for sec in ("fine", "main"):
        f = ROOT / "results" / dataset / model / "robustness" / seed / f"robustness_attacks_{sec}_max1024.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        for k, v in r.items():
            if not (k.startswith(attack) and isinstance(v, dict) and "robust_accuracy" in v):
                continue
            if attack == "PGD":
                coll = v.get("pred_distribution", {}).get("collapse", {}).get("adv_majority_fraction", 0)
                if coll > 0.97:   # collapsed to one class -> artifact, exclude
                    continue
            e = _eps255(k)
            if e is not None:
                pts[round(e, 4)] = v["robust_accuracy"]["full_robust_accuracy"]
    return pts


def _robust_at_eps(dataset, model, seed, target=0.1):
    pts = attack_points(dataset, model, seed, "PGD")
    if not pts:
        return None
    e = min(pts, key=lambda k: abs(k - target))
    return pts[e] if abs(e - target) <= 0.03 else None


def make_h1_dual_view():
    """Two views of H1 in one image: top = robustness vs attack budget (eps),
    bottom = robustness vs MODEL COMPLEXITY at fixed eps (makes the U-shape explicit)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    xpos = np.arange(len(MODELS))
    xlab = [LABELS[m].replace("ResNet-", "R") for m in MODELS]

    # --- top row: vs eps ---
    for j, (ds, title, seeds) in enumerate(DATASETS):
        ax = axes[0][j]
        for m in MODELS:
            per = {}
            for s in seeds:
                for e, acc in attack_points(ds, m, s, "PGD").items():
                    if e <= 0.3:
                        per.setdefault(e, []).append(acc)
            if not per:
                continue
            xs = sorted(per)
            mean = np.array([np.mean(per[e]) for e in xs]) * 100
            ax.plot(xs, mean, "-o", color=COLORS[m], ms=3, lw=1.4, label=LABELS[m])
            if len(seeds) > 1:
                std = np.array([np.std(per[e]) for e in xs]) * 100
                ax.fill_between(xs, mean - std, mean + std, color=COLORS[m], alpha=0.13)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\epsilon$ (/255, log)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-3, 100)
    axes[0][0].set_ylabel("Robust accuracy (%)\nvs attack budget", fontsize=10)
    axes[0][2].legend(fontsize=7, loc="upper right")

    # --- bottom row: vs model complexity at eps = 0.1/255 ---
    for j, (ds, title, seeds) in enumerate(DATASETS):
        ax = axes[1][j]
        ys = []
        for m in MODELS:
            vals = [v for v in (_robust_at_eps(ds, m, s, 0.1) for s in seeds) if v is not None]
            ys.append(np.mean(vals) * 100 if vals else np.nan)
        ax.plot(xpos, ys, "-", color="#666666", lw=2, zorder=1)
        for i, y in enumerate(ys):
            c = "#2ca02c" if i in (0, 4) else ("#cc3333" if i in (2, 3) else "#888888")
            ax.plot(i, y, "o", ms=10, color=c, zorder=2)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlab, fontsize=9)
        ax.set_xlabel("model complexity  →", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        top = max([y for y in ys if not np.isnan(y)] or [1])
        ax.set_ylim(0, top * 1.3)
    axes[1][0].set_ylabel("Robust accuracy (%)\n@ ε = 0.1/255", fontsize=10)
    # legend for the color coding
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", ms=9, label="ends (R18 / R152) — robust"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#cc3333", ms=9, label="middle (R50 / R101) — fragile")]
    axes[1][2].legend(handles=handles, fontsize=7, loc="upper right")

    fig.suptitle("H1 — Robustness vs attack budget (top)  and  vs model complexity (bottom): non-monotonic U-shape in all three datasets",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"H1_dual_view.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'H1_dual_view.png'}")


def make_paper_style():
    """Paper Figure-1 layout: rows = datasets, cols = {FGSM, PGD}, full eps range."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True)
    for i, (ds, title, seeds) in enumerate(DATASETS):
        for j, atk in enumerate(["FGSM", "PGD"]):
            ax = axes[i][j]
            for m in MODELS:
                per_eps = {}
                for s in seeds:
                    for e, acc in attack_points(ds, m, s, atk).items():
                        per_eps.setdefault(e, []).append(acc)
                if not per_eps:
                    continue
                xs = sorted(per_eps)
                mean = np.array([np.mean(per_eps[e]) for e in xs]) * 100
                ax.plot(xs, mean, "-o", color=COLORS[m], label=LABELS[m], ms=3, lw=1.5)
                if len(seeds) > 1:
                    std = np.array([np.std(per_eps[e]) for e in xs]) * 100
                    ax.fill_between(xs, mean - std, mean + std, color=COLORS[m], alpha=0.15)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3, 100)
            if i == 0:
                ax.set_title(f"{atk} Attack", fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{title}\nFull robust accuracy (%)", fontsize=10)
            if i == len(DATASETS) - 1:
                ax.set_xlabel(r"Perturbation budget $\epsilon$ (/255, log)")
    axes[0][1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Accuracy vs perturbation budget across model complexity\n(rows: datasets — Chest X-ray / Malaria / OCT2017; columns: FGSM / PGD)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"paper_style_fig1.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'paper_style_fig1.png'}")


def at_points(dataset, model, seed="seed42"):
    """AT clean + robust@8/255 (PGD-50/5restart) for one model."""
    f = ROOT / "results" / dataset / model / "defense_PGD-AT" / seed / "defense_results_max1024.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    clean = r.get("clean_accuracy_defended")
    rob8 = None
    for k, v in r.items():
        if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
            if abs((_eps255(k) or 0) - 8) < 0.5:
                rob8 = v["robust_accuracy"]["full_robust_accuracy"]
    return clean, rob8


def make_h1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (ds, title, seeds) in zip(axes, DATASETS):
        for m in MODELS:
            # mean +/- std over available seeds, per eps
            per_eps = {}
            for s in seeds:
                for e, acc in pgd_points(ds, m, s).items():
                    if e <= EPS_MAX_255:
                        per_eps.setdefault(e, []).append(acc)
            if not per_eps:
                continue
            xs = sorted(per_eps)
            mean = np.array([np.mean(per_eps[e]) for e in xs]) * 100
            ax.plot(xs, mean, "-o", color=COLORS[m], label=LABELS[m], ms=4, lw=1.8)
            if len(seeds) > 1:
                std = np.array([np.std(per_eps[e]) for e in xs]) * 100
                ax.fill_between(xs, mean - std, mean + std, color=COLORS[m], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Perturbation budget $\epsilon$ (/255, log)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 100)
    axes[0].set_ylabel("Full robust accuracy (%)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("H1 — Standard training: PGD robustness across model complexity (non-monotonic, replicated across datasets)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"H1_pgd_across_datasets.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'H1_pgd_across_datasets.png'}")


def make_h2():
    at_models = ["resnet18", "resnet50", "resnet152"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(at_models))
    w = 0.38
    for ax, (ds, title, _) in zip(axes, DATASETS):
        clean_v, rob_v = [], []
        for m in at_models:
            ap = at_points(ds, m)
            clean_v.append((ap[0] or 0) * 100 if ap else 0)
            rob_v.append((ap[1] or 0) * 100 if ap else 0)
        ax.bar(x - w / 2, clean_v, w, label="AT clean acc", color="#4C72B0")
        ax.bar(x + w / 2, rob_v, w, label="AT robust acc @8/255", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m].replace("ResNet-", "R") for m in at_models])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(fontsize=9, loc="upper left")
    fig.suptitle("H2 — Adversarial training: clean vs robust accuracy (standard models ~0 @8/255 under strong PGD)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"H2_at_across_datasets.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'H2_at_across_datasets.png'}")


if __name__ == "__main__":
    make_h1()
    make_h2()
    make_paper_style()
    make_h1_dual_view()
