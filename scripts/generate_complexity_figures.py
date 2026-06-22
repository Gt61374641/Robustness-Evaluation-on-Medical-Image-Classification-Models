"""Complexity-vs-robustness figures (the signature analysis).

Overlays the ResNet complexity ladder on a single accuracy-vs-epsilon axis (one
line per model complexity), separately for FGSM and PGD — the medical-imaging
analogue of Rodriguez et al. 2022, Fig 1. Also emits a complexity-vs-robustness
summary table (Table 1) and a standard-vs-PGD-AT comparison (Fig 2 / Table 2).

The y-axis is FULL robust accuracy (clean-correct AND still-correct-after-attack
over ALL samples), anchored at epsilon=0 by the recorded clean accuracy.

Usage:
    python scripts/generate_complexity_figures.py --dataset chest_xray_pneumonia --seed seed42
    python scripts/generate_complexity_figures.py --dataset oct2017 --models resnet18 resnet50
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_style import (
    apply_publication_style,
    add_panel_label,
    finalize_figure,
    make_grouped_bar,
    make_trend,
    PALETTE,
)

LADDER = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
DISPLAY = {
    "resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
    "resnet101": "ResNet-101", "resnet152": "ResNet-152",
}
# Static fallback param counts (M); get_model_info is used first if timm is available.
PARAMS_M = {"resnet18": 11.7, "resnet34": 21.8, "resnet50": 25.6,
            "resnet101": 44.5, "resnet152": 60.2}


def _params_m(model: str) -> float:
    try:
        from src.models.model_factory import get_model_info
        return get_model_info(model)["num_params_M"]
    except Exception:
        return PARAMS_M.get(model, float("nan"))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_results(base: Path, stem: str) -> Path | None:
    """Prefer the plain file, else the largest-sample _max variant."""
    plain = base / f"{stem}.json"
    if plain.exists():
        return plain
    candidates = sorted(base.glob(f"{stem}_max*.json"))
    return candidates[-1] if candidates else None


def _parse_eps_255(key: str):
    """Return (attack_name, eps_in_255) for keys like 'PGD_eps=0.031373'.

    eps_255 is kept as a FLOAT (rounded to 4 dp) so sub-1/255 fine-grid points
    (0.05, 0.1, 0.15 /255) stay distinct instead of all rounding to 0.
    """
    if "_eps=" not in key:
        return None, None
    attack, eps_text = key.split("_eps=", 1)
    return attack, round(float(eps_text) * 255, 4)


def _series(results: dict, attack_prefix: str, clean_acc: float):
    """Extract sorted (eps_255 -> full robust acc + bootstrap CI), clean-anchored at eps=0."""
    pts = {}
    if clean_acc is not None:
        pts[0] = (clean_acc, clean_acc, clean_acc)
    for key, metrics in results.items():
        if key.startswith("_") or not isinstance(metrics, dict) or "robust_accuracy" not in metrics:
            continue
        attack, eps255 = _parse_eps_255(key)
        if attack is None or not attack.startswith(attack_prefix):
            continue
        ra = metrics["robust_accuracy"]
        val = ra.get("full_robust_accuracy", ra.get("robust_accuracy"))
        pts[eps255] = (val, ra.get("full_ci_low", val), ra.get("full_ci_high", val))
    if not pts:
        return [], [], [], []
    xs = sorted(pts)
    return xs, [pts[x][0] for x in xs], [pts[x][1] for x in xs], [pts[x][2] for x in xs]


def _value_at(results: dict, attack_prefix: str, eps_255: float, field: str):
    """Robust-accuracy field for a given attack prefix at a specific eps (in /255)."""
    for key, metrics in results.items():
        if key.startswith("_") or not isinstance(metrics, dict) or "robust_accuracy" not in metrics:
            continue
        attack, e = _parse_eps_255(key)
        if attack and attack.startswith(attack_prefix) and e is not None and abs(e - eps_255) < 1e-4:
            ra = metrics["robust_accuracy"]
            return ra.get(field, ra.get("robust_accuracy"))
    return None


def load_model_curves(results_dir: Path, dataset: str, model: str, seed: str):
    """Merge fine + main robustness results (the PGD separation lives sub-1/255,
    so the fine probe must be included). Returns {results, clean} or None."""
    base = results_dir / dataset / model / "robustness" / seed
    merged, clean = {}, None
    for stem in ("robustness_attacks_fine", "robustness_attacks_main"):
        path = _find_results(base, stem)
        if path is None:
            continue
        j = load_json(path)
        clean = clean or j.get("_meta", {}).get("clean_accuracy")
        for k, v in j.items():
            if not k.startswith("_"):
                merged[k] = v
    if not merged:
        return None
    return {"results": merged, "clean": clean}


def complexity_colors(n: int):
    cmap = plt.get_cmap("viridis")
    return [cmap(t) for t in np.linspace(0.12, 0.88, n)]


def _agg_series(seed_list, attack_prefix):
    """Aggregate full-robust-acc vs eps across seeds.

    seed_list: list of {results, clean} (one per seed). With >1 seed the band is
    mean +/- std ACROSS seeds (between-run variance); with 1 seed it falls back to
    that seed's bootstrap CI (within-run). Returns (eps, mean, lo, hi) on eps points
    present in every seed.
    """
    per_seed = []
    for d in seed_list:
        xs, ys, lo, hi = _series(d["results"], attack_prefix, d["clean"])
        per_seed.append({x: (y, l, h) for x, y, l, h in zip(xs, ys, lo, hi)})
    if not per_seed:
        return [], [], [], []
    common = set(per_seed[0])
    for s in per_seed[1:]:
        common &= set(s)
    xs = sorted(x for x in common if x > 0)  # drop clean(0) for the log axis
    if not xs:
        return [], [], [], []
    n = len(per_seed)
    mean, lo, hi = [], [], []
    for x in xs:
        vals = [s[x][0] for s in per_seed]
        m = float(np.mean(vals)); mean.append(m)
        if n > 1:
            sd = float(np.std(vals))
            lo.append(max(0.0, m - sd)); hi.append(min(1.0, m + sd))
        else:
            lo.append(per_seed[0][x][1]); hi.append(per_seed[0][x][2])
    return xs, mean, lo, hi


def plot_curves(model_data, models, attack_prefix, panel, title, out_path, n_seeds=1, max_eps_255=None):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    colors = complexity_colors(len(models))
    plotted = 0
    for model, color in zip(models, colors):
        seed_list = model_data.get(model)
        if not seed_list:
            continue
        xs, ys, lo, hi = _agg_series(seed_list, attack_prefix)
        if max_eps_255 is not None:  # restrict to the meaningful regime (avoid large-eps collapse artifact)
            keep = [i for i, x in enumerate(xs) if x <= max_eps_255]
            xs = [xs[i] for i in keep]; ys = [ys[i] for i in keep]
            lo = [lo[i] for i in keep]; hi = [hi[i] for i in keep]
        if not xs:
            continue
        make_trend(ax, xs, [ys], [DISPLAY.get(model, model)], colors=[color])
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xscale("log")
    band = f"mean $\\pm$ s.d. over {n_seeds} seeds" if n_seeds > 1 else "bootstrap 95% CI"
    ax.set_xlabel(r"Perturbation budget $\epsilon$ (/255, log)")
    ax.set_ylabel(f"Full robust accuracy\n(band: {band})")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6, loc="upper right")
    add_panel_label(ax, panel)
    ax.set_title(title)
    return finalize_figure(fig, out_path, pad=1.0)


def _mean_std_across_seeds(seed_list, prefix, eps, field):
    vals = [v for v in (_value_at(d["results"], prefix, eps, field) for d in seed_list) if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def build_summary_table(model_data, models, table_eps_255, out_csv):
    rows = []
    for model in models:
        seed_list = model_data.get(model)
        if not seed_list:
            continue
        n = len(seed_list)
        clean_vals = [d["clean"] for d in seed_list if d["clean"] is not None]
        fgsm_m, fgsm_sd = _mean_std_across_seeds(seed_list, "FGSM", table_eps_255, "full_robust_accuracy")
        pgd_m, pgd_sd = _mean_std_across_seeds(seed_list, "PGD", table_eps_255, "full_robust_accuracy")
        pgd_cm, _ = _mean_std_across_seeds(seed_list, "PGD", table_eps_255, "conditional_robust_accuracy")
        rows.append({
            "model": model,
            "params_M": _params_m(model),
            "n_seeds": n,
            "clean_accuracy": float(np.mean(clean_vals)) if clean_vals else None,
            "clean_std": float(np.std(clean_vals)) if len(clean_vals) > 1 else 0.0,
            "fgsm_full_robust": fgsm_m,
            "fgsm_full_robust_std": fgsm_sd,
            "pgd_full_robust": pgd_m,
            "pgd_full_robust_std": pgd_sd,
            "pgd_conditional_robust": pgd_cm,
        })
    df = pd.DataFrame(rows).sort_values("params_M").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_at_curves(results_dir, dataset, models, seed, out_dir):
    """AT acc-vs-eps curves (paper Fig 2): per complexity, PGD-AT solid vs Standard dashed."""
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    colors = complexity_colors(len(models))
    plotted = 0
    for model, color in zip(models, colors):
        std_path = _find_results(results_dir / dataset / model / "robustness" / seed,
                                 "robustness_attacks_main")
        at_path = _find_results(results_dir / dataset / model / "defense_PGD-AT" / seed,
                                "defense_results")
        if at_path is None:
            continue
        at_res = load_json(at_path)
        at_clean = at_res.get("clean_accuracy_defended", at_res.get("_meta", {}).get("clean_accuracy_defended"))
        axs, ays, alo, ahi = _series(at_res, "PGD", at_clean)
        keep = [i for i, x in enumerate(axs) if x > 0]
        if keep:
            axs = [axs[i] for i in keep]; ays = [ays[i] for i in keep]
            alo = [alo[i] for i in keep]; ahi = [ahi[i] for i in keep]
            make_trend(ax, axs, [ays], [DISPLAY.get(model, model)], colors=[color])
            ax.fill_between(axs, alo, ahi, color=color, alpha=0.13, linewidth=0)
            plotted += 1
        if std_path is not None:
            std = load_json(std_path)
            sxs, sys_, _, _ = _series(std, "PGD", std.get("_meta", {}).get("clean_accuracy"))
            sk = [i for i, x in enumerate(sxs) if x > 0]
            if sk:
                ax.plot([sxs[i] for i in sk], [sys_[i] for i in sk],
                        color=color, lw=1.0, linestyle="--", alpha=0.55, marker=None)
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xscale("log")
    ax.set_xlabel(r"Perturbation budget $\epsilon$ (/255, log)")
    ax.set_ylabel("Full robust accuracy (PGD)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6, loc="upper right", title="PGD-AT (— solid), Standard (-- dashed)",
              title_fontsize=5)
    add_panel_label(ax, "a")
    ax.set_title("Adversarial training across complexity")
    return finalize_figure(fig, out_dir / "complexity_at_curves", pad=1.0)


def plot_at_comparison(results_dir, dataset, models, seed, table_eps_255, out_dir):
    """Standard vs PGD-AT robustness at the headline eps, per complexity.

    Includes AutoAttack (gold-standard) for the defended model when available.
    """
    std_vals, at_vals, aa_vals, cats = [], [], [], []
    for model in models:
        std_path = _find_results(results_dir / dataset / model / "robustness" / seed,
                                 "robustness_attacks_main")
        at_path = _find_results(results_dir / dataset / model / "defense_PGD-AT" / seed,
                                "defense_results")
        if std_path is None or at_path is None:
            continue
        at_json = load_json(at_path)
        std = _value_at(load_json(std_path), "PGD", table_eps_255, "full_robust_accuracy")
        at = _value_at(at_json, "PGD", table_eps_255, "full_robust_accuracy")          # PGD50-5restart
        aa = _value_at(at_json, "AutoAttack", table_eps_255, "full_robust_accuracy")   # may be None
        if std is None or at is None:
            continue
        cats.append(DISPLAY.get(model, model))
        std_vals.append(std)
        at_vals.append(at)
        aa_vals.append(aa)

    if not cats:
        return [], None

    have_aa = any(v is not None for v in aa_vals)
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    series = [np.array(std_vals, dtype=float), np.array(at_vals, dtype=float)]
    labels = ["Standard", "PGD-AT (PGD-50)"]
    colors = [PALETTE["neutral_mid"], PALETTE["blue_main"]]
    if have_aa:
        series.append(np.array([np.nan if v is None else v for v in aa_vals], dtype=float))
        labels.append("PGD-AT (AutoAttack)")
        colors.append(PALETTE["red_strong"])
    make_grouped_bar(ax, cats, series, labels, ylabel="Full robust accuracy",
                     colors=colors, bar_width=0.78)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.03)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=5, loc="upper left")
    add_panel_label(ax, "c")
    ax.set_title(f"Standard vs PGD-AT @ {table_eps_255}/255")
    paths = finalize_figure(fig, out_dir / "complexity_at_comparison", pad=1.0)

    df = pd.DataFrame({"model": cats, "standard_pgd_full_robust": std_vals,
                       "pgd_at_pgd50_full_robust": at_vals,
                       "pgd_at_autoattack_full_robust": aa_vals})
    csv = out_dir / "at_comparison_table.csv"
    df.to_csv(csv, index=False)
    return paths, csv


def main():
    parser = argparse.ArgumentParser(description="Complexity-vs-robustness figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "complexity")
    parser.add_argument("--dataset", default="chest_xray_pneumonia")
    parser.add_argument("--seeds", nargs="+", default=["seed42"],
                        help="One or more seed dirs (e.g. seed42 seed43 seed44). >1 -> mean+/-std bands.")
    parser.add_argument("--models", nargs="+", default=LADDER)
    parser.add_argument("--table-eps", type=float, default=0.1,
                        help="Headline eps in /255 (float) for the STANDARD complexity table. Default "
                             "0.1 sits in the discriminating fine regime.")
    parser.add_argument("--at-eps", type=float, default=8.0,
                        help="Headline eps in /255 for the AT comparison (where adversarial training's "
                             "benefit shows and standard models are already ~0). Default 8.")
    args = parser.parse_args()

    apply_publication_style()
    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    model_data = {}
    for model in args.models:
        seed_list = [load_model_curves(args.results_dir, args.dataset, model, s) for s in args.seeds]
        seed_list = [d for d in seed_list if d is not None]
        if not seed_list:
            print(f"[warn] no robustness results for {model}; skipping.")
        else:
            model_data[model] = seed_list
    if not model_data:
        sys.exit(f"No robustness results found under {args.results_dir/args.dataset}.")

    n_seeds = max(len(v) for v in model_data.values())
    print(f"Using up to {n_seeds} seed(s): {args.seeds}")

    outputs = []
    outputs += plot_curves(model_data, args.models, "FGSM", "a",
                           "FGSM robustness across complexity",
                           out_dir / "complexity_robustness_curves_fgsm", n_seeds=n_seeds)
    outputs += plot_curves(model_data, args.models, "PGD", "b",
                           "PGD robustness across complexity",
                           out_dir / "complexity_robustness_curves_pgd", n_seeds=n_seeds,
                           max_eps_255=0.3)  # PGD on standard models is only meaningful sub-0.25/255

    table_csv = out_dir / "complexity_summary_table.csv"
    df = build_summary_table(model_data, args.models, args.table_eps, table_csv)
    outputs.append(table_csv)
    print(df.to_string(index=False))

    # AT comparison uses the primary (first) seed only.
    primary = args.seeds[0]
    outputs += plot_at_curves(args.results_dir, args.dataset, args.models, primary, out_dir)
    at_paths, at_csv = plot_at_comparison(args.results_dir, args.dataset, args.models,
                                          primary, args.at_eps, out_dir)
    outputs += at_paths
    if at_csv:
        outputs.append(at_csv)
    else:
        print("[info] no PGD-AT results yet; skipped standard-vs-AT comparison.")

    print(f"\nGenerated {len(outputs)} files:")
    for p in outputs:
        print(" ", p)


if __name__ == "__main__":
    main()
