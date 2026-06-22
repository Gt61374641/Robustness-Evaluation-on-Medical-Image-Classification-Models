"""Generate Nature/SCI-style figures from robustness experiment results.

Reads the corrected clean and robustness JSON files and exports
publication-grade SVG/PDF/PNG figures with consistent typography and a
unified, semantically-grounded color palette. All styling is delegated to
``src.utils.plot_style`` so every figure in this project obeys the same
contract (Arial sans-serif, editable SVG text, 600 dpi raster fallback).
"""

import argparse
import json
import math
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
    ATTACK_COLORS,
    PALETTE,
    PALETTE_NMI_PASTEL,
    SIGNAL_COLORS,
    add_panel_label,
    apply_publication_style,
    class_color,
    finalize_figure,
    make_grouped_bar,
    make_trend,
)


ATTACK_ORDER = ["FGSM", "PGD", "AutoPGD", "SquareAttack", "DeepFool"]
REPRESENTATIVE = {
    "FGSM": 8,
    "PGD": 8,
    "AutoPGD": 8,
    "SquareAttack": 8,
    "DeepFool": None,
}


def _display_attack(name: str) -> str:
    """Compact display label (Nature figures avoid 'SquareAttack')."""
    return "Square" if name == "SquareAttack" else name


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_attack_key(key: str) -> tuple[str, float | None, int | None, str]:
    """Parse keys such as FGSM_eps=0.031372 or DeepFool_default."""
    if "_eps=" in key:
        attack, eps_text = key.split("_eps=", maxsplit=1)
        eps = float(eps_text)
        eps_255 = int(round(eps * 255))
        return attack, eps, eps_255, f"{_display_attack(attack)} {eps_255}/255"

    if key.endswith("_default"):
        attack = key.replace("_default", "")
        return attack, None, None, _display_attack(attack)

    return key, None, None, key


def metrics_to_rows(results: dict) -> list[dict]:
    rows = []
    for key, metrics in results.items():
        if key.startswith("_") or not isinstance(metrics, dict):
            continue
        if "error" in metrics or "robust_accuracy" not in metrics:
            continue

        attack, eps, eps_255, label = parse_attack_key(key)
        row = {
            "attack_key": key,
            "attack": attack,
            "eps": eps,
            "eps_255": eps_255,
            "label": label,
            "robust_accuracy": metrics["robust_accuracy"]["robust_accuracy"],
            "asr": metrics["asr"],
            "accuracy_drop": metrics["accuracy_drop"]["accuracy_drop"],
            "clean_accuracy": metrics["accuracy_drop"]["clean_accuracy"],
            "adversarial_accuracy": metrics["accuracy_drop"]["adversarial_accuracy"],
            "ece_clean": metrics.get("ece_clean", {}).get("ece", np.nan),
            "ece_adv": metrics.get("ece_adv", {}).get("ece", np.nan),
        }

        for class_name, class_metrics in metrics.get("per_class", {}).items():
            row[f"asr_{class_name}"] = class_metrics.get("asr", np.nan)
            row[f"robust_accuracy_{class_name}"] = class_metrics.get("robust_accuracy", np.nan)
            row[f"clean_accuracy_{class_name}"] = class_metrics.get("clean_accuracy", np.nan)

        rows.append(row)
    return rows


def load_experiment_frame(results_dir: Path, dataset: str, model: str, seed: str) -> pd.DataFrame:
    base = results_dir / dataset / model / "robustness" / seed
    paths = [
        base / "robustness_attacks_main.json",
        base / "robustness_attacks_extended.json",
    ]

    rows = []
    for path in paths:
        if path.exists():
            rows.extend(metrics_to_rows(load_json(path)))

    if not rows:
        raise FileNotFoundError(f"No robustness results found under {base}")

    df = pd.DataFrame(rows)
    attack_rank = {attack: i for i, attack in enumerate(ATTACK_ORDER)}
    df["_attack_rank"] = df["attack"].map(attack_rank).fillna(len(ATTACK_ORDER))
    df["_eps_sort"] = df["eps_255"].fillna(math.inf)
    return df.sort_values(["_attack_rank", "_eps_sort", "attack_key"]).reset_index(drop=True)


def load_clean_results(results_dir: Path, dataset: str, model: str, seed: str) -> dict:
    path = results_dir / dataset / model / "clean" / seed / "clean_results.json"
    if not path.exists():
        return {}
    return load_json(path)


def representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for attack in ATTACK_ORDER:
        subset = df[df["attack"] == attack]
        if subset.empty:
            continue
        eps_255 = REPRESENTATIVE[attack]
        if eps_255 is None:
            row = subset.iloc[0]
        else:
            exact = subset[subset["eps_255"] == eps_255]
            row = exact.iloc[0] if not exact.empty else subset.iloc[-1]
        selected.append(row)

    return pd.DataFrame(selected)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def plot_robust_accuracy_curves(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    eps_grid = sorted(df["eps_255"].dropna().unique())
    attack_lines = [
        attack for attack in ["FGSM", "PGD", "AutoPGD", "SquareAttack"]
        if not df[(df["attack"] == attack) & df["eps_255"].notna()].empty
    ]
    y_series, labels, colors = [], [], []
    for attack in attack_lines:
        subset = df[(df["attack"] == attack) & df["eps_255"].notna()].sort_values("eps_255")
        # align to global eps grid (NaN where missing)
        series = [
            float(subset.loc[subset["eps_255"] == eps, "robust_accuracy"].iloc[0])
            if (subset["eps_255"] == eps).any() else np.nan
            for eps in eps_grid
        ]
        y_series.append(series)
        labels.append(_display_attack(attack))
        colors.append(ATTACK_COLORS.get(attack, PALETTE["neutral_dark"]))

    make_trend(
        ax,
        eps_grid,
        y_series,
        labels,
        colors=colors,
        ylabel="Robust accuracy",
        xlabel=r"Perturbation budget, $\epsilon$ (/255)",
    )

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(eps_grid)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(ncol=2, loc="upper right")
    add_panel_label(ax, "a")
    ax.set_title("Robustness across attack budgets")

    return finalize_figure(fig, output_dir / "sci_robust_accuracy_curves", pad=1.0)


def plot_attack_summary(rep: pd.DataFrame, output_dir: Path) -> list[Path]:
    categories = list(rep["label"])
    series = [rep["robust_accuracy"].to_numpy(), rep["asr"].to_numpy()]
    labels = ["Robust accuracy", "Attack success rate"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    make_grouped_bar(
        ax,
        categories,
        series,
        labels,
        ylabel="Metric value",
        colors=[PALETTE["blue_main"], PALETTE["red_strong"]],
        bar_width=0.72,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(loc="upper left")
    add_panel_label(ax, "b")
    ax.set_title("Attack-level performance")

    return finalize_figure(fig, output_dir / "sci_attack_summary_bars", pad=1.0)


def plot_calibration_shift(rep: pd.DataFrame, clean_results: dict, output_dir: Path) -> list[Path]:
    categories = list(rep["label"])
    clean_ece = clean_results.get("ece")
    if clean_ece is None:
        clean_ece_values = rep["ece_clean"].dropna()
        clean_ece = float(clean_ece_values.iloc[0]) if not clean_ece_values.empty else 0.0
    series = [
        np.full(len(rep), clean_ece, dtype=float),
        rep["ece_adv"].to_numpy(dtype=float),
    ]
    labels = ["Clean ECE", "Adversarial ECE"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    make_grouped_bar(
        ax,
        categories,
        series,
        labels,
        ylabel="Expected calibration error",
        colors=[SIGNAL_COLORS["Clean"], SIGNAL_COLORS["Adv"]],
        bar_width=0.72,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(1.0, float(rep["ece_adv"].max()) * 1.08))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(loc="upper left")
    add_panel_label(ax, "c")
    ax.set_title("Calibration under attack")

    return finalize_figure(fig, output_dir / "sci_calibration_ece_shift", pad=1.0)


def plot_per_class_vulnerability(rep: pd.DataFrame, output_dir: Path) -> list[Path]:
    categories = list(rep["label"])
    classes = [col.replace("asr_", "") for col in rep.columns if col.startswith("asr_")]
    if not classes:
        return []

    series = [rep[f"asr_{c}"].to_numpy(dtype=float) for c in classes]
    colors = [class_color(c, idx) for idx, c in enumerate(classes)]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    make_grouped_bar(
        ax,
        categories,
        series,
        classes,
        ylabel="Attack success rate",
        colors=colors,
        bar_width=0.72,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(loc="upper left", ncol=min(len(classes), 3))
    add_panel_label(ax, "d")
    ax.set_title("Class-specific vulnerability")

    return finalize_figure(fig, output_dir / "sci_per_class_vulnerability", pad=1.0)


def plot_accuracy_drop_ranking(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    ranked = df.sort_values("accuracy_drop", ascending=True).copy()
    ranked["display_label"] = ranked["label"]
    colors = [ATTACK_COLORS.get(attack, PALETTE["neutral_mid"]) for attack in ranked["attack"]]

    fig_height = max(2.6, 0.16 * len(ranked) + 0.8)
    fig, ax = plt.subplots(figsize=(3.5, fig_height))
    ax.barh(
        ranked["display_label"],
        ranked["accuracy_drop"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Accuracy drop")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, min(1.0, max(0.05, float(ranked["accuracy_drop"].max()) * 1.08)))
    add_panel_label(ax, "e")
    ax.set_title("Accuracy degradation ranking")

    return finalize_figure(fig, output_dir / "sci_accuracy_drop_ranking", pad=1.0)


def plot_clean_performance(clean_results: dict, output_dir: Path) -> list[Path]:
    if not clean_results:
        return []

    rows = [{"label": "Overall", "accuracy": clean_results["accuracy"]}]
    for class_name, metrics in clean_results.get("per_class", {}).items():
        rows.append({"label": class_name, "accuracy": metrics["accuracy"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    colors = [PALETTE["neutral_dark"]] + [
        class_color(label, idx) for idx, label in enumerate(df["label"].iloc[1:])
    ]
    ax.bar(
        df["label"],
        df["accuracy"],
        color=colors,
        edgecolor="black",
        linewidth=0.45,
    )
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    add_panel_label(ax, "f")
    ax.set_title("Clean-test performance")

    return finalize_figure(fig, output_dir / "sci_clean_performance", pad=1.0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature/SCI-style robustness figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "sci")
    parser.add_argument("--dataset", default="chest_xray_pneumonia")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--seed", default="seed42")
    args = parser.parse_args()

    apply_publication_style()
    fig_dir = args.output_dir / args.dataset / args.model
    df = load_experiment_frame(args.results_dir, args.dataset, args.model, args.seed)
    clean_results = load_clean_results(args.results_dir, args.dataset, args.model, args.seed)
    rep = representative_rows(df)

    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fig_dir / "sci_summary_metrics.csv"
    df.drop(columns=["_attack_rank", "_eps_sort"]).to_csv(summary_path, index=False)

    outputs = [summary_path]
    outputs.extend(plot_robust_accuracy_curves(df, fig_dir))
    outputs.extend(plot_attack_summary(rep, fig_dir))
    outputs.extend(plot_calibration_shift(rep, clean_results, fig_dir))
    outputs.extend(plot_per_class_vulnerability(rep, fig_dir))
    outputs.extend(plot_accuracy_drop_ranking(df, fig_dir))
    outputs.extend(plot_clean_performance(clean_results, fig_dir))

    print(f"Generated {len(outputs) - 1} figure files and 1 CSV:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
