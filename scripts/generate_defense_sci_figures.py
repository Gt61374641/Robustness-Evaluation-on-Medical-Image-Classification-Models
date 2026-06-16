"""Generate SCI-style defense comparison figures.

This script compares the standard model, adversarial-training defenses, and
preprocessing baselines using the corrected defense_results.json files.
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
    METHOD_COLORS,
    PALETTE,
    add_panel_label,
    apply_publication_style,
    class_color,
    finalize_figure,
)


METHOD_ORDER = [
    "Standard",
    "PGD-AT",
    "TRADES",
    "SpatialSmoothing",
    "JpegCompression",
    "FeatureSqueezing",
]

CATEGORY = {
    "Standard": "Standard model",
    "PGD-AT": "Adversarial training",
    "TRADES": "Adversarial training",
    "SpatialSmoothing": "Preprocessing baseline",
    "JpegCompression": "Preprocessing baseline",
    "FeatureSqueezing": "Preprocessing baseline",
}

COLORS = {
    **METHOD_COLORS,
    "Normal": class_color("Normal"),
    "Pneumonia": class_color("Pneumonia"),
}


def apply_sci_style() -> None:
    apply_publication_style(font_size=8.0, axes_linewidth=0.8, grid=False)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_attack_key(key: str) -> tuple[str, float | None, int | None, str]:
    if "_eps=" in key:
        attack, eps_text = key.split("_eps=", maxsplit=1)
        eps = float(eps_text)
        eps_255 = int(round(eps * 255))
        return attack, eps, eps_255, f"{attack} {eps_255}/255"
    if key.endswith("_default"):
        attack = key.replace("_default", "")
        return attack, None, None, attack
    return key, None, None, key


def append_result_rows(rows: list[dict], method: str, results: dict, clean_accuracy: float) -> None:
    for key, metrics in results.items():
        if key.startswith("_") or key == "clean_accuracy_defended":
            continue
        if not isinstance(metrics, dict) or "error" in metrics or "robust_accuracy" not in metrics:
            continue

        attack, eps, eps_255, label = parse_attack_key(key)
        row = {
            "method": method,
            "category": CATEGORY[method],
            "attack_key": key,
            "attack": attack,
            "eps": eps,
            "eps_255": eps_255,
            "label": label,
            "clean_accuracy": clean_accuracy,
            "robust_accuracy": metrics["robust_accuracy"]["robust_accuracy"],
            "asr": metrics["asr"],
            "accuracy_drop": metrics["accuracy_drop"]["accuracy_drop"],
            "ece_clean": metrics.get("ece_clean", {}).get("ece", np.nan),
            "ece_adv": metrics.get("ece_adv", {}).get("ece", np.nan),
        }

        for class_name, class_metrics in metrics.get("per_class", {}).items():
            row[f"asr_{class_name}"] = class_metrics.get("asr", np.nan)
            row[f"robust_accuracy_{class_name}"] = class_metrics.get("robust_accuracy", np.nan)
            row[f"clean_accuracy_{class_name}"] = class_metrics.get("clean_accuracy", np.nan)

        rows.append(row)


def load_defense_frame(results_dir: Path, dataset: str, model: str, seed: str) -> pd.DataFrame:
    rows = []

    clean_path = results_dir / dataset / model / "clean" / seed / "clean_results.json"
    standard_clean = load_json(clean_path)["accuracy"]
    standard_results = load_json(
        results_dir / dataset / model / "robustness" / seed / "robustness_attacks_main.json"
    )
    append_result_rows(rows, "Standard", standard_results, standard_clean)

    for method in METHOD_ORDER:
        if method == "Standard":
            continue
        path = results_dir / dataset / model / f"defense_{method}" / seed / "defense_results.json"
        if not path.exists():
            continue
        results = load_json(path)
        clean_accuracy = results.get("clean_accuracy_defended")
        if clean_accuracy is None:
            clean_accuracy = results.get("_meta", {}).get("clean_accuracy_defended", np.nan)
        append_result_rows(rows, method, results, clean_accuracy)

    if not rows:
        raise FileNotFoundError("No defense results were found.")

    df = pd.DataFrame(rows)
    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}
    df["_method_rank"] = df["method"].map(method_rank).fillna(len(METHOD_ORDER))
    df["_eps_sort"] = df["eps_255"].fillna(math.inf)
    return df.sort_values(["_method_rank", "attack", "_eps_sort"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    return finalize_figure(fig, output_dir / name, pad=1.0)


def get_attack_subset(df: pd.DataFrame, attack: str, eps_255: int | None = None) -> pd.DataFrame:
    subset = df[df["attack"] == attack].copy()
    if eps_255 is not None:
        subset = subset[subset["eps_255"] == eps_255]
    order = {method: i for i, method in enumerate(METHOD_ORDER)}
    subset["_order"] = subset["method"].map(order)
    return subset.sort_values("_order")


def plot_pgd_curves(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    for method in METHOD_ORDER:
        subset = df[(df["method"] == method) & (df["attack"] == "PGD")]
        subset = subset[subset["eps_255"].notna()].sort_values("eps_255")
        if subset.empty:
            continue
        linestyle = "--" if CATEGORY[method] == "Preprocessing baseline" else "-"
        ax.plot(
            subset["eps_255"],
            subset["robust_accuracy"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            linestyle=linestyle,
            color=COLORS[method],
            label=method,
        )

    ax.set_xlabel(r"Perturbation budget, $\epsilon$ (/255)")
    ax.set_ylabel("Robust accuracy under PGD")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_ylim(-0.03, 1.03)
    ax.legend(ncol=2, loc="upper right", handlelength=1.7)
    add_panel_label(ax, "a")
    ax.set_title("PGD robustness after defense")
    return save_figure(fig, output_dir, "sci_defense_pgd_curves")


def plot_pgd8_bars(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = get_attack_subset(df, "PGD", eps_255=8)
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.bar(
        x,
        subset["robust_accuracy"],
        color=[COLORS[m] for m in subset["method"]],
        edgecolor="black",
        linewidth=0.45,
    )
    ax.set_ylabel("Robust accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["method"], rotation=35, ha="right")
    add_panel_label(ax, "b")
    ax.set_title("Defense comparison at PGD 8/255")
    return save_figure(fig, output_dir, "sci_defense_pgd8_bars")


def plot_clean_pgd_tradeoff(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = get_attack_subset(df, "PGD", eps_255=8)
    label_offsets = {
        "Standard": (-46, 8),
        "PGD-AT": (6, 6),
        "TRADES": (6, 6),
        "SpatialSmoothing": (8, 6),
        "JpegCompression": (8, 18),
        "FeatureSqueezing": (8, 30),
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for _, row in subset.iterrows():
        marker = "s" if row["category"] == "Preprocessing baseline" else "o"
        ax.scatter(
            row["clean_accuracy"],
            row["robust_accuracy"],
            s=32,
            marker=marker,
            color=COLORS[row["method"]],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        offset = label_offsets.get(row["method"], (4, 3))
        ax.annotate(
            row["method"],
            (row["clean_accuracy"], row["robust_accuracy"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.5,
        )

    ax.set_xlabel("Clean accuracy")
    ax.set_ylabel("PGD 8/255 robust accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.74, 0.94)
    ax.set_ylim(-0.03, 1.03)
    add_panel_label(ax, "c")
    ax.set_title("Clean-robustness trade-off")
    return save_figure(fig, output_dir, "sci_defense_clean_pgd_tradeoff")


def plot_deepfool_vs_pgd(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    pgd = get_attack_subset(df, "PGD", eps_255=8)[["method", "robust_accuracy"]]
    deepfool = get_attack_subset(df, "DeepFool")[["method", "robust_accuracy"]]
    merged = pgd.merge(deepfool, on="method", suffixes=("_pgd8", "_deepfool"))
    merged["_order"] = merged["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    merged = merged.sort_values("_order")

    x = np.arange(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    ax.bar(
        x - width / 2,
        merged["robust_accuracy_pgd8"],
        width=width,
        color=ATTACK_COLORS.get("PGD", PALETTE["blue_main"]),
        edgecolor="black",
        linewidth=0.45,
        label="PGD 8/255",
    )
    ax.bar(
        x + width / 2,
        merged["robust_accuracy_deepfool"],
        width=width,
        color=ATTACK_COLORS.get("DeepFool", PALETTE["violet"]),
        edgecolor="black",
        linewidth=0.45,
        label="DeepFool",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(merged["method"], rotation=35, ha="right")
    ax.set_ylabel("Robust accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    add_panel_label(ax, "d")
    ax.set_title("Attack-dependent defense behavior")
    return save_figure(fig, output_dir, "sci_defense_pgd8_vs_deepfool")


def plot_per_class_asr(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = get_attack_subset(df, "PGD", eps_255=8)
    classes = [col.replace("asr_", "") for col in subset.columns if col.startswith("asr_")]
    if not classes:
        return []

    x = np.arange(len(subset))
    width = 0.72 / len(classes)
    offsets = (np.arange(len(classes)) - (len(classes) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(3.65, 2.5))
    for offset, class_name in zip(offsets, classes):
        ax.bar(
            x + offset,
            subset[f"asr_{class_name}"],
            width=width,
            color=COLORS.get(class_name, PALETTE["neutral_mid"]),
            edgecolor="black",
            linewidth=0.45,
            label=class_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["method"], rotation=35, ha="right")
    ax.set_ylabel("Attack success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    add_panel_label(ax, "e")
    ax.set_title("Class-specific ASR under PGD 8/255")
    return save_figure(fig, output_dir, "sci_defense_per_class_asr_pgd8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SCI-style defense comparison figures")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "sci_defense")
    parser.add_argument("--dataset", default="chest_xray_pneumonia")
    parser.add_argument("--model", default="densenet121")
    parser.add_argument("--seed", default="seed42")
    args = parser.parse_args()

    apply_sci_style()
    fig_dir = args.output_dir / args.dataset / args.model
    df = load_defense_frame(args.results_dir, args.dataset, args.model, args.seed)

    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fig_dir / "sci_defense_summary_metrics.csv"
    df.drop(columns=["_method_rank", "_eps_sort"]).to_csv(summary_path, index=False)

    outputs = [summary_path]
    outputs.extend(plot_pgd_curves(df, fig_dir))
    outputs.extend(plot_pgd8_bars(df, fig_dir))
    outputs.extend(plot_clean_pgd_tradeoff(df, fig_dir))
    outputs.extend(plot_deepfool_vs_pgd(df, fig_dir))
    outputs.extend(plot_per_class_asr(df, fig_dir))

    print(f"Generated {len(outputs) - 1} defense figure files and 1 CSV:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
