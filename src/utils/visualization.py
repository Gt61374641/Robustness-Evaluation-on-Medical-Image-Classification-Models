"""Publication-style visualization utilities for robustness experiments.

These helpers keep the older report-generation API, but all styling and export
rules now come from :mod:`src.utils.plot_style`: editable SVG text, vector PDF,
600 dpi raster fallbacks, restrained semantic colors, and clean Nature-style
axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from src.utils.plot_style import (
    ATTACK_COLORS,
    METHOD_COLORS,
    PALETTE,
    SIGNAL_COLORS,
    add_panel_label,
    apply_publication_style,
    class_color,
    make_grouped_bar,
    make_heatmap,
    save_or_show_figure,
    style_dark_image_ax,
)


apply_publication_style(font_size=8.0, axes_linewidth=0.8, grid=False)


def _label(index: int, class_names: list | None) -> str:
    if class_names:
        return str(class_names[index])
    return str(index)


def plot_adversarial_examples(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    class_names: list | None = None,
    n_samples: int = 5,
    save_path: str | None = None,
):
    """Visualize clean image, amplified perturbation, and adversarial image."""
    n_samples = min(n_samples, len(x_clean))
    fig, axes = plt.subplots(
        n_samples,
        3,
        figsize=(7.1, max(1.8, 1.25 * n_samples)),
        squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.32},
    )

    for i in range(n_samples):
        img_clean = np.transpose(x_clean[i], (1, 2, 0))
        img_adv = np.transpose(x_adv[i], (1, 2, 0))
        perturbation_vis = np.clip(0.5 + (img_adv - img_clean) * 10, 0, 1)

        panels = [
            (img_clean, f"Clean: {_label(int(y_pred_clean[i]), class_names)}"),
            (perturbation_vis, "Perturbation x10"),
            (img_adv, f"Adv: {_label(int(y_pred_adv[i]), class_names)}"),
        ]
        true_label = _label(int(y_true[i]), class_names)

        for j, (image, title) in enumerate(panels):
            ax = axes[i, j]
            style_dark_image_ax(ax, facecolor="black")
            ax.imshow(np.clip(image, 0, 1))
            subtitle = f"{title}\nTrue: {true_label}" if j == 0 else title
            ax.set_title(subtitle, color="black", pad=2)

    add_panel_label(axes[0, 0], "a", x=-0.1, y=1.12)
    return save_or_show_figure(fig, save_path, pad=0.2)


def plot_eps_vs_accuracy(
    results: dict,
    attack_name: str = "PGD",
    save_path: str | None = None,
):
    """Plot perturbation budget against robust accuracy for one attack."""
    eps_values = []
    robust_accs = []

    for key, metrics in results.items():
        if not key.startswith(attack_name) or not isinstance(metrics, dict) or "error" in metrics:
            continue
        parts = key.split("eps=")
        if len(parts) == 2:
            eps = float(parts[1])
            eps_values.append(eps)
            robust_accs.append(metrics["robust_accuracy"]["robust_accuracy"])

    if not eps_values:
        print(f"No results found for attack: {attack_name}")
        return []

    eps_values, robust_accs = zip(*sorted(zip(eps_values, robust_accs)))
    eps_255 = [round(eps * 255) for eps in eps_values]

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    color = ATTACK_COLORS.get(attack_name, PALETTE["blue_main"])
    ax.plot(
        eps_255,
        robust_accs,
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        color=color,
        markeredgecolor=color,
    )
    ax.set_xlabel(r"Perturbation budget, $\epsilon$ (/255)")
    ax.set_ylabel("Robust accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(eps_255)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{attack_name} robustness curve")
    add_panel_label(ax, "a")

    return save_or_show_figure(fig, save_path, pad=1.0)


def plot_robustness_heatmap(
    results_by_model: dict,
    eps: float | None = None,
    save_path: str | None = None,
):
    """Plot a model-by-attack robust-accuracy heatmap."""
    models = list(results_by_model.keys())
    attacks = sorted({
        key
        for model_results in results_by_model.values()
        for key, value in model_results.items()
        if isinstance(value, dict) and "error" not in value
    })

    matrix = np.full((len(models), len(attacks)), np.nan)
    for i, model in enumerate(models):
        for j, attack in enumerate(attacks):
            metrics = results_by_model[model].get(attack)
            if isinstance(metrics, dict) and "error" not in metrics:
                matrix[i, j] = metrics["robust_accuracy"]["robust_accuracy"]

    fig, ax = plt.subplots(figsize=(max(3.6, len(attacks) * 0.5), max(2.2, len(models) * 0.35 + 1.2)))
    make_heatmap(
        ax,
        matrix,
        x_labels=attacks,
        y_labels=models,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_label="Robust accuracy",
        annotate=True,
        fmt="{:.2f}",
    )
    ax.set_xlabel("Attack")
    ax.set_ylabel("Model")
    ax.set_title("Robust accuracy across attacks")
    add_panel_label(ax, "a")

    return save_or_show_figure(fig, save_path, pad=1.0)


def plot_defense_comparison(
    standard_results: dict,
    at_results: dict,
    trades_results: dict,
    attack_name: str = "PGD",
    eps: float = 8 / 255,
    save_path: str | None = None,
):
    """Plot robust accuracy for standard, PGD-AT, and TRADES models."""
    key = f"{attack_name}_eps={eps:.6f}"
    labels = ["Standard", "PGD-AT", "TRADES"]
    result_sets = [standard_results, at_results, trades_results]
    robust_accs = [
        result.get(key, {}).get("robust_accuracy", {}).get("robust_accuracy", 0.0)
        if isinstance(result, dict) else 0.0
        for result in result_sets
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    make_grouped_bar(
        ax,
        labels,
        [robust_accs],
        ["Robust accuracy"],
        ylabel="Robust accuracy",
        colors=[METHOD_COLORS["PGD-AT"]],
        annotate=True,
        bar_width=0.58,
    )
    for bar, method in zip(ax.patches, labels):
        bar.set_facecolor(METHOD_COLORS.get(method, PALETTE["neutral_mid"]))

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title(rf"Defense comparison ({attack_name}, $\epsilon$={eps * 255:.0f}/255)")
    add_panel_label(ax, "a")

    return save_or_show_figure(fig, save_path, pad=1.0)


def plot_confidence_histogram(
    clean_confs_correct: np.ndarray,
    clean_confs_wrong: np.ndarray,
    adv_confs_correct: np.ndarray,
    adv_confs_wrong: np.ndarray,
    save_path: str | None = None,
):
    """Plot confidence distributions split by correctness and attack status."""
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), sharey=True)

    groups = [
        (axes[0], clean_confs_correct, clean_confs_wrong, "Clean predictions"),
        (axes[1], adv_confs_correct, adv_confs_wrong, "Adversarial predictions"),
    ]
    for idx, (ax, correct, wrong, title) in enumerate(groups):
        ax.hist(correct, bins=bins, alpha=0.75, label="Correct", color=SIGNAL_COLORS["Correct"], density=True)
        if len(wrong) > 0:
            ax.hist(wrong, bins=bins, alpha=0.75, label="Incorrect", color=SIGNAL_COLORS["Incorrect"], density=True)
        ax.set_xlabel("Maximum softmax probability")
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.legend(loc="upper left")
        add_panel_label(ax, chr(ord("a") + idx))

    axes[0].set_ylabel("Density")
    return save_or_show_figure(fig, save_path, pad=1.0)


def plot_per_class_asr(
    per_class_results: dict,
    save_path: str | None = None,
):
    """Plot per-class attack success rate as a horizontal bar chart."""
    classes = list(per_class_results.keys())
    asrs = [per_class_results[class_name]["asr"] for class_name in classes]
    colors = [class_color(class_name, idx) for idx, class_name in enumerate(classes)]

    fig, ax = plt.subplots(figsize=(3.3, max(1.8, len(classes) * 0.38 + 1.1)))
    bars = ax.barh(classes, asrs, color=colors, edgecolor="black", linewidth=0.45)

    for bar, val in zip(bars, asrs):
        ax.text(
            min(val + 0.015, 1.08),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=6.5,
        )

    ax.set_xlabel("Attack success rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1.15)
    ax.set_title("Class-specific vulnerability")
    add_panel_label(ax, "a")

    return save_or_show_figure(fig, save_path, pad=1.0)
