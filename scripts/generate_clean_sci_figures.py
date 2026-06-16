"""Generate SCI-style clean-performance diagnostics.

Outputs confusion matrix, ROC curve, reliability diagram, class metrics, and
confidence distributions from the standard checkpoint.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataloaders
from src.evaluation.metrics import compute_ece
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.utils.plot_style import (
    PALETTE,
    SIGNAL_COLORS,
    add_panel_label,
    apply_publication_style,
    class_color,
    finalize_figure,
)
from src.utils.reproducibility import load_config, set_seed


COLORS = {
    "Normal": class_color("Normal"),
    "Pneumonia": class_color("Pneumonia"),
    "Correct": SIGNAL_COLORS["Correct"],
    "Incorrect": SIGNAL_COLORS["Incorrect"],
}


def apply_sci_style() -> None:
    apply_publication_style(font_size=8.0, axes_linewidth=0.8, grid=False)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    return finalize_figure(fig, output_dir / name, pad=1.0)


def get_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Clean inference", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_clean_metrics(probs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    class_rows = []
    for idx, class_name in enumerate(class_names):
        class_total = cm[idx].sum()
        class_accuracy = cm[idx, idx] / class_total if class_total else 0.0
        class_rows.append({
            "class": class_name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "class_accuracy": float(class_accuracy),
            "support": int(support[idx]),
        })

    if probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = np.array([]), np.array([]), np.nan

    ece = compute_ece(confs, preds, labels)
    return {
        "preds": preds,
        "confs": confs,
        "confusion_matrix": cm,
        "class_metrics": pd.DataFrame(class_rows),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "ece": ece,
        "fpr": fpr,
        "tpr": tpr,
        "auc": float(roc_auc),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_dir: Path) -> list[Path]:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(2.7, 2.45))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.grid(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    add_panel_label(ax, "a")
    ax.set_title("Clean confusion matrix")
    return save_figure(fig, output_dir, "sci_clean_confusion_matrix")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(2.8, 2.45))
    ax.plot(fpr, tpr, color=PALETTE["blue_main"], linewidth=1.4, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    add_panel_label(ax, "b")
    ax.set_title("ROC curve")
    return save_figure(fig, output_dir, "sci_clean_roc_curve")


def plot_reliability(ece: dict, output_dir: Path) -> list[Path]:
    bin_conf = np.array(ece["bin_confidences"])
    bin_acc = np.array(ece["bin_accuracies"])
    bin_counts = np.array(ece["bin_counts"])
    mask = bin_counts > 0

    fig, ax = plt.subplots(figsize=(2.8, 2.45))
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--", label="Perfect")
    ax.scatter(
        bin_conf[mask],
        bin_acc[mask],
        s=np.clip(bin_counts[mask] * 1.4, 18, 95),
        color=PALETTE["blue_main"],
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
        label=f"ECE = {ece['ece']:.3f}",
    )
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    add_panel_label(ax, "c")
    ax.set_title("Reliability diagram")
    return save_figure(fig, output_dir, "sci_clean_reliability")


def plot_class_metrics(class_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(class_metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    palette = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"]]
    for idx, metric in enumerate(metrics):
        ax.bar(
            x + (idx - 1) * width,
            class_metrics[metric],
            width=width,
            color=palette[idx],
            edgecolor="black",
            linewidth=0.4,
            label=metric.capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_metrics["class"])
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    add_panel_label(ax, "d")
    ax.set_title("Class-wise clean metrics")
    return save_figure(fig, output_dir, "sci_clean_class_metrics")


def plot_confidence_distribution(confs: np.ndarray, preds: np.ndarray, labels: np.ndarray, output_dir: Path) -> list[Path]:
    correct = confs[preds == labels]
    wrong = confs[preds != labels]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    ax.hist(correct, bins=bins, density=True, alpha=0.75, color=COLORS["Correct"], label="Correct")
    ax.hist(wrong, bins=bins, density=True, alpha=0.75, color=COLORS["Incorrect"], label="Incorrect")
    ax.set_xlabel("Maximum softmax probability")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="upper left")
    add_panel_label(ax, "e")
    ax.set_title("Confidence distribution")
    return save_figure(fig, output_dir, "sci_clean_confidence_distribution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean SCI figures and metrics")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "sci_clean")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    apply_sci_style()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    dataset = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]
    fig_dir = args.output_dir / dataset / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_dataloaders(cfg)
    class_names = data.get("class_names", [str(i) for i in range(data["num_classes"])])

    model = create_model(model_name, data["num_classes"], pretrained=False)
    model = load_checkpoint(model, str(args.checkpoint), device=device)
    model = model.to(device)

    probs, labels = get_probabilities(model, data["test"], device)
    metrics = compute_clean_metrics(probs, labels, class_names)

    np.savez_compressed(
        fig_dir / "clean_predictions.npz",
        probs=probs,
        labels=labels,
        preds=metrics["preds"],
        confidences=metrics["confs"],
        class_names=np.array(class_names),
    )

    metrics["class_metrics"].to_csv(fig_dir / "clean_classification_metrics.csv", index=False)
    summary = {
        "_meta": {
            "created_at": datetime.now().isoformat(),
            "confidence_type": "softmax_probability",
            "checkpoint": str(args.checkpoint),
        },
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auc": metrics["auc"],
        "ece": metrics["ece"]["ece"],
        "num_samples": int(len(labels)),
        "class_names": class_names,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(fig_dir / "clean_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    outputs = [
        fig_dir / "clean_predictions.npz",
        fig_dir / "clean_classification_metrics.csv",
        fig_dir / "clean_diagnostics_summary.json",
    ]
    outputs.extend(plot_confusion_matrix(metrics["confusion_matrix"], class_names, fig_dir))
    outputs.extend(plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"], fig_dir))
    outputs.extend(plot_reliability(metrics["ece"], fig_dir))
    outputs.extend(plot_class_metrics(metrics["class_metrics"], fig_dir))
    outputs.extend(plot_confidence_distribution(metrics["confs"], metrics["preds"], labels, fig_dir))

    print(f"Generated {len(outputs) - 3} clean figure files and 3 data files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
