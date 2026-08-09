"""Generate publication-ready qualitative figures for the manuscript.

Outputs two traceable Nature-style image plates:

1. Three-dataset clean / perturbation / PGD-adversarial examples.
2. One deterministic test example for every dataset class.

All quantitative pixels come from the project datasets and checkpoints. The
script records the exact source sample, selection rule, model, attack, and
display transform in JSON manifests beside the figures.

Usage:
    python scripts/generate_qualitative_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.attack_factory import create_attack
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.utils.reproducibility import load_config, set_seed


PALETTE = {
    "blue": "#0F4D92",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "green": "#2E7D32",
    "red": "#B64342",
    "neutral": "#767676",
    "dark": "#272727",
    "light": "#CFCECE",
}

DATASETS = [
    {
        "slug": "chest_xray_pneumonia",
        "display": "Chest X-ray",
        "panel": "a",
        "color": PALETTE["blue"],
    },
    {
        "slug": "malaria",
        "display": "Malaria",
        "panel": "b",
        "color": PALETTE["teal"],
    },
    {
        "slug": "oct2017",
        "display": "OCT",
        "panel": "c",
        "color": PALETTE["violet"],
    },
]


def apply_style() -> None:
    """Apply a compact, editable Nature-style matplotlib theme."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    """Export editable vectors plus high-resolution review/submission rasters."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def unwrap_dataset(dataset, index: int):
    """Resolve a nested Subset index back to its source dataset and source index."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    return dataset, index


def sample_record(dataset, index: int) -> tuple[Path | None, int]:
    """Return source path and class index without altering the image."""
    base, base_index = unwrap_dataset(dataset, index)
    if hasattr(base, "samples"):
        path, label = base.samples[base_index]
        return Path(path), int(label)
    _, label = dataset[index]
    return None, int(label)


def relative_source(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def first_index_per_class(dataset, num_classes: int) -> list[int]:
    """Select the first deterministic test item for each class."""
    found: dict[int, int] = {}
    for index in range(len(dataset)):
        _, label = sample_record(dataset, index)
        if label not in found:
            found[label] = index
        if len(found) == num_classes:
            break
    missing = sorted(set(range(num_classes)) - set(found))
    if missing:
        raise RuntimeError(f"Missing test examples for classes {missing}")
    return [found[label] for label in range(num_classes)]


def balanced_candidate_indices(dataset, num_classes: int, per_class: int) -> list[int]:
    """Build a deterministic round-robin pool with equal candidates per class."""
    buckets: list[list[int]] = [[] for _ in range(num_classes)]
    for index in range(len(dataset)):
        _, label = sample_record(dataset, index)
        if len(buckets[label]) < per_class:
            buckets[label].append(index)
        if all(len(bucket) >= per_class for bucket in buckets):
            break

    if any(not bucket for bucket in buckets):
        missing = [i for i, bucket in enumerate(buckets) if not bucket]
        raise RuntimeError(f"No candidates found for classes {missing}")

    n_rounds = min(len(bucket) for bucket in buckets)
    return [
        buckets[class_index][rank]
        for rank in range(n_rounds)
        for class_index in range(num_classes)
    ]


def load_dataset_bundle(dataset_spec: dict, model_name: str, seed: int):
    config_path = PROJECT_ROOT / "configs" / f"{dataset_spec['slug']}_{model_name}.yaml"
    cfg = load_config(str(config_path))
    cfg["seed"] = seed
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = min(int(cfg["data"].get("batch_size", 32)), 32)
    data = get_dataloaders(cfg)
    return cfg, data


def make_classifier(cfg: dict, data: dict, checkpoint: Path, device: torch.device):
    model = create_model(cfg["model"]["name"], data["num_classes"], pretrained=False)
    model = load_checkpoint(model, str(checkpoint), device=device).to(device).eval()
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=data["num_classes"],
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return model, classifier


def choose_successful_attack(
    dataset_spec: dict,
    cfg: dict,
    data: dict,
    model_name: str,
    seed: int,
    checkpoint_tag: str,
    eps: float,
    max_iter: int,
    per_class: int,
    device: torch.device,
) -> dict:
    """Return the first successful attack from a deterministic balanced pool."""
    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / f"{dataset_spec['slug']}_{model_name}_seed{seed}{checkpoint_tag}.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    test_dataset = data["test"].dataset
    candidate_indices = balanced_candidate_indices(
        test_dataset, data["num_classes"], per_class
    )
    images, labels = [], []
    for index in candidate_indices:
        image, label = test_dataset[index]
        images.append(image.numpy())
        labels.append(int(label))
    x_clean = np.stack(images).astype(np.float32)
    y_true = np.asarray(labels, dtype=np.int64)

    model, classifier = make_classifier(cfg, data, checkpoint, device)
    clean_logits = classifier.predict(x_clean, batch_size=32)
    clean_pred = np.argmax(clean_logits, axis=1)

    attack_cfg = {
        "name": "PGD",
        "max_iter": max_iter,
        "num_random_init": 1,
        "eps_step": eps / 10.0,
    }
    attack = create_attack(classifier, attack_cfg, eps=eps)
    x_adv = attack.generate(x=x_clean)
    adv_logits = classifier.predict(x_adv, batch_size=32)
    adv_pred = np.argmax(adv_logits, axis=1)

    success = np.flatnonzero((clean_pred == y_true) & (adv_pred != y_true))
    if len(success) == 0:
        raise RuntimeError(
            f"No successful PGD attack for {dataset_spec['display']} in "
            f"{len(candidate_indices)} balanced candidates."
        )

    chosen_position = int(success[0])
    dataset_index = int(candidate_indices[chosen_position])
    source_path, source_label = sample_record(test_dataset, dataset_index)
    if source_label != int(y_true[chosen_position]):
        raise RuntimeError("Sample traceability check failed: label mismatch.")

    clean = np.clip(x_clean[chosen_position], 0.0, 1.0)
    adversarial = np.clip(x_adv[chosen_position], 0.0, 1.0)
    delta = adversarial - clean
    perturbation_display = np.clip(0.5 + delta / (2.0 * eps), 0.0, 1.0)

    class_names = list(data["class_names"])
    record = {
        "dataset": dataset_spec["slug"],
        "dataset_display": dataset_spec["display"],
        "dataset_index": dataset_index,
        "source_path": relative_source(source_path),
        "true_index": int(y_true[chosen_position]),
        "true_label": class_names[int(y_true[chosen_position])],
        "clean_prediction_index": int(clean_pred[chosen_position]),
        "clean_prediction": class_names[int(clean_pred[chosen_position])],
        "clean_confidence": float(
            torch.softmax(torch.from_numpy(clean_logits[chosen_position]), dim=0).max()
        ),
        "adversarial_prediction_index": int(adv_pred[chosen_position]),
        "adversarial_prediction": class_names[int(adv_pred[chosen_position])],
        "adversarial_confidence": float(
            torch.softmax(torch.from_numpy(adv_logits[chosen_position]), dim=0).max()
        ),
        "linf": float(np.max(np.abs(delta))),
        "l2": float(np.linalg.norm(delta.reshape(-1), ord=2)),
        "candidate_pool_size": len(candidate_indices),
        "candidate_pool_per_class": per_class,
        "candidate_position": chosen_position,
        "selection_rule": (
            "First successful attack among a deterministic round-robin, "
            "class-balanced test pool."
        ),
        "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "model": model_name,
        "seed": seed,
        "attack": "untargeted white-box PGD",
        "eps": eps,
        "eps_255": eps * 255.0,
        "eps_step": eps / 10.0,
        "max_iter": max_iter,
        "num_random_init": 1,
        "display": {
            "clean_and_adversarial": "clipped to [0, 1], no local adjustment",
            "perturbation": "RGB display = clip(0.5 + delta / (2*eps), 0, 1)",
        },
        "arrays": {
            "clean": clean,
            "perturbation": delta,
            "perturbation_display": perturbation_display,
            "adversarial": adversarial,
        },
    }

    del classifier, model, attack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (1, 2, 0))


def style_image_axis(ax: plt.Axes, border_color: str, linewidth: float = 1.1) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(linewidth)


def build_attack_figure(records: list[dict], output_dir: Path, eps: float) -> list[Path]:
    """Build the three-row clean/perturbation/adversarial image plate."""
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.05))
    fig.subplots_adjust(
        left=0.13, right=0.995, top=0.91, bottom=0.055, wspace=0.055, hspace=0.39
    )

    headers = ["Clean image", "Perturbation (rescaled)", "PGD adversarial"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, pad=7, color=PALETTE["dark"], fontweight="bold")

    for row, (spec, record) in enumerate(zip(DATASETS, records)):
        clean = chw_to_hwc(record["arrays"]["clean"])
        perturbation_display = chw_to_hwc(record["arrays"]["perturbation_display"])
        adversarial = chw_to_hwc(record["arrays"]["adversarial"])

        images = [clean, perturbation_display, adversarial]
        borders = [PALETTE["green"], PALETTE["neutral"], PALETTE["red"]]
        for col, (image, border) in enumerate(zip(images, borders)):
            ax = axes[row, col]
            style_image_axis(ax, border)
            ax.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest")

        clean_text = (
            f"true: {record['true_label']}  |  pred: {record['clean_prediction']}"
        )
        perturbation_text = (
            rf"$\|\delta\|_\infty$ = {record['linf'] * 255:.1f}/255"
        )
        adv_text = f"pred: {record['adversarial_prediction']}"
        subtitles = [clean_text, perturbation_text, adv_text]
        subtitle_colors = [PALETTE["green"], PALETTE["neutral"], PALETTE["red"]]

        for col, (text, color) in enumerate(zip(subtitles, subtitle_colors)):
            axes[row, col].text(
                0.5,
                -0.075,
                text,
                transform=axes[row, col].transAxes,
                ha="center",
                va="top",
                fontsize=6.4,
                color=color,
                fontweight="bold" if col != 1 else "normal",
            )

        pos = axes[row, 0].get_position()
        fig.text(
            0.018,
            (pos.y0 + pos.y1) / 2,
            spec["panel"],
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        fig.text(
            0.045,
            (pos.y0 + pos.y1) / 2,
            spec["display"],
            ha="left",
            va="center",
            rotation=90,
            fontsize=8,
            fontweight="bold",
            color=spec["color"],
        )

    return export_figure(fig, output_dir / "qualitative_attack_examples")


def build_overview_figure(
    overview_groups: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Build one aligned test-image example for each of the eight classes."""
    total_classes = sum(len(group["examples"]) for group in overview_groups)
    fig, axes = plt.subplots(1, total_classes, figsize=(7.2, 1.92))
    fig.subplots_adjust(
        left=0.012, right=0.995, top=0.79, bottom=0.17, wspace=0.045
    )

    axis_index = 0
    for spec, group in zip(DATASETS, overview_groups):
        group_axes = []
        for example in group["examples"]:
            ax = axes[axis_index]
            group_axes.append(ax)
            style_image_axis(ax, PALETTE["light"], linewidth=0.75)
            ax.imshow(chw_to_hwc(example["image"]), interpolation="nearest")
            ax.text(
                0.5,
                -0.09,
                example["class_name"],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.3,
                color=PALETTE["dark"],
            )
            axis_index += 1

        first_pos = group_axes[0].get_position()
        last_pos = group_axes[-1].get_position()
        x0, x1 = first_pos.x0, last_pos.x1
        center = (x0 + x1) / 2
        fig.text(
            x0,
            0.94,
            spec["panel"],
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        fig.text(
            center,
            0.94,
            spec["display"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        fig.add_artist(
            mlines.Line2D(
                [x0, x1],
                [0.885, 0.885],
                transform=fig.transFigure,
                color=spec["color"],
                linewidth=1.6,
                solid_capstyle="round",
            )
        )

    fig.text(
        0.995,
        0.015,
        "Deterministic test examples · common 224 × 224 evaluation resize",
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=PALETTE["neutral"],
    )
    return export_figure(fig, output_dir / "dataset_class_overview")


def write_legends(
    output_dir: Path,
    eps: float,
    max_iter: int,
    model_name: str,
    checkpoint_tag: str,
) -> None:
    model_display = model_name.replace("resnet", "ResNet-")
    training_description = (
        "PGD-adversarially trained"
        if checkpoint_tag == "_pgd_at"
        else "standard"
    )
    attack_legend = (
        "Fig. X | Representative PGD adversarial examples across three medical "
        "image modalities. a Chest X-ray, b malaria cell microscopy and c optical "
        "coherence tomography (OCT). Clean images were correctly classified by "
        f"{training_description} {model_display} models. Adversarial images were "
        "generated with "
        "an untargeted white-box projected gradient descent attack under an "
        f"L-infinity budget of epsilon = {eps * 255:.0f}/255 ({max_iter} iterations, "
        "step size epsilon/10 and one random initialization). Perturbations are "
        "rescaled for visualization as clip(0.5 + delta/(2 epsilon), 0, 1); clean "
        "and adversarial images are shown in the model input range [0,1] without "
        "local contrast or gamma adjustment. For each dataset, the displayed "
        "example is the first successful attack in a deterministic class-balanced "
        "test pool."
    )
    overview_legend = (
        "Fig. X | Representative test images from the three medical image "
        "classification datasets. a Chest X-ray pneumonia dataset (NORMAL and "
        "PNEUMONIA). b NIH malaria cell-image dataset (Parasitized and Uninfected). "
        "c OCT2017 dataset (CNV, DME, DRUSEN and NORMAL). The first deterministic "
        "test item from each class is shown after the common 224 x 224 evaluation "
        "resize, without local contrast or gamma adjustment."
    )
    (output_dir / "qualitative_attack_examples_legend.txt").write_text(
        attack_legend + "\n", encoding="utf-8"
    )
    (output_dir / "dataset_class_overview_legend.txt").write_text(
        overview_legend + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--pool-per-class", type=int, default=16)
    parser.add_argument(
        "--checkpoint-tag",
        default="_pgd_at",
        help=(
            "Checkpoint suffix placed after the seed, e.g. '_pgd_at'. "
            "Use an empty string for standard checkpoints."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures") / "qualitative",
    )
    args = parser.parse_args()

    apply_style()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_records = []
    overview_groups = []
    overview_manifest = {
        "selection_rule": "First deterministic test item for each class.",
        "image_processing": (
            "Common evaluation transform: resize to 224 x 224 and convert to "
            "float tensor in [0,1]; no local contrast or gamma adjustment."
        ),
        "datasets": [],
    }

    for dataset_spec in DATASETS:
        print(f"[{dataset_spec['display']}] loading data", flush=True)
        cfg, data = load_dataset_bundle(dataset_spec, args.model, args.seed)
        test_dataset = data["test"].dataset

        overview_indices = first_index_per_class(test_dataset, data["num_classes"])
        examples = []
        manifest_examples = []
        for class_index, dataset_index in enumerate(overview_indices):
            image, label = test_dataset[dataset_index]
            if int(label) != class_index:
                raise RuntimeError("Overview traceability check failed: label mismatch.")
            source_path, _ = sample_record(test_dataset, dataset_index)
            examples.append(
                {
                    "class_index": class_index,
                    "class_name": data["class_names"][class_index],
                    "image": np.clip(image.numpy(), 0.0, 1.0),
                }
            )
            manifest_examples.append(
                {
                    "class_index": class_index,
                    "class_name": data["class_names"][class_index],
                    "dataset_index": int(dataset_index),
                    "source_path": relative_source(source_path),
                }
            )
        overview_groups.append({"dataset": dataset_spec["slug"], "examples": examples})
        overview_manifest["datasets"].append(
            {
                "dataset": dataset_spec["slug"],
                "dataset_display": dataset_spec["display"],
                "examples": manifest_examples,
            }
        )

        print(
            f"[{dataset_spec['display']}] generating PGD candidates "
            f"(eps={args.eps * 255:.0f}/255)",
            flush=True,
        )
        record = choose_successful_attack(
            dataset_spec=dataset_spec,
            cfg=cfg,
            data=data,
            model_name=args.model,
            seed=args.seed,
            checkpoint_tag=args.checkpoint_tag,
            eps=args.eps,
            max_iter=args.max_iter,
            per_class=args.pool_per_class,
            device=device,
        )
        print(
            f"[{dataset_spec['display']}] selected {record['true_label']} -> "
            f"{record['adversarial_prediction']} "
            f"(Linf={record['linf'] * 255:.2f}/255)",
            flush=True,
        )
        attack_records.append(record)

    attack_manifest = {
        "figure": "qualitative_attack_examples",
        "model": args.model,
        "seed": args.seed,
        "checkpoint_tag": args.checkpoint_tag,
        "attack": "untargeted white-box PGD",
        "eps": args.eps,
        "eps_255": args.eps * 255.0,
        "max_iter": args.max_iter,
        "examples": [
            {key: value for key, value in record.items() if key != "arrays"}
            for record in attack_records
        ],
    }

    np.savez_compressed(
        output_dir / "qualitative_attack_examples_source_data.npz",
        **{
            f"{record['dataset']}_{kind}": record["arrays"][kind]
            for record in attack_records
            for kind in ("clean", "perturbation", "adversarial")
        },
    )
    (output_dir / "qualitative_attack_examples_manifest.json").write_text(
        json.dumps(attack_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_class_overview_manifest.json").write_text(
        json.dumps(overview_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    attack_outputs = build_attack_figure(attack_records, output_dir, args.eps)
    overview_outputs = build_overview_figure(overview_groups, output_dir)
    write_legends(
        output_dir,
        args.eps,
        args.max_iter,
        args.model,
        args.checkpoint_tag,
    )

    print("[done] qualitative attack outputs:", flush=True)
    for path in attack_outputs:
        print(f"  {path.relative_to(PROJECT_ROOT)}", flush=True)
    print("[done] dataset overview outputs:", flush=True)
    for path in overview_outputs:
        print(f"  {path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
