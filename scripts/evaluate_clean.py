"""Evaluate clean (baseline) accuracy of a trained model.

Usage:
    python scripts/evaluate_clean.py --config configs/config.yaml --checkpoint checkpoints/chest_xray_pneumonia_densenet121_seed42.pth
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed, load_config, save_config_snapshot, get_results_dir
from src.utils.logger import get_logger
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.evaluation.metrics import compute_ece, compute_clean_metrics


def get_predictions(model, loader, device):
    """Get predictions, confidences, full probabilities, and labels."""
    model.eval()
    all_preds = []
    all_confs = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_confs),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (for multi-seed runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]

    results_dir = get_results_dir("results", dataset_name, model_name, "clean", seed)
    save_config_snapshot(cfg, results_dir)
    logger = get_logger("evaluate_clean", log_dir=results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]

    # Load model
    model = create_model(model_name, num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    preds, confs, probs, labels = get_predictions(model, data["test"], device)

    class_names = data.get("class_names", [str(i) for i in range(num_classes)])
    clean_metrics = compute_clean_metrics(preds, probs, labels, class_names)
    ece_result = compute_ece(confs, preds, labels)

    logger.info(f"Test Accuracy: {clean_metrics['accuracy']:.4f}, "
                f"Balanced Acc: {clean_metrics['balanced_accuracy']:.4f}, "
                f"ROC-AUC: {clean_metrics['roc_auc']}")
    logger.info(f"Macro F1: {clean_metrics['macro']['f1']:.4f}, ECE: {ece_result['ece']:.4f}")
    for cls_name, m in clean_metrics["per_class"].items():
        logger.info(f"  {cls_name}: recall={m['recall']:.4f} precision={m['precision']:.4f} "
                    f"f1={m['f1']:.4f} (n={m['support']})")

    # per_class kept backward-compatible (accuracy == recall) plus the richer fields.
    per_class = {
        name: {"accuracy": m["recall"], "recall": m["recall"], "precision": m["precision"],
               "f1": m["f1"], "count": m["support"]}
        for name, m in clean_metrics["per_class"].items()
    }

    # Save results
    results = {
        "_meta": {
            "schema_version": 3,
            "created_at": datetime.now().isoformat(),
            "confidence_type": "softmax_probability",
        },
        "accuracy": clean_metrics["accuracy"],
        "balanced_accuracy": clean_metrics["balanced_accuracy"],
        "roc_auc": clean_metrics["roc_auc"],
        "macro": clean_metrics["macro"],
        "weighted": clean_metrics["weighted"],
        "confusion_matrix": clean_metrics["confusion_matrix"],
        "ece": ece_result["ece"],
        "num_samples": int(len(labels)),
        "per_class": per_class,
        "dataset": dataset_name,
        "model": model_name,
        "checkpoint": args.checkpoint,
    }
    with open(results_dir / "clean_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
