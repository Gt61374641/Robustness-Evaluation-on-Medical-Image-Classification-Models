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
from src.evaluation.metrics import compute_ece


def get_predictions(model, loader, device):
    """Get predictions, confidences, and labels from a dataloader."""
    model.eval()
    all_preds = []
    all_confs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_confs),
        np.concatenate(all_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    preds, confs, labels = get_predictions(model, data["test"], device)

    accuracy = (preds == labels).mean()
    ece_result = compute_ece(confs, preds, labels)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"ECE: {ece_result['ece']:.4f}")

    # Per-class accuracy
    class_names = data.get("class_names", [str(i) for i in range(num_classes)])
    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).mean()
            per_class[cls_name] = {
                "accuracy": float(cls_acc),
                "count": int(mask.sum()),
            }
            logger.info(f"  {cls_name}: {cls_acc:.4f} ({mask.sum()} samples)")

    # Save results
    results = {
        "_meta": {
            "schema_version": 2,
            "created_at": datetime.now().isoformat(),
            "confidence_type": "softmax_probability",
        },
        "accuracy": float(accuracy),
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
