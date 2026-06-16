"""Model training script with AMP and gradient accumulation support.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --model densenet121 --dataset chest_xray_pneumonia
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed, load_config, save_config_snapshot, get_results_dir, get_checkpoint_path
from src.utils.logger import get_logger
from src.datasets import get_dataloaders, NUM_CLASSES
from src.models import create_model
from src.training.imbalance import (
    compute_class_counts,
    compute_class_weights,
    replace_loader_with_balanced_sampler,
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Train for one epoch with AMP and gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device)

        with autocast("cuda", enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a model on a medical image dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.seed:
        cfg["seed"] = args.seed

    seed = cfg.get("seed", 42)
    set_seed(seed)

    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]

    # Setup results directory and logger
    results_dir = get_results_dir("results", dataset_name, model_name, "train", seed)
    save_config_snapshot(cfg, results_dir)
    logger = get_logger("train", log_dir=results_dir)

    logger.info(f"Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Data
    logger.info("Loading data...")
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    cfg["model"]["num_classes"] = num_classes
    logger.info(f"Classes: {num_classes} ({data.get('class_names', 'N/A')})")
    logger.info(f"Train: {len(data['train'].dataset)}, Val: {len(data['val'].dataset)}, Test: {len(data['test'].dataset)}")

    # Model
    logger.info(f"Creating model: {model_name} (pretrained={cfg['model']['pretrained']})")
    model = create_model(model_name, num_classes, pretrained=cfg["model"]["pretrained"])
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {num_params:.1f}M")

    # Training setup
    train_cfg = cfg["train"]
    balance_cfg = train_cfg.get("class_balance", {})
    loss_mode = balance_cfg.get("loss", "none")
    sampler_mode = balance_cfg.get("sampler", "none")

    class_counts = compute_class_counts(data["train"].dataset, num_classes)
    logger.info(f"Train class counts: {[int(x) for x in class_counts.tolist()]}")

    if sampler_mode in {"balanced", "balanced_sampler"}:
        data["train"] = replace_loader_with_balanced_sampler(data["train"], num_classes, seed=seed)
        logger.info("Class balance sampler: enabled")
    elif sampler_mode in {"none", None}:
        logger.info("Class balance sampler: disabled")
    else:
        raise ValueError(f"Unknown class_balance.sampler mode: {sampler_mode}")

    if loss_mode in {"weighted", "weighted_cross_entropy"}:
        class_weights = compute_class_weights(data["train"].dataset, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Weighted cross entropy: enabled, weights={class_weights.detach().cpu().tolist()}")
    elif loss_mode in {"none", None}:
        criterion = nn.CrossEntropyLoss()
        logger.info("Weighted cross entropy: disabled")
    else:
        raise ValueError(f"Unknown class_balance.loss mode: {loss_mode}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )

    # AMP scaler
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    logger.info(f"AMP: {use_amp}, Gradient accumulation: {accumulation_steps}")

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    checkpoint_path = get_checkpoint_path("checkpoints", dataset_name, model_name, seed)
    logger.info(f"Checkpoint will be saved to: {checkpoint_path}")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, data["train"], criterion, optimizer, scaler, device, accumulation_steps
        )
        val_loss, val_acc = evaluate(model, data["val"], criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch}/{train_cfg['epochs']} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} — "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }, checkpoint_path)
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Final test evaluation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True)["model_state_dict"])
    test_loss, test_acc = evaluate(model, data["test"], criterion, device)
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save training history
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["test_acc"] = test_acc
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
