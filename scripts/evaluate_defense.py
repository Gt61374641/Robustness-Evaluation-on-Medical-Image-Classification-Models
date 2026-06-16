"""Defense evaluation script.

Evaluates both main defenses (adversarial training) and baseline defenses
(preprocessors) against adversarial attacks.

Usage:
    # Adversarial training (main defense — retrains the model)
    python scripts/evaluate_defense.py --config configs/config.yaml --defense PGD-AT

    # Preprocessor defense (baseline only — applies at inference time)
    python scripts/evaluate_defense.py --config configs/config.yaml --defense SpatialSmoothing --checkpoint checkpoints/xxx.pth
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier

from src.utils.reproducibility import set_seed, load_config, save_config_snapshot, get_results_dir, get_checkpoint_path
from src.utils.logger import get_logger
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.attacks import create_attacks_from_config
from src.defenses import create_defense_trainer, create_preprocessor_defense
from src.evaluation.metrics import evaluate_robustness
from scripts.evaluate_robustness import collect_test_data, get_predictions_and_confidences


MAIN_DEFENSES = {"PGD-AT", "TRADES"}
BASELINE_DEFENSES = {"SpatialSmoothing", "JpegCompression", "FeatureSqueezing"}
RESULT_SCHEMA_VERSION = 2


def find_defense_config(cfg, defense_name):
    """Find the defense config from the config file."""
    for section in ["defenses_main", "defenses_baseline"]:
        for defense_cfg in cfg.get(section, []):
            if defense_cfg["name"] == defense_name:
                return defense_cfg
    raise ValueError(f"Defense '{defense_name}' not found in config")


def make_run_metadata(args, defense_name, checkpoint_path, num_samples, clean_acc):
    checkpoint_mtime = None
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            checkpoint_mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat()

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "defense": defense_name,
        "checkpoint": str(Path(checkpoint_path)) if checkpoint_path else None,
        "checkpoint_mtime": checkpoint_mtime,
        "max_samples": args.max_samples,
        "num_samples": int(num_samples),
        "clean_accuracy_defended": float(clean_acc),
        "confidence_type": "softmax_probability",
    }


def run_adversarial_training(cfg, defense_cfg, device, logger, max_samples=None):
    """Run adversarial training and return the robust model's classifier."""
    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]
    seed = cfg.get("seed", 42)

    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]

    # Create fresh model for adversarial training
    model = create_model(model_name, num_classes, pretrained=cfg["model"]["pretrained"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # Collect training data
    logger.info("Collecting training data for adversarial training...")
    if max_samples is not None:
        logger.info(f"Limiting adversarial-training data to {max_samples} samples")
    x_train, y_train = collect_test_data(data["train"], max_samples=max_samples)
    # One-hot encode labels for ART
    y_train_oh = np.eye(num_classes)[y_train]

    # Create and run adversarial trainer
    defense_name = defense_cfg["name"]
    logger.info(f"Starting adversarial training with {defense_name}...")
    trainer = create_defense_trainer(classifier, defense_cfg)
    trainer.fit(
        x_train,
        y_train_oh,
        nb_epochs=defense_cfg.get("nb_epochs", 20),
        batch_size=defense_cfg.get("batch_size", cfg["data"]["batch_size"]),
    )

    # Save adversarially trained model
    suffix = defense_name.lower().replace("-", "_")
    ckpt_path = get_checkpoint_path("checkpoints", dataset_name, model_name, seed, suffix=suffix)
    torch.save({
        "model_state_dict": model.state_dict(),
        "defense": defense_name,
        "config": cfg,
    }, ckpt_path)
    logger.info(f"Adversarially trained model saved to {ckpt_path}")

    return classifier, data, ckpt_path


def load_classifier_from_checkpoint(cfg, checkpoint_path, device, logger):
    """Load a trained model checkpoint as an ART classifier."""
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    model_name = cfg["model"]["name"]

    logger.info(f"Loading defended checkpoint from {checkpoint_path}")
    model = create_model(model_name, num_classes, pretrained=False)
    model = load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    return classifier, data


def run_preprocessor_defense(cfg, defense_cfg, checkpoint_path, device, logger):
    """Apply preprocessor defense and return defended predictions."""
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    model_name = cfg["model"]["name"]

    model = create_model(model_name, num_classes, pretrained=False)
    model = load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    preprocessor = create_preprocessor_defense(defense_cfg)
    return classifier, preprocessor, data


def main():
    parser = argparse.ArgumentParser(description="Evaluate defense methods")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--defense", type=str, required=True,
                        choices=list(MAIN_DEFENSES | BASELINE_DEFENSES))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for preprocessor defenses (not needed for AT)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]
    defense_name = args.defense

    results_dir = get_results_dir("results", dataset_name, model_name, f"defense_{defense_name}", seed)
    save_config_snapshot(cfg, results_dir)
    logger = get_logger("evaluate_defense", log_dir=results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Defense: {defense_name}, Device: {device}")

    defense_cfg = find_defense_config(cfg, defense_name)

    checkpoint_for_meta = args.checkpoint

    if defense_name in MAIN_DEFENSES:
        if args.checkpoint:
            classifier, data = load_classifier_from_checkpoint(cfg, args.checkpoint, device, logger)
        else:
            # Adversarial training
            classifier, data, checkpoint_for_meta = run_adversarial_training(
                cfg,
                defense_cfg,
                device,
                logger,
                max_samples=args.max_samples,
            )
        preprocessor = None
    else:
        # Preprocessor defense
        if not args.checkpoint:
            parser.error(f"--checkpoint required for preprocessor defense {defense_name}")
        classifier, preprocessor, data = run_preprocessor_defense(
            cfg, defense_cfg, args.checkpoint, device, logger
        )
        logger.warning(
            f"NOTE: {defense_name} is a preprocessor defense used as a BASELINE only. "
            "It may cause gradient obfuscation. Do NOT claim it 'effectively improves robustness' "
            "in the paper without adaptive attack verification."
        )

    # Collect test data
    x_test, y_test = collect_test_data(data["test"], max_samples=args.max_samples)
    class_names = data.get("class_names", None)

    # Clean predictions (with defense applied if preprocessor)
    if preprocessor is not None:
        x_test_defended, _ = preprocessor(x_test)
        x_test_defended = x_test_defended.astype(np.float32)
        clean_preds, clean_confs = get_predictions_and_confidences(classifier, x_test_defended)
    else:
        clean_preds, clean_confs = get_predictions_and_confidences(classifier, x_test)

    clean_acc = (clean_preds == y_test).mean()
    logger.info(f"Clean accuracy (with defense): {clean_acc:.4f}")

    # Run main attacks against defended model
    attacks = create_attacks_from_config(classifier, cfg, section="attacks_main")
    all_results = {
        "_meta": make_run_metadata(args, defense_name, checkpoint_for_meta, len(x_test), clean_acc),
        "clean_accuracy_defended": float(clean_acc),
    }

    for attack_name, eps_val, attack in attacks:
        eps_str = f"eps={eps_val:.6f}" if eps_val is not None else "default"
        run_key = f"{attack_name}_{eps_str}"
        logger.info(f"\nAttacking defended model: {attack_name} ({eps_str})...")

        try:
            x_adv = attack.generate(x=x_test)

            # Apply preprocessor defense to adversarial examples if applicable
            if preprocessor is not None:
                x_adv_defended, _ = preprocessor(x_adv)
                x_adv_defended = x_adv_defended.astype(np.float32)
                adv_preds, adv_confs = get_predictions_and_confidences(classifier, x_adv_defended)
            else:
                adv_preds, adv_confs = get_predictions_and_confidences(classifier, x_adv)

            metrics = evaluate_robustness(
                clean_preds, adv_preds, y_test, clean_confs, adv_confs, class_names
            )
            all_results[run_key] = metrics

            ra = metrics["robust_accuracy"]["robust_accuracy"]
            asr = metrics["asr"]
            logger.info(f"  Robust Accuracy: {ra:.4f}, ASR: {asr:.4f}")

        except Exception as e:
            logger.error(f"  Attack {run_key} failed: {e}")
            all_results[run_key] = {"error": str(e)}

    # Save
    output_name = "defense_results.json"
    if args.max_samples is not None:
        output_name = f"defense_results_max{args.max_samples}.json"
    output_file = results_dir / output_name
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
