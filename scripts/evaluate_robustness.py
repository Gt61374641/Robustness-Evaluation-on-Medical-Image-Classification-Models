"""Adversarial robustness evaluation script (CORE experiment).

Runs attack methods against a trained model and computes all robustness metrics.
Priority: main attacks first (FGSM, PGD, CW/DeepFool), then extended.

Usage:
    python scripts/evaluate_robustness.py --config configs/config.yaml --checkpoint checkpoints/xxx.pth
    python scripts/evaluate_robustness.py --config configs/config.yaml --checkpoint checkpoints/xxx.pth --attacks-section attacks_extended
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier

from src.utils.reproducibility import set_seed, load_config, save_config_snapshot, get_results_dir
from src.utils.logger import get_logger
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.attacks import create_attacks_from_config, create_defense_eval_attacks
from src.evaluation.metrics import evaluate_robustness
from src.evaluation.subset import get_attack_subset


def collect_test_data(loader, max_samples=None):
    """Collect test data into numpy arrays for ART.

    ART expects numpy arrays with shape (N, C, H, W) and values in [0, 1].
    """
    all_images = []
    all_labels = []
    count = 0

    for images, labels in loader:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        count += images.size(0)
        if max_samples and count >= max_samples:
            break

    x = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    if max_samples and len(x) > max_samples:
        x = x[:max_samples]
        y = y[:max_samples]

    return x, y


RESULT_SCHEMA_VERSION = 2


def _as_probabilities(outputs: np.ndarray) -> np.ndarray:
    """Convert classifier outputs to probabilities if ART returns logits."""
    outputs = np.asarray(outputs, dtype=np.float64)
    row_sums = outputs.sum(axis=1)
    is_prob_like = (
        np.all(outputs >= -1e-6)
        and np.all(outputs <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    )
    if is_prob_like:
        return np.clip(outputs, 0.0, 1.0)

    shifted = outputs - np.max(outputs, axis=1, keepdims=True)
    exp_outputs = np.exp(shifted)
    return exp_outputs / exp_outputs.sum(axis=1, keepdims=True)


def get_predictions_and_confidences(classifier, x):
    """Get predictions and calibrated confidence values from ART classifier."""
    outputs = classifier.predict(x, batch_size=64)
    probs = _as_probabilities(outputs)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    return preds, confs


def _make_run_metadata(args, checkpoint_path, num_samples, clean_acc):
    checkpoint = Path(checkpoint_path)
    checkpoint_mtime = None
    if checkpoint.exists():
        checkpoint_mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat()

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_mtime": checkpoint_mtime,
        "attacks_section": args.attacks_section,
        "strong_pgd": bool(getattr(args, "strong_pgd", False)),
        "max_samples": args.max_samples,
        "num_samples": int(num_samples),
        "clean_accuracy": float(clean_acc),
        "confidence_type": "softmax_probability",
    }


def _can_resume(existing_results, args, checkpoint_path):
    meta = existing_results.get("_meta")
    if not isinstance(meta, dict):
        return False
    # Invalidate the cache if the checkpoint was retrained (mtime changed): reusing
    # old attack metrics against a new checkpoint silently mixes experiments.
    checkpoint = Path(checkpoint_path)
    current_mtime = (datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat()
                     if checkpoint.exists() else None)
    return (
        meta.get("schema_version") == RESULT_SCHEMA_VERSION
        and meta.get("checkpoint") == str(checkpoint)
        and meta.get("checkpoint_mtime") == current_mtime
        and meta.get("attacks_section") == args.attacks_section
        and meta.get("strong_pgd", False) == bool(getattr(args, "strong_pgd", False))
        and meta.get("max_samples") == args.max_samples
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attacks-section", type=str, default="attacks_main",
                        choices=["attacks_main", "attacks_extended", "attacks_stress", "attacks_fine"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit test samples (useful for slow attacks like CW)")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (for multi-seed runs)")
    parser.add_argument("--strong-pgd", action="store_true",
                        help="Re-check standard models with the SAME strong protocol as the AT "
                             "defense eval (config 'defense_eval': PGD-50 + 5 restarts + matching "
                             "eps grid/step). Makes standard-vs-AT strictly same-protocol and lets "
                             "per-eps pred_distribution expose any large-eps class collapse.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]

    results_dir = get_results_dir("results", dataset_name, model_name, "robustness", seed)
    save_config_snapshot(cfg, results_dir)
    logger = get_logger("evaluate_robustness", log_dir=results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    class_names = data.get("class_names", None)

    # Load model
    model = create_model(model_name, num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Wrap model with ART
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

    # Collect FULL test data, then take a FIXED stratified subset when
    # --max-samples is set. The subset is cached per dataset+seed+size so EVERY
    # model and attack is evaluated on exactly the same images (fair comparison).
    logger.info("Collecting test data...")
    x_full, y_full = collect_test_data(data["test"], max_samples=None)
    if args.max_samples is not None and args.max_samples < len(x_full):
        subset_path = Path("results") / dataset_name / f"attack_subset_seed{seed}_n{args.max_samples}.json"
        idx = get_attack_subset(y_full, args.max_samples, seed, subset_path)
        x_test, y_test = x_full[idx], y_full[idx]
        logger.info(f"Fixed stratified subset: {len(x_test)}/{len(x_full)} samples (cache {subset_path})")
    else:
        x_test, y_test = x_full, y_full
        logger.info(f"Full test set: {len(x_test)} samples")

    # Clean predictions
    logger.info("Computing clean predictions...")
    clean_preds, clean_confs = get_predictions_and_confidences(classifier, x_test)
    clean_acc = (clean_preds == y_test).mean()
    logger.info(f"Clean accuracy: {clean_acc:.4f}")

    # Create attacks
    if args.strong_pgd:
        # Strong re-check using the EXACT same protocol as the AT (defense) eval —
        # the 'defense_eval' section (PGD-50 + 5 restarts + matching eps grid/step).
        # This makes standard-vs-AT a strictly same-protocol comparison.
        attacks = create_defense_eval_attacks(classifier, cfg)
        if not attacks:
            parser.error("--strong-pgd requires a 'defense_eval' section in the config "
                         "(it reuses the AT evaluation protocol for a fair comparison).")
        logger.info(f"STRONG eval (same protocol as AT defense eval): {[a[0] for a in attacks]}")
    else:
        logger.info(f"Creating attacks from [{args.attacks_section}]...")
        attacks = create_attacks_from_config(classifier, cfg, section=args.attacks_section)
        logger.info(f"Total attack configurations: {len(attacks)}")

    # Run attacks. Results are saved incrementally so Ctrl+C preserves completed work.
    output_stem = "robustness_strongpgd" if args.strong_pgd else f"robustness_{args.attacks_section}"
    if args.max_samples is not None:
        output_stem += f"_max{args.max_samples}"
    output_file = results_dir / f"{output_stem}.json"
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
        if _can_resume(existing_results, args, args.checkpoint):
            all_results = existing_results
            completed = sum(
                1 for key, value in all_results.items()
                if key != "_meta" and isinstance(value, dict) and "error" not in value
            )
            logger.info(f"Resuming from existing results: {completed} attacks already done")
        else:
            logger.warning(
                "Existing results are from an older or different run; starting a fresh result file."
            )
            all_results = {}
    else:
        all_results = {}

    all_results["_meta"] = _make_run_metadata(args, args.checkpoint, len(x_test), clean_acc)

    for attack_name, eps_val, attack in attacks:
        eps_str = f"eps={eps_val:.6f}" if eps_val is not None else "default"
        run_key = f"{attack_name}_{eps_str}"

        if run_key in all_results and isinstance(all_results[run_key], dict) and "error" not in all_results[run_key]:
            logger.info(f"Skipping {run_key} (already computed)")
            continue

        logger.info(f"\nRunning {attack_name} ({eps_str})...")

        try:
            # Generate adversarial examples
            x_adv = attack.generate(x=x_test)

            # Adversarial predictions
            adv_preds, adv_confs = get_predictions_and_confidences(classifier, x_adv)

            # Compute all metrics
            metrics = evaluate_robustness(
                clean_preds=clean_preds,
                adv_preds=adv_preds,
                true_labels=y_test,
                clean_confidences=clean_confs,
                adv_confidences=adv_confs,
                class_names=class_names,
                bootstrap=True,
                bootstrap_seed=seed,
            )

            # Compute actual perturbation magnitude
            perturbation = x_adv - x_test
            l2_norm = float(np.sqrt((perturbation ** 2).sum(axis=(1, 2, 3)).mean()))
            linf_norm = float(np.abs(perturbation).max())
            metrics["perturbation"] = {"l2_mean": l2_norm, "linf_max": linf_norm}

            all_results[run_key] = metrics

            robust_acc = metrics["robust_accuracy"]["robust_accuracy"]
            asr = metrics["asr"]
            logger.info(f"  Robust Accuracy: {robust_acc:.4f}, ASR: {asr:.4f}")
            logger.info(f"  L2 perturbation (mean): {l2_norm:.4f}, Linf (max): {linf_norm:.6f}")

        except Exception as e:
            logger.error(f"  Attack {run_key} failed: {e}")
            all_results[run_key] = {"error": str(e)}

        # Incremental save after every attack
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved to {output_file}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Attack':<30} {'Robust Acc':>12} {'ASR':>8} {'Acc Drop':>10}")
    logger.info("-" * 70)
    for key, metrics in all_results.items():
        if key == "_meta":
            continue
        if "error" in metrics:
            logger.info(f"{key:<30} {'ERROR':>12}")
        else:
            ra = metrics["robust_accuracy"]["robust_accuracy"]
            asr = metrics["asr"]
            drop = metrics["accuracy_drop"]["accuracy_drop"]
            logger.info(f"{key:<30} {ra:>12.4f} {asr:>8.4f} {drop:>10.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
