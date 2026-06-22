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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier

from src.utils.reproducibility import set_seed, load_config, save_config_snapshot, get_results_dir, get_checkpoint_path
from src.utils.logger import get_logger
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.attacks import create_attacks_from_config, create_defense_eval_attacks
from src.defenses import create_defense_trainer, create_preprocessor_defense
from src.training.imbalance import compute_class_weights
from src.evaluation.metrics import evaluate_robustness
from src.evaluation.subset import get_attack_subset
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


def _pgd_perturb(model, x, y, eps, eps_step, max_iter, random_start=True):
    """L-inf PGD in [0, 1] pixel space against ``model`` (ImageNet normalization is
    applied inside the model wrapper). The caller puts the model in eval mode so BN
    running stats stay stable while generating the perturbation."""
    x_adv = x.clone().detach()
    if random_start:
        x_adv = torch.clamp(x_adv + torch.empty_like(x_adv).uniform_(-eps, eps), 0.0, 1.0)
    for _ in range(max_iter):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + eps_step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def _robust_balanced_acc(model, loader, eps, eps_step, max_iter, device, num_classes):
    """Mean per-class recall under inner PGD on ``loader`` (robust balanced accuracy)
    — the AT checkpoint-selection metric. Guards against robust overfitting, where
    clean val accuracy keeps rising while adversarial accuracy has already peaked."""
    model.eval()
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x_adv = _pgd_perturb(model, images, labels, eps, eps_step, max_iter)
        with torch.no_grad():
            preds = model(x_adv).argmax(1).cpu()
        labels_cpu = labels.cpu()
        for c in range(num_classes):
            mask = labels_cpu == c
            class_total[c] += int(mask.sum())
            class_correct[c] += int((preds[mask] == c).sum())
    recalls = class_correct.float() / class_total.clamp_min(1).float()
    valid = class_total > 0
    return recalls[valid].mean().item() if bool(valid.any()) else 0.0


def run_adversarial_training(cfg, defense_cfg, device, logger, train_max_samples=None):
    """Dispatch adversarial training. PGD-AT uses a custom loop harmonized with
    standard training (scripts/train.py); TRADES keeps the ART trainer path.

    train_max_samples is for SMOKE TESTS ONLY. Formal AT must use the full training
    set (leave it None); limiting training data here would invalidate the comparison.
    """
    if defense_cfg["name"] == "PGD-AT":
        return _run_pgd_at(cfg, defense_cfg, device, logger, train_max_samples)
    return _run_art_trainer(cfg, defense_cfg, device, logger, train_max_samples)


def _run_pgd_at(cfg, defense_cfg, device, logger, train_max_samples=None):
    """Madry PGD adversarial training whose loss/optimizer/scheduler/AMP exactly
    mirror scripts/train.py, so the ONLY deliberate difference from the standard
    model is the adversarial inner loop. Best checkpoint is selected on val ROBUST
    balanced accuracy (per-class recall under PGD)."""
    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]
    seed = cfg.get("seed", 42)

    # AT needs ~3x the memory of standard training; honor the defense batch_size.
    if defense_cfg.get("batch_size"):
        cfg = {**cfg, "data": {**cfg["data"], "batch_size": defense_cfg["batch_size"]}}

    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    model = create_model(model_name, num_classes, pretrained=cfg["model"]["pretrained"]).to(device)

    # --- loss / optimizer / scheduler / AMP mirror scripts/train.py ---
    train_cfg = cfg["train"]
    loss_mode = train_cfg.get("class_balance", {}).get("loss", "none")
    if loss_mode in {"weighted", "weighted_cross_entropy"}:
        weights = compute_class_weights(data["train"].dataset, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"AT loss: weighted CE, weights={weights.detach().cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("AT loss: plain CE")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    nb_epochs = defense_cfg.get("nb_epochs", train_cfg["epochs"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs)
                 if train_cfg.get("scheduler") == "cosine" else None)

    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    amp_dtype = (torch.bfloat16 if str(train_cfg.get("amp_dtype", "float16")).lower()
                 in {"bf16", "bfloat16"} else torch.float16)
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    eps = defense_cfg.get("eps", 8 / 255)
    eps_step = defense_cfg.get("eps_step", 2 / 255)
    max_iter = defense_cfg.get("max_iter", 7)
    logger.info(
        f"PGD-AT: eps={eps:.5f}, eps_step={eps_step:.5f}, inner_iter={max_iter}, "
        f"nb_epochs={nb_epochs}, batch_size={cfg['data']['batch_size']}, "
        f"weight_decay={train_cfg.get('weight_decay', 0.0)}, amp={use_amp}({amp_dtype})"
    )
    if train_max_samples is not None:
        logger.warning(f"SMOKE MODE: limiting AT to ~{train_max_samples} samples/epoch. "
                       "Do NOT use for formal results.")

    suffix = defense_cfg["name"].lower().replace("-", "_")
    ckpt_path = get_checkpoint_path("checkpoints", dataset_name, model_name, seed, suffix=suffix)

    best_robust_bal, best_epoch = -1.0, 0
    for epoch in range(1, nb_epochs + 1):
        model.train()
        seen, running = 0, 0.0
        for images, labels in tqdm(data["train"], desc=f"AT {epoch}/{nb_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            model.eval()                                   # stable BN during attack gen
            x_adv = _pgd_perturb(model, images, labels, eps, eps_step, max_iter)
            model.train()
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = criterion(model(x_adv), labels)     # Madry: train on adv only
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            if train_max_samples is not None and seen >= train_max_samples:
                break
        if scheduler:
            scheduler.step()

        robust_bal = _robust_balanced_acc(model, data["val"], eps, eps_step, max_iter, device, num_classes)
        logger.info(f"AT epoch {epoch}/{nb_epochs} — train loss {running / max(seen, 1):.4f} — "
                    f"val robust balanced acc {robust_bal:.4f}")
        if robust_bal > best_robust_bal:
            best_robust_bal, best_epoch = robust_bal, epoch
            torch.save({"model_state_dict": model.state_dict(), "defense": defense_cfg["name"],
                        "epoch": epoch, "val_robust_balanced_acc": robust_bal, "config": cfg}, ckpt_path)
            logger.info(f"  -> new best AT checkpoint (val robust balanced acc={robust_bal:.4f})")

    logger.info(f"AT complete. Best val robust balanced acc {best_robust_bal:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    model.eval()

    classifier = PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"]),
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=num_classes, clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier, data, ckpt_path


def _run_art_trainer(cfg, defense_cfg, device, logger, train_max_samples=None):
    """ART-trainer path (used for TRADES). NOTE: ART's trainer cannot apply the
    cosine schedule / AMP / val-checkpoint selection used by standard training, so
    TRADES is not yet fully harmonized — keep that in mind if comparing it directly."""
    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]
    seed = cfg.get("seed", 42)

    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]

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

    logger.info("Collecting training data for adversarial training...")
    if train_max_samples is not None:
        logger.warning(
            f"SMOKE MODE: limiting adversarial-training data to {train_max_samples} "
            "samples. Do NOT use this for formal results."
        )
    x_train, y_train = collect_test_data(data["train"], max_samples=train_max_samples)
    y_train_oh = np.eye(num_classes)[y_train]

    defense_name = defense_cfg["name"]
    logger.info(f"Starting adversarial training with {defense_name} (ART trainer)...")
    trainer = create_defense_trainer(classifier, defense_cfg)
    trainer.fit(
        x_train,
        y_train_oh,
        nb_epochs=defense_cfg.get("nb_epochs", 20),
        batch_size=defense_cfg.get("batch_size", cfg["data"]["batch_size"]),
    )

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
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the ATTACK-EVALUATION subset only (fixed stratified). "
                             "Does NOT limit adversarial-training data.")
    parser.add_argument("--smoke-train-samples", type=int, default=None,
                        help="SMOKE TEST ONLY: limit adversarial-training data. "
                             "Leave unset for formal AT (uses the full training set).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (for multi-seed AT, e.g. seed43).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
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
            # Adversarial training (FULL training set unless smoke-testing).
            classifier, data, checkpoint_for_meta = run_adversarial_training(
                cfg,
                defense_cfg,
                device,
                logger,
                train_max_samples=args.smoke_train_samples,
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

    # Collect FULL test data, then a FIXED stratified subset (shared across models).
    x_full, y_full = collect_test_data(data["test"], max_samples=None)
    if args.max_samples is not None and args.max_samples < len(x_full):
        subset_path = Path("results") / dataset_name / f"attack_subset_seed{seed}_n{args.max_samples}.json"
        idx = get_attack_subset(y_full, args.max_samples, seed, subset_path)
        x_test, y_test = x_full[idx], y_full[idx]
        logger.info(f"Fixed stratified subset: {len(x_test)}/{len(x_full)} samples")
    else:
        x_test, y_test = x_full, y_full
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

    # Adversarially trained models get a STRONG evaluation (PGD-50 + restarts +
    # AutoAttack) to avoid overestimating robustness. Preprocessor baselines keep
    # the standard attacks_main sweep.
    if defense_name in MAIN_DEFENSES:
        attacks = create_defense_eval_attacks(classifier, cfg)
        if not attacks:
            logger.warning("No 'defense_eval' section found; falling back to attacks_main.")
            attacks = create_attacks_from_config(classifier, cfg, section="attacks_main")
        else:
            logger.info(f"Strong defense evaluation: {[a[0] for a in attacks]}")
    else:
        attacks = create_attacks_from_config(classifier, cfg, section="attacks_main")
    # Output path (computed early so we can resume already-finished attacks, e.g.
    # to add AutoAttack to a defense_results.json that already has the PGD-50 sweep).
    output_name = "defense_results.json"
    if args.max_samples is not None:
        output_name = f"defense_results_max{args.max_samples}.json"
    output_file = results_dir / output_name

    all_results = {
        "_meta": make_run_metadata(args, defense_name, checkpoint_for_meta, len(x_test), clean_acc),
        "clean_accuracy_defended": float(clean_acc),
    }
    if output_file.exists():
        try:
            existing = json.load(open(output_file))
            existing_meta = existing.get("_meta", {})
            current_meta = all_results["_meta"]
            # Only reuse cached attacks if they came from the SAME checkpoint+run.
            # After retraining AT the checkpoint mtime changes, so old metrics must
            # be discarded — otherwise a new checkpoint gets paired with stale attack
            # results and the JSON silently mixes two experiments.
            same_run = (
                isinstance(existing_meta, dict)
                and existing_meta.get("checkpoint") == current_meta.get("checkpoint")
                and existing_meta.get("checkpoint_mtime") == current_meta.get("checkpoint_mtime")
                and existing_meta.get("max_samples") == current_meta.get("max_samples")
            )
            if same_run:
                for k, v in existing.items():
                    if k not in ("_meta",) and isinstance(v, (dict, float)) and not (
                        isinstance(v, dict) and "error" in v):
                        all_results[k] = v
                logger.info(f"Resuming: kept {sum(1 for k in all_results if k not in ('_meta','clean_accuracy_defended'))} existing attack results.")
            else:
                logger.warning(
                    "Existing defense_results are from a different checkpoint/run "
                    "(mtime or path changed); ignoring them and re-attacking from scratch."
                )
        except (json.JSONDecodeError, OSError):
            pass

    for attack_name, eps_val, attack in attacks:
        eps_str = f"eps={eps_val:.6f}" if eps_val is not None else "default"
        run_key = f"{attack_name}_{eps_str}"
        if run_key in all_results and isinstance(all_results[run_key], dict) and "error" not in all_results[run_key]:
            logger.info(f"Skipping {run_key} (already computed).")
            continue
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
                clean_preds, adv_preds, y_test, clean_confs, adv_confs, class_names,
                bootstrap=True, bootstrap_seed=seed,
            )
            all_results[run_key] = metrics

            ra = metrics["robust_accuracy"]["robust_accuracy"]
            asr = metrics["asr"]
            logger.info(f"  Robust Accuracy: {ra:.4f}, ASR: {asr:.4f}")

        except Exception as e:
            logger.error(f"  Attack {run_key} failed: {e}")
            all_results[run_key] = {"error": str(e)}

        # Incremental save so a crash/interrupt preserves completed attacks.
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
