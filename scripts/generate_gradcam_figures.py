"""Generate clean-vs-adversarial Grad-CAM comparison panels.

Picks successful adversarial examples (correctly classified clean, misclassified
after attack) and saves, for each, a 2x2 panel: clean image / clean Grad-CAM and
adversarial image / adversarial Grad-CAM. This visualises whether the attack
shifts the model's attention away from the lesion.

Usage:
    python scripts/generate_gradcam_figures.py --config configs/chest_xray_pneumonia_resnet50.yaml \
        --checkpoint checkpoints/chest_xray_pneumonia_resnet50_seed42.pth --attack PGD --eps 0.031373 --num-samples 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier

from src.utils.reproducibility import set_seed, load_config
from src.utils.logger import get_logger
from src.utils.gradcam import GradCAM, _save_panel
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.attacks.attack_factory import create_attack


def resolve_layer(model: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted attribute path (supporting integer indices) to a module."""
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def auto_target_layer(model: nn.Module):
    """Pick a sensible Grad-CAM target: the last Conv2d, else the last LayerNorm."""
    last_conv = last_ln = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
        elif isinstance(module, nn.LayerNorm):
            last_ln = module
    return last_conv if last_conv is not None else last_ln


def main():
    parser = argparse.ArgumentParser(description="Generate clean-vs-adversarial Grad-CAM panels")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attack", type=str, default="PGD")
    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of successful adversarial examples to visualise")
    parser.add_argument("--pool-size", type=int, default=128,
                        help="Number of test samples to scan for successful attacks")
    parser.add_argument("--target-layer", type=str, default=None,
                        help="Dotted path to the Grad-CAM target layer (default: auto)")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    dataset_name = cfg["data"]["dataset"]
    model_name = cfg["model"]["name"]

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "figures" / "gradcam" / dataset_name / model_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("generate_gradcam", log_dir=out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_dataloaders(cfg)
    num_classes = data["num_classes"]
    class_names = data.get("class_names") or [str(i) for i in range(num_classes)]

    model = create_model(model_name, num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device=device).to(device).eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    classifier = PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=num_classes, clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # Collect a pool of test samples.
    xs, ys, count = [], [], 0
    for images, labels in data["test"]:
        xs.append(images.numpy()); ys.append(labels.numpy())
        count += len(labels)
        if count >= args.pool_size:
            break
    x_pool = np.concatenate(xs)[: args.pool_size]
    y_pool = np.concatenate(ys)[: args.pool_size]

    clean_preds = np.argmax(classifier.predict(x_pool, batch_size=64), axis=1)

    # Generate adversarial examples for the pool.
    attack = create_attack(classifier, {"name": args.attack, "max_iter": 20}, eps=args.eps)
    logger.info(f"Generating {args.attack} (eps={args.eps:.6f}) on {len(x_pool)} samples...")
    x_adv = attack.generate(x=x_pool)
    adv_preds = np.argmax(classifier.predict(x_adv, batch_size=64), axis=1)

    # Successful adversarial examples: correct on clean, wrong after attack.
    success = np.where((clean_preds == y_pool) & (adv_preds != y_pool))[0]
    if len(success) == 0:
        logger.warning("No successful adversarial examples found; falling back to correctly-classified.")
        success = np.where(clean_preds == y_pool)[0]
    chosen = success[: args.num_samples]
    logger.info(f"Visualising {len(chosen)} examples (of {len(success)} candidates).")

    # Grad-CAM target layer.
    target = resolve_layer(model, args.target_layer) if args.target_layer else auto_target_layer(model)
    if target is None:
        logger.error("Could not find a Grad-CAM target layer. Pass --target-layer explicitly.")
        return
    logger.info(f"Grad-CAM target layer: {type(target).__name__}")
    cam = GradCAM(model, target_layer=target)

    for rank, idx in enumerate(chosen):
        clean_t = torch.from_numpy(x_pool[idx:idx + 1]).float().to(device)
        adv_t = torch.from_numpy(x_adv[idx:idx + 1]).float().to(device)
        clean_cam = cam(clean_t, class_idx=int(clean_preds[idx]))
        adv_cam = cam(adv_t, class_idx=int(adv_preds[idx]))

        stem = out_dir / f"sample_{rank:03d}"
        _save_panel(
            stem,
            clean_t[0].cpu(), adv_t[0].cpu(), clean_cam, adv_cam,
            true_label=class_names[int(y_pool[idx])],
            clean_pred=class_names[int(clean_preds[idx])],
            adv_pred=class_names[int(adv_preds[idx])],
        )
        logger.info(f"  saved {stem.name}: true={class_names[int(y_pool[idx])]} "
                    f"clean={class_names[int(clean_preds[idx])]} adv={class_names[int(adv_preds[idx])]}")

    cam.close()
    logger.info(f"\nGrad-CAM panels saved to {out_dir}")


if __name__ == "__main__":
    main()
