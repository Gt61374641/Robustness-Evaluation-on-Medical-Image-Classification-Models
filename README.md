# Medical Image Adversarial Robustness

This repository contains the current DenseNet121-focused experimental pipeline
for adversarial robustness evaluation on medical image classification tasks.

## Current Scope

- Model: DenseNet121.
- Primary dataset: ISIC 2020.
- Attacks: FGSM, PGD, DeepFool, AutoPGD, SquareAttack.
- Defenses: PGD-AT and TRADES.
- Outputs: clean metrics, robustness metrics, SCI-style figures, and paper tables.

Large local artifacts are intentionally excluded from git:

- `data/`
- `checkpoints/`
- `results/`
- `figures/`
- temporary pytest and Python cache directories

## Typical Commands

Run from this directory.

```powershell
python scripts/train.py --config configs/isic2020_densenet121.yaml
python scripts/evaluate_clean.py --config configs/isic2020_densenet121.yaml --checkpoint checkpoints/isic2020_densenet121_seed42.pth
python scripts/evaluate_robustness.py --config configs/isic2020_densenet121.yaml --checkpoint checkpoints/isic2020_densenet121_seed42.pth --max-samples 1024
```

Use `configs/isic2020_densenet121_balanced.yaml` for the next class-imbalance
correction run.
