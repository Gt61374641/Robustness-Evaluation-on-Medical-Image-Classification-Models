# Model Complexity and Adversarial Robustness in Medical Image Classification

Experimental framework for studying how **model complexity** affects **adversarial
robustness** in medical image classification, and how **adversarial training**
changes that relationship. The design follows Rodriguez et al. 2022 (BMC,
*On the role of deep learning model complexity in adversarial robustness for
medical images*): one architecture family at several complexities, swept across a
range of attack budgets, then adversarially trained and re-evaluated.

## Scope

- **Models** — ResNet complexity ladder: `resnet18, resnet34, resnet50, resnet101,
  resnet152` (~11.7M → ~60.2M params), ImageNet-pretrained fine-tuning.
  Architecture comparison (chest only): `deit_small` (ViT-S/16) + `convnext_tiny`,
  parameter-matched to ResNet-50, IN1k-only supervised weights.
- **Datasets** — `chest_xray_pneumonia` (binary, **primary**), `oct2017` (4-class),
  `malaria` (binary, color; NIH cell images, **patient-level split**).
- **Attacks** — FGSM + PGD core eps-sweep `{1,2,4,8,16}/255` (stress `{32,64}/255`);
  attack-method comparison (`attacks_extra`, chest): CW, DeepFool, AutoAttack
  (binary-safe reduced ensemble), SquareAttack.
- **Adversarial training** — PGD-AT (all five ResNets + both new architectures,
  full training set), TRADES + MART as second/third methods (chest 18/50/152);
  defended models evaluated with a **strong** protocol (PGD-50 + 5 restarts;
  AutoAttack where ≥3 classes).
- **Outputs** — clean metrics, robustness metrics (full + conditional robust
  accuracy), complexity-vs-eps curves, complexity/AT tables, Grad-CAM.

> **Research hypotheses (to test, not assume):** H1 — do lower-complexity models
> show higher robustness? H2 — does adversarial training change the ranking?

Large local artifacts are git-excluded: `data/`, `checkpoints/`, `results/`,
`figures/`, caches. Archived (pre-pivot) ISIC/detection material lives in `_archive/`.

## Setup

```powershell
# inside the (medimg-robust) env
python scripts/download_data.py --dataset chest_xray_pneumonia
python scripts/download_data.py --dataset oct2017
python scripts/download_data.py --dataset malaria
python scripts/make_configs.py        # generate the 15 per-model configs
```

## Typical commands

```powershell
# one (dataset, model)
python scripts/train.py --config configs/chest_xray_pneumonia_resnet18.yaml
python scripts/evaluate_clean.py --config configs/chest_xray_pneumonia_resnet18.yaml --checkpoint checkpoints/chest_xray_pneumonia_resnet18_seed42.pth
python scripts/evaluate_robustness.py --config configs/chest_xray_pneumonia_resnet18.yaml --checkpoint checkpoints/chest_xray_pneumonia_resnet18_seed42.pth --max-samples 1024

# adversarial training (FULL training set; --max-samples limits only the attack-eval subset)
python scripts/evaluate_defense.py --config configs/chest_xray_pneumonia_resnet50.yaml --defense PGD-AT --max-samples 1024

# signature figures (after the ladder is evaluated)
python scripts/generate_complexity_figures.py --dataset chest_xray_pneumonia --seed seed42

# or drive a whole dataset end-to-end (train -> clean -> robustness -> figures)
bash run_dataset.sh chest_xray_pneumonia
```

See `../MEDICAL_ROBUSTNESS_PLAN.md` (project root) for the full plan, experiment
matrix, and methodology (eps grid, robust-accuracy definitions, strong-attack
evaluation protocol, fair-comparison subset).
