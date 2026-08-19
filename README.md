# Robustness Evaluation on Medical Image Classification Models

Code, configurations, per-run experimental outputs, and figures for the
dissertation *Robustness Evaluation on Medical Image Classification Models*.

The study is a systematic evaluation of adversarial robustness under one fixed
protocol: three medical imaging datasets (paediatric chest X-ray, thin-blood-smear
malaria microscopy, retinal OCT), a five-depth ResNet complexity ladder plus a
DeiT-S/ConvNeXt-T cross-architecture case study, FGSM/PGD budget sweeps summarised
by a conservative audited attack envelope, a Chest X-ray multi-attack audit
(AutoAttack, Square, CW, DeepFool), and three adversarial-training defences
(PGD-AT, TRADES, MART) with every training seed retained and convergence failures
disclosed.

## What is (and is not) in this repository

| Included | Not included |
|---|---|
| `src/` — datasets, models, attacks, defences, metrics | Datasets (public; see download instructions below) |
| `configs/` — one YAML per dataset × architecture | Model checkpoints (≈ GB scale; retrainable with the pipeline below) |
| `scripts/` — training, evaluation, audits, figure generation | The dissertation text itself |
| `results/` — all per-run JSON outputs (clean, attacks, defences) and run configs | Training/evaluation logs |
| `figures/thesis_main/` — the 14 dissertation figures (PNG/PDF/SVG) with provenance manifest | |
| `figures/thesis_tables/` — the main-table CSVs | |

Because every per-run output is included, **all dissertation figures and tables can
be regenerated without a GPU and without retraining anything** (see step 5 below).

## Repository layout

```
src/                    Reusable library code
  datasets/             Loaders + patient-grouped Malaria split
  models/               timm-based model factory (ImageNet-1k init)
  attacks/              ART attack factories (FGSM, PGD, AutoAttack, Square, CW, DeepFool)
  defenses/             Adversarial-training factories (PGD-AT, TRADES, MART)
  evaluation/           Metrics, conditional robust accuracy, stratified subsets
configs/                Experiment configs (3 datasets × 7 architectures)
scripts/                Entry points (see pipeline below)
results/<dataset>/<model>/
  train/seed<..>/       Training history + config snapshot
  clean/seed<..>/       Clean test metrics
  robustness/seed<..>/  FGSM/PGD sweeps; Chest X-ray also extra/stress/strong-PGD attacks
  defense_<METHOD>/seed<..>/  Defended-model evaluations (PGD-50, 5 restarts)
figures/thesis_main/    fig01–fig14 as cited in the dissertation + figure_manifest.json
figures/thesis_tables/  Main tables as CSV
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# A Kaggle API token is required at ~/.kaggle/kaggle.json
python scripts/download_data.py --dataset chest_xray_pneumonia
python scripts/download_data.py --dataset malaria
python scripts/download_data.py --dataset oct2017
```

Datasets (all public): Kermany paediatric chest X-ray, NIH/NLM malaria cell
images, and Kermany OCT2017.

For a clean-server installation, a functionality-to-file map, expected outputs,
GPU checks, and troubleshooting, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

## Reproduction pipeline

Each stage writes JSON to `results/` and is run per dataset/model/seed
(seeds 42, 43, 44 throughout).

```bash
# 1. Standard training (per config; ~minutes to hours on one GPU)
python scripts/train.py --config configs/chest_xray_pneumonia_resnet50.yaml --seed 42

# 2. Clean evaluation
python scripts/evaluate_clean.py --config configs/chest_xray_pneumonia_resnet50.yaml \
    --checkpoint checkpoints/<...>.pth --seed 42

# 3. Attack sweeps (FGSM + PGD-20 over ten budgets; Chest X-ray also runs
#    the extra/stress attack sections and strong PGD)
python scripts/evaluate_robustness.py --config configs/chest_xray_pneumonia_resnet50.yaml \
    --checkpoint checkpoints/<...>.pth --seed 42

# 4. Adversarial training + defended evaluation (PGD-AT | TRADES | MART)
python scripts/evaluate_defense.py --config configs/chest_xray_pneumonia_resnet50.yaml \
    --defense PGD-AT --seed 42

# 5. Rebuild every dissertation figure and table from stored results (no GPU needed)
python scripts/build_thesis_evidence_package.py

# Optional: AutoAttack audit of the defended Chest X-ray checkpoints
python scripts/run_autoattack_audit.py --datasets chest_xray_pneumonia
```

Use `python scripts/<entry-point>.py --help` to inspect all supported options.
Training and attack evaluation are computationally expensive; rebuilding the
stored evidence package and dissertation figures is CPU-only.

## Mapping to the dissertation

- `figures/thesis_main/fig01…fig14` are the figures cited as Figures 1–14;
  `figure_manifest.json` records the source data behind each one and
  `figure_legends.md` holds the captions.
- `figures/thesis_tables/table01…table04` are dissertation Tables 1–4.
- The audited attack-set envelope (per-budget FGSM/PGD minimum, then a cumulative
  minimum over increasing budget) is computed in
  `scripts/build_thesis_evidence_package.py`, which also assigns each defence run
  its optimisation regime (successful / partial / collapsed).

## Reproducibility notes

- The formal experiments use seeds 42, 43, and 44.
- Each run stores a configuration snapshot and originating git commit under
  `results/<dataset>/<model>/<experiment>/seed<N>/`.
- Downloaded data and checkpoints are intentionally excluded from version
  control. Checkpoints can be regenerated from the included configurations.
- Do not commit Kaggle credentials, patient data, downloaded datasets, or model
  checkpoints.
