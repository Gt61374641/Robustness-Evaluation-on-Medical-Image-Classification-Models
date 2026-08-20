# Developer Guide

This guide explains how to navigate, install, run, and extend the medical-image
robustness evaluation code. The top-level `README.md` is the short reproduction
route; this file is the operational reference for a new developer or server.

## 1. System overview

The project is a configuration-driven PyTorch pipeline:

1. A YAML file selects the dataset, model, training protocol, attacks, and
   defences.
2. Dataset loaders return a common train/validation/test interface.
3. A `timm` backbone is wrapped so ImageNet normalisation occurs inside the model
   while attacks operate directly in the `[0, 1]` pixel space.
4. Training writes checkpoints and structured per-run results.
5. Clean, robustness, and defence entry points write JSON outputs.
6. The evidence builder aggregates stored JSON files into the dissertation tables,
   figures, provenance manifest, and audit reports.

## 2. Functionality-to-code map

| Functionality | Primary location | Entry point or main symbol |
|---|---|---|
| Dataset selection | `src/datasets/__init__.py` | `get_dataloaders` |
| Chest X-ray loading | `src/datasets/chest_xray.py` | `get_chest_xray_loaders` |
| Malaria loading and grouped split | `src/datasets/malaria.py` | `get_malaria_loaders` |
| Retinal OCT loading | `src/datasets/retinal.py` | `get_retinal_loaders` |
| Train/evaluation transforms | `src/datasets/transforms.py` | `get_transforms` |
| Model construction and normalisation | `src/models/model_factory.py` | `create_model`, `NormalizedModel` |
| Attack construction | `src/attacks/attack_factory.py` | `create_attack`, `create_attacks_from_config` |
| Defence construction | `src/defenses/defense_factory.py` | `create_defense_trainer`, `create_preprocessor_defense` |
| Clean and robustness metrics | `src/evaluation/metrics.py` | metric functions used by the evaluation scripts |
| Fixed evaluation subsets | `src/evaluation/subset.py` | stratified subset utilities |
| Seed and provenance handling | `src/utils/reproducibility.py` | `set_seed`, `save_config_snapshot` |
| Standard training | `scripts/train.py` | command-line entry point |
| Clean evaluation | `scripts/evaluate_clean.py` | command-line entry point |
| Attack evaluation | `scripts/evaluate_robustness.py` | command-line entry point |
| Adversarial training and defence evaluation | `scripts/evaluate_defense.py` | command-line entry point |
| AutoAttack audit | `scripts/run_autoattack_audit.py` | command-line entry point |
| Spectral-energy diagnostic | `scripts/analyze_spectral_energy.py` | command-line entry point |
| Tables, figures, and audit package | `scripts/build_thesis_evidence_package.py` | command-line entry point |

## 3. Dependencies

The tested environment is Python 3.11 with PyTorch 2.8.0, torchvision, timm
1.0.26, and Adversarial Robustness Toolbox (ART) 1.20.1. Lower bounds are
recorded in `requirements.txt` so a server image can provide a CUDA-compatible
PyTorch build without `pip` replacing it with a mismatched build.

The dependency groups are:

- deep learning: `torch`, `torchvision`, `timm`;
- adversarial evaluation: `adversarial-robustness-toolbox`, `multiprocess`;
- analysis: `numpy`, `scipy`, `scikit-learn`, `pandas`;
- plots and images: `matplotlib`, `seaborn`, `Pillow`;
- utilities: `tqdm`, `PyYAML`, `kaggle`.

A CUDA-capable GPU is strongly recommended for training and attack generation.
The evidence builder can run on CPU from the stored JSON results.

## 4. Clean installation on a Linux server

The following sequence assumes Ubuntu or another Linux distribution with Git,
Python 3.11, and an NVIDIA driver already installed.

```bash
git clone <repository-url>
cd Robustness-Evaluation-on-Medical-Image-Classification-Models

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a GPU server, use a PyTorch build compatible with the installed NVIDIA driver
and CUDA runtime. A managed image containing PyTorch 2.x and CUDA 12.x is the
simplest option. Verify the environment before downloading data:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import timm, art; print(timm.__version__); print(art.__version__)"
```

If the first command reports `False` for CUDA availability, correct the NVIDIA
driver/PyTorch build pairing before starting a GPU experiment. Do not mix a CPU
PyTorch wheel with a CUDA torchvision wheel.

## 5. Data preparation

Create a Kaggle API token in the Kaggle account settings and place it at
`~/.kaggle/kaggle.json`. On Linux, restrict its permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_data.py --dataset chest_xray_pneumonia
python scripts/download_data.py --dataset malaria
python scripts/download_data.py --dataset oct2017
```

The loaders expect these roots:

```text
data/chest_xray_pneumonia/{train,val,test}/<class>/
data/malaria/cell_images/{Parasitized,Uninfected}/
data/oct2017/{train,test}/<class>/
```

The OCT loader also detects the nested directory variants produced by different
versions of the Kaggle archive. Data are excluded from version control.

## 6. Configuration

`configs/` contains one YAML file per dataset and architecture. Start from the
configuration that matches the intended experiment, for example
`configs/chest_xray_pneumonia_resnet50.yaml`. The main sections are:

- `data`: dataset key, root directory, image size, batch size, and workers;
- `model`: public model name, pretraining flag, and class count;
- `train`: epochs, optimiser settings, precision, scheduler, and imbalance policy;
- `attacks_main`, `attacks_fine`, `attacks_stress`, `attacks_extended`,
  `attacks_extra`: attack protocols and perturbation grids;
- `defenses_main`: PGD-AT, TRADES, MART, and the labelled rescue protocol;
- `defense_eval`: the strong evaluation protocol for defended models.

Copy a configuration before changing an experimental protocol. Result folders
contain a snapshot of the effective YAML and the originating git commit, which
makes later comparisons auditable.

## 7. End-to-end run

Run commands from the repository root. The example below uses Chest X-ray,
ResNet-50, and seed 42.

```bash
python scripts/train.py \
  --config configs/chest_xray_pneumonia_resnet50.yaml \
  --seed 42

python scripts/evaluate_clean.py \
  --config configs/chest_xray_pneumonia_resnet50.yaml \
  --checkpoint checkpoints/chest_xray_pneumonia_resnet50_seed42.pth \
  --seed 42

python scripts/evaluate_robustness.py \
  --config configs/chest_xray_pneumonia_resnet50.yaml \
  --checkpoint checkpoints/chest_xray_pneumonia_resnet50_seed42.pth \
  --attacks-section attacks_main \
  --seed 42

python scripts/evaluate_defense.py \
  --config configs/chest_xray_pneumonia_resnet50.yaml \
  --defense PGD-AT \
  --seed 42

python scripts/build_thesis_evidence_package.py
```

Repeat the relevant commands for seeds 43 and 44 and for each dataset/model
configuration required by the experiment. `--max-samples` is suitable for smoke
tests of attack evaluation, but it must not be used as a substitute for the
formal full-sample protocol. For adversarial training, only
`--smoke-train-samples` deliberately reduces the training set and is explicitly
labelled as a smoke-test option.

## 8. Outputs and provenance

Standard checkpoints are written as:

```text
checkpoints/<dataset>_<model>_seed<N>.pth
```

Adversarially trained checkpoints add a method suffix. Results are organised as:

```text
results/<dataset>/<model>/train/seed<N>/
results/<dataset>/<model>/clean/seed<N>/
results/<dataset>/<model>/robustness/seed<N>/
results/<dataset>/<model>/defense_<METHOD>/seed<N>/
```

Each result directory contains JSON outputs and a configuration snapshot. The
evidence builder writes dissertation-facing artefacts to
`figures/thesis_main/`, `figures/thesis_tables/`, and
`reports/thesis_evidence/`. `figures/thesis_main/figure_manifest.json` maps each
numbered figure to its source files, and records for each one the formats that
are actually present on disk rather than an assumed PNG/PDF/SVG triple.

One figure is not plotted by the builder. The workflow schematic (dissertation
Figure 4) is authored in diagrams.net and kept as
`figures/thesis_main/fig02_evaluation_workflow_unified.drawio` with its `.png`
and `.svg` exports. To change it, edit the `.drawio` file, re-export both, and
rerun the builder. The builder raises `FileNotFoundError` if the PNG export is
missing, so a lost export fails the rebuild instead of quietly dropping the
figure or reinstating an older plotted version.

`reports/thesis_evidence/analysis_note.md` is regenerated on every build. Its
"AutoAttack audit" section re-derives the regime grouping used by dissertation
Table 4 by joining `autoattack_audit.csv` to the per-seed regimes in
`defense_summary.csv`, and lists any audited cell whose published PGD-50 figure
could not be located on disk. Keep that section derived rather than hand-edited:
an earlier hand-maintained version of the table drifted out of step with the
regime classifier.

## 9. Reproducibility and safe development

- Keep the formal seeds at 42, 43, and 44 unless a new protocol is clearly named.
- Do not silently overwrite a published configuration or mix outputs from
  different checkpoints. The evaluation scripts record checkpoint paths and
  modification times to prevent stale results from being reused.
- Preserve `[0, 1]` input pixels. `NormalizedModel` applies ImageNet
  normalisation at the model boundary so ART epsilon values remain pixel-space
  quantities.
- Treat rescue runs and reduced-sample tests as separate protocols; do not merge
  them into the primary result grid.
- Never commit Kaggle credentials, downloaded medical images, checkpoints, or
  machine-specific absolute paths.

## 10. Troubleshooting

**CUDA is unavailable.** Check `nvidia-smi`, then compare the driver capability
with the CUDA runtime reported by PyTorch. Reinstall a matching PyTorch and
torchvision pair rather than changing project code.

**GPU memory is exhausted.** Reduce `data.batch_size` or use the existing
gradient-accumulation setting for standard training. For a smoke test, reduce the
evaluation sample count. A reduced run must not be reported as the formal result.

**Kaggle download fails.** Confirm that the token exists, is valid, and has mode
`600`. The script prints the exact dataset slug and destination.

**A dataset is not found.** Compare the extracted tree with Section 5. The OCT
loader handles common nested archive layouts; the other loaders report the
expected directory in their error message.

**A checkpoint cannot be loaded.** Confirm that the dataset, architecture, class
count, and seed match the selected YAML. Legacy unwrapped timm checkpoints are
accepted with a warning, but they should not be compared with newly trained
normalised models without retraining.

**Attack evaluation is unexpectedly slow.** CW, DeepFool, AutoAttack, and large
query budgets are intentionally expensive. First validate the pipeline with
`attacks_main` or a labelled `--max-samples` smoke test, then run the full protocol.

**An interrupted robustness run is restarted.** The robustness and defence
scripts save completed attacks incrementally and only resume when run metadata
matches the same checkpoint and protocol.

## 11. Extending the project

To add a dataset, implement a loader returning the common dictionary interface,
register it in `src/datasets/__init__.py`, and add a YAML configuration. To add a
model, add a public-to-timm mapping in `src/models/model_factory.py` or supply a
valid timm architecture name. To add an attack or defence, register the class in
the corresponding factory and expose all protocol parameters in YAML. Add a
small smoke run before launching the full seed grid, and keep new result
namespaces separate until the evidence builder explicitly supports them.
