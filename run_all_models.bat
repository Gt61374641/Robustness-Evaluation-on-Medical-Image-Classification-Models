@echo off
REM ============================================================
REM  Full study driver (good for an overnight unattended run):
REM    1) standard training + robustness for the ResNet ladder on all 3 datasets
REM    2) PGD-AT on the primary dataset (chest_xray) for resnet18/50/152 (FULL train set)
REM    3) complexity figures per dataset
REM
REM  Usage (inside the activated (medimg-robust) env):
REM    run_all_models.bat
REM ============================================================
setlocal enabledelayedexpansion
cd /d %~dp0

set DATASETS=chest_xray_pneumonia oct2017 malaria
set LADDER=resnet18 resnet34 resnet50 resnet101 resnet152
set AT_MODELS=resnet18 resnet50 resnet152

echo === [1] Standard training + robustness (5 models x 3 datasets) ===
for %%D in (%DATASETS%) do (
  for %%M in (%LADDER%) do (
    call run_pipeline.bat %%D %%M
  )
)

echo === [2] PGD-AT on primary (chest_xray) — FULL training set ===
for %%M in (%AT_MODELS%) do (
  python scripts\evaluate_defense.py --config configs\chest_xray_pneumonia_%%M.yaml --defense PGD-AT --max-samples 1024
  REM Grad-CAM on the adversarially trained model (paper Fig 4)
  python scripts\generate_gradcam_figures.py --config configs\chest_xray_pneumonia_%%M.yaml --checkpoint checkpoints\chest_xray_pneumonia_%%M_seed42_pgd_at.pth --attack PGD --eps 0.031373 --num-samples 8 --out-dir figures\gradcam\chest_xray_pneumonia\%%M_at
)

echo === [3] Complexity figures per dataset ===
for %%D in (%DATASETS%) do (
  python scripts\generate_complexity_figures.py --dataset %%D --seed seed42
)

echo === [4] Decision-boundary t-SNE (primary: standard + AT) ===
python scripts\generate_decision_boundary_figures.py --dataset chest_xray_pneumonia --models resnet18 resnet152
python scripts\generate_decision_boundary_figures.py --dataset chest_xray_pneumonia --models resnet18 resnet152 --checkpoint-suffix _pgd_at

echo ============================================================
echo  ALL DONE (check output above for any per-step errors)
echo ============================================================
endlocal
