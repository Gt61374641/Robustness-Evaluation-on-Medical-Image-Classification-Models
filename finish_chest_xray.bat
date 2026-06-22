@echo off
REM ============================================================
REM  Finish ALL remaining chest_xray experiments (run overnight).
REM  Run inside the activated (medimg-robust) env, from code1\.
REM
REM  Sections (comment out any you don't want):
REM    [A] seed42 gap-fill (resnet152 clean/main/gradcam, resnet18 clean re-run)
REM    [B] multi-seed standard line: seeds 43 & 44, all 5 models
REM    [C] PGD-AT on primary (resnet18/50/152, FULL training set) + AT Grad-CAM
REM    [D] decision-boundary t-SNE (standard + AT)
REM    [E] regenerate all figures with 3-seed mean +/- std bands
REM
REM  Rough time: A ~10min, B ~2-3h, C ~4-6h, D/E ~minutes.  Total ~7-10h.
REM ============================================================
setlocal enabledelayedexpansion
cd /d %~dp0

set DS=chest_xray_pneumonia
set LADDER=resnet18 resnet34 resnet50 resnet101 resnet152
set AT_MODELS=resnet18 resnet50 resnet152
set EPS8=0.031373

echo ============================================================
echo [A] seed42 gap-fill
echo ============================================================
REM resnet18 clean re-run (was computed before enriched AUC/F1 metrics existed)
python scripts\evaluate_clean.py --config configs\%DS%_resnet18.yaml --checkpoint checkpoints\%DS%_resnet18_seed42.pth --seed 42
REM resnet152 missing clean + main + standard Grad-CAM
python scripts\evaluate_clean.py --config configs\%DS%_resnet152.yaml --checkpoint checkpoints\%DS%_resnet152_seed42.pth --seed 42
python scripts\evaluate_robustness.py --config configs\%DS%_resnet152.yaml --checkpoint checkpoints\%DS%_resnet152_seed42.pth --max-samples 1024 --seed 42
python scripts\generate_gradcam_figures.py --config configs\%DS%_resnet152.yaml --checkpoint checkpoints\%DS%_resnet152_seed42.pth --attack PGD --eps %EPS8% --num-samples 8

echo ============================================================
echo [B] multi-seed standard line (seeds 43, 44 x 5 models)
echo ============================================================
for %%S in (43 44) do (
  for %%M in (%LADDER%) do (
    echo --- seed %%S / %%M : train ---
    python scripts\train.py --config configs\%DS%_%%M.yaml --seed %%S
    echo --- seed %%S / %%M : clean ---
    python scripts\evaluate_clean.py --config configs\%DS%_%%M.yaml --checkpoint checkpoints\%DS%_%%M_seed%%S.pth --seed %%S
    echo --- seed %%S / %%M : robustness main + fine ---
    python scripts\evaluate_robustness.py --config configs\%DS%_%%M.yaml --checkpoint checkpoints\%DS%_%%M_seed%%S.pth --max-samples 1024 --seed %%S
    python scripts\evaluate_robustness.py --config configs\%DS%_%%M.yaml --checkpoint checkpoints\%DS%_%%M_seed%%S.pth --attacks-section attacks_fine --max-samples 1024 --seed %%S
  )
)

echo ============================================================
echo [C] PGD-AT on primary (FULL training set, seed42) + AT Grad-CAM
echo ============================================================
for %%M in (%AT_MODELS%) do (
  echo --- PGD-AT %%M ---
  python scripts\evaluate_defense.py --config configs\%DS%_%%M.yaml --defense PGD-AT --max-samples 1024
  python scripts\generate_gradcam_figures.py --config configs\%DS%_%%M.yaml --checkpoint checkpoints\%DS%_%%M_seed42_pgd_at.pth --attack PGD --eps %EPS8% --num-samples 8 --out-dir figures\gradcam\%DS%\%%M_at
)

echo ============================================================
echo [D] decision-boundary t-SNE (standard + AT)
echo ============================================================
python scripts\generate_decision_boundary_figures.py --dataset %DS% --models resnet18 resnet152
python scripts\generate_decision_boundary_figures.py --dataset %DS% --models resnet18 resnet152 --checkpoint-suffix _pgd_at

echo ============================================================
echo [E] regenerate figures with 3-seed mean +/- std bands
echo ============================================================
python scripts\generate_complexity_figures.py --dataset %DS% --seeds seed42 seed43 seed44

echo ============================================================
echo  chest_xray DONE. Check figures\complexity\%DS%\ and the summary table.
echo ============================================================
endlocal
