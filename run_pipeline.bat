@echo off
REM ============================================================
REM  Standard-training pipeline for ONE (dataset, model):
REM    train -> clean eval -> robustness eps-sweep -> [primary] grad-cam
REM
REM  Usage (inside the activated (medimg-robust) env, from anywhere):
REM    run_pipeline.bat chest_xray_pneumonia resnet18
REM    run_pipeline.bat oct2017 resnet50
REM
REM  Adversarial training is run separately (see run_all_models.bat AT block).
REM ============================================================
setlocal
cd /d %~dp0

if "%~2"=="" (
  echo Usage: run_pipeline.bat ^<dataset^> ^<model^>
  echo   dataset: chest_xray_pneumonia ^| oct2017 ^| malaria
  echo   model:   resnet18 ^| resnet34 ^| resnet50 ^| resnet101 ^| resnet152
  exit /b 1
)

set DATASET=%~1
set MODEL=%~2
set CFG=configs\%DATASET%_%MODEL%.yaml
set CKPT=checkpoints\%DATASET%_%MODEL%_seed42.pth

if not exist "%CFG%" (
  echo Config not found: %CFG%   ^(run: python scripts\make_configs.py^)
  exit /b 1
)

echo ============================================================
echo  Pipeline for %DATASET% / %MODEL%    config=%CFG%
echo ============================================================

echo [1/4] Training...
python scripts\train.py --config %CFG%
if errorlevel 1 goto :error

echo [2/4] Clean evaluation...
python scripts\evaluate_clean.py --config %CFG% --checkpoint %CKPT%
if errorlevel 1 goto :error

echo [3/4] Robustness eps-sweep (fixed stratified subset @1024)...
python scripts\evaluate_robustness.py --config %CFG% --checkpoint %CKPT% --max-samples 1024
if errorlevel 1 goto :error
echo       ... fine sub-1/255 probe (where PGD separates by complexity)...
python scripts\evaluate_robustness.py --config %CFG% --checkpoint %CKPT% --attacks-section attacks_fine --max-samples 1024
if errorlevel 1 goto :error

echo [4/4] Grad-CAM (primary dataset only)...
if /i "%DATASET%"=="chest_xray_pneumonia" (
  python scripts\generate_gradcam_figures.py --config %CFG% --checkpoint %CKPT% --attack PGD --eps 0.031373 --num-samples 8
) else (
  echo   Skipping Grad-CAM for secondary dataset %DATASET%.
)

echo ============================================================
echo  DONE: %DATASET% / %MODEL%
echo ============================================================
endlocal
exit /b 0

:error
echo.
echo *** ERROR at a step for %DATASET% / %MODEL%. Stopping. ***
endlocal
exit /b 1
