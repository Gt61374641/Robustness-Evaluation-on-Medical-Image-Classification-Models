@echo off
REM Pure cmd.exe runner for the corrected Chest X-ray robustness pipeline.
REM Run directly from Anaconda Prompt:
REM   run_corrected_pipeline.bat            (all: strong-PGD then AT seed42+43)
REM   run_corrected_pipeline.bat strong     (only the fast strong-PGD recheck)
REM   run_corrected_pipeline.bat at         (only PGD-AT re-train)
setlocal enabledelayedexpansion

set "PY=%USERPROFILE%\anaconda3\envs\medimg-robust\python.exe"
if not exist "%PY%" set "PY=python"

set "PHASE=%~1"
if "%PHASE%"=="" set "PHASE=all"
set "MAX=1024"

echo Python: %PY%
echo Phase : %PHASE%

if /I "%PHASE%"=="all"    goto strong
if /I "%PHASE%"=="strong" goto strong
if /I "%PHASE%"=="at"     goto at
echo Unknown phase "%PHASE%". Use: all ^| strong ^| at
exit /b 1

:strong
echo.
echo ##### PHASE: strong-PGD recheck (standard models, seed42) #####
for %%m in (resnet18 resnet34 resnet50 resnet101 resnet152) do (
  set "CKPT=checkpoints\chest_xray_pneumonia_%%m_seed42.pth"
  if not exist "!CKPT!" (
    echo [skip] missing checkpoint: !CKPT!
  ) else (
    echo.
    echo ==== strong-PGD  %%m  seed42 ====
    "%PY%" scripts\evaluate_robustness.py --config configs\chest_xray_pneumonia_%%m.yaml --checkpoint "!CKPT!" --strong-pgd --max-samples %MAX%
  )
)
if /I "%PHASE%"=="strong" goto done

:at
echo.
echo ##### PHASE: PGD-AT re-train (resnet18/50/152, seeds 42 43) #####
echo (retrains and overwrites *_pgd_at.pth; resume guard discards stale attack JSONs)
for %%m in (resnet18 resnet50 resnet152) do (
  for %%s in (42 43) do (
    echo.
    echo ==== PGD-AT  %%m  seed%%s ====
    "%PY%" scripts\evaluate_defense.py --config configs\chest_xray_pneumonia_%%m.yaml --defense PGD-AT --max-samples %MAX% --seed %%s
  )
)

:done
echo.
echo ===== pipeline finished =====
endlocal
