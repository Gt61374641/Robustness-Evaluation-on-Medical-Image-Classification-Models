@echo off
REM Standard pipeline (train -> clean -> robustness main+fine) for ALL 5 ResNets
REM on one dataset. Reuses run_pipeline.bat per model (seed42).
REM   Usage:  run_dataset.bat malaria
REM           run_dataset.bat oct2017
REM Adversarial training is NOT run here (do it later via evaluate_defense.py).
setlocal
cd /d %~dp0

set DATASET=%~1
if "%DATASET%"=="" (
  echo Usage: run_dataset.bat ^<dataset^>     [ chest_xray_pneumonia ^| oct2017 ^| malaria ]
  exit /b 1
)

for %%m in (resnet18 resnet34 resnet50 resnet101 resnet152) do (
  echo.
  echo #################### %DATASET% / %%m ####################
  call run_pipeline.bat %DATASET% %%m
)

echo.
echo ===== standard pipeline done for all 5 models: %DATASET% =====
endlocal
