<#
.SYNOPSIS
  Re-run the Chest X-ray robustness experiments after the methodology fixes
  (harmonized PGD-AT loop, stratified val, resume-mtime guard, strong-PGD aligned
  to the AT eval protocol).

.DESCRIPTION
  Phase "strong" : re-check the 5 standard models with the SAME strong protocol as
                   the AT eval (config 'defense_eval': PGD-50 + 5 restarts). Fast,
                   no training. Run this FIRST to confirm the large-eps behaviour.
  Phase "at"     : re-train + strong-evaluate PGD-AT for resnet18/50/152 across the
                   requested seeds. Slow (retrains models). Overwrites the old
                   *_pgd_at.pth checkpoints; the resume guard discards stale attack
                   JSONs automatically.

.EXAMPLE
  # everything, recommended order (strong first, then AT seed42+43)
  .\run_corrected_pipeline.ps1

.EXAMPLE
  # only the fast strong-PGD recheck
  .\run_corrected_pipeline.ps1 -Phase strong

.EXAMPLE
  # only AT, only seed43
  .\run_corrected_pipeline.ps1 -Phase at -AtSeeds 43
#>
param(
    [ValidateSet("all", "strong", "at")] [string]$Phase = "all",
    [int[]]$AtSeeds = @(42, 43),
    [int]$StrongSeed = 42,
    [int]$MaxSamples = 1024
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# --- Resolve the medimg-robust environment's python -------------------------
$envPy = Join-Path $env:USERPROFILE "anaconda3\envs\medimg-robust\python.exe"
if (Test-Path $envPy) {
    $py = $envPy
} else {
    Write-Warning "env python not found at $envPy; falling back to 'python' on PATH (make sure 'conda activate medimg-robust' was run)."
    $py = "python"
}
Write-Host "Python: $py"

$standardModels = @("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
$atModels       = @("resnet18", "resnet50", "resnet152")

$failures = New-Object System.Collections.Generic.List[string]
$skipped  = New-Object System.Collections.Generic.List[string]
$startAll = Get-Date

function Invoke-Step {
    param([string]$Desc, [string[]]$PyArgs)
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Desc)
    Write-Host ("=" * 72)
    $t0 = Get-Date
    & $py @PyArgs
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Warning ("FAILED (exit {0}, {1} min): {2}" -f $code, $mins, $Desc)
        $failures.Add($Desc)
    } else {
        Write-Host ("OK ({0} min): {1}" -f $mins, $Desc) -ForegroundColor Green
    }
}

# --- Phase: strong-PGD recheck on the standard models -----------------------
if ($Phase -eq "all" -or $Phase -eq "strong") {
    Write-Host "`n##### PHASE: strong-PGD recheck (standard models, seed$StrongSeed) #####"
    foreach ($m in $standardModels) {
        $config = "configs/chest_xray_pneumonia_$m.yaml"
        $ckpt   = "checkpoints/chest_xray_pneumonia_${m}_seed${StrongSeed}.pth"
        if (-not (Test-Path $ckpt)) {
            Write-Warning "checkpoint missing, skipping: $ckpt"
            $skipped.Add("strong $m seed$StrongSeed (no checkpoint)")
            continue
        }
        Invoke-Step "strong-PGD  $m  seed$StrongSeed" @(
            "scripts/evaluate_robustness.py",
            "--config", $config,
            "--checkpoint", $ckpt,
            "--strong-pgd",
            "--max-samples", "$MaxSamples"
        )
    }
}

# --- Phase: PGD-AT re-train + strong eval -----------------------------------
if ($Phase -eq "all" -or $Phase -eq "at") {
    Write-Host "`n##### PHASE: PGD-AT re-train (resnet18/50/152, seeds: $($AtSeeds -join ',')) #####"
    Write-Warning "This RETRAINS and overwrites the *_pgd_at.pth checkpoints. The resume guard discards stale attack JSONs automatically."
    foreach ($m in $atModels) {
        $config = "configs/chest_xray_pneumonia_$m.yaml"
        foreach ($s in $AtSeeds) {
            Invoke-Step "PGD-AT      $m  seed$s" @(
                "scripts/evaluate_defense.py",
                "--config", $config,
                "--defense", "PGD-AT",
                "--max-samples", "$MaxSamples",
                "--seed", "$s"
            )
        }
    }
}

# --- Summary ----------------------------------------------------------------
$elapsed = [math]::Round(((Get-Date) - $startAll).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 72)
Write-Host ("DONE in {0} min.  failures: {1}  skipped: {2}" -f $elapsed, $failures.Count, $skipped.Count)
if ($skipped.Count)  { $skipped  | ForEach-Object { Write-Host "  SKIPPED: $_" -ForegroundColor Yellow } }
if ($failures.Count) { $failures | ForEach-Object { Write-Host "  FAILED:  $_" -ForegroundColor Red } }
Write-Host ("=" * 72)
if ($failures.Count) { exit 1 }
