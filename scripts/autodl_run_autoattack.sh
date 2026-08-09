#!/usr/bin/env bash
# AutoDL runner: AutoAttack audit of the Chest X-ray defended models.
#
# Runs the 16 Chest X-ray adversarial-training checkpoints (PGD-AT / TRADES / MART)
# through AutoAttack at 8/255, to check whether the PGD-50 robust accuracies
# reported in Section 4.5 survive a stronger attack.
#
#   bash scripts/autodl_run_autoattack.sh preflight   # no GPU, no deps needed
#   bash scripts/autodl_run_autoattack.sh setup       # mirrors + pip install
#   bash scripts/autodl_run_autoattack.sh calibrate   # measure this GPU, then stop
#   bash scripts/autodl_run_autoattack.sh run         # full 16-cell audit (nohup)
#   bash scripts/autodl_run_autoattack.sh all         # setup -> calibrate -> run
#
# Nothing here touches the published defense_results*.json that Table 4 is built
# from; the audit writes to its own autoattack_audit_* namespace.
set -euo pipefail

# Derive the project root from this script's own location (it lives in <proj>/scripts),
# so the repo can sit under any directory name. Override with PROJ=... if needed.
# On AutoDL keep the project on /root/autodl-tmp: that disk survives SHUTDOWN but
# NOT RELEASE, and the system disk is small.
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SAMPLES="${SAMPLES:-256}"
PYBIN="${PYBIN:-python}"
LOGDIR="$PROJ/logs"
STAGE="${1:-all}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cd "$PROJ" 2>/dev/null || die "Project not found at $PROJ (set PROJ=... to override)"
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- preflight ---
# Deliberately dependency-free: run this before installing anything, and before
# uploading gigabytes, to confirm the checkpoints and data are actually present.
preflight() {
  say "Preflight: $PROJ"
  echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

  local n_ck n_data n_all
  n_ck=$(ls checkpoints/chest_xray_pneumonia_*_{pgd_at,trades,mart}.pth 2>/dev/null | wc -l)
  n_all=$(ls checkpoints/*_{pgd_at,trades,mart}.pth 2>/dev/null | wc -l)
  # No hardcoded expectation: what is on disk here is the ground truth, and a
  # local working copy may hold only a subset of it.
  echo "Chest AT checkpoints present: $n_ck   (all datasets: $n_all)"
  [ "$n_ck" -eq 0 ] && die "No checkpoints. Upload them before continuing (see README note below)."

  n_data=$(find data/chest_xray_pneumonia -type f 2>/dev/null | wc -l)
  echo "Chest data files present:     $n_data"
  [ "$n_data" -eq 0 ] && die "data/chest_xray_pneumonia is empty. Upload it before continuing."

  echo "Free space on data disk:"; df -h "$PROJ" | tail -1

  say "Plan (no GPU work)"
  $PYBIN scripts/run_autoattack_audit.py --dry-run \
      --datasets chest_xray_pneumonia --max-samples "$SAMPLES"
}

# -------------------------------------------------------------------- setup ---
setup() {
  say "Network + mirrors"
  # AutoDL's academic accelerator: only helps international endpoints (GitHub,
  # HuggingFace). Harmless if absent, so guard it rather than assume it.
  if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo && echo "  academic acceleration: ON"
  else
    echo "  /etc/network_turbo not found, skipping (fine on non-AutoDL hosts)"
  fi
  # A domestic PyPI mirror is faster than the accelerator for pip itself.
  $PYBIN -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null
  $PYBIN -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn >/dev/null
  echo "  pip index: $($PYBIN -m pip config get global.index-url)"

  say "Dependencies"
  # torch/torchvision come from the AutoDL base image. Reinstalling them from a
  # mirror can pull a CUDA build that mismatches the driver, so they are excluded
  # here even though requirements.txt lower-bounds them.
  grep -vE '^(torch|torchvision)\b' requirements.txt > /tmp/req_no_torch.txt
  $PYBIN -m pip install -q -r /tmp/req_no_torch.txt
  $PYBIN - <<'PY'
import torch, timm, art
print(f"  torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"| timm {timm.__version__} | ART {art.__version__}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY
}

# ---------------------------------------------------------------- calibrate ---
# Cost scales with the SURVIVING sample fraction, so calibrate on a converged
# cell (TRADES converged 3/3 everywhere). Calibrating on a collapsed PGD-AT cell
# would badly underestimate the full run.
calibrate() {
  say "Calibrating on a converged TRADES cell (32 samples)"
  $PYBIN scripts/run_autoattack_audit.py --calibrate 32 \
      --datasets chest_xray_pneumonia --methods TRADES --max-samples "$SAMPLES" \
      2>&1 | tee "$LOGDIR/calibrate.log"
}

# --------------------------------------------------------------------- run ---
run() {
  local ts log
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOGDIR/autoattack_chest_${ts}.log"
  # Never hardcode the cell count here: the checkpoints present on this machine
  # are the ground truth, and they may differ from any local working copy. The
  # Python script prints the real count as its first line of output.
  say "Full audit: all Chest X-ray cells on disk x $SAMPLES samples -> $log"
  # nohup so the run survives an SSH drop; AutoDL sessions disconnect routinely.
  nohup $PYBIN -u scripts/run_autoattack_audit.py \
      --datasets chest_xray_pneumonia --max-samples "$SAMPLES" > "$log" 2>&1 &
  echo "  PID $! started"
  cat <<EOF

  Monitor:   tail -f $log
  Progress:  grep -E '^\[[0-9]+/[0-9]+\]' $log | tail -1
             # matches only the [i/N] cell markers. A bare '^\[' would also count
             # the logger's [timestamp] lines and overstate progress.
  Verdict:   grep -c 'binds: True' $log   # cells where AutoAttack beat PGD-50
  Results:   reports/thesis_evidence/autoattack_audit.csv
  Stop:      kill $!

  When it finishes, pull ONLY the results back (they are small):
    rsync -avz -e 'ssh -p <PORT>' \\
      root@<HOST>:$PROJ/reports/thesis_evidence/autoattack_audit.csv ./reports/thesis_evidence/
    rsync -avz -e 'ssh -p <PORT>' --include='*/' --include='autoattack_audit_n*.json' \\
      --exclude='*' root@<HOST>:$PROJ/results/ ./results/
EOF
}

case "$STAGE" in
  preflight) preflight ;;
  setup)     setup ;;
  calibrate) calibrate ;;
  run)       run ;;
  all)       setup; calibrate; run ;;
  *)         die "Unknown stage '$STAGE' (preflight|setup|calibrate|run|all)" ;;
esac
