#!/usr/bin/env bash
# Shared helpers for the three-dataset alignment batch (run_align_p{0,1,2}.sh).
# Source this from a partition script; do not run it directly.
#
# Scope of the batch:
#   P0  oct R18/R152 x {TRADES,MART} x seed{42,43,44}   -> three-method comparison
#                                                          reaches all 3 datasets
#   P1  PGD-AT seed43/44 for every n=1 cell in table8   -> collapse verdicts stop
#                                                          resting on a single seed
# OCT AutoAttack stays OFF (configs/oct2017_base.yaml): its SquareAttack component
# costs 6-8 h per model. The gradient-masking check runs separately afterwards via
# attacks_extended (Square @1000 queries).

align_setup () {
  PROJ="${PROJ:-$(pwd)}"
  MAX="${MAX:-1024}"
  PART="${PART:-?}"

  echo "############### AutoDL academic acceleration ###############"
  if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo && echo "[ok] sourced /etc/network_turbo"
  else
    echo "[warn] /etc/network_turbo not found; continuing without it"
  fi
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PYTHONUNBUFFERED=1

  cd "$PROJ" || { echo "[abort] cannot cd to $PROJ"; exit 1; }
  echo "[info] partition : p${PART}"
  echo "[info] project   : $(pwd)"
  echo "[info] GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset!>}"
  echo "[info] commit    : $(git log --oneline -1 2>/dev/null || echo n/a)"

  echo "############### preflight ###############"
  # 1. deps
  python -c "import torch, timm, art, sklearn; print('[ok] deps: torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ART', art.__version__)" || exit 1

  # 2. exactly one GPU must be bound, or all three partitions land on card 0
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[abort] CUDA_VISIBLE_DEVICES is unset. Launch as:"
    echo "        CUDA_VISIBLE_DEVICES=<n> nohup bash run_align_p${PART}.sh > p${PART}.log 2>&1 &"
    exit 1
  fi

  # 3. regenerate per-model configs, then assert OCT AutoAttack is really off.
  #    If it is on, every OCT defense eval gains a 6-8 h tail and this batch
  #    silently turns into a multi-day run.
  python scripts/make_configs.py || exit 1
  for f in configs/oct2017_resnet18.yaml configs/oct2017_resnet152.yaml; do
    if [ "$(grep -c '^  autoattack:' "$f")" != "0" ]; then
      echo "[abort] AutoAttack is ENABLED in $f — comment out the autoattack block"
      echo "        in configs/oct2017_base.yaml and rerun scripts/make_configs.py"
      exit 1
    fi
  done
  echo "[ok] OCT AutoAttack disabled"

  # 4. training-side code present
  grep -q "grad_clip" scripts/evaluate_defense.py || { echo "[abort] evaluate_defense.py missing grad_clip"; exit 1; }

  # 5. ImageNet weights must be cached: AT starts from pretrained=True, so an
  #    uncached model + flaky network kills the run at minute one.
  python -c "
from src.models import create_model
import sys
bad = []
for m in ['resnet18', 'resnet34', 'resnet101', 'resnet152']:
    try:
        create_model(m, 4, pretrained=True)
        print('[ok] pretrained weights:', m)
    except Exception as e:
        bad.append(m); print('[FAIL]', m, type(e).__name__, str(e)[:120])
sys.exit(1 if bad else 0)
" || { echo "[abort] pretrained weights unavailable (see above)"; exit 1; }

  # 6. datasets present
  for d in data/oct2017 data/malaria data/chest_xray_pneumonia; do
    [ -d "$d" ] || { echo "[abort] missing dataset dir: $d"; exit 1; }
  done
  echo "[ok] preflight passed"
  echo ""
}

# run_defense DATASET MODEL DEFENSE SEED
# Resumes: an existing checkpoint means training is skipped and only the strong
# evaluation reruns. A failure is logged and the partition continues.
run_defense () {
  local DS="$1" M="$2" DEF="$3" S="$4"
  local suffix CKPT CFG
  suffix=$(echo "$DEF" | tr 'A-Z-' 'a-z_')
  CKPT="checkpoints/${DS}_${M}_seed${S}_${suffix}.pth"
  CFG="configs/${DS}_${M}.yaml"

  echo ""
  echo "### [$(date '+%F %T')] ${DEF} ${DS}/${M}/seed${S} ###"
  if [ ! -f "$CFG" ]; then
    echo "[skip] missing config: $CFG"
    return 0
  fi

  if [ -f "$CKPT" ]; then
    echo "[resume] checkpoint exists -> strong re-eval only: $CKPT"
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" \
      --checkpoint "$CKPT" --max-samples "$MAX" --seed "$S" \
      || echo "[fail] eval ${DEF} ${DS}/${M}/seed${S}"
  else
    echo "[train] checkpoint missing -> train + strong eval"
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" \
      --max-samples "$MAX" --seed "$S" \
      || echo "[fail] train/eval ${DEF} ${DS}/${M}/seed${S}"
  fi
}

align_finish () {
  local P="$1"
  touch ".align_done_p${P}"
  echo ""
  echo "===== [$(date '+%F %T')] partition p${P} done ====="
  grep -c "^\[fail\]" "p${P}.log" 2>/dev/null | xargs -I{} echo "[summary] {} failed task(s) in p${P}.log"
}
