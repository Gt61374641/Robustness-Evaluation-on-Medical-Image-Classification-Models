#!/usr/bin/env bash
# Backfill partition 2/3 — chest ResNet-152 line + malaria multi-seed (slow model).
#   chest R152 x {PGD-AT,TRADES,MART} x seed{43,44}  (Task 1)
#   malaria R50/R152 PGD-AT x seed{43,44}            (Task 1)
# Run:  nohup bash run_bf_p2.sh > p2.log 2>&1 & ; tail -f p2.log
set -uo pipefail
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX=1024; CHEST=chest_xray_pneumonia
python scripts/make_configs.py

run_defense () {  # DS MODEL DEFENSE SEED
  local DS="$1" M="$2" DEF="$3" S="$4"
  local suffix; suffix=$(echo "$DEF" | tr 'A-Z-' 'a-z_')
  local CKPT="checkpoints/${DS}_${M}_seed${S}_${suffix}.pth"
  local CFG="configs/${DS}_${M}.yaml"
  echo "### ${DEF} ${DS}/${M}/seed${S} ###"
  if [ -f "$CKPT" ]; then
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" --checkpoint "$CKPT" --max-samples $MAX --seed "$S" || echo "[fail] $DEF eval $DS/$M/$S"
  else
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" --max-samples $MAX --seed "$S" || echo "[fail] $DEF $DS/$M/$S"
  fi
}

for S in 43 44; do
  for DEF in PGD-AT TRADES MART; do run_defense "$CHEST" resnet152 "$DEF" "$S"; done
  for M in resnet50 resnet152; do run_defense malaria "$M" PGD-AT "$S"; done
done
echo "===== p2 done ====="
