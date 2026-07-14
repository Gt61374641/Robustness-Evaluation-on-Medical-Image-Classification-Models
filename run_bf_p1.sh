#!/usr/bin/env bash
# Backfill partition 1/3 — chest ResNet-50 line (medium load).
#   chest R50 x {PGD-AT,TRADES,MART} x seed{43,44}  (Task 1)
#   chest R18 PGD-AT-rescue seed42                   (Task 2)
#   sci_defense figures R18/R50/R152 seed42          (Task 4, fast, runs last)
# Run:  nohup bash run_bf_p1.sh > p1.log 2>&1 & ; tail -f p1.log
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
  for DEF in PGD-AT TRADES MART; do run_defense "$CHEST" resnet50 "$DEF" "$S"; done
done
run_defense "$CHEST" resnet18 PGD-AT-rescue 42
# Optional 3rd rescue point:
# run_defense "$CHEST" resnet101 PGD-AT-rescue 42

echo "### Task 4: sci_defense figures (no GPU) ###"
for M in resnet18 resnet50 resnet152; do
  python scripts/generate_defense_sci_figures.py --dataset "$CHEST" --model "$M" --seed seed42 || echo "[fail] sci_defense $M"
done
echo "===== p1 done ====="
