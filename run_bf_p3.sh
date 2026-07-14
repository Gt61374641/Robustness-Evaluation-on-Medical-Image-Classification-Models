#!/usr/bin/env bash
# Backfill partition 3/3 — cross-dataset TRADES + oct heavy + decision boundaries.
#   malaria R50/R152 TRADES seed42; oct R50 TRADES seed42   (Task 3a)
#   oct R152 PGD-AT-rescue seed42                            (Task 2, slowest: 83k train)
#   decision-boundary figures for malaria + oct             (Task 3b)
# The oct runs are the long pole -> this terminal isolates them.
# Run:  nohup bash run_bf_p3.sh > p3.log 2>&1 & ; tail -f p3.log
set -uo pipefail
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX=1024
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

# Task 3a — cross-dataset TRADES ablation (seed42)
run_defense malaria resnet50  TRADES 42
run_defense malaria resnet152 TRADES 42
run_defense oct2017 resnet50  TRADES 42
# Task 2 — oct R152 rescue (long pole)
run_defense oct2017 resnet152 PGD-AT-rescue 42

# Task 3b — decision boundaries (standard R18-vs-R152; AT where a checkpoint exists)
for DS in malaria oct2017; do
  python scripts/generate_decision_boundary_figures.py --dataset "$DS" --models resnet18 resnet152 --eps 0.031373 --max-samples 300 || echo "[fail] boundary std $DS"
done
[ -f checkpoints/malaria_resnet152_seed42_pgd_at.pth ] && python scripts/generate_decision_boundary_figures.py --dataset malaria --models resnet50 resnet152 --checkpoint-suffix _pgd_at --max-samples 300 || echo "[skip] boundary malaria _pgd_at"
[ -f checkpoints/oct2017_resnet50_seed42_pgd_at.pth ]  && python scripts/generate_decision_boundary_figures.py --dataset oct2017 --models resnet18 resnet50  --checkpoint-suffix _pgd_at --max-samples 300 || echo "[skip] boundary oct _pgd_at"
echo "===== p3 done ====="
