#!/usr/bin/env bash
# Adversarial training (PGD-AT) + strong evaluation for the representative models
# (resnet18 / resnet50 / resnet152) on one dataset. Run AFTER the standard pipeline.
#
#   Usage:  bash run_at.sh malaria              # AT, seed42
#           bash run_at.sh malaria "42 43"      # AT, seed42 + seed43
#           bash run_at.sh oct2017              # NOTE: 4-class -> AutoAttack (slow)
#
# Each call: retrain PGD-AT (harmonized loop) then strong-evaluate (PGD-50 + 5
# restarts; AutoAttack too for >=3 classes). Overwrites *_pgd_at.pth; the resume
# guard discards stale attack JSONs automatically.
set -uo pipefail

DATASET="${1:-}"
SEEDS="${2:-42}"
if [ -z "$DATASET" ]; then
  echo "Usage: bash run_at.sh <dataset> [seeds]   e.g. bash run_at.sh malaria \"42 43\""
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
MAX=1024

for m in resnet18 resnet50 resnet152; do
  CFG="configs/${DATASET}_${m}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[skip] missing config: $CFG"
    continue
  fi
  for s in $SEEDS; do
    echo ""
    echo "############### AT  ${DATASET} / ${m} / seed${s} ###############"
    python scripts/evaluate_defense.py --config "$CFG" --defense PGD-AT --max-samples "$MAX" --seed "$s" \
      || echo "[fail] AT failed for ${m} seed${s}; continuing."
  done
done

echo ""
echo "===== AT done: ${DATASET} (models: resnet18/50/152, seeds: ${SEEDS}) ====="
