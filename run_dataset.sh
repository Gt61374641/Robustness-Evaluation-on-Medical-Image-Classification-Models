#!/usr/bin/env bash
# Standard pipeline (train -> clean -> robustness main+fine) for ALL 5 ResNets on
# one dataset. Linux equivalent of run_dataset.bat. Adversarial training is run
# separately via scripts/evaluate_defense.py.
#
#   Usage:  bash run_dataset.sh malaria
#           bash run_dataset.sh oct2017
#           bash run_dataset.sh chest_xray_pneumonia
#
# Tip: run inside tmux so it survives a dropped SSH connection:
#   tmux new -s run    (detach: Ctrl-b then d ; reattach: tmux attach -t run)
set -uo pipefail   # NOT -e: a single model failing must not abort the rest

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
  echo "Usage: bash run_dataset.sh <dataset>   [ chest_xray_pneumonia | oct2017 | malaria ]"
  exit 1
fi

# timm pretrained weights + (AutoDL) acceleration, in case this is a fresh shell.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true

MAX=1024
MODELS="resnet18 resnet34 resnet50 resnet101 resnet152"

for m in $MODELS; do
  CFG="configs/${DATASET}_${m}.yaml"
  CKPT="checkpoints/${DATASET}_${m}_seed42.pth"
  echo ""
  echo "############### ${DATASET} / ${m} ###############"
  if [ ! -f "$CFG" ]; then
    echo "[skip] missing config: $CFG (run: python scripts/make_configs.py)"
    continue
  fi

  echo "[1/4] train"
  if ! python scripts/train.py --config "$CFG"; then
    echo "[fail] training failed for ${m}; skipping its evaluation."
    continue
  fi
  echo "[2/4] clean eval"
  python scripts/evaluate_clean.py --config "$CFG" --checkpoint "$CKPT"
  echo "[3/4] robustness (main eps grid)"
  python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --max-samples "$MAX"
  echo "[4/4] robustness (fine sub-1/255 probe)"
  python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --attacks-section attacks_fine --max-samples "$MAX"
done

echo ""
echo "===== standard pipeline done for all 5 models: ${DATASET} ====="
