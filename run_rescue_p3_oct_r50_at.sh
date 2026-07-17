#!/usr/bin/env bash
# Terminal 3: add OCT2017 ResNet-50 PGD-AT multi-seed evidence.
set -uo pipefail
LOG="${LOG:-run_rescue_p3_oct_r50_at.log}"
exec > >(tee -a "$LOG") 2>&1

source "$(dirname "$0")/run_rescue_common.sh"
rescue_setup

echo "############### P3: OCT2017 ResNet-50 PGD-AT seed43/44 ###############"
for S in 43 44; do
  run_defense oct2017 resnet50 PGD-AT "$S"
done

echo "===== P3 done: OCT2017 ResNet-50 PGD-AT multi-seed ====="

