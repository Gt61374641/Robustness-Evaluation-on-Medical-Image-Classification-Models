#!/usr/bin/env bash
# Terminal 1: rescue the collapsed OCT2017 ResNet-152 PGD-AT point.
set -uo pipefail
LOG="${LOG:-run_rescue_p1_oct_r152.log}"
exec > >(tee -a "$LOG") 2>&1

source "$(dirname "$0")/run_rescue_common.sh"
rescue_setup

echo "############### P1: OCT2017 ResNet-152 rescue ###############"
run_defense oct2017 resnet152 PGD-AT-rescue 42

echo "===== P1 done: OCT2017 ResNet-152 rescue ====="

