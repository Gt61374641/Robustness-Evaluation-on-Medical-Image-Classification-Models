#!/usr/bin/env bash
# Terminal 2: add malaria PGD-AT multi-seed evidence for stable points.
set -uo pipefail
LOG="${LOG:-run_rescue_p2_malaria_at.log}"
exec > >(tee -a "$LOG") 2>&1

source "$(dirname "$0")/run_rescue_common.sh"
rescue_setup

echo "############### P2: malaria PGD-AT seed43/44 ###############"
for S in 43 44; do
  for M in resnet50 resnet101 resnet152; do
    run_defense malaria "$M" PGD-AT "$S"
  done
done

echo "===== P2 done: malaria PGD-AT multi-seed ====="

