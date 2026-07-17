#!/usr/bin/env bash
# Optional terminal: run TRADES/MART on the stable malaria/OCT points.
# This is compute-heavy. Skip it if you only want the core rescue/backfill.
set -uo pipefail
LOG="${LOG:-run_rescue_p4_optional_trades_mart.log}"
exec > >(tee -a "$LOG") 2>&1

source "$(dirname "$0")/run_rescue_common.sh"
rescue_setup

echo "############### P4 optional: malaria TRADES/MART ###############"
for S in 42 43 44; do
  for M in resnet50 resnet152; do
    run_defense malaria "$M" TRADES "$S"
    run_defense malaria "$M" MART "$S"
  done
done

echo "############### P4 optional: OCT2017 ResNet-50 TRADES/MART ###############"
for S in 42 43 44; do
  run_defense oct2017 resnet50 TRADES "$S"
  run_defense oct2017 resnet50 MART "$S"
done

echo "===== P4 done: optional TRADES/MART ====="

