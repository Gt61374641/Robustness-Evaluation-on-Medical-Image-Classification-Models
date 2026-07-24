#!/usr/bin/env bash
# Alignment partition 0/3  — est. 18.1 h
#   P0: oct R152 x TRADES x seed{42,43,44}      (heaviest block in the batch)
#   P1: oct R101 PGD-AT x seed{43,44}
#
# Launch:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_align_p0.sh > p0.log 2>&1 &
#   tail -f p0.log
set -uo pipefail
PART=0
source "$(dirname "$0")/run_align_common.sh"
align_setup

# --- P0: three-method comparison, oct ResNet-152 (load-bearing; run first) ---
for S in 42 43 44; do
  run_defense oct2017 resnet152 TRADES "$S"
done

# --- P1: multi-seed backfill for table8 cells that are still n=1 ---
for S in 43 44; do
  run_defense oct2017 resnet101 PGD-AT "$S"
done

align_finish 0
