#!/usr/bin/env bash
# Alignment partition 2/3  — est. 17.9 h
#   P0: oct R18 x {TRADES,MART} x seed{42,43,44}
#       oct R18 collapses under PGD-AT, so this is the third dataset for the
#       "TRADES/MART rescue a PGD-AT collapse" claim (chest N=1, malaria N=2).
#   P1: oct     R18  PGD-AT x seed{43,44}
#   P1: oct     R34  PGD-AT x seed{43,44}
#   P1: malaria R18  PGD-AT x seed{43,44}
#   P1: chest   R101 PGD-AT x seed{43,44}
#   P1: chest   R34  PGD-AT x seed{43,44}   (moved off p0 to rebalance)
#
# Launch:
#   CUDA_VISIBLE_DEVICES=2 nohup bash run_align_p2.sh > p2.log 2>&1 &
#   tail -f p2.log
set -uo pipefail
PART=2
source "$(dirname "$0")/run_align_common.sh"
align_setup

# --- P0: three-method comparison, oct ResNet-18 (load-bearing; run first) ---
for S in 42 43 44; do
  run_defense oct2017 resnet18 TRADES "$S"
done
for S in 42 43 44; do
  run_defense oct2017 resnet18 MART "$S"
done

# --- P1: multi-seed backfill ---
for S in 43 44; do
  run_defense oct2017 resnet18 PGD-AT "$S"
done
for S in 43 44; do
  run_defense oct2017 resnet34 PGD-AT "$S"
done
for S in 43 44; do
  run_defense malaria resnet18 PGD-AT "$S"
done
for S in 43 44; do
  run_defense chest_xray_pneumonia resnet101 PGD-AT "$S"
done
for S in 43 44; do
  run_defense chest_xray_pneumonia resnet34 PGD-AT "$S"
done

align_finish 2
