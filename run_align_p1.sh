#!/usr/bin/env bash
# Alignment partition 1/3  — est. 16.8 h
#   P0: oct R152 x MART x seed{42,43,44}
#   P1: oct     R152 PGD-AT x seed{43,44}   (its collapse verdict is currently n=1)
#   P1: malaria R34  PGD-AT x seed{43,44}
#   P1: chest   R18  PGD-AT x seed44        (only cell at n=2)
#
# Launch:
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_align_p1.sh > p1.log 2>&1 &
#   tail -f p1.log
set -uo pipefail
PART=1
source "$(dirname "$0")/run_align_common.sh"
align_setup

# --- P0: three-method comparison, oct ResNet-152 (load-bearing; run first) ---
for S in 42 43 44; do
  run_defense oct2017 resnet152 MART "$S"
done

# --- P1: multi-seed backfill ---
# NOTE: oct R152 PGD-AT seed42 collapsed under the original protocol and was
# recovered only by the separate rescue protocol. seed43/44 here stay on the
# ORIGINAL protocol on purpose — they test whether that collapse was seed luck.
# Do not substitute PGD-AT-rescue: it is a different protocol and must not enter
# the table8 ladder.
for S in 43 44; do
  run_defense oct2017 resnet152 PGD-AT "$S"
done
for S in 43 44; do
  run_defense malaria resnet34 PGD-AT "$S"
done
run_defense chest_xray_pneumonia resnet18 PGD-AT 44

align_finish 1
