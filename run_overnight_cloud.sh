#!/usr/bin/env bash
# Overnight unattended DEFENSE phase for the two secondary datasets (malaria +
# oct2017): PGD-AT (resnet18/50/152) + strong evaluation. Run on the cloud
# instance AFTER the standard pipeline for both datasets is done.
#
#   Launch so it survives an SSH disconnect (you can close your laptop):
#     nohup bash run_overnight_cloud.sh > overnight.log 2>&1 &
#   Watch progress:
#     tail -f overnight.log
#
# Notes:
#  - malaria is binary -> AutoAttack auto-skipped (faster).
#  - oct2017 is 4-class -> strong eval includes AutoAttack (slower).
#  - Single seed (42) for these secondary datasets.
#  - Each model saves incrementally; a failure skips that model, not the batch.
set -uo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true

echo "################ OVERNIGHT DEFENSE RUN — START ################"
date

echo ""
echo "===== [1/2] malaria AT (binary; AutoAttack auto-skipped) ====="
bash run_at.sh malaria

echo ""
echo "===== [2/2] oct2017 AT (4-class; includes AutoAttack, slow) ====="
bash run_at.sh oct2017

echo ""
echo "################ OVERNIGHT DEFENSE RUN — DONE ################"
date
echo "Next: pack results (tar czf results.tgz results/) and download them."
