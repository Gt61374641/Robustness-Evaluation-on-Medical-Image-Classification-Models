#!/usr/bin/env bash
# Optional: wait for all three alignment partitions to finish, then power off so
# the instance stops billing overnight. Start it AFTER the three partitions.
#
#   nohup bash run_align_shutdown.sh > shutdown_watch.log 2>&1 &
#
# Cancel at any time with:  pkill -f run_align_shutdown.sh
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "[$(date '+%F %T')] waiting for .align_done_p{0,1,2}"
while true; do
  n=0
  for p in 0 1 2; do [ -f ".align_done_p${p}" ] && n=$((n + 1)); done
  if [ "$n" -eq 3 ]; then break; fi
  sleep 300
done

echo "[$(date '+%F %T')] all three partitions finished"
python status_align.py || true

echo "[$(date '+%F %T')] powering off in 5 minutes — pkill -f run_align_shutdown.sh to cancel"
sleep 300
shutdown -h now
