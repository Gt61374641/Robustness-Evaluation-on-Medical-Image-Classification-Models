#!/usr/bin/env bash
# Final terminal: regenerate diagnostics/figures/tables and package cloud results.
# Run this only after P1/P2/P3 have finished.
set -uo pipefail
LOG="${LOG:-run_rescue_pack_results.log}"
exec > >(tee -a "$LOG") 2>&1

source "$(dirname "$0")/run_rescue_common.sh"
rescue_setup

echo "############### decision-boundary diagnostics ###############"
python scripts/generate_decision_boundary_figures.py --dataset malaria --models resnet18 resnet152 --eps 0.031373 --max-samples 300 \
  || echo "[fail] malaria standard decision boundary"
python scripts/generate_decision_boundary_figures.py --dataset oct2017 --models resnet18 resnet152 --eps 0.031373 --max-samples 300 \
  || echo "[fail] OCT standard decision boundary"
[ -f checkpoints/malaria_resnet152_seed42_pgd_at.pth ] && \
  python scripts/generate_decision_boundary_figures.py --dataset malaria --models resnet50 resnet152 --checkpoint-suffix _pgd_at --max-samples 300 \
  || echo "[skip/fail] malaria _pgd_at decision boundary"
[ -f checkpoints/oct2017_resnet50_seed42_pgd_at.pth ] && \
  python scripts/generate_decision_boundary_figures.py --dataset oct2017 --models resnet18 resnet50 --checkpoint-suffix _pgd_at --max-samples 300 \
  || echo "[skip/fail] OCT _pgd_at decision boundary"

echo "############### optional new-architecture clean diagnostics ###############"
CHEST="chest_xray_pneumonia"
for M in deit_small convnext_tiny; do
  CKPT="checkpoints/${CHEST}_${M}_seed42.pth"
  if [ -f "$CKPT" ]; then
    python scripts/generate_clean_sci_figures.py --config "configs/${CHEST}_${M}.yaml" --checkpoint "$CKPT" \
      || echo "[fail] clean sci ${CHEST}/${M}"
  else
    echo "[skip] clean sci ${M}: missing $CKPT"
  fi
done

echo "############### regenerate figures/tables ###############"
python scripts/extract_figure_data.py || echo "[fail] extract_figure_data"
python scripts/generate_main_figures.py || echo "[fail] generate_main_figures"
python scripts/generate_at_ladder_figure.py || echo "[fail] generate_at_ladder_figure"
python scripts/generate_comparison_tables.py || echo "[fail] generate_comparison_tables"

echo "############### package for local download ###############"
find results -name '*.json' -o -name '*.yaml' > /tmp/rescue_parallel_results.txt
tar czf rescue_parallel_results_json.tgz -T /tmp/rescue_parallel_results.txt
echo "  rescue_parallel_results_json.tgz : $(du -h rescue_parallel_results_json.tgz | cut -f1)"

FIGURE_ITEMS=()
for d in figures/data figures/main figures/at_ladder figures/paper_tables figures/decision_boundary figures/sci_clean; do
  [ -e "$d" ] && FIGURE_ITEMS+=("$d")
done
if [ "${#FIGURE_ITEMS[@]}" -gt 0 ]; then
  tar czf rescue_parallel_figures.tgz "${FIGURE_ITEMS[@]}" \
    2>/dev/null || echo "[warn] some figure dirs failed while packaging"
else
  echo "[warn] no figure dirs found to package"
fi
[ -f rescue_parallel_figures.tgz ] && echo "  rescue_parallel_figures.tgz      : $(du -h rescue_parallel_figures.tgz | cut -f1)"

LOG_ITEMS=()
for f in *.log; do
  [ -f "$f" ] && LOG_ITEMS+=("$f")
done
for d in results/malaria results/oct2017; do
  [ -e "$d" ] && LOG_ITEMS+=("$d")
done
if [ "${#LOG_ITEMS[@]}" -gt 0 ]; then
  tar --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
    -czf rescue_parallel_logs.tgz "${LOG_ITEMS[@]}" \
    2>/dev/null || echo "[warn] log package skipped/partial"
else
  echo "[warn] no logs/results found to package"
fi
[ -f rescue_parallel_logs.tgz ] && echo "  rescue_parallel_logs.tgz         : $(du -h rescue_parallel_logs.tgz | cut -f1)"

echo "===== packaging done ====="
