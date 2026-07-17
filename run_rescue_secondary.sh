#!/usr/bin/env bash
# AutoDL secondary rescue/backfill batch.
#
# Goals:
#   1. Rescue the collapsed OCT2017 ResNet-152 PGD-AT point with the stronger
#      PGD-AT-rescue defense block.
#   2. Add multi-seed PGD-AT evidence for the stable malaria/OCT points.
#   3. Regenerate malaria/OCT decision-boundary diagnostics and package the
#      results/figures for local merging.
#
# Run on AutoDL from the repo root:
#   cd /root/autodl-tmp/Robustness-Evaluation-on-Medical-Image-Classification-Models
#   tmux new -s rescue
#   bash run_rescue_secondary.sh 2>&1 | tee run_rescue_secondary.log
# Detach tmux with: Ctrl-b then d
#
# Optional toggles:
#   RUN_TRADES=1              also run malaria/oct TRADES on stable points
#   RUN_MART=1                also run malaria/oct MART on stable points
#   RUN_RESCUE_EXTRA=1        also try PGD-AT-rescue on malaria R18/R34 and OCT R34/R101
#   RUN_NEW_ARCH_CLEAN=1      also generate clean sci diagnostics for chest new architectures
#
# This script is idempotent: if a defense checkpoint exists, it re-evaluates
# rather than retraining; finished JSON files are kept and attack eval resumes.
set -uo pipefail

PROJ="${PROJ:-$(pwd)}"
MAX="${MAX:-1024}"
CHEST="chest_xray_pneumonia"
SEEDS_EXTRA="${SEEDS_EXTRA:-43 44}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "############### 0. AutoDL academic acceleration ###############"
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
  echo "[ok] sourced /etc/network_turbo"
else
  echo "[warn] /etc/network_turbo not found; continuing without AutoDL accelerator"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-$PIP_MIRROR}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
echo "[info] project     : $PROJ"
echo "[info] HF_ENDPOINT : $HF_ENDPOINT"
echo "[info] PIP_INDEX   : $PIP_INDEX_URL"
cd "$PROJ" || { echo "[abort] cannot cd to $PROJ"; exit 1; }

echo "############### 1. code/config preflight ###############"
FAIL=0
python -c "import torch, timm, art, sklearn; print('[ok] deps:', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ART', art.__version__)" || FAIL=1
python -m py_compile scripts/evaluate_defense.py scripts/generate_decision_boundary_figures.py scripts/extract_figure_data.py scripts/generate_main_figures.py scripts/generate_at_ladder_figure.py || FAIL=1
grep -q "PGD-AT-rescue" scripts/evaluate_defense.py || { echo "[x] evaluate_defense.py missing PGD-AT-rescue"; FAIL=1; }
grep -q "grad_clip" scripts/evaluate_defense.py || { echo "[x] evaluate_defense.py missing grad_clip support"; FAIL=1; }
python scripts/make_configs.py || FAIL=1
for cfg in configs/oct2017_resnet152.yaml configs/malaria_resnet50.yaml configs/oct2017_resnet50.yaml; do
  [ -f "$cfg" ] || { echo "[x] missing config: $cfg"; FAIL=1; continue; }
done
grep -q "PGD-AT-rescue" configs/oct2017_resnet152.yaml || { echo "[x] OCT R152 config lacks PGD-AT-rescue"; FAIL=1; }
grep -q "TRADES" configs/malaria_resnet50.yaml || { echo "[x] malaria configs lack TRADES"; FAIL=1; }
for d in data results checkpoints; do
  [ -d "$d" ] && echo "[ok] $d/ present" || echo "[warn] $d/ missing; related steps may train/download/fail"
done
if [ "$FAIL" != 0 ]; then
  echo "[abort] preflight failed. Fix the [x] lines above before running experiments."
  exit 1
fi

run_defense () {
  local DS="$1" M="$2" DEF="$3" S="$4"
  local suffix CKPT CFG
  suffix=$(echo "$DEF" | tr 'A-Z-' 'a-z_')
  CKPT="checkpoints/${DS}_${M}_seed${S}_${suffix}.pth"
  CFG="configs/${DS}_${M}.yaml"
  echo ""
  echo "### ${DEF} ${DS}/${M}/seed${S} ###"
  if [ ! -f "$CFG" ]; then
    echo "[skip] missing config: $CFG"
    return 0
  fi
  if [ -f "$CKPT" ]; then
    echo "[resume] checkpoint exists -> strong re-eval only: $CKPT"
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" --checkpoint "$CKPT" --max-samples "$MAX" --seed "$S" \
      || echo "[fail] eval ${DEF} ${DS}/${M}/seed${S}"
  else
    echo "[train] checkpoint missing -> train + strong eval"
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" --max-samples "$MAX" --seed "$S" \
      || echo "[fail] train/eval ${DEF} ${DS}/${M}/seed${S}"
  fi
}

run_boundary () {
  local DS="$1" SUFFIX="$2"
  shift 2
  echo ""
  echo "### decision boundary ${DS} suffix='${SUFFIX:-standard}' models=$* ###"
  if [ -n "$SUFFIX" ]; then
    python scripts/generate_decision_boundary_figures.py --dataset "$DS" --models "$@" --checkpoint-suffix "$SUFFIX" --max-samples 300 \
      || echo "[fail] decision boundary ${DS} ${SUFFIX}"
  else
    python scripts/generate_decision_boundary_figures.py --dataset "$DS" --models "$@" --eps 0.031373 --max-samples 300 \
      || echo "[fail] decision boundary ${DS} standard"
  fi
}

echo "############### 2. rescue OCT2017 ResNet-152 ###############"
run_defense oct2017 resnet152 PGD-AT-rescue 42

echo "############### 3. malaria/OCT stable-point multi-seed PGD-AT ###############"
for S in $SEEDS_EXTRA; do
  for M in resnet50 resnet101 resnet152; do
    run_defense malaria "$M" PGD-AT "$S"
  done
  run_defense oct2017 resnet50 PGD-AT "$S"
done

if [ "${RUN_TRADES:-0}" = "1" ]; then
  echo "############### 4a. optional TRADES malaria/OCT ###############"
  for S in 42 $SEEDS_EXTRA; do
    for M in resnet50 resnet152; do
      run_defense malaria "$M" TRADES "$S"
    done
    run_defense oct2017 resnet50 TRADES "$S"
  done
fi

if [ "${RUN_MART:-0}" = "1" ]; then
  echo "############### 4b. optional MART malaria/OCT ###############"
  for S in 42 $SEEDS_EXTRA; do
    for M in resnet50 resnet152; do
      run_defense malaria "$M" MART "$S"
    done
    run_defense oct2017 resnet50 MART "$S"
  done
fi

if [ "${RUN_RESCUE_EXTRA:-0}" = "1" ]; then
  echo "############### 4c. optional extra collapse rescue points ###############"
  for M in resnet18 resnet34; do
    run_defense malaria "$M" PGD-AT-rescue 42
  done
  for M in resnet34 resnet101; do
    run_defense oct2017 "$M" PGD-AT-rescue 42
  done
fi

echo "############### 5. malaria/OCT decision-boundary diagnostics ###############"
run_boundary malaria "" resnet18 resnet152
run_boundary oct2017 "" resnet18 resnet152
[ -f checkpoints/malaria_resnet152_seed42_pgd_at.pth ] && run_boundary malaria _pgd_at resnet50 resnet152 || echo "[skip] malaria _pgd_at boundary: missing checkpoint"
[ -f checkpoints/oct2017_resnet50_seed42_pgd_at.pth ] && run_boundary oct2017 _pgd_at resnet18 resnet50 || echo "[skip] OCT _pgd_at boundary: missing checkpoint"

if [ "${RUN_NEW_ARCH_CLEAN:-0}" = "1" ]; then
  echo "############### 6. optional new-architecture clean diagnostics ###############"
  for M in deit_small convnext_tiny; do
    CKPT="checkpoints/${CHEST}_${M}_seed42.pth"
    if [ -f "$CKPT" ]; then
      python scripts/generate_clean_sci_figures.py --config "configs/${CHEST}_${M}.yaml" --checkpoint "$CKPT" \
        || echo "[fail] clean sci ${CHEST}/${M}"
    else
      echo "[skip] clean sci ${M}: missing $CKPT"
    fi
  done
fi

echo "############### 7. regenerate local figure data/tables on AutoDL ###############"
python scripts/extract_figure_data.py || echo "[fail] extract_figure_data"
python scripts/generate_main_figures.py || echo "[fail] generate_main_figures"
python scripts/generate_at_ladder_figure.py || echo "[fail] generate_at_ladder_figure"
python scripts/generate_comparison_tables.py || echo "[fail] generate_comparison_tables"

echo "############### 8. package for local download ###############"
find results -name '*.json' -o -name '*.yaml' > /tmp/rescue_secondary_results.txt
tar czf rescue_secondary_results_json.tgz -T /tmp/rescue_secondary_results.txt
echo "  rescue_secondary_results_json.tgz : $(du -h rescue_secondary_results_json.tgz | cut -f1)"

tar czf rescue_secondary_figures.tgz \
  figures/data figures/main figures/at_ladder figures/paper_tables figures/decision_boundary figures/sci_clean \
  2>/dev/null || echo "[warn] some figure dirs missing while packaging"
[ -f rescue_secondary_figures.tgz ] && echo "  rescue_secondary_figures.tgz      : $(du -h rescue_secondary_figures.tgz | cut -f1)"

LOG_ITEMS=(results/malaria results/oct2017)
[ -f run_rescue_secondary.log ] && LOG_ITEMS+=(run_rescue_secondary.log)
tar --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
  -czf rescue_secondary_logs.tgz "${LOG_ITEMS[@]}" \
  2>/dev/null || echo "[warn] log package skipped/partial"
[ -f rescue_secondary_logs.tgz ] && echo "  rescue_secondary_logs.tgz         : $(du -h rescue_secondary_logs.tgz | cut -f1)"

echo ""
echo "===== rescue secondary batch done ====="
echo "Pull from local machine with rsync, e.g.:"
echo "  rsync -P --append-verify -e ssh <user>@<autodl-host>:${PROJ}/rescue_secondary_results_json.tgz ."
echo "  rsync -P --append-verify -e ssh <user>@<autodl-host>:${PROJ}/rescue_secondary_figures.tgz ."
