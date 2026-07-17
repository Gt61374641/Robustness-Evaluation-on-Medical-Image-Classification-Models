#!/usr/bin/env bash
# Shared helpers for AutoDL rescue/backfill partition scripts.
# Source this file from a partition script; do not run it directly.

rescue_setup () {
  PROJ="${PROJ:-$(pwd)}"
  MAX="${MAX:-1024}"
  PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

  echo "############### AutoDL academic acceleration ###############"
  if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "[ok] sourced /etc/network_turbo"
  else
    echo "[warn] /etc/network_turbo not found; continuing without it"
  fi
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-$PIP_MIRROR}"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PYTHONUNBUFFERED=1

  cd "$PROJ" || { echo "[abort] cannot cd to $PROJ"; exit 1; }
  echo "[info] project     : $(pwd)"
  echo "[info] HF_ENDPOINT : $HF_ENDPOINT"
  echo "[info] PIP_INDEX   : $PIP_INDEX_URL"

  echo "############### preflight ###############"
  python -c "import torch, timm, art, sklearn; print('[ok] deps:', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ART', art.__version__)" || exit 1
  python -m py_compile scripts/evaluate_defense.py scripts/generate_decision_boundary_figures.py scripts/extract_figure_data.py || exit 1
  grep -q "PGD-AT-rescue" scripts/evaluate_defense.py || { echo "[abort] evaluate_defense.py missing PGD-AT-rescue"; exit 1; }
  grep -q "grad_clip" scripts/evaluate_defense.py || { echo "[abort] evaluate_defense.py missing grad_clip support"; exit 1; }
  python scripts/make_configs.py || exit 1
}

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

