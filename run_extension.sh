#!/usr/bin/env bash
# Extension batch (2026-07-10 plan), to run on the AutoDL GPU instance:
#
#   0. Regenerate configs (17: +deit_small/+convnext_tiny for chest)
#   1. chest NEW-ARCH standard training (ViT-S/DeiT-S + ConvNeXt-T, 3 seeds)
#      -> architecture-vs-robustness comparison against the ResNet ladder
#   2. chest attack-method comparison: attacks_extra (CW/DeepFool/AutoAttack/Square)
#      on all 7 standard models (5 ResNets + 2 new archs), seed42
#   3. PGD-AT completion: R34/R101 on ALL THREE datasets (full 5-model AT ladder)
#   4. chest NEW-ARCH PGD-AT (deit_small + convnext_tiny, seed42)
#   5. chest MART (third AT method) on R18/R50/R152, seed42
#   6. Grad-CAM for convnext_tiny (deit skipped: conv-hook Grad-CAM does not
#      support ViT token maps -- documented limitation)
#   7. Transfer-lean packaging (same as run_extras.sh)
#
# The UK<->China SSH link WILL drop; run under tmux or nohup:
#   nohup bash run_extension.sh > run_extension.log 2>&1 &
#   tail -f run_extension.log
#
# Rough GPU-time guide (RTX 4090-class): section 1 ~2-3 h; section 2 is the
# slow one (CW 1000 grad steps + AA + Square x 7 models, ~6-12 h); section 3
# chest ~2 h, malaria ~4 h, oct ~12-16 h (83k train set); sections 4-5 ~4 h.
# Every step resumes: finished attacks/checkpoints are skipped on rerun.
#
# set -uo pipefail (NOT -e): a single model failing must not abort the rest.
set -uo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
MAX=1024
CHEST=chest_xray_pneumonia

########################################################################
# 0. Regenerate configs (adds chest deit_small/convnext_tiny; byte-identical
#    train/attack blocks across models within a dataset).
########################################################################
echo "############### 0. make_configs + preflight ###############"
python scripts/make_configs.py
# Fail fast if the installed ART cannot take a custom AutoAttack ensemble
# (needed for the binary-safe AutoAttack in attacks_extra).
python - <<'EOF'
import inspect
from art.attacks.evasion import AutoAttack
assert "attacks" in inspect.signature(AutoAttack.__init__).parameters, \
    "ART AutoAttack lacks the 'attacks' kwarg -- upgrade ART (>=1.20)"
print("preflight OK: ART AutoAttack supports custom ensembles")
EOF

########################################################################
# 1. chest NEW-ARCH standard training, 3 seeds (matches the ResNet ladder:
#    train -> clean -> robustness main + fine, --max-samples 1024 for the
#    same result-file naming the figure scripts expect).
########################################################################
echo "############### 1. chest new-arch standard (3 seeds) ###############"
for m in deit_small convnext_tiny; do
  for S in 42 43 44; do
    CFG="configs/${CHEST}_${m}.yaml"
    CKPT="checkpoints/${CHEST}_${m}_seed${S}.pth"
    echo "### STD ${CHEST}/${m}/seed${S} ###"
    # Idempotent: an existing checkpoint is NOT retrained (train.py would
    # overwrite it and invalidate finished attack results via the mtime check).
    if [ -f "$CKPT" ]; then
      echo "  checkpoint exists -> skip training, top up evals only"
    elif ! python scripts/train.py --config "$CFG" --seed "$S"; then
      echo "[fail] train ${CHEST}/${m}/seed${S}; skipping its eval."; continue
    fi
    python scripts/evaluate_clean.py      --config "$CFG" --checkpoint "$CKPT" --seed "$S"
    python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --max-samples $MAX --seed "$S"
    python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --attacks-section attacks_fine --max-samples $MAX --seed "$S"
  done
done

########################################################################
# 2. chest attack-method comparison (attacks_extra) on all 7 standard models,
#    seed42. CW/DeepFool are minimal-perturbation (report their perturbation
#    L2/Linf, not just robust acc); AutoAttack uses the binary-safe reduced
#    ensemble (APGD-CE + DeepFool + Square, no DLR); Square is black-box.
#    SLOWEST section -- safe to split across nights (results resume per attack).
########################################################################
echo "############### 2. chest attacks_extra (7 models) ###############"
for m in resnet18 resnet34 resnet50 resnet101 resnet152 deit_small convnext_tiny; do
  CKPT="checkpoints/${CHEST}_${m}_seed42.pth"
  [ -f "$CKPT" ] || { echo "[skip] ${m}: no seed42 checkpoint"; continue; }
  python scripts/evaluate_robustness.py --config "configs/${CHEST}_${m}.yaml" \
    --checkpoint "$CKPT" --attacks-section attacks_extra --max-samples $MAX --seed 42 \
    || echo "[fail] attacks_extra ${CHEST}/${m}"
done

########################################################################
# 3. PGD-AT completion: R34/R101 on all three datasets -> full 5-model AT
#    ladder (H2 becomes a real complexity curve, not 3 points). oct is by far
#    the most expensive (83k train images x PGD-7); it runs last.
########################################################################
echo "############### 3. PGD-AT completion R34/R101 (3 datasets) ###############"
for DS in $CHEST malaria oct2017; do
  for m in resnet34 resnet101; do
    echo "### PGD-AT ${DS}/${m}/seed42 ###"
    # Idempotent: if the AT checkpoint exists, re-evaluate only (resumes any
    # missing attacks) instead of retraining from scratch.
    ATCKPT="checkpoints/${DS}_${m}_seed42_pgd_at.pth"
    if [ -f "$ATCKPT" ]; then
      python scripts/evaluate_defense.py --config "configs/${DS}_${m}.yaml" \
        --defense PGD-AT --checkpoint "$ATCKPT" --max-samples $MAX --seed 42 \
        || echo "[fail] PGD-AT eval ${DS}/${m}"
    else
      python scripts/evaluate_defense.py --config "configs/${DS}_${m}.yaml" \
        --defense PGD-AT --max-samples $MAX --seed 42 \
        || echo "[fail] PGD-AT ${DS}/${m}"
    fi
  done
done

########################################################################
# 4. chest NEW-ARCH PGD-AT (seed42): does the "complexity helps under AT"
#    finding hold across architectures?
########################################################################
echo "############### 4. chest new-arch PGD-AT ###############"
for m in deit_small convnext_tiny; do
  ATCKPT="checkpoints/${CHEST}_${m}_seed42_pgd_at.pth"
  if [ -f "$ATCKPT" ]; then
    python scripts/evaluate_defense.py --config "configs/${CHEST}_${m}.yaml" \
      --defense PGD-AT --checkpoint "$ATCKPT" --max-samples $MAX --seed 42 \
      || echo "[fail] PGD-AT eval ${CHEST}/${m}"
  else
    python scripts/evaluate_defense.py --config "configs/${CHEST}_${m}.yaml" \
      --defense PGD-AT --max-samples $MAX --seed 42 \
      || echo "[fail] PGD-AT ${CHEST}/${m}"
  fi
done

########################################################################
# 5. chest MART on R18/R50/R152 (matches the TRADES trio) -> three-way AT
#    method comparison: PGD-AT vs TRADES vs MART. MART shares the custom loop
#    (same warmup stabilisers as PGD-AT), unlike TRADES (ART trainer, no warmup).
########################################################################
echo "############### 5. chest MART (R18/R50/R152) ###############"
for m in resnet18 resnet50 resnet152; do
  ATCKPT="checkpoints/${CHEST}_${m}_seed42_mart.pth"
  if [ -f "$ATCKPT" ]; then
    python scripts/evaluate_defense.py --config "configs/${CHEST}_${m}.yaml" \
      --defense MART --checkpoint "$ATCKPT" --max-samples $MAX --seed 42 \
      || echo "[fail] MART eval ${CHEST}/${m}"
  else
    python scripts/evaluate_defense.py --config "configs/${CHEST}_${m}.yaml" \
      --defense MART --max-samples $MAX --seed 42 \
      || echo "[fail] MART ${CHEST}/${m}"
  fi
done

########################################################################
# 6. Grad-CAM for convnext_tiny (standard + AT if it converged). deit_small is
#    SKIPPED: auto_target_layer picks the last Conv2d, which for ViT is the
#    patch embedding -- meaningless CAM (documented limitation; attention
#    rollout would be the proper tool).
########################################################################
echo "############### 6. Grad-CAM convnext_tiny ###############"
python scripts/generate_gradcam_figures.py \
  --config "configs/${CHEST}_convnext_tiny.yaml" \
  --checkpoint "checkpoints/${CHEST}_convnext_tiny_seed42.pth" \
  --attack PGD --eps 0.031373 --num-samples 8 \
  || echo "[fail] gradcam convnext_tiny"
if [ -f "checkpoints/${CHEST}_convnext_tiny_seed42_pgd_at.pth" ]; then
  python scripts/generate_gradcam_figures.py \
    --config "configs/${CHEST}_convnext_tiny.yaml" \
    --checkpoint "checkpoints/${CHEST}_convnext_tiny_seed42_pgd_at.pth" \
    --attack PGD --eps 0.031373 --num-samples 8 \
    --out-dir "figures/gradcam/${CHEST}/convnext_tiny_at" \
    || echo "[fail] gradcam convnext_tiny_at"
fi

########################################################################
# 7. Package for local download (transfer-lean; see run_extras.sh notes).
########################################################################
echo "############### 7. packaging (transfer-lean) ###############"
find results -name '*.json' -o -name '*.yaml' > /tmp/flist.txt
tar czf results_json.tgz -T /tmp/flist.txt
echo "  results_json.tgz : $(du -h results_json.tgz | cut -f1)  <-- essential, pull this first"
find figures/gradcam -name '*.png' > /tmp/glist.txt
tar czf gradcam_png.tgz -T /tmp/glist.txt
echo "  gradcam_png.tgz  : $(du -h gradcam_png.tgz | cut -f1)  <-- optional"
echo ""
echo "===== extension done ====="
echo "Pull to the local repo (rsync resumes a dropped cross-border transfer):"
echo "  rsync -P --append-verify -e ssh <user>@<autodl-host>:.../results_json.tgz ."
echo "then locally:  tar xzf results_json.tgz"
