#!/usr/bin/env bash
# One-shot "extras" for the dissertation, to run on the AutoDL GPU instance.
# Covers four independent tasks; each section can be run on its own (comment the
# others out) or all in sequence. RUN INSIDE tmux so it survives SSH drops:
#   tmux new -s extras     (detach: Ctrl-b then d ; reattach: tmux attach -t extras)
#
#   0. chest R50 warmup AT rerun  (protocol consistency with R18/R152)
#   1. Multi-seed STANDARD training for malaria + oct  (adds error bars to H1)
#   2. Grad-CAM for malaria + oct  (standard + AT models)
#   3. TRADES on chest R18/R50/R152  (second-method ablation for H2)
#
# set -uo pipefail (NOT -e): a single model failing must not abort the rest.
set -uo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
MAX=1024

########################################################################
# 0. chest R50 warmup AT rerun (overwrites the old no-warmup _pgd_at.pth).
#    No --checkpoint => retrains from ImageNet with warmup (config already has
#    eps_warmup=5/lr_warmup=3). chest is binary => AutoAttack auto-skips.
########################################################################
echo "############### 0. chest R50 warmup AT rerun ###############"
# back up the old no-warmup result first (in case you want to compare)
cp results/chest_xray_pneumonia/resnet50/defense_PGD-AT/seed42/defense_results_max1024.json \
   results/chest_xray_pneumonia/resnet50/defense_PGD-AT/seed42/defense_results_max1024.nowarmup.bak.json 2>/dev/null || true
python scripts/evaluate_defense.py --config configs/chest_xray_pneumonia_resnet50.yaml \
  --defense PGD-AT --max-samples $MAX --seed 42 || echo "[fail] chest R50 AT seed42"
# optional: seed43 too, so chest R50 double-seed is also harmonized
python scripts/evaluate_defense.py --config configs/chest_xray_pneumonia_resnet50.yaml \
  --defense PGD-AT --max-samples $MAX --seed 43 || echo "[fail] chest R50 AT seed43"

########################################################################
# 1. Multi-seed STANDARD training for malaria + oct (H1 error bars).
#    5 models x seeds {43,44} x {train -> clean -> robustness main+fine}.
#    chest already has seed42/43/44; here we bring malaria/oct up to 3 seeds.
########################################################################
echo "############### 1. malaria/oct multi-seed standard ###############"
for DS in malaria oct2017; do
  for S in 43 44; do
    for m in resnet18 resnet34 resnet50 resnet101 resnet152; do
      CFG="configs/${DS}_${m}.yaml"
      CKPT="checkpoints/${DS}_${m}_seed${S}.pth"
      echo "### STD ${DS}/${m}/seed${S} ###"
      if ! python scripts/train.py --config "$CFG" --seed "$S"; then
        echo "[fail] train ${DS}/${m}/seed${S}; skipping its eval."; continue
      fi
      python scripts/evaluate_clean.py      --config "$CFG" --checkpoint "$CKPT" --seed "$S"
      python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --max-samples $MAX --seed "$S"
      python scripts/evaluate_robustness.py --config "$CFG" --checkpoint "$CKPT" --attacks-section attacks_fine --max-samples $MAX --seed "$S"
    done
  done
done

########################################################################
# 2. Grad-CAM for malaria + oct (seed42).
#    Standard: R18 & R152 (the two ends shown for chest).
#    AT: only the models that CONVERGED (collapsed AT models give meaningless
#    saliency): malaria R50+R152, oct R50 only (oct R152 AT collapsed).
########################################################################
echo "############### 2. malaria/oct Grad-CAM ###############"
for DS in malaria oct2017; do
  # standard models
  for m in resnet18 resnet152; do
    python scripts/generate_gradcam_figures.py \
      --config "configs/${DS}_${m}.yaml" \
      --checkpoint "checkpoints/${DS}_${m}_seed42.pth" \
      --attack PGD --eps 0.031373 --num-samples 8 \
      || echo "[fail] gradcam ${DS}/${m}"
  done
done
# AT models (converged only), tag out-dir with _at
for m in resnet50 resnet152; do   # malaria: both converged
  python scripts/generate_gradcam_figures.py \
    --config "configs/malaria_${m}.yaml" \
    --checkpoint "checkpoints/malaria_${m}_seed42_pgd_at.pth" \
    --attack PGD --eps 0.031373 --num-samples 8 \
    --out-dir "figures/gradcam/malaria/${m}_at" || echo "[fail] gradcam malaria/${m}_at"
done
python scripts/generate_gradcam_figures.py \
  --config "configs/oct2017_resnet50.yaml" \
  --checkpoint "checkpoints/oct2017_resnet50_seed42_pgd_at.pth" \
  --attack PGD --eps 0.031373 --num-samples 8 \
  --out-dir "figures/gradcam/oct2017/resnet50_at" || echo "[fail] gradcam oct2017/resnet50_at"

########################################################################
# 3. TRADES on chest R18/R50/R152 (second-method ablation for H2).
#    NOTE: TRADES uses ART's trainer -> it does NOT get the eps/lr warmup the
#    custom PGD-AT loop has. So R18 (and possibly others) may collapse for the
#    same no-warmup reason -> interpret a TRADES-R18 collapse with that caveat,
#    not as "TRADES is weak". chest is binary => AutoAttack auto-skips.
########################################################################
echo "############### 3. chest TRADES (R18/R50/R152) ###############"
for m in resnet18 resnet50 resnet152; do
  python scripts/evaluate_defense.py --config "configs/chest_xray_pneumonia_${m}.yaml" \
    --defense TRADES --max-samples $MAX --seed 42 \
    || echo "[fail] TRADES chest/${m}"
done

########################################################################
# 4. Package for local download.
########################################################################
echo "############### 4. packaging ###############"
tar czf results_extras.tgz results/ figures/gradcam/
echo "===== extras done. Download results_extras.tgz to the local repo and unpack. ====="
