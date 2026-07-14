#!/usr/bin/env bash
# Backfill batch (2026-07-14), to run on the AutoDL GPU instance.
# Four tasks that close the remaining gaps before writing Results/Discussion:
#
#   1. H2 multi-seed: give the STABLE AT successes seed43/44 so H2 has mean±std
#      like H1 (chest R50/R152 x {PGD-AT,TRADES,MART}; malaria R50/R152 x PGD-AT).
#   2. Rescue representative COLLAPSE points with stronger stabilisers
#      (eps_warmup=8 / lr_warmup=5 / nb_epochs=30 / LR halved / grad clip) via the
#      new "PGD-AT-rescue" defense -> distinguishes "un-trainable" vs "under-stabilised".
#      Results go to defense_PGD-AT-rescue/ (does NOT overwrite the collapse evidence).
#   3. Cross-dataset TRADES ablation (malaria/oct) + decision-boundary figures for
#      malaria/oct -> strengthens the cross-dataset argument.
#   4. Regenerate the per-model sci_defense diagnostic figures with 3 methods
#      (Standard/PGD-AT/TRADES/MART) instead of the old 2-method data.
#
# PREREQUISITE (already done locally, but re-run to be safe): configs regenerated
# from the updated base files (adds TRADES/MART to malaria+oct, PGD-AT-rescue to all).
#
# The UK<->China SSH link WILL drop; run under tmux or nohup:
#   nohup bash run_backfill.sh > run_backfill.log 2>&1 &
#   tail -f run_backfill.log
#
# Rough GPU-time guide (RTX 4090-class):
#   Task 1 chest: 6 trainings (R50/R152 x 3 methods) x 2 seeds = 12 AT runs ~ 8-12 h
#   Task 1 malaria: 2 models x 2 seeds = 4 PGD-AT runs ~ 3-4 h
#   Task 2 rescue: 2-3 runs x 30 epochs ~ 3-5 h (oct is slow: 83k train set)
#   Task 3 TRADES malaria/oct + decision boundary ~ 4-6 h
#   Task 4 figures: minutes (no GPU needed; pure plotting)
# Every step resumes: existing checkpoints -> re-evaluate only; finished attacks skipped.
#
# set -uo pipefail (NOT -e): one model failing must not abort the rest.
set -uo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
MAX=1024
CHEST=chest_xray_pneumonia

echo "############### 0. regenerate configs (idempotent) ###############"
python scripts/make_configs.py

# Helper: train (if no checkpoint) + strong-eval a defense for one (dataset,model,seed).
# Idempotent: an existing AT checkpoint is re-evaluated only (never retrained).
run_defense () {
  local DS="$1" M="$2" DEF="$3" S="$4"
  local suffix; suffix=$(echo "$DEF" | tr 'A-Z-' 'a-z_')
  local CKPT="checkpoints/${DS}_${M}_seed${S}_${suffix}.pth"
  local CFG="configs/${DS}_${M}.yaml"
  echo "### ${DEF} ${DS}/${M}/seed${S} (ckpt=${CKPT}) ###"
  if [ -f "$CKPT" ]; then
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" \
      --checkpoint "$CKPT" --max-samples $MAX --seed "$S" \
      || echo "[fail] ${DEF} eval ${DS}/${M}/seed${S}"
  else
    python scripts/evaluate_defense.py --config "$CFG" --defense "$DEF" \
      --max-samples $MAX --seed "$S" \
      || echo "[fail] ${DEF} train+eval ${DS}/${M}/seed${S}"
  fi
}

########################################################################
# TASK 1 — H2 multi-seed on the STABLE successes (give H2 mean±std).
#   chest R50/R152 across all three AT methods; malaria R50/R152 PGD-AT.
#   (R18/R34/R101 collapsed -> not part of the "complexity helps" claim, so no
#    multi-seed needed here; their trainability is Task 2's question instead.)
########################################################################
echo "############### TASK 1. H2 multi-seed ###############"
for S in 43 44; do
  for M in resnet50 resnet152; do
    for DEF in PGD-AT TRADES MART; do
      run_defense "$CHEST" "$M" "$DEF" "$S"
    done
  done
  for M in resnet50 resnet152; do
    run_defense malaria "$M" PGD-AT "$S"
  done
done

########################################################################
# TASK 2 — Rescue representative COLLAPSE points (PGD-AT-rescue defense).
#   Representative choices: oct R152 (big model, collapsed only on hardest task)
#   and chest R18 (smallest, collapsed under PGD-AT but TRADES rescued it on chest
#   -> tests whether stronger PGD-AT stabilisers alone suffice). chest R101 optional.
#   Verdict rule: if PGD@8 full-robust-acc rises well above 0 AND clean predictions
#   are NOT a single constant class -> "under-stabilised" (rescuable); else "un-trainable".
########################################################################
echo "############### TASK 2. rescue collapse points ###############"
run_defense oct2017 resnet152 PGD-AT-rescue 42
run_defense "$CHEST" resnet18  PGD-AT-rescue 42
# Optional third representative (uncomment to include):
# run_defense "$CHEST" resnet101 PGD-AT-rescue 42

########################################################################
# TASK 3a — Cross-dataset TRADES ablation (seed42). malaria R50/R152, oct R50
#   (the datasets' stable PGD-AT points) -> "TRADES helps across datasets too".
########################################################################
echo "############### TASK 3a. cross-dataset TRADES ###############"
for M in resnet50 resnet152; do
  run_defense malaria "$M" TRADES 42
done
run_defense oct2017 resnet50 TRADES 42

########################################################################
# TASK 3b — Decision-boundary figures for malaria + oct (chest already done).
#   Standard R18-vs-R152 (clean vs adversarial t-SNE), plus AT versions where a
#   converged AT checkpoint exists (malaria R152 _pgd_at, oct R50 _pgd_at).
########################################################################
echo "############### TASK 3b. decision boundary malaria/oct ###############"
for DS in malaria oct2017; do
  python scripts/generate_decision_boundary_figures.py --dataset "$DS" \
    --models resnet18 resnet152 --eps 0.031373 --max-samples 300 \
    || echo "[fail] decision_boundary standard ${DS}"
done
# AT versions (only where a converged AT checkpoint exists):
[ -f checkpoints/malaria_resnet152_seed42_pgd_at.pth ] && \
  python scripts/generate_decision_boundary_figures.py --dataset malaria \
    --models resnet50 resnet152 --checkpoint-suffix _pgd_at --max-samples 300 \
    || echo "[skip/fail] decision_boundary malaria _pgd_at"
[ -f checkpoints/oct2017_resnet50_seed42_pgd_at.pth ] && \
  python scripts/generate_decision_boundary_figures.py --dataset oct2017 \
    --models resnet18 resnet50 --checkpoint-suffix _pgd_at --max-samples 300 \
    || echo "[skip/fail] decision_boundary oct2017 _pgd_at"

########################################################################
# TASK 4 — Regenerate per-model sci_defense diagnostics with MART (3 methods).
#   No GPU needed (reads the defense_results JSON). chest R18/R50/R152 have all
#   three AT methods at seed42.
########################################################################
echo "############### TASK 4. sci_defense figures (3 methods) ###############"
for M in resnet18 resnet50 resnet152; do
  python scripts/generate_defense_sci_figures.py \
    --dataset "$CHEST" --model "$M" --seed seed42 \
    || echo "[fail] sci_defense ${M}"
done

########################################################################
# Package result JSONs for local download (rsync resumes a dropped transfer).
########################################################################
echo "############### packaging ###############"
find results -name '*.json' -o -name '*.yaml' > /tmp/flist.txt
tar czf backfill_json.tgz -T /tmp/flist.txt
echo "  backfill_json.tgz : $(du -h backfill_json.tgz | cut -f1)"
echo ""
echo "===== backfill done ====="
echo "Pull to local:  rsync -P --append-verify -e ssh <user>@<host>:.../backfill_json.tgz ."
echo "Then locally regenerate the main H2 figures/tables with the new seeds:"
echo "  python scripts/extract_figure_data.py"
echo "  python scripts/generate_at_ladder_figure.py"
echo "  python scripts/generate_comparison_tables.py"
