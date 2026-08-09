"""AutoAttack audit of adversarially trained models.

Why this exists
---------------
The defended-model results in Section 4.5 were produced under PGD-50 with five
restarts. Chapter 5 argues that a robustness number is only as trustworthy as the
strongest attack behind it, and cites AutoAttack and RobustBench, but never applies
AutoAttack to the defences themselves. This script closes that gap.

Two constraints shape the design.

1. ``create_defense_eval_attacks`` deliberately SKIPS AutoAttack for binary tasks,
   so that every defended model in the published tables shares one protocol
   (src/attacks/attack_factory.py). This script bypasses that skip on purpose, and
   therefore writes to a SEPARATE results namespace so the published
   ``defense_results*.json`` files, and Table 4 built from them, are never touched.

2. Only some adversarial-training checkpoints still exist on disk. This script
   evaluates what is present and reports what is missing, rather than retraining.
   Retraining is far more expensive than the audit itself and is out of scope here.

Usage
-----
    # what would run, and on how many samples (no GPU work)
    python scripts/run_autoattack_audit.py --dry-run

    # measure this machine's throughput and project the full plan (minutes)
    python scripts/run_autoattack_audit.py --calibrate 32

    # the real audit
    python scripts/run_autoattack_audit.py --max-samples 256

Outputs ``results/<dataset>/<model>/defense_<method>/seed<N>/autoattack_audit_n<M>.json``
and appends one row per model to ``reports/thesis_evidence/autoattack_audit.csv``.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# torch, ART and the src/ package are imported lazily inside the functions that
# need them. Keeping them out of module scope is what lets --dry-run report the
# plan on a machine with no deep-learning environment installed, which is exactly
# the machine you want to check the plan on before renting a GPU.

CKPT_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<model>resnet\d+|deit_small|convnext_tiny)"
    r"_seed(?P<seed>\d+)_(?P<method>pgd_at|trades|mart)\.pth$"
)
METHOD_LABEL = {"pgd_at": "PGD-AT", "trades": "TRADES", "mart": "MART"}
EPS_8_255 = 8.0 / 255.0
EVID = PROJECT_ROOT / "reports" / "thesis_evidence"


def discover(ckpt_dir: Path) -> list[dict]:
    """Every adversarial-training checkpoint present on disk, with its config path."""
    found = []
    for p in sorted(ckpt_dir.glob("*.pth")):
        m = CKPT_RE.match(p.name)
        if not m:
            continue
        d = m.groupdict()
        cfg = PROJECT_ROOT / "configs" / f"{d['dataset']}_{d['model']}.yaml"
        if not cfg.exists():
            continue
        found.append(
            {
                "dataset": d["dataset"],
                "model": d["model"],
                "seed": int(d["seed"]),
                "method": METHOD_LABEL[d["method"]],
                "checkpoint": p,
                "config": cfg,
            }
        )
    return found


def published_robust_acc(cell: dict) -> float | None:
    """PGD-50 robust accuracy already reported for this cell at 8/255, for comparison.

    The published JSONs key each attack as e.g. "PGD50-5restart_eps=0.031373", so the
    budget lives in the KEY, not in a field, and the figure itself is nested under
    ``["robust_accuracy"]["full_robust_accuracy"]``. An earlier version of this
    function looked for a top-level "eps" field and a flat accuracy value, matched
    nothing across all 37 cells, and silently returned None everywhere.

    Returns None when no matching entry exists, which is recorded as such rather
    than filled in.
    """
    rd = (
        PROJECT_ROOT / "results" / cell["dataset"] / cell["model"]
        / f"defense_{cell['method']}" / f"seed{cell['seed']}"
    )
    key_eps = re.compile(r"_eps=([0-9.]+)$")
    for f in sorted(rd.glob("defense_results*.json")):
        try:
            blob = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for key, val in blob.items():
            if not isinstance(val, dict) or "PGD" not in key:
                continue
            m = key_eps.search(key)
            if not m or abs(float(m.group(1)) - EPS_8_255) > 1e-6:
                continue
            ra = val.get("robust_accuracy")
            if isinstance(ra, dict):
                # full_ is over all test samples, matching this audit's denominator.
                for field in ("full_robust_accuracy", "robust_accuracy"):
                    if field in ra:
                        return float(ra[field])
            elif isinstance(ra, (int, float)):
                return float(ra)
    return None


def evaluate_cell(cell: dict, n_samples: int, device, logger) -> dict:
    """Run AutoAttack at 8/255 on one defended checkpoint. Returns a result row."""
    import numpy as np
    import torch

    from src.utils.reproducibility import set_seed, load_config
    from src.attacks.attack_factory import create_attack
    from src.evaluation.subset import get_attack_subset
    from scripts.evaluate_defense import load_classifier_from_checkpoint
    from scripts.evaluate_robustness import collect_test_data, get_predictions_and_confidences

    cfg = load_config(str(cell["config"]))
    cfg["seed"] = cell["seed"]
    set_seed(cell["seed"])

    classifier, data = load_classifier_from_checkpoint(
        cfg, str(cell["checkpoint"]), device, logger
    )

    x_full, y_full = collect_test_data(data["test"], max_samples=None)
    if n_samples is not None and n_samples < len(x_full):
        cache = (
            PROJECT_ROOT / "results" / cell["dataset"]
            / f"attack_subset_seed{cell['seed']}_n{n_samples}.json"
        )
        idx = get_attack_subset(y_full, n_samples, cell["seed"], cache)
        x_test, y_test = x_full[idx], y_full[idx]
    else:
        x_test, y_test = x_full, y_full

    clean_preds, _ = get_predictions_and_confidences(classifier, x_test)
    clean_acc = float((clean_preds == y_test).mean())

    # create_attack drops the DLR member for <3 classes; AutoAttack still rejects
    # any candidate exceeding eps, so the reduced ensemble stays budget-respecting.
    nb = getattr(classifier, "nb_classes", None)
    attack = create_attack(classifier, {"name": "AutoAttack"}, eps=EPS_8_255)

    t0 = time.perf_counter()
    x_adv = attack.generate(x=x_test, y=y_test)
    elapsed = time.perf_counter() - t0

    adv_preds, _ = get_predictions_and_confidences(classifier, x_adv)
    robust_acc = float((adv_preds == y_test).mean())
    max_pert = float(np.abs(x_adv - x_test).max())

    # The published PGD-50 figure is a three-seed mean over 1024 samples, so it
    # cannot be differenced against a single-seed AutoAttack run on a 256-sample
    # subset. Re-run PGD-50 with five restarts on THESE samples so the comparison
    # shares one denominator, and take the per-cell minimum, which is the defended
    # analogue of the attack-set envelope defined in Section 3.4.
    pgd = create_attack(
        classifier,
        {"name": "PGD", "max_iter": 50, "num_random_init": 5},
        eps=EPS_8_255,
    )
    t1 = time.perf_counter()
    x_pgd = pgd.generate(x=x_test, y=y_test)
    pgd_elapsed = time.perf_counter() - t1
    pgd_preds, _ = get_predictions_and_confidences(classifier, x_pgd)
    pgd_acc = float((pgd_preds == y_test).mean())

    return {
        "dataset": cell["dataset"],
        "model": cell["model"],
        "method": cell["method"],
        "seed": cell["seed"],
        "n_samples": int(len(x_test)),
        "nb_classes": int(nb) if nb is not None else None,
        "clean_accuracy": clean_acc,
        "autoattack_robust_accuracy_8_255": robust_acc,
        "pgd50_robust_accuracy_8_255_same_subset": pgd_acc,
        "audited_min_robust_accuracy_8_255": min(robust_acc, pgd_acc),
        "autoattack_binds": bool(robust_acc < pgd_acc - 1e-9),
        "pgd50_robust_accuracy_8_255_published": published_robust_acc(cell),
        "max_perturbation_linf": max_pert,
        "seconds": round(elapsed, 1),
        "seconds_pgd50": round(pgd_elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def write_outputs(row: dict, cell: dict) -> None:
    from src.utils.reproducibility import get_results_dir

    rd = get_results_dir(
        "results", cell["dataset"], cell["model"],
        f"defense_{cell['method']}", cell["seed"],
    )
    out = Path(rd) / f"autoattack_audit_n{row['n_samples']}.json"
    json.dump(row, open(out, "w", encoding="utf-8"), indent=2)

    EVID.mkdir(parents=True, exist_ok=True)
    csv = EVID / "autoattack_audit.csv"
    cols = list(row.keys())
    if not csv.exists():
        csv.write_text(",".join(cols) + "\n", encoding="utf-8")
    with open(csv, "a", encoding="utf-8") as fh:
        fh.write(",".join(str(row[c]) for c in cols) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoAttack audit of defended models")
    ap.add_argument("--max-samples", type=int, default=256,
                    help="Evaluation subset size (fixed stratified, shared with prior runs)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="PGD-AT TRADES MART")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="Time ONE cell on N samples, then project the full plan")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    all_cells = discover(PROJECT_ROOT / args.checkpoint_dir)
    cells = all_cells
    for key, want in (("dataset", args.datasets), ("method", args.methods),
                      ("model", args.models), ("seed", args.seeds)):
        if want:
            cells = [c for c in cells if c[key] in want]

    if not cells:
        print("No adversarial-training checkpoints matched. Nothing to do.")
        return

    print(f"{len(cells)} checkpoint(s) available, {args.max_samples} samples each\n")
    by_group: dict[tuple[str, str], int] = {}
    for c in cells:
        by_group[(c["dataset"], c["method"])] = by_group.get((c["dataset"], c["method"]), 0) + 1
    for (ds, meth), n in sorted(by_group.items()):
        print(f"  {ds:22s} {meth:8s} {n:2d} seed-level run(s)")

    # Coverage gaps are a finding, not an error: report them explicitly. Compute
    # them from the UNFILTERED discovery, so that a method excluded by --methods is
    # never reported as an absent checkpoint.
    on_disk = {(c["dataset"], c["method"]) for c in all_cells}
    missing = [
        f"{ds}/{meth}"
        for ds in sorted({c["dataset"] for c in all_cells})
        for meth in ("PGD-AT", "TRADES", "MART")
        if (ds, meth) not in on_disk
    ]
    if missing:
        print("\n  No checkpoint on disk anywhere (would need retraining): "
              + ", ".join(missing))
    if len(cells) < len(all_cells):
        print(f"  ({len(all_cells) - len(cells)} further cell(s) present on disk "
              f"but excluded by the filters above)")

    if args.dry_run:
        print(f"\nPlan: {len(cells)} cell(s) x {args.max_samples} samples, "
              f"AutoAttack at 8/255 (eps={EPS_8_255:.6f}).")
        print("Writes autoattack_audit_n<M>.json per cell plus "
              "reports/thesis_evidence/autoattack_audit.csv.")
        print("Published defense_results*.json are NOT modified.")
        print("\nRuntime is not predictable from the plan alone: AutoAttack cost scales")
        print("with the surviving-sample fraction. Measure it with --calibrate 32 on a")
        print("converged cell (e.g. --datasets chest_xray_pneumonia --methods TRADES).")
        print("\nDry run: nothing executed.")
        return

    import torch
    from src.utils.logger import get_logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("\nWARNING: no CUDA device. AutoAttack on CPU is not practical.")

    if args.calibrate:
        cell = cells[0]
        logger = get_logger("autoattack_calibrate", log_dir=None)
        print(f"\nCalibrating on {cell['dataset']}/{cell['model']}/"
              f"{cell['method']}/seed{cell['seed']} with {args.calibrate} samples...")
        row = evaluate_cell(cell, args.calibrate, device, logger)
        per_sample = row["seconds"] / row["n_samples"]
        print(f"\n  GPU:                    {row['gpu']}")
        print(f"  measured:               {row['seconds']:.1f} s AutoAttack + "
              f"{row['seconds_pgd50']:.1f} s PGD-50 "
              f"for {row['n_samples']} samples ({per_sample:.2f} s/sample total)")
        print(f"  AutoAttack robust acc:  {row['autoattack_robust_accuracy_8_255']:.4f}")
        print(f"  PGD-50 same subset:     {row['pgd50_robust_accuracy_8_255_same_subset']:.4f}"
              f"   <- the like-for-like comparison")
        print(f"  audited minimum:        {row['audited_min_robust_accuracy_8_255']:.4f}"
              f"   (AutoAttack binds: {row['autoattack_binds']})")
        print("\n  NOTE: cost scales with the SURVIVING sample fraction, so a converged")
        print("  model projects higher than a collapsed one. Calibrate on a converged")
        print("  cell for a representative estimate.")
        total_h = per_sample * args.max_samples * len(cells) / 3600
        print(f"\n  Projected for {len(cells)} cell(s) x {args.max_samples} samples: "
              f"{total_h:.1f} h")
        return

    logger = get_logger("autoattack_audit", log_dir=None)
    started = time.perf_counter()
    for i, cell in enumerate(cells, 1):
        tag = f"{cell['dataset']}/{cell['model']}/{cell['method']}/seed{cell['seed']}"
        print(f"\n[{i}/{len(cells)}] {tag}")
        try:
            row = evaluate_cell(cell, args.max_samples, device, logger)
        except Exception as exc:                      # keep the sweep alive
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        write_outputs(row, cell)
        print(f"  clean {row['clean_accuracy']:.4f} | "
              f"PGD-50 {row['pgd50_robust_accuracy_8_255_same_subset']:.4f} -> "
              f"AA {row['autoattack_robust_accuracy_8_255']:.4f} | "
              f"min {row['audited_min_robust_accuracy_8_255']:.4f} | "
              f"{(row['seconds'] + row['seconds_pgd50']) / 60:.1f} min")
    print(f"\nTotal wall clock: {(time.perf_counter() - started) / 3600:.2f} h")
    print(f"Summary: {EVID / 'autoattack_audit.csv'}")


if __name__ == "__main__":
    main()
