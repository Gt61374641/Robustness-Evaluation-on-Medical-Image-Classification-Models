#!/usr/bin/env python
"""Progress report for the three-dataset alignment batch (run_align_p{0,1,2}.sh).

Usage:  python status_align.py

Prints one line per expected (dataset, model, defense, seed) cell: whether the
result JSON exists, its defended clean accuracy and PGD@8/255 full robust
accuracy, and whether the point collapsed. Also tallies [fail] markers in the
partition logs. Safe to run while the batch is still going.
"""
import json
import os
import re

MAX = 1024
EPS8 = "PGD50-5restart_eps=0.031373"
CHEST, MAL, OCT = "chest_xray_pneumonia", "malaria", "oct2017"

# (partition, dataset, model, defense, seeds)
JOBS = [
    (0, OCT,   "resnet152", "TRADES", [42, 43, 44]),
    (0, OCT,   "resnet101", "PGD-AT", [43, 44]),

    (1, OCT,   "resnet152", "MART",   [42, 43, 44]),
    (1, OCT,   "resnet152", "PGD-AT", [43, 44]),
    (1, MAL,   "resnet34",  "PGD-AT", [43, 44]),
    (1, CHEST, "resnet18",  "PGD-AT", [44]),

    (2, OCT,   "resnet18",  "TRADES", [42, 43, 44]),
    (2, OCT,   "resnet18",  "MART",   [42, 43, 44]),
    (2, OCT,   "resnet18",  "PGD-AT", [43, 44]),
    (2, OCT,   "resnet34",  "PGD-AT", [43, 44]),
    (2, MAL,   "resnet18",  "PGD-AT", [43, 44]),
    (2, CHEST, "resnet101", "PGD-AT", [43, 44]),
    (2, CHEST, "resnet34",  "PGD-AT", [43, 44]),
]

# A defended model counts as collapsed if PGD@8 robustness is ~0, or if clean
# accuracy has fallen to the majority-class rate (constant predictor).
MAJORITY = {CHEST: 0.625, MAL: 0.50, OCT: 0.25}


def cell(ds, model, defense, seed):
    path = os.path.join("results", ds, model, f"defense_{defense}",
                        f"seed{seed}", f"defense_results_max{MAX}.json")
    if not os.path.exists(path):
        running = os.path.exists(os.path.join(os.path.dirname(path),
                                              "evaluate_defense.log"))
        return "RUNNING" if running else "pending", None, None, False
    try:
        j = json.load(open(path))
    except Exception:
        return "corrupt", None, None, False
    clean = j.get("clean_accuracy_defended")
    rob = None
    if EPS8 in j:
        rob = j[EPS8].get("robust_accuracy", {}).get("full_robust_accuracy")
    collapsed = False
    if rob is not None and clean is not None:
        collapsed = rob < 0.01 or clean <= MAJORITY.get(ds, 0.0) + 0.02
    status = "done" if rob is not None else "partial"
    return status, clean, rob, collapsed


def main():
    counts = {}
    print("%-7s %-22s %-10s %-8s %-5s %-8s %7s %7s  %s" %
          ("part", "dataset", "model", "defense", "seed", "status",
           "clean", "rob@8", "note"))
    print("-" * 100)
    for part, ds, model, defense, seeds in JOBS:
        for s in seeds:
            status, clean, rob, collapsed = cell(ds, model, defense, s)
            counts[status] = counts.get(status, 0) + 1
            print("%-7s %-22s %-10s %-8s %-5s %-8s %7s %7s  %s" % (
                f"p{part}", ds, model, defense, s, status,
                "-" if clean is None else f"{clean:.3f}",
                "-" if rob is None else f"{rob:.3f}",
                "COLLAPSED" if collapsed else ""))

    total = sum(counts.values())
    print("-" * 100)
    print("total %d cells: %s" % (
        total, ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))))

    for p in (0, 1, 2):
        log = f"p{p}.log"
        if not os.path.exists(log):
            continue
        text = open(log, encoding="utf-8", errors="replace").read()
        fails = re.findall(r"^\[fail\].*$", text, re.M)
        done = os.path.exists(f".align_done_p{p}")
        print(f"p{p}: {'FINISHED' if done else 'in progress'}, "
              f"{len(fails)} failed task(s)")
        for f in fails:
            print("    " + f)


if __name__ == "__main__":
    main()
