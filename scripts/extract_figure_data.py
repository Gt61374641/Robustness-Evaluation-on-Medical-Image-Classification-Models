"""Extract all main-figure data to figures/data/*.json.

Single source of truth so the Python and R figure backends render byte-identical
numbers. One JSON per hero figure:

  at_ladder_h2.json           H2 5-model AT ladder, PGD@8 full robust acc,
                              seed-mean over 42/43/44 (+std/n_seeds); PGD-AT only
  at_rescue.json              PGD-AT-rescue points vs original PGD-AT (separate
                              protocol -- stability diagnostic, not in the ladder)
  h1_pgd_curves.json          H1 robustness vs eps, FGSM & PGD, per dataset/model,
                              mean+std over seeds (PGD collapse-artifact excluded)
  h1_complexity_fixedeps.json H1 U-shape: robust acc at eps=0.1/255 vs capacity
  defense_methods.json        Standard / PGD-AT / TRADES / MART, chest R18/R50/R152
  attack_methods.json         CW / DeepFool / AutoAttack / Square on 7 chest models

Run:  python scripts/extract_figure_data.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)

LADDER = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
PARAMS_M = {"resnet18": 11.7, "resnet34": 21.8, "resnet50": 25.6,
            "resnet101": 44.5, "resnet152": 60.2}
DISPLAY = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
           "resnet101": "ResNet-101", "resnet152": "ResNet-152",
           "deit_small": "DeiT-S", "convnext_tiny": "ConvNeXt-T"}
DATASETS = [
    ("chest_xray_pneumonia", "Chest X-ray", ["seed42", "seed43", "seed44"]),
    ("malaria", "Malaria", ["seed42", "seed43", "seed44"]),
    ("oct2017", "OCT", ["seed42", "seed43", "seed44"]),
]


def eps255(key):
    m = re.search(r"eps=([0-9.]+)", key)
    return round(float(m.group(1)) * 255, 4) if m else None


def _rob_json(ds, model, seed, sec):
    f = ROOT / "results" / ds / model / "robustness" / seed / f"robustness_attacks_{sec}_max1024.json"
    return json.load(open(f)) if f.exists() else None


def attack_points(ds, model, seed, attack):
    """{eps255: full_robust_acc} for FGSM/PGD; PGD large-eps class-collapse dropped."""
    pts = {}
    for sec in ("fine", "main"):
        r = _rob_json(ds, model, seed, sec)
        if not r:
            continue
        for k, v in r.items():
            if not (k.startswith(attack) and isinstance(v, dict) and "robust_accuracy" in v):
                continue
            if attack == "PGD":
                coll = v.get("pred_distribution", {}).get("collapse", {}).get("adv_majority_fraction", 0)
                if coll > 0.97:
                    continue
            e = eps255(k)
            if e is not None:
                pts[e] = v["robust_accuracy"]["full_robust_accuracy"]
    return pts


def aggregate_curves():
    """H1 curves: per dataset/model/attack -> [{eps, mean, std, n}] over seeds."""
    out = {"datasets": [], "ladder": LADDER, "display": DISPLAY, "params_m": PARAMS_M}
    for ds, disp, seeds in DATASETS:
        d = {"dataset": ds, "display": disp, "models": {}}
        for m in LADDER:
            d["models"][m] = {}
            for atk in ("FGSM", "PGD"):
                per = {}
                for s in seeds:
                    for e, acc in attack_points(ds, m, s, atk).items():
                        per.setdefault(e, []).append(acc)
                series = [{"eps": e, "mean": float(np.mean(per[e])),
                           "std": float(np.std(per[e])), "n": len(per[e])}
                          for e in sorted(per)]
                d["models"][m][atk] = series
        out["datasets"].append(d)
    json.dump(out, open(OUT / "h1_pgd_curves.json", "w"), indent=2)
    print("wrote h1_pgd_curves.json")


def fixed_eps_ushape(target=0.1):
    """H1 U-shape: robust acc at eps~=target/255 vs capacity, mean over seeds."""
    out = {"target_eps255": target, "ladder": LADDER, "display": DISPLAY,
           "params_m": PARAMS_M, "datasets": []}
    for ds, disp, seeds in DATASETS:
        row = {"dataset": ds, "display": disp, "models": {}}
        for m in LADDER:
            vals = []
            for s in seeds:
                pts = attack_points(ds, m, s, "PGD")
                if not pts:
                    continue
                e = min(pts, key=lambda k: abs(k - target))
                if abs(e - target) <= 0.03:
                    vals.append(pts[e])
            row["models"][m] = {"mean": float(np.mean(vals)) if vals else None,
                                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                                "n": len(vals)}
        out["datasets"].append(row)
    json.dump(out, open(OUT / "h1_complexity_fixedeps.json", "w"), indent=2)
    print("wrote h1_complexity_fixedeps.json")


def _defense_clean_rob8(ds, model, defense, seed="seed42"):
    """(clean_defended, robust@8 full/conditional, asr, collapsed) for a defense JSON."""
    f = ROOT / "results" / ds / model / f"defense_{defense}" / seed / "defense_results_max1024.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    clean = r.get("clean_accuracy_defended")
    rob8 = rob8_cond = asr8 = None
    for k, v in r.items():
        if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
            if abs((eps255(k) or 0) - 8) < 0.5:
                rob8 = v["robust_accuracy"]["full_robust_accuracy"]
                rob8_cond = v["robust_accuracy"].get("conditional_robust_accuracy")
                asr8 = v.get("asr")
    # collapse: constant clean prediction OR ~0 robust
    kk = [k for k in r if k.startswith("PGD")]
    constant = False
    if kk:
        fr = [x["fraction"] for x in r[kk[0]]["pred_distribution"]["clean"].values()]
        constant = max(fr) >= 0.99
    collapsed = bool(constant or (rob8 is not None and rob8 < 0.02))
    return {"clean": clean, "rob8": rob8, "rob8_cond": rob8_cond, "asr8": asr8,
            "collapsed": collapsed}


def _standard_clean_rob8(ds, model, seed="seed42"):
    """Standard (undefended) model clean + PGD robust@8 (strong sweep)."""
    clean = None
    cf = ROOT / "results" / ds / model / "clean" / seed / "clean_results.json"
    if cf.exists():
        cj = json.load(open(cf))
        clean = cj.get("accuracy") or cj.get("clean_accuracy") or \
            cj.get("metrics", {}).get("accuracy")
    rob8 = rob8_cond = asr8 = None
    for sec in ("main", "fine"):
        r = _rob_json(ds, model, seed, sec)
        if not r:
            continue
        for k, v in r.items():
            if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
                if abs((eps255(k) or 0) - 8) < 0.5:
                    rob8 = v["robust_accuracy"]["full_robust_accuracy"]
                    rob8_cond = v["robust_accuracy"].get("conditional_robust_accuracy")
                    asr8 = v.get("asr")
    return {"clean": clean, "rob8": rob8, "rob8_cond": rob8_cond, "asr8": asr8,
            "collapsed": False}


def _agg_defense(ds, model, method, seeds):
    """Aggregate one (model, method) cell over seeds.

    Keeps `rob8`/`clean` as the seed-mean (back-compat: the R backend reads
    those scalars) and adds `rob8_std`/`clean_std`/`n_seeds` for error bars.
    `collapsed` = majority vote over the per-seed collapse flags.
    """
    recs = []
    for s in seeds:
        rec = _standard_clean_rob8(ds, model, s) if method == "Standard" \
            else _defense_clean_rob8(ds, model, method, s)
        if rec is not None:
            recs.append(rec)
    if not recs:
        return {"clean": None, "rob8": None, "rob8_std": 0.0, "clean_std": 0.0,
                "n_seeds": 0, "collapsed": False, "seeds": []}

    def stat(key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        if not vals:
            return None, 0.0
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    rob8_m, rob8_s = stat("rob8")
    clean_m, clean_s = stat("clean")
    n_coll = sum(1 for r in recs if r.get("collapsed"))
    return {"clean": clean_m, "clean_std": clean_s,
            "rob8": rob8_m, "rob8_std": rob8_s,
            "n_seeds": len(recs), "collapsed": n_coll * 2 > len(recs),
            "seeds": [s for s in seeds]}


def defense_methods():
    ds = "chest_xray_pneumonia"
    models = ["resnet18", "resnet50", "resnet152"]
    methods = ["Standard", "PGD-AT", "TRADES", "MART"]
    seeds = ["seed42", "seed43", "seed44"]
    out = {"dataset": ds, "display": "Chest X-ray", "models": models,
           "display_names": DISPLAY, "methods": methods, "seeds": seeds, "rows": []}
    for m in models:
        for meth in methods:
            rec = _agg_defense(ds, m, meth, seeds)
            out["rows"].append({"model": m, "method": meth, **rec})
    json.dump(out, open(OUT / "defense_methods.json", "w"), indent=2)
    print("wrote defense_methods.json")


def at_ladder_h2():
    """H2 5-model AT ladder x 3 datasets, PGD@8 full robust acc (PGD-AT).

    Aggregated over seed42/43/44 (whatever exists per cell). Keeps `clean`/
    `robust8` as the seed-mean so the R/py figure backends and table8 keep
    reading those scalars; adds `clean_std`/`robust8_std`/`n_seeds` for error
    bars. Uses the ORIGINAL unified PGD-AT protocol only -- the PGD-AT-rescue
    runs are a different (stronger-stabilisation) protocol and live in
    at_rescue.json so the two are never mixed in one ladder.
    """
    seeds = ["seed42", "seed43", "seed44"]
    rows = []
    for ds, disp, _ in DATASETS:
        for m in LADDER:
            rec = _agg_defense(ds, m, "PGD-AT", seeds)
            rows.append({
                "dataset": ds, "dataset_display": disp,
                "model": m, "params_m": PARAMS_M[m],
                "clean": rec["clean"], "clean_std": rec["clean_std"],
                "robust8": rec["rob8"], "robust8_std": rec["rob8_std"],
                "n_seeds": rec["n_seeds"], "collapsed": rec["collapsed"],
            })
    out = {
        "metric": "PGD@8/255 full robust accuracy (PGD-AT, seed-mean over 42/43/44)",
        "protocol": "unified PGD-AT (eps_warmup=5, lr_warmup=3, nb_epochs aligned); "
                    "PGD-50+5restart strong eval",
        "ladder": LADDER, "params_m": PARAMS_M,
        "datasets": [ds for ds, _, _ in DATASETS],
        "rows": rows,
    }
    json.dump(out, open(OUT / "at_ladder_h2.json", "w"), indent=2)
    print("wrote at_ladder_h2.json")


def at_rescue():
    """PGD-AT-rescue points (stronger stabilisation) vs their ORIGINAL PGD-AT.

    Separate from the H2 ladder on purpose: rescue is a different protocol, so
    it is reported as a stability/diagnostic table (original collapsed -> rescue
    recovered), not silently substituted into the PGD-AT ladder.
    """
    seeds = ["seed42", "seed43", "seed44"]
    disp_ds = {ds: disp for ds, disp, _ in DATASETS}
    rows = []
    for ds, _, _ in DATASETS:
        for m in LADDER:
            rdir = ROOT / "results" / ds / m / "defense_PGD-AT-rescue"
            if not rdir.is_dir():
                continue
            rescue = _agg_defense(ds, m, "PGD-AT-rescue", seeds)
            orig = _agg_defense(ds, m, "PGD-AT", seeds)
            rows.append({
                "dataset": ds, "dataset_display": disp_ds[ds],
                "model": m, "params_m": PARAMS_M[m],
                "orig_clean": orig["clean"], "orig_robust8": orig["rob8"],
                "orig_collapsed": orig["collapsed"], "orig_n_seeds": orig["n_seeds"],
                "rescue_clean": rescue["clean"], "rescue_clean_std": rescue["clean_std"],
                "rescue_robust8": rescue["rob8"], "rescue_robust8_std": rescue["rob8_std"],
                "rescue_collapsed": rescue["collapsed"], "rescue_n_seeds": rescue["n_seeds"],
            })
    out = {
        "metric": "PGD@8/255 full robust accuracy: original PGD-AT vs PGD-AT-rescue",
        "rescue_protocol": "stronger stabilisation (e.g. eps_warmup=8/lr_warmup=5/"
                           "longer schedule/LR halved/grad-clip); PGD-50+5restart eval",
        "note": "reported separately from the H2 PGD-AT ladder; different protocol, "
                "not substituted into it.",
        "rows": rows,
    }
    json.dump(out, open(OUT / "at_rescue.json", "w"), indent=2)
    print("wrote at_rescue.json")


def attack_methods():
    ds = "chest_xray_pneumonia"
    models = LADDER + ["deit_small", "convnext_tiny"]
    out = {"dataset": ds, "display": "Chest X-ray", "models": models,
           "display_names": DISPLAY, "rows": []}
    for m in models:
        f = ROOT / "results" / ds / m / "robustness" / "seed42" / "robustness_attacks_extra_max1024.json"
        if not f.exists():
            continue
        r = json.load(open(f))

        def cond(key):
            return r[key]["robust_accuracy"]["conditional_robust_accuracy"] if key in r else None

        def l2(key):
            return r[key]["perturbation"]["l2_mean"] if key in r else None

        out["rows"].append({
            "model": m,
            "clean": r.get("_meta", {}).get("clean_accuracy"),
            "AutoAttack8": cond("AutoAttack_eps=0.031373"),
            "Square8": cond("SquareAttack_eps=0.031373"),
            "CW_l2": l2("CW_default"),
            "DeepFool_l2": l2("DeepFool_default"),
        })
    json.dump(out, open(OUT / "attack_methods.json", "w"), indent=2)
    print("wrote attack_methods.json")


if __name__ == "__main__":
    aggregate_curves()
    fixed_eps_ushape()
    defense_methods()
    at_ladder_h2()
    at_rescue()
    attack_methods()
    print("done -> figures/data/")
