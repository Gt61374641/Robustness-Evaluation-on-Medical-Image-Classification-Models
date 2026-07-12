"""Extract all main-figure data to figures/data/*.json.

Single source of truth so the Python and R figure backends render byte-identical
numbers. One JSON per hero figure:

  at_ladder_h2.json           H2 5-model AT ladder, PGD@8 full robust acc (already
                              written by the at_ladder step; re-emitted here too)
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
    """(clean_defended, robust@8 full, collapsed) for a defense JSON, or None."""
    f = ROOT / "results" / ds / model / f"defense_{defense}" / seed / "defense_results_max1024.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    clean = r.get("clean_accuracy_defended")
    rob8 = None
    for k, v in r.items():
        if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
            if abs((eps255(k) or 0) - 8) < 0.5:
                rob8 = v["robust_accuracy"]["full_robust_accuracy"]
    # collapse: constant clean prediction OR ~0 robust
    kk = [k for k in r if k.startswith("PGD")]
    constant = False
    if kk:
        fr = [x["fraction"] for x in r[kk[0]]["pred_distribution"]["clean"].values()]
        constant = max(fr) >= 0.99
    collapsed = bool(constant or (rob8 is not None and rob8 < 0.02))
    return {"clean": clean, "rob8": rob8, "collapsed": collapsed}


def _standard_clean_rob8(ds, model, seed="seed42"):
    """Standard (undefended) model clean + PGD robust@8 (strong sweep)."""
    clean = None
    cf = ROOT / "results" / ds / model / "clean" / seed / "clean_results.json"
    if cf.exists():
        cj = json.load(open(cf))
        clean = cj.get("accuracy") or cj.get("clean_accuracy") or \
            cj.get("metrics", {}).get("accuracy")
    rob8 = None
    for sec in ("main", "fine"):
        r = _rob_json(ds, model, seed, sec)
        if not r:
            continue
        for k, v in r.items():
            if k.startswith("PGD") and isinstance(v, dict) and "robust_accuracy" in v:
                if abs((eps255(k) or 0) - 8) < 0.5:
                    rob8 = v["robust_accuracy"]["full_robust_accuracy"]
    return {"clean": clean, "rob8": rob8, "collapsed": False}


def defense_methods():
    ds = "chest_xray_pneumonia"
    models = ["resnet18", "resnet50", "resnet152"]
    methods = ["Standard", "PGD-AT", "TRADES", "MART"]
    out = {"dataset": ds, "display": "Chest X-ray", "models": models,
           "display_names": DISPLAY, "methods": methods, "rows": []}
    for m in models:
        for meth in methods:
            rec = _standard_clean_rob8(ds, m) if meth == "Standard" \
                else _defense_clean_rob8(ds, m, meth)
            if rec is None:
                rec = {"clean": None, "rob8": None, "collapsed": False}
            out["rows"].append({"model": m, "method": meth, **rec})
    json.dump(out, open(OUT / "defense_methods.json", "w"), indent=2)
    print("wrote defense_methods.json")


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
    attack_methods()
    print("done -> figures/data/")
