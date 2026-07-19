"""Cross-model comparison tables (three-defense + attack-method + H2 ladder).

Modernized replacement for the stale 2-method defense tables. Reads the SAME
shared data as the figures (figures/data/*.json, written by
scripts/extract_figure_data.py) so tables and figures never disagree.

Emits, per table, a raw CSV, a display-formatted CSV, and a LaTeX file:
  table5_defense_methods    Standard/PGD-AT/TRADES/MART, chest R18/R50/R152
                            (clean, robust@8 full+conditional, ASR, collapse)
  table6_attack_methods     CW/DeepFool mean L2 + AutoAttack/Square robust@8,
                            7 chest models
  table8_h2_at_ladder       5-model AT ladder x 3 datasets (clean, robust@8, collapse)
  table9_at_rescue          original PGD-AT vs PGD-AT-rescue (stability diagnostic,
                            separate protocol -- e.g. OCT ResNet-152 recovers)

Output: figures/paper_tables/<dataset>/ (dataset-level; defense/attack are chest).

Run:  python scripts/generate_comparison_tables.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data"
OUT = ROOT / "figures" / "paper_tables"

DISP = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
        "resnet101": "ResNet-101", "resnet152": "ResNet-152",
        "deit_small": "DeiT-S", "convnext_tiny": "ConvNeXt-T"}


def _pct(v, d=1):
    return "" if v is None or pd.isna(v) else f"{float(v) * 100:.{d}f}"


def _num(v, d=3):
    return "" if v is None or pd.isna(v) else f"{float(v):.{d}f}"


def _save(raw, formatted, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / f"{name}.csv", index=False)
    formatted.to_csv(out_dir / f"{name}_formatted.csv", index=False)
    formatted.to_latex(out_dir / f"{name}.tex", index=False, escape=False)
    return [out_dir / f"{name}{s}" for s in (".csv", "_formatted.csv", ".tex")]


def defense_methods_table():
    d = json.load(open(DATA / "defense_methods.json"))
    rows = []
    for r in d["rows"]:
        rows.append({
            "model": DISP.get(r["model"], r["model"]),
            "method": r["method"],
            "clean_accuracy": r["clean"],
            "robust_acc_pgd8_full": r["rob8"],
            "robust_acc_pgd8_conditional": r.get("rob8_cond"),
            "asr_pgd8": r.get("asr8"),
            "collapsed": r["collapsed"],
        })
    raw = pd.DataFrame(rows)
    fmt = raw.copy()
    for c in ("clean_accuracy", "robust_acc_pgd8_full",
              "robust_acc_pgd8_conditional", "asr_pgd8"):
        fmt[c] = fmt[c].map(lambda v: _pct(v))
    fmt["collapsed"] = fmt["collapsed"].map(lambda b: "yes" if b else "")
    fmt = fmt.rename(columns={
        "clean_accuracy": "Clean (%)", "robust_acc_pgd8_full": "Robust@8 full (%)",
        "robust_acc_pgd8_conditional": "Robust@8 cond. (%)", "asr_pgd8": "ASR@8 (%)",
        "model": "Model", "method": "Method", "collapsed": "Collapsed"})
    return raw, fmt, d.get("display", "Chest X-ray")


def attack_methods_table():
    d = json.load(open(DATA / "attack_methods.json"))
    rows = []
    for r in d["rows"]:
        rows.append({
            "model": DISP.get(r["model"], r["model"]),
            "clean_accuracy": r["clean"],
            "cw_l2_mean": r["CW_l2"],
            "deepfool_l2_mean": r["DeepFool_l2"],
            "autoattack_robust_acc_8": r["AutoAttack8"],
            "square_robust_acc_8": r["Square8"],
        })
    raw = pd.DataFrame(rows)
    fmt = raw.copy()
    fmt["clean_accuracy"] = fmt["clean_accuracy"].map(lambda v: _pct(v))
    fmt["cw_l2_mean"] = fmt["cw_l2_mean"].map(lambda v: _num(v))
    fmt["deepfool_l2_mean"] = fmt["deepfool_l2_mean"].map(lambda v: _num(v))
    fmt["autoattack_robust_acc_8"] = fmt["autoattack_robust_acc_8"].map(lambda v: _pct(v))
    fmt["square_robust_acc_8"] = fmt["square_robust_acc_8"].map(lambda v: _pct(v))
    fmt = fmt.rename(columns={
        "model": "Model", "clean_accuracy": "Clean (%)",
        "cw_l2_mean": "CW L2", "deepfool_l2_mean": "DeepFool L2",
        "autoattack_robust_acc_8": "AutoAttack@8 (%)", "square_robust_acc_8": "Square@8 (%)"})
    return raw, fmt, d.get("display", "Chest X-ray")


def h2_ladder_table():
    d = json.load(open(DATA / "at_ladder_h2.json"))
    disp_ds = {r["dataset"]: r["dataset_display"] for r in d["rows"]}
    rows = []
    for r in d["rows"]:
        rows.append({
            "dataset": disp_ds[r["dataset"]],
            "model": DISP.get(r["model"], r["model"]),
            "params_m": r["params_m"],
            "at_clean_accuracy": r["clean"],
            "at_robust_acc_pgd8": r["robust8"],
            "collapsed": r["collapsed"],
        })
    raw = pd.DataFrame(rows)
    fmt = raw.copy()
    fmt["at_clean_accuracy"] = fmt["at_clean_accuracy"].map(lambda v: _pct(v))
    fmt["at_robust_acc_pgd8"] = fmt["at_robust_acc_pgd8"].map(lambda v: _pct(v))
    fmt["collapsed"] = fmt["collapsed"].map(lambda b: "yes" if b else "")
    fmt = fmt.rename(columns={
        "dataset": "Dataset", "model": "Model", "params_m": "Params (M)",
        "at_clean_accuracy": "AT clean (%)", "at_robust_acc_pgd8": "AT robust@8 (%)",
        "collapsed": "Collapsed"})
    return raw, fmt


def at_rescue_table():
    """Original PGD-AT vs PGD-AT-rescue (stronger stabilisation), separate protocol."""
    d = json.load(open(DATA / "at_rescue.json"))
    rows = []
    for r in d["rows"]:
        rows.append({
            "dataset": r["dataset_display"],
            "model": DISP.get(r["model"], r["model"]),
            "params_m": r["params_m"],
            "orig_clean": r["orig_clean"],
            "orig_robust_pgd8": r["orig_robust8"],
            "orig_collapsed": r["orig_collapsed"],
            "rescue_clean": r["rescue_clean"],
            "rescue_robust_pgd8": r["rescue_robust8"],
            "rescue_collapsed": r["rescue_collapsed"],
        })
    raw = pd.DataFrame(rows)
    fmt = raw.copy()
    for c in ("orig_clean", "orig_robust_pgd8", "rescue_clean", "rescue_robust_pgd8"):
        fmt[c] = fmt[c].map(lambda v: _pct(v))
    for c in ("orig_collapsed", "rescue_collapsed"):
        fmt[c] = fmt[c].map(lambda b: "yes" if b else "")
    fmt = fmt.rename(columns={
        "dataset": "Dataset", "model": "Model", "params_m": "Params (M)",
        "orig_clean": "PGD-AT clean (%)", "orig_robust_pgd8": "PGD-AT robust@8 (%)",
        "orig_collapsed": "PGD-AT collapsed",
        "rescue_clean": "Rescue clean (%)", "rescue_robust_pgd8": "Rescue robust@8 (%)",
        "rescue_collapsed": "Rescue collapsed"})
    return raw, fmt


def main():
    outputs = []
    # defense + attack are chest-only
    draw, dfmt, ds_disp = defense_methods_table()
    araw, afmt, _ = attack_methods_table()
    chest_dir = OUT / "chest_xray_pneumonia"
    outputs += _save(draw, dfmt, chest_dir, "table5_defense_methods")
    outputs += _save(araw, afmt, chest_dir, "table6_attack_methods")
    # H2 ladder spans all datasets
    hraw, hfmt = h2_ladder_table()
    outputs += _save(hraw, hfmt, OUT, "table8_h2_at_ladder")
    # PGD-AT rescue (optimisation-stability diagnostic), separate protocol
    rraw, rfmt = at_rescue_table()
    outputs += _save(rraw, rfmt, OUT, "table9_at_rescue")

    print(f"Generated {len(outputs)} comparison-table files:")
    for p in outputs:
        print(" ", p)


if __name__ == "__main__":
    main()
