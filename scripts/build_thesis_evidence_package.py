#!/usr/bin/env python
"""Build the thesis-facing evidence package from existing experiment outputs.

This script does not train or re-evaluate a model. It:
1. audits and aggregates the existing clean/FGSM/PGD/defence JSON files;
2. applies a conservative attack-set envelope for headline robustness metrics;
3. creates the numbered main-figure package and four thesis tables; and
4. records figure/table provenance and captions.

The conservative envelope is defined per seed and epsilon as:
    cumulative_min_eps(min(FGSM conditional robust accuracy,
                           PGD conditional robust accuracy)).
The cumulative minimum enforces the nesting property of L-infinity threat sets.
Raw attack curves remain available for audit; the envelope is a post-hoc
conservative summary of observed attacks, not a newly executed attack.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_FIG = FIGURES / "thesis_main"
OUT_TAB = FIGURES / "thesis_tables"
OUT_AUDIT = ROOT / "reports" / "thesis_evidence"

DATASETS = {
    "chest_xray_pneumonia": "Chest X-ray",
    "malaria": "Malaria",
    "oct2017": "OCT",
}
DATASET_SHORT = {"chest_xray_pneumonia": "chest", "malaria": "malaria", "oct2017": "oct"}
MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "deit_small": "DeiT-S",
    "convnext_tiny": "ConvNeXt-T",
}
PARAMS_M = {
    "resnet18": 11.7,
    "resnet34": 21.8,
    "resnet50": 25.6,
    "resnet101": 44.5,
    "resnet152": 60.2,
    "deit_small": 22.1,
    "convnext_tiny": 28.6,
}
SEEDS = ["seed42", "seed43", "seed44"]
EPS_GRID = np.array([0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, 4, 8, 16], dtype=float)
COLORS = {
    "resnet18": "#0072B2",
    "resnet34": "#56B4E9",
    "resnet50": "#009E73",
    "resnet101": "#E69F00",
    "resnet152": "#D55E00",
    "deit_small": "#CC79A7",
    "convnext_tiny": "#6B4C9A",
}
MARKERS = {
    "resnet18": "o",
    "resnet34": "s",
    "resnet50": "^",
    "resnet101": "D",
    "resnet152": "P",
    "deit_small": "X",
    "convnext_tiny": "v",
}
ATTACK_RE = re.compile(r"^(FGSM|PGD)_eps=([0-9.]+)$")


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(OUT_FIG / f"{stem}{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mean_sd(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, 0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def _canonical_eps(raw: float) -> float | None:
    eps255 = raw * 255.0
    idx = int(np.argmin(np.abs(EPS_GRID - eps255)))
    if abs(EPS_GRID[idx] - eps255) <= 0.06:
        return float(EPS_GRID[idx])
    return None


def _load_clean_rows(models: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for ds in DATASETS:
        for model in models:
            for seed in SEEDS:
                path = RESULTS / ds / model / "clean" / seed / "clean_results.json"
                if not path.exists():
                    continue
                rec = _read_json(path)
                rows.append(
                    {
                        "dataset": ds,
                        "dataset_label": DATASETS[ds],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "params_m": PARAMS_M[model],
                        "accuracy": rec.get("accuracy"),
                        "balanced_accuracy": rec.get("balanced_accuracy"),
                        "macro_f1": rec.get("macro", {}).get("f1"),
                        "auroc": rec.get("roc_auc"),
                        "ece": rec.get("ece"),
                        "num_samples": rec.get("num_samples"),
                        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                    }
                )
    return pd.DataFrame(rows)


def _load_attack_curves(models: list[str]) -> pd.DataFrame:
    """Return raw conditional robust accuracy for each attack/seed/epsilon."""
    merged: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    sources: dict[tuple[str, str, str, str, float], list[str]] = defaultdict(list)
    for ds in DATASETS:
        for model in models:
            for seed in SEEDS:
                run_dir = RESULTS / ds / model / "robustness" / seed
                for section in ("fine", "main"):
                    path = run_dir / f"robustness_attacks_{section}_max1024.json"
                    if not path.exists():
                        continue
                    data = _read_json(path)
                    for key, rec in data.items():
                        match = ATTACK_RE.match(key)
                        if not match or not isinstance(rec, dict):
                            continue
                        eps = _canonical_eps(float(match.group(2)))
                        if eps is None:
                            continue
                        attack = match.group(1)
                        robust = rec.get("robust_accuracy", {}).get("conditional_robust_accuracy")
                        if robust is None:
                            robust = rec.get("robust_accuracy", {}).get("robust_accuracy")
                        if robust is None:
                            continue
                        idx = (ds, model, seed, attack, eps)
                        merged[idx].append(float(robust))
                        sources[idx].append(str(path.relative_to(ROOT)).replace("\\", "/"))
    rows = []
    for (ds, model, seed, attack, eps), values in merged.items():
        # Duplicate epsilon points can occur in fine and main sweeps. Keeping the
        # lower observation is conservative and avoids arbitrary file precedence.
        rows.append(
            {
                "dataset": ds,
                "dataset_label": DATASETS[ds],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": seed,
                "attack": attack,
                "epsilon_255": eps,
                "conditional_robust_accuracy": min(values),
                "source_count": len(values),
                "sources": ";".join(sources[(ds, model, seed, attack, eps)]),
            }
        )
    return pd.DataFrame(rows)


def _build_envelope(attack_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "model", "seed"]
    for (ds, model, seed), group in attack_df.groupby(group_cols):
        pivot = group.pivot_table(
            index="epsilon_255",
            columns="attack",
            values="conditional_robust_accuracy",
            aggfunc="min",
        ).sort_index()
        pivot = pivot.reindex(EPS_GRID)
        if pivot[["FGSM", "PGD"]].isna().any().any():
            missing = pivot[pivot[["FGSM", "PGD"]].isna().any(axis=1)].index.tolist()
            raise RuntimeError(f"Incomplete FGSM/PGD grid for {ds}/{model}/{seed}: {missing}")
        observed = pivot[["FGSM", "PGD"]].min(axis=1).to_numpy(dtype=float)
        audited = np.minimum.accumulate(observed)
        raw_pgd = pivot["PGD"].to_numpy(dtype=float)
        rebounds = np.maximum(0.0, np.diff(raw_pgd))
        for i, eps in enumerate(EPS_GRID):
            rows.append(
                {
                    "dataset": ds,
                    "dataset_label": DATASETS[ds],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seed": seed,
                    "epsilon_255": float(eps),
                    "fgsm": float(pivot.loc[eps, "FGSM"]),
                    "pgd": float(pivot.loc[eps, "PGD"]),
                    "attack_set_min": float(observed[i]),
                    "audited_envelope": float(audited[i]),
                    "adjusted_for_nonmonotonicity": bool(audited[i] < observed[i] - 1e-12),
                    "max_pgd_rebound": float(rebounds.max()) if rebounds.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _seed_metrics(envelope_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    clean_lookup = clean_df.set_index(["dataset", "model", "seed"])["accuracy"].to_dict()
    rows = []
    x = np.log10(EPS_GRID)
    x_span = float(x[-1] - x[0])
    for (ds, model, seed), group in envelope_df.groupby(["dataset", "model", "seed"]):
        group = group.sort_values("epsilon_255")
        y = group["audited_envelope"].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x=x) / x_span)
        below = np.flatnonzero(y <= 0.5)
        critical = float(EPS_GRID[below[0]]) if below.size else np.nan
        rows.append(
            {
                "dataset": ds,
                "dataset_label": DATASETS[ds],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": seed,
                "params_m": PARAMS_M[model],
                "clean_accuracy": clean_lookup.get((ds, model, seed), np.nan),
                "normalized_log_epsilon_auc": auc,
                "critical_epsilon_255": critical,
                "critical_epsilon_censored_above_16": bool(not below.size),
                "robust_accuracy_0.1_255": float(y[np.where(EPS_GRID == 0.1)[0][0]]),
                "robust_accuracy_8_255": float(y[np.where(EPS_GRID == 8)[0][0]]),
                "relative_accuracy_drop_8_255": float(1.0 - y[np.where(EPS_GRID == 8)[0][0]]),
                "max_pgd_rebound": float(group["max_pgd_rebound"].max()),
                "n_monotonic_adjustments": int(group["adjusted_for_nonmonotonicity"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_core(seed_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "auroc"]
    for ds in DATASETS:
        for model in MODELS:
            rec: dict[str, object] = {
                "dataset": ds,
                "dataset_label": DATASETS[ds],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "params_m": PARAMS_M[model],
            }
            c = clean_df[(clean_df.dataset == ds) & (clean_df.model == model)]
            s = seed_df[(seed_df.dataset == ds) & (seed_df.model == model)]
            for metric in clean_metrics:
                mean, sd, n = _mean_sd(c[metric].tolist())
                rec[f"{metric}_mean"] = mean
                rec[f"{metric}_sd"] = sd
                rec[f"{metric}_n"] = n
            for metric in [
                "normalized_log_epsilon_auc",
                "robust_accuracy_0.1_255",
                "robust_accuracy_8_255",
                "relative_accuracy_drop_8_255",
                "max_pgd_rebound",
            ]:
                mean, sd, n = _mean_sd(s[metric].tolist())
                rec[f"{metric}_mean"] = mean
                rec[f"{metric}_sd"] = sd
                rec[f"{metric}_n"] = n
            finite_crit = s["critical_epsilon_255"].dropna().tolist()
            crit_mean, crit_sd, crit_n = _mean_sd(finite_crit)
            rec["critical_epsilon_255_mean"] = crit_mean
            rec["critical_epsilon_255_sd"] = crit_sd
            rec["critical_epsilon_255_n_observed"] = crit_n
            rec["critical_epsilon_255_n_censored"] = int(s["critical_epsilon_censored_above_16"].sum())
            rec["n_seeds"] = int(len(s))
            rec["n_monotonic_adjustments"] = int(s["n_monotonic_adjustments"].sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def _load_defense_table() -> pd.DataFrame:
    allowed_models = {
        "PGD-AT": set(MODELS),
        "TRADES": {"resnet18", "resnet50", "resnet152"},
        "MART": {"resnet18", "resnet50", "resnet152"},
    }
    raw_rows = []
    for ds in DATASETS:
        for defense, models in allowed_models.items():
            for model in models:
                for seed in SEEDS:
                    path = RESULTS / ds / model / f"defense_{defense}" / seed / "defense_results_max1024.json"
                    if not path.exists():
                        continue
                    data = _read_json(path)
                    clean = data.get("clean_accuracy_defended")
                    robust8 = None
                    constant_clean = False
                    for key, value in data.items():
                        if not (key.startswith("PGD") and isinstance(value, dict)):
                            continue
                        match = re.search(r"eps=([0-9.]+)", key)
                        if not match or abs(float(match.group(1)) * 255 - 8.0) > 0.5:
                            continue
                        robust8 = value.get("robust_accuracy", {}).get("full_robust_accuracy")
                        dist = value.get("pred_distribution", {}).get("clean", {})
                        fractions = [
                            v.get("fraction", 0.0)
                            for v in dist.values()
                            if isinstance(v, dict)
                        ]
                        constant_clean = bool(fractions and max(fractions) >= 0.99)
                        break
                    if robust8 is None:
                        continue
                    attack_did_nothing = clean is not None and abs(float(robust8) - float(clean)) < 0.005
                    if constant_clean or attack_did_nothing or float(robust8) < 0.05:
                        regime = "collapsed"
                    elif float(robust8) > 0.50:
                        regime = "success"
                    else:
                        regime = "partial"
                    raw_rows.append(
                        {
                            "dataset": ds,
                            "model": model,
                            "defense": defense,
                            "seed": seed,
                            "clean": float(clean),
                            "robust8": float(robust8),
                            "regime": regime,
                            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                        }
                    )
    raw = pd.DataFrame(raw_rows)
    rows = []
    for (ds, model, defense), group in raw.groupby(["dataset", "model", "defense"]):
        success = group[group.regime == "success"]
        clean_mean, clean_sd, n_all = _mean_sd(group.clean.tolist())
        rob_mean, rob_sd, _ = _mean_sd(group.robust8.tolist())
        s_clean_mean, s_clean_sd, n_success = _mean_sd(success.clean.tolist())
        s_rob_mean, s_rob_sd, _ = _mean_sd(success.robust8.tolist())
        n_collapsed = int((group.regime == "collapsed").sum())
        n_partial = int((group.regime == "partial").sum())
        status = f"{n_success} success; {n_partial} partial; {n_collapsed} collapsed"
        rows.append(
            {
                "dataset": ds,
                "dataset_label": DATASETS[ds],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "defense": defense,
                "n_seeds": n_all,
                "n_success": n_success,
                "n_success_over_3": f"{n_success}/{n_all}",
                "n_partial": n_partial,
                "n_collapsed": n_collapsed,
                "collapse_status": status,
                "seed_regimes": ";".join(
                    f"{row.seed}:{row.regime}" for row in group.sort_values("seed").itertuples()
                ),
                "clean_accuracy_all_mean": clean_mean,
                "clean_accuracy_all_sd": clean_sd,
                "robust_accuracy_8_255_all_mean": rob_mean,
                "robust_accuracy_8_255_all_sd": rob_sd,
                "clean_robust_gap_all": clean_mean - rob_mean,
                "clean_accuracy_success_mean": s_clean_mean,
                "clean_accuracy_success_sd": s_clean_sd,
                "robust_accuracy_8_255_success_mean": s_rob_mean,
                "robust_accuracy_8_255_success_sd": s_rob_sd,
                "clean_robust_gap_success": s_clean_mean - s_rob_mean if n_success else np.nan,
            }
        )
    order = pd.CategoricalDtype(["PGD-AT", "TRADES", "MART"], ordered=True)
    out = pd.DataFrame(rows)
    out["defense"] = out["defense"].astype(order)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODELS)})
    return out.sort_values(["dataset", "model_order", "defense"]).drop(columns="model_order")


def _write_tables(core: pd.DataFrame, defenses: pd.DataFrame) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    dataset_table = pd.DataFrame(
        [
            {
                "dataset": "Chest X-ray",
                "source": "Kermany paediatric chest X-ray collection (official split)",
                "classes": "NORMAL; PNEUMONIA",
                "n_classes": 2,
                "train": 4634,
                "validation": 514,
                "test": 624,
                "split_note": "Official train/test; stratified 10% validation carve from train",
            },
            {
                "dataset": "Malaria",
                "source": "NIH/NLM malaria cell-image collection",
                "classes": "Parasitized; Uninfected",
                "n_classes": 2,
                "train": 20099,
                "validation": 2342,
                "test": 5117,
                # The seed-42 sizes alone would misread as fixed. Patients are
                # assigned by seed and contribute unequal numbers of cells, so
                # the other two partitions are recorded here as well.
                "split_note": (
                    "Patient-group split; no patient shared across partitions; sizes shown "
                    "are seed 42 — partition sizes are seed-dependent "
                    "(seed 43: 19038/2049/6471; seed 44: 20358/2274/4926; "
                    "27558 images in total at every seed)"
                ),
            },
            {
                "dataset": "OCT2017",
                "source": "Kermany OCT2017 collection",
                "classes": "CNV; DME; DRUSEN; NORMAL",
                "n_classes": 4,
                "train": 15028,
                "validation": 8349,
                "test": 968,
                "split_note": "Official test; stratified validation carve; fixed 20% train subsample",
            },
        ]
    )
    dataset_table.to_csv(OUT_TAB / "table01_datasets.csv", index=False)

    model_table = pd.DataFrame(
        [
            {
                "component": "Model",
                "setting": "ResNet-18 / 34 / 50 / 101 / 152",
                "details": "11.7 / 21.8 / 25.6 / 44.5 / 60.2 M parameters; ImageNet-1k initialisation",
            },
            {
                "component": "Cross-architecture extension",
                "setting": "DeiT-S; ConvNeXt-T",
                "details": "22.1 M; 28.6 M parameters; Chest X-ray case study only",
            },
            {
                "component": "Standard training",
                "setting": "Adam optimiser; cosine schedule; LR 1e-4; WD 1e-4",
                "details": "20 / 15 / 10 epochs for Chest X-ray / Malaria / OCT; three seeds",
            },
            {
                "component": "Core attacks",
                "setting": "FGSM and PGD-20, L-infinity",
                "details": "epsilon={0.1,0.15,0.2,0.25,0.5,1,2,4,8,16}/255; PGD one random start",
            },
            {
                "component": "Attack audit",
                "setting": "Per-epsilon minimum then cumulative-minimum envelope",
                "details": "Used for headline AUC/critical epsilon; raw PGD retained for audit",
            },
            {
                "component": "Defences (held fixed)",
                "setting": "PGD-AT; TRADES; MART",
                "details": "Training epsilon=8/255; eps_step=2/255; PGD-7 inner loop; beta=6 for TRADES/MART",
            },
            {
                "component": "Defences (varies by method)",
                "setting": "Batch size; warm-up schedule",
                "details": (
                    "Batch 16/32/32 for PGD-AT and 8/16/16 for TRADES and MART on "
                    "Chest X-ray/Malaria/OCT; eps+LR warm-up of 5/3 epochs for PGD-AT "
                    "and MART, none for TRADES; defence epochs 20/15/20"
                ),
            },
            {
                "component": "Defence evaluation",
                "setting": "PGD-50, five restarts, L-infinity",
                "details": "Headline robust accuracy at 8/255; all seeds retained, including collapse",
            },
        ]
    )
    model_table.to_csv(OUT_TAB / "table02_protocol.csv", index=False)

    cols = [
        "dataset_label",
        "model_label",
        "params_m",
        "accuracy_mean",
        "accuracy_sd",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "auroc_mean",
        "normalized_log_epsilon_auc_mean",
        "normalized_log_epsilon_auc_sd",
        "critical_epsilon_255_mean",
        "critical_epsilon_255_n_observed",
        "critical_epsilon_255_n_censored",
        "robust_accuracy_0.1_255_mean",
        "robust_accuracy_8_255_mean",
        "relative_accuracy_drop_8_255_mean",
        "n_seeds",
        "n_monotonic_adjustments",
    ]
    core[cols].to_csv(OUT_TAB / "table03_core_metrics.csv", index=False)
    defenses.to_csv(OUT_TAB / "table04_defenses.csv", index=False)

    readme = """# Thesis main tables

The file number and the dissertation table number are not the same, because
dissertation Table 4 is the AutoAttack audit and is not rendered here.

| File | Dissertation table | Contents |
|---|---|---|
| `table01_datasets.csv` | Table 1 | Effective partitions used by the experiments |
| `table02_protocol.csv` | Table 2 | Model, attack, audit, and defence settings |
| `table03_core_metrics.csv` | Table 3 | Three-seed clean and audited robustness summary |
| `table04_defenses.csv` | Table 5 | All-seed and success-only defence summaries |
| — | Table 4 | Grouped from `reports/thesis_evidence/autoattack_audit.csv`; see the "AutoAttack audit" section of `analysis_note.md` |

`robust_accuracy_8_255_all_mean` is full adversarial accuracy, so it is directly
comparable to clean accuracy. `n_success_over_3` and `collapse_status` must always
be shown beside any success-only value. Critical epsilon is right-censored when
the audited conditional robust accuracy does not fall to 0.5 by 16/255.
"""
    (OUT_TAB / "README.md").write_text(readme, encoding="utf-8")


def _plot_clean(core: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    for ax, (ds, label) in zip(axes, DATASETS.items()):
        data = core[core.dataset == ds].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        y = data["accuracy_mean"].to_numpy() * 100
        err = data["accuracy_sd"].to_numpy() * 100
        # The connecting line is a reading guide, not a series, so it stays neutral.
        # Identity lives on the markers, coloured by model with the same palette used
        # in Figures 6 to 8, so that colour means "model" everywhere in the document
        # rather than one thing here and another there. The x labels repeat the
        # identity, so nothing depends on colour alone.
        ax.plot(x, y, color="#9AA0A6", linewidth=1.1, zorder=1)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#5F6368",
                    elinewidth=0.9, capsize=2.3, zorder=2)
        for xi, yi, model in zip(x, y, MODELS):
            ax.plot(xi, yi, marker=MARKERS[model], markersize=5.2,
                    color=COLORS[model], markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3)
        ax.set_title(label)
        ax.set_xticks(x, [m.replace("resnet", "R") for m in MODELS], rotation=35)
        ax.set_ylim(45, 101)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Clean accuracy (%)")
    fig.supxlabel("ResNet depth (mean ± SD, n=3)", y=0.00, fontsize=8.5)
    fig.tight_layout()
    _save_figure(fig, "fig04_clean_performance")


def _figure_envelope_construction(envelope: pd.DataFrame) -> None:
    """Show how the audited envelope of Section 3.4 is built, on one run.

    Chest X-ray ResNet-50 at seed 42 is chosen because it is the pathological case,
    not a typical one: it carries the largest raw PGD rebound in the whole core
    sweep. That makes it the run on which each step of the construction is visible.
    """
    run = envelope[(envelope.dataset == "chest_xray_pneumonia")
                   & (envelope.model == "resnet50")
                   & (envelope.seed == "seed42")].sort_values("epsilon_255")
    eps = run["epsilon_255"].to_numpy()
    fgsm = run["fgsm"].to_numpy() * 100
    pgd = run["pgd"].to_numpy() * 100
    amin = run["attack_set_min"].to_numpy() * 100
    env = run["audited_envelope"].to_numpy() * 100
    adj = run["adjusted_for_nonmonotonicity"].to_numpy().astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True)

    ax = axes[0]
    ax.plot(eps, fgsm, color="#E69F00", marker="s", markersize=3.5, label="Raw FGSM")
    ax.plot(eps, pgd, color="#0072B2", marker="o", markersize=3.5, label="Raw PGD-20")
    # Drawn thicker and dashed underneath PGD: the two coincide at every budget,
    # which is the point the panel is making.
    ax.plot(eps, amin, color="#111111", linestyle="--", linewidth=2.4, alpha=0.55,
            zorder=1, label="Attack-set minimum")
    ax.set_title("a  Per-budget minimum of the two attacks", loc="left")
    ax.legend(frameon=False, loc="center left")

    ax = axes[1]
    ax.plot(eps, amin, color="#111111", linestyle="--", linewidth=1.6,
            marker="o", markersize=3.2, label="Attack-set minimum")
    ax.plot(eps, env, color="#009E73", linewidth=2.0, label="Audited envelope")
    if adj.any():
        ax.scatter(eps[adj], amin[adj], s=70, facecolors="none", edgecolors="#D55E00",
                   linewidths=1.4, zorder=5, label="Corrected by the audit")
        for e, hi, lo in zip(eps[adj], amin[adj], env[adj]):
            ax.annotate("", xy=(e, lo), xytext=(e, hi),
                        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.1))
    ax.set_title("b  Cumulative minimum over increasing budget", loc="left")
    ax.legend(frameon=False, loc="center left")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(-4, 104)
        ax.set_xticks([0.1, 1, 8, 16], ["0.1", "1", "8", "16"])
        ax.grid(color="#E0E0E0", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Conditional robust accuracy (%)")
    fig.supxlabel(r"Perturbation budget, $\epsilon$ (/255)", y=0.00, fontsize=8.5)
    fig.tight_layout()
    _save_figure(fig, "fig03_audited_envelope_construction")


def _plot_curves(envelope: pd.DataFrame, kind: str, stem: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, sharey=True)
    for ax, (ds, label) in zip(axes, DATASETS.items()):
        for model in MODELS:
            data = envelope[(envelope.dataset == ds) & (envelope.model == model)]
            summary = data.groupby("epsilon_255")[kind].agg(["mean", "std", "count"]).reindex(EPS_GRID)
            mean = summary["mean"].to_numpy() * 100
            # 95% t approximation for n=3, used as a descriptive uncertainty band.
            ci = 4.303 * summary["std"].fillna(0).to_numpy() / np.sqrt(summary["count"].clip(lower=1)) * 100
            ax.plot(
                EPS_GRID,
                mean,
                label=MODEL_LABELS[model],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=3.5,
            )
            ax.fill_between(EPS_GRID, np.maximum(0, mean - ci), np.minimum(100, mean + ci), color=COLORS[model], alpha=0.10)
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_ylim(-2, 102)
        ax.set_xticks([0.1, 1, 8, 16], ["0.1", "1", "8", "16"])
        ax.grid(color="#E0E0E0", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    fig.supxlabel(r"Perturbation budget, $\epsilon$ (/255)", y=0.00, fontsize=8.5)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _save_figure(fig, stem)


def _plot_complexity(core: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), sharex="col")
    for col, (ds, label) in enumerate(DATASETS.items()):
        data = core[core.dataset == ds].set_index("model").loc[MODELS]
        x = data["params_m"].to_numpy()
        y_auc = data["normalized_log_epsilon_auc_mean"].to_numpy()
        axes[0, col].plot(x, y_auc, color="#333333", linewidth=0.8, zorder=1)
        axes[0, col].scatter(x, y_auc, c=[COLORS[m] for m in MODELS], s=32, zorder=2)
        rho_auc = pd.Series(x).corr(pd.Series(y_auc), method="spearman")
        axes[0, col].text(0.04, 0.94, f"Spearman ρ={rho_auc:.2f}", transform=axes[0, col].transAxes, va="top", fontsize=7.4)
        axes[0, col].set_title(label)
        crit_y = []
        censored = []
        for _, row in data.iterrows():
            if row["critical_epsilon_255_n_observed"] > 0:
                crit_y.append(row["critical_epsilon_255_mean"])
                censored.append(False)
            else:
                crit_y.append(16.0)
                censored.append(True)
        crit_y = np.asarray(crit_y, dtype=float)
        axes[1, col].plot(x, crit_y, color="#333333", linewidth=0.8, zorder=1)
        for i, model in enumerate(MODELS):
            axes[1, col].scatter(
                x[i],
                crit_y[i],
                c=COLORS[model],
                s=38,
                marker="^" if censored[i] else "o",
                zorder=2,
            )
        rho_crit = pd.Series(x).corr(pd.Series(crit_y), method="spearman")
        axes[1, col].text(0.04, 0.94, f"Spearman ρ={rho_crit:.2f}", transform=axes[1, col].transAxes, va="top", fontsize=7.4)
        for row in range(2):
            axes[row, col].grid(color="#E0E0E0", linewidth=0.5)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Normalised robustness AUC")
    axes[1, 0].set_ylabel(r"Critical $\epsilon$ (/255)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_yticks([0.1, 1, 8, 16], ["0.1", "1", "8", "16 or >16"])
    for ax in axes[1]:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 1, 8, 16], ["0.1", "1", "8", "16 or >16"])
        ax.set_xlabel("Parameters (M)")
    fig.text(0.5, 0.005, "Triangle: threshold not reached by 16/255 (right-censored)", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, "fig08_complexity_vs_robustness")


def _plot_cross_arch(clean_df: pd.DataFrame, attack_df: pd.DataFrame) -> None:
    models = MODELS + ["deit_small", "convnext_tiny"]
    chest_clean = clean_df[clean_df.dataset == "chest_xray_pneumonia"]
    chest_attack = attack_df[attack_df.dataset == "chest_xray_pneumonia"]
    envelope = _build_envelope(chest_attack)
    seed = _seed_metrics(envelope, chest_clean)
    rows = []
    for model in models:
        c = chest_clean[chest_clean.model == model]
        s = seed[seed.model == model]
        clean_mean, clean_sd, n = _mean_sd(c.accuracy.tolist())
        auc_mean, auc_sd, _ = _mean_sd(s.normalized_log_epsilon_auc.tolist())
        rob8_mean, rob8_sd, _ = _mean_sd(s.robust_accuracy_8_255.tolist())
        rows.append(
            {
                "model": model,
                "clean_mean": clean_mean,
                "clean_sd": clean_sd,
                "auc_mean": auc_mean,
                "auc_sd": auc_sd,
                "rob8_mean": rob8_mean,
                "rob8_sd": rob8_sd,
                "n": n,
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(OUT_AUDIT / "chest_cross_architecture_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    x = np.arange(len(data))
    for ax, metric, err, ylabel in [
        (axes[0], "clean_mean", "clean_sd", "Clean accuracy (%)"),
        (axes[1], "auc_mean", "auc_sd", "Normalised robustness AUC"),
    ]:
        values = data[metric].to_numpy() * (100 if metric == "clean_mean" else 1)
        errors = data[err].to_numpy() * (100 if metric == "clean_mean" else 1)
        ax.bar(x, values, yerr=errors, capsize=2.2, color=[COLORS[m] for m in data.model], edgecolor="#333333", linewidth=0.45)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in data.model], rotation=42, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvspan(4.5, 6.5, color="#F4F0F7", zorder=-2)
    axes[1].axvspan(4.5, 6.5, color="#F4F0F7", zorder=-2)
    for ax in axes:
        ax.axvline(4.5, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_title("Clean performance", pad=8)
    axes[1].set_title("Audited robustness", pad=8)
    fig.text(0.29, 0.965, "ResNet depth ladder", ha="center", fontsize=7.5, color="#444444")
    fig.text(0.83, 0.965, "Chest-only architecture extension", ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout()
    _save_figure(fig, "fig09_chest_cross_architecture")


def _plot_attack_audit() -> None:
    data = _read_json(FIGURES / "data" / "attack_methods.json")
    rows = pd.DataFrame(data["rows"])
    rows["label"] = rows["model"].map(MODEL_LABELS)
    rows.to_csv(OUT_AUDIT / "chest_attack_audit_seed42.csv", index=False)
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    axes[0].bar(x - width / 2, rows["AutoAttack8"] * 100, width, label="AutoAttack", color="#0072B2")
    axes[0].bar(x + width / 2, rows["Square8"] * 100, width, label="Square Attack", color="#E69F00")
    axes[0].set_ylabel("Conditional robust accuracy (%)")
    axes[0].set_title(r"Bounded attacks at $\epsilon=8/255$")
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, rows["CW_l2"], width, label="CW", color="#009E73")
    axes[1].bar(x + width / 2, rows["DeepFool_l2"], width, label="DeepFool", color="#CC79A7")
    axes[1].set_ylabel(r"Mean minimal perturbation ($L_2$)")
    axes[1].set_title("Minimal-perturbation attacks")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, rows["label"], rotation=42, ha="right")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "Chest X-ray case study, seed 42; panels use different metrics and must not be pooled.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, "fig10_chest_multi_attack_audit")


def _copy_existing_numbered() -> list[dict]:
    specs = [
        (
            "fig01_dataset_class_overview",
            FIGURES / "qualitative" / "dataset_class_overview",
            "Reused dataset/class overview generated from the local evaluation datasets.",
        ),
        (
            "fig05_qualitative_attack_examples",
            FIGURES / "qualitative" / "qualitative_attack_examples",
            "Reused clean–perturbation–adversarial examples for the three datasets.",
        ),
        (
            "fig11_chest_defense_comparison",
            FIGURES / "main" / "defense_methods",
            "Reused Chest X-ray defence comparison.",
        ),
        (
            "fig12_malaria_defense_comparison",
            FIGURES / "main" / "defense_methods_malaria",
            "Reused Malaria defence comparison.",
        ),
        (
            "fig13_oct_defense_comparison",
            FIGURES / "main" / "defense_methods_oct2017",
            "Reused OCT defence comparison.",
        ),
    ]
    records = []
    for out_stem, source_stem, note in specs:
        copied = []
        for suffix in (".png", ".pdf", ".svg"):
            src = source_stem.with_suffix(suffix)
            if not src.exists():
                continue
            dst = OUT_FIG / f"{out_stem}{suffix}"
            shutil.copy2(src, dst)
            copied.append(str(src.relative_to(ROOT)).replace("\\", "/"))
        if not copied:
            raise FileNotFoundError(f"No source artifacts found for {source_stem}")
        records.append({"figure": out_stem, "sources": copied, "note": note})
    return records


def _plot_interpretation_composite() -> None:
    grad_paths = [
        FIGURES / "gradcam" / "chest_xray_pneumonia" / "resnet50" / "sample_000.png",
        FIGURES / "gradcam" / "malaria" / "resnet18" / "sample_000.png",
        FIGURES / "gradcam" / "oct2017" / "resnet18" / "sample_000.png",
    ]
    boundary_paths = [
        FIGURES / "decision_boundary" / "chest_xray_pneumonia" / "decision_boundary_standard.png",
        FIGURES / "decision_boundary" / "malaria" / "decision_boundary_standard.png",
        FIGURES / "decision_boundary" / "oct2017" / "decision_boundary_standard.png",
    ]
    for path in grad_paths + boundary_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], hspace=0.15, wspace=0.04)
    for col, (label, path) in enumerate(zip(DATASETS.values(), grad_paths)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(label, fontsize=9.5, pad=4)
        if col == 0:
            ax.text(-0.05, 1.02, "a", transform=ax.transAxes, weight="bold", fontsize=11)
    for col, path in enumerate(boundary_paths):
        ax = fig.add_subplot(gs[1, col])
        image = mpimg.imread(path)
        # Each decision-region source contains two model panels. Retain the
        # left representative panel so symbols and regions remain readable.
        image = image[:, : image.shape[1] // 2]
        ax.imshow(image)
        ax.axis("off")
        if col == 0:
            ax.text(-0.05, 1.02, "b", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.text(
        0.5,
        0.015,
        "Penultimate-layer 2D feature projections with surrogate decision regions; not the true high-dimensional boundary.",
        ha="center",
        fontsize=7.5,
    )
    _save_figure(fig, "fig14_interpretation_composite")


def _captions() -> dict[str, str]:
    return {
        "fig01_dataset_class_overview": "Representative classes from the Chest X-ray, Malaria, and OCT2017 evaluation datasets. This descriptive overview establishes modality and label-space differences and is not used as quantitative evidence.",
        "fig02_evaluation_workflow_unified": "Unified evaluation workflow. The cross-dataset core comprises three imaging datasets, a five-depth ResNet ladder, matched FGSM/PGD budgets, audited robustness metrics, and three adversarial-training methods. DeiT-S, ConvNeXt-T, and the multi-attack suite are Chest X-ray case studies.",
        # Figures are numbered in order of appearance in the dissertation: clean
        # baselines (§4.1) precede the attack landscape (§4.2). The blueprint's
        # register grouped them by topic instead, which numbered the attack
        # examples third even though they appear second; the thesis ordering wins.
        "fig03_audited_envelope_construction": "Construction of the audited attack-set envelope, illustrated on Chest X-ray ResNet-50 at seed 42. Panel a shows the raw FGSM and PGD-20 curves with their per-budget minimum, which coincides with PGD at every budget in this run. Panel b applies the cumulative minimum over increasing budget, which is the step that removes the high-budget rebound. This run is an illustrative pathological example chosen because it carries the largest raw PGD rebound of the core sweep, not a representative one.",
        "fig04_clean_performance": "Clean test accuracy of the five ResNet depths across datasets. Points show the three-seed mean and error bars show one standard deviation.",
        "fig05_qualitative_attack_examples": "Representative clean images, scaled perturbations, and PGD adversarial images for the three datasets. Perturbations are amplified for visibility; quantitative conclusions rely on the attack sweeps rather than visual salience.",
        "fig06_fgsm_robustness_curves": "Observed FGSM conditional robust accuracy across perturbation budgets. Lines show three-seed means and shaded bands show descriptive 95% confidence intervals.",
        "fig07_audited_attack_envelope": "Conservative attack-ensemble robustness curves. For each seed and budget, the lower of FGSM and PGD conditional robust accuracy is retained, followed by a cumulative minimum over increasing budgets. This prevents high-budget PGD optimisation failures from being interpreted as recovered robustness.",
        "fig08_complexity_vs_robustness": "Model parameters versus normalised log-epsilon robustness AUC and the critical perturbation budget at which audited conditional robust accuracy first falls to 0.5. Spearman correlations are descriptive; upward triangles denote right-censoring above 16/255.",
        "fig09_chest_cross_architecture": "Chest X-ray cross-architecture extension. Clean accuracy and audited robustness AUC are compared for the ResNet ladder, DeiT-S, and ConvNeXt-T. This single-dataset extension is a case study and is not pooled with the cross-dataset ResNet analysis.",
        "fig10_chest_multi_attack_audit": "Chest X-ray multi-attack audit for seed 42. Bounded-attack robust accuracy and CW/DeepFool minimal L2 perturbations are shown in separate panels because the metrics are not directly comparable.",
        "fig11_chest_defense_comparison": "Chest X-ray defence comparison for representative ResNet depths. All-seed estimates retain collapsed runs; successful-seed summaries must be accompanied by the success count.",
        "fig12_malaria_defense_comparison": "Malaria defence comparison for representative ResNet depths under the same reporting rule.",
        "fig13_oct_defense_comparison": "OCT defence comparison for representative ResNet depths under the same reporting rule.",
        "fig14_interpretation_composite": "Interpretive visualisations. Grad-CAM illustrates representative attention shifts, while two-dimensional penultimate-layer feature projections are overlaid with surrogate decision regions. Neither component is used to prove quantitative superiority or to claim recovery of the true high-dimensional decision boundary.",
    }


def _write_manifest(reused: list[dict], audit_stats: dict) -> None:
    captions = _captions()
    generated_sources = {
        "fig02_evaluation_workflow_unified": ["figures/thesis_main/fig02_evaluation_workflow_unified.drawio"],
        "fig03_audited_envelope_construction": ["reports/thesis_evidence/audited_envelope_seed.csv"],
        "fig04_clean_performance": ["reports/thesis_evidence/core_model_summary.csv"],
        "fig06_fgsm_robustness_curves": ["reports/thesis_evidence/attack_curve_seed.csv"],
        "fig07_audited_attack_envelope": ["reports/thesis_evidence/audited_envelope_seed.csv"],
        "fig08_complexity_vs_robustness": ["reports/thesis_evidence/core_model_summary.csv"],
        "fig09_chest_cross_architecture": ["reports/thesis_evidence/chest_cross_architecture_summary.csv"],
        "fig10_chest_multi_attack_audit": ["figures/data/attack_methods.json"],
        "fig14_interpretation_composite": [
            "figures/gradcam/*/resnet*/sample_000.png",
            "figures/decision_boundary/*/decision_boundary_standard.png",
        ],
    }
    reused_lookup = {r["figure"]: r for r in reused}
    records = []
    for i in range(1, 15):
        prefix = f"fig{i:02d}_"
        stem = next(k for k in captions if k.startswith(prefix))
        source = reused_lookup.get(stem, {}).get("sources", generated_sources.get(stem, []))
        records.append(
            {
                # Thesis Figures 1-2 are reproduced from the literature and sit
                # outside this package, so the generated figures are numbered 3-16.
                "number": i + 2,
                "stem": stem,
                "caption": captions[stem],
                "sources": source,
                "formats": sorted(
                    p.name for p in OUT_FIG.glob(f"{stem}.*")
                    if p.suffix in (".drawio", ".png", ".pdf", ".svg")
                ),
            }
        )
    _write_json(
        OUT_FIG / "figure_manifest.json",
        {
            "title": "A Systematic Evaluation of Adversarial Robustness in Medical Image Classification: Model Complexity, Imaging Modality, and Adversarial Training",
            "audit_rule": "per-seed min(FGSM, PGD) at each epsilon, followed by cumulative minimum over epsilon",
            "audit_statistics": audit_stats,
            "figures": records,
        },
    )
    lines = ["# Main figure legends", ""]
    for rec in records:
        lines.extend([f"## Figure {rec['number']}", "", rec["caption"], ""])
    (OUT_FIG / "figure_legends.md").write_text("\n".join(lines), encoding="utf-8")


def _autoattack_audit_note(defenses: pd.DataFrame) -> list[str]:
    """Regime-grouped summary of the AutoAttack audit, and its coverage gaps.

    Table 4 of the dissertation is this grouping. It is derived here rather than
    kept by hand so that the regime labels always come from the same classifier
    that produced ``defense_summary.csv``: an earlier hand-maintained version of
    the table split off a "partial" row for Chest X-ray ResNet-50 seed 44, a cell
    the classifier in fact assigns to collapse on the constant-prediction rule.

    ``pgd50_robust_accuracy_8_255_published`` is recovered by the audit script
    from the stored defence JSONs. Where no such run exists on disk the field
    stays empty; it is never inferred from the audit's own subset measurement.
    """
    audit_csv = OUT_AUDIT / "autoattack_audit.csv"
    if not audit_csv.exists():
        return ["## AutoAttack audit", "", "`autoattack_audit.csv` is not present.", ""]
    audit = pd.read_csv(audit_csv)

    regime: dict[tuple, str] = {}
    for _, row in defenses.iterrows():
        for token in str(row.seed_regimes).split(";"):
            seed, label = token.split(":")
            regime[(row.dataset, row.model, row.defense, int(seed[4:]))] = label
    audit["regime"] = [
        regime.get((r.dataset, r.model, r.method, r.seed)) for _, r in audit.iterrows()
    ]
    ladder = audit[audit.regime.notna()].copy()
    ladder["reduction"] = (
        ladder.pgd50_robust_accuracy_8_255_same_subset
        - ladder.autoattack_robust_accuracy_8_255
    )

    lines = [
        "## AutoAttack audit",
        "",
        f"Audited cells: {len(audit)} ({len(ladder)} in the defence tables, "
        f"{len(audit) - len(ladder)} on case-study backbones reported separately).",
        "",
        "| Regime | Cells | AutoAttack lower | Largest reduction | Mean RA under AutoAttack |",
        "|---|---|---|---|---|",
    ]
    for label in ["success", "partial", "collapsed"]:
        group = ladder[ladder.regime == label]
        if group.empty:
            continue
        lower = int(
            (
                group.autoattack_robust_accuracy_8_255
                < group.pgd50_robust_accuracy_8_255_same_subset
            ).sum()
        )
        lines.append(
            f"| {label} | {len(group)} | {lower} | {group.reduction.max():.3f} | "
            f"{group.autoattack_robust_accuracy_8_255.mean():.3f} |"
        )

    unmatched = audit[audit.pgd50_robust_accuracy_8_255_published.isna()]
    lines.append("")
    if unmatched.empty:
        lines.append("Every audited cell has a published PGD-50 figure to compare against.")
    else:
        lines.append(
            f"{len(unmatched)} audited cell(s) carry no published PGD-50 figure. The audit "
            "script reads that field from `results/<dataset>/<model>/defense_<method>/"
            "seed<N>/defense_results*.json`; where no such run was ever stored the field is "
            "left empty rather than filled from the audit's own 256-image subset, which has "
            "a different denominator. Affected cells:"
        )
        for _, row in unmatched.iterrows():
            lines.append(
                f"- `{row.dataset}/{row.model}/{row.method}/seed{row.seed}`: "
                "no stored defence evaluation under the published protocol."
            )
    return lines


def _write_analysis_note(core: pd.DataFrame, defenses: pd.DataFrame, seed_metrics: pd.DataFrame) -> None:
    lines = [
        "# Thesis evidence audit",
        "",
        "## Headline interpretation rule",
        "",
        "The headline robustness metric is the per-seed minimum of FGSM and PGD conditional robust accuracy at each epsilon, followed by a cumulative minimum over increasing epsilon. The raw PGD values remain in `attack_curve_seed.csv`. A high-budget rebound is treated as an attack-optimisation warning, not recovered robustness.",
        "",
        "## Descriptive findings",
        "",
    ]
    for ds, label in DATASETS.items():
        data = core[core.dataset == ds]
        rho = data["params_m"].corr(data["normalized_log_epsilon_auc_mean"], method="spearman")
        best = data.loc[data["normalized_log_epsilon_auc_mean"].idxmax()]
        lines.append(
            f"- **{label}:** parameter count versus robustness AUC has Spearman rho={rho:.2f}; "
            f"the highest mean audited AUC is observed for {best['model_label']} "
            f"({best['normalized_log_epsilon_auc_mean']:.3f})."
        )
    n_adjust = int(seed_metrics["n_monotonic_adjustments"].sum())
    affected = int((seed_metrics["n_monotonic_adjustments"] > 0).sum())
    max_rebound = float(seed_metrics["max_pgd_rebound"].max())
    lines.extend(
        [
            "",
            f"- The strict audit adjusted {n_adjust} seed-budget observations across {affected}/45 core runs; the largest raw PGD rebound was {max_rebound:.3f}.",
            "- These descriptive results support a dataset- and budget-dependent account; they do not support a universal monotonic relationship between model size and robustness.",
            "",
            "## Defence reporting",
            "",
        ]
    )
    for defense in ["PGD-AT", "TRADES", "MART"]:
        d = defenses[defenses.defense == defense]
        lines.append(
            f"- **{defense}:** {int(d.n_success.sum())}/{int(d.n_seeds.sum())} evaluated seeds met the non-collapse criterion in the planned table cells."
        )
    lines.extend(
        [
            "",
            "Success-only defence means are diagnostic supplements. The primary table retains every seed and reports `n_success/3` and collapse status beside any success-only value.",
            "",
        ]
    )
    lines.extend(_autoattack_audit_note(defenses))
    lines.extend(
        [
            "",
            "## Scope controls",
            "",
            "- Cross-dataset claims use only the complete three-dataset ResNet matrix.",
            "- DeiT-S, ConvNeXt-T, AutoAttack, Square Attack, CW, and DeepFool are labelled Chest X-ray case studies.",
            "- Grad-CAM and 2D projections are explanatory observations only.",
        ]
    )
    (OUT_AUDIT / "analysis_note.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _setup_style()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)

    all_models = MODELS + ["deit_small", "convnext_tiny"]
    clean_df = _load_clean_rows(all_models)
    attack_df = _load_attack_curves(all_models)
    core_attack = attack_df[attack_df.model.isin(MODELS)].copy()
    core_envelope = _build_envelope(core_attack)
    core_clean = clean_df[clean_df.model.isin(MODELS)].copy()
    seed_metrics = _seed_metrics(core_envelope, core_clean)
    core_summary = _summarize_core(seed_metrics, core_clean)
    defenses = _load_defense_table()

    clean_df.to_csv(OUT_AUDIT / "clean_metrics_seed.csv", index=False)
    attack_df.to_csv(OUT_AUDIT / "attack_curve_seed.csv", index=False)
    core_envelope.to_csv(OUT_AUDIT / "audited_envelope_seed.csv", index=False)
    seed_metrics.to_csv(OUT_AUDIT / "core_seed_metrics.csv", index=False)
    core_summary.to_csv(OUT_AUDIT / "core_model_summary.csv", index=False)
    defenses.to_csv(OUT_AUDIT / "defense_summary.csv", index=False)

    _write_tables(core_summary, defenses)
    reused = _copy_existing_numbered()
    # Figure 4 (workflow) is authored in diagrams.net, not generated here; the
    # .drawio source and its exports live in figures/thesis_main/.
    if not (OUT_FIG / "fig02_evaluation_workflow_unified.png").exists():
        raise FileNotFoundError("fig02_evaluation_workflow_unified.png missing; export it from the .drawio source")
    _plot_clean(core_summary)
    _figure_envelope_construction(core_envelope)
    _plot_curves(core_envelope, "fgsm", "fig06_fgsm_robustness_curves", "Conditional robust accuracy (%)")
    _plot_curves(core_envelope, "audited_envelope", "fig07_audited_attack_envelope", "Audited robust accuracy (%)")
    _plot_complexity(core_summary)
    _plot_cross_arch(clean_df, attack_df)
    _plot_attack_audit()
    _plot_interpretation_composite()

    audit_stats = {
        "core_runs_expected": 45,
        "core_runs_observed": int(seed_metrics.shape[0]),
        "runs_with_monotonic_adjustment": int((seed_metrics.n_monotonic_adjustments > 0).sum()),
        "adjusted_seed_budget_observations": int(seed_metrics.n_monotonic_adjustments.sum()),
        "largest_raw_pgd_rebound": float(seed_metrics.max_pgd_rebound.max()),
        "defense_seed_rows": int(defenses.n_seeds.sum()),
        "defense_successful_seed_rows": int(defenses.n_success.sum()),
    }
    _write_json(OUT_AUDIT / "audit_summary.json", audit_stats)
    _write_manifest(reused, audit_stats)
    _write_analysis_note(core_summary, defenses, seed_metrics)

    expected = [OUT_FIG / f"{stem}.png" for stem in _captions()]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing numbered figures: {missing}")
    if audit_stats["core_runs_observed"] != audit_stats["core_runs_expected"]:
        raise RuntimeError(f"Core matrix incomplete: {audit_stats}")
    print(json.dumps(audit_stats, indent=2))
    print(f"Figures: {OUT_FIG}")
    print(f"Tables: {OUT_TAB}")
    print(f"Audit: {OUT_AUDIT}")


if __name__ == "__main__":
    main()
