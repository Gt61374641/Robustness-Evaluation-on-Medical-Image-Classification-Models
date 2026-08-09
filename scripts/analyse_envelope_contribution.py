"""Quantify how much each step of the audited attack-set envelope actually changes.

The envelope of Section 3.4 is built in two steps:

  step A  attack-set minimum   per seed and budget, min(FGSM, PGD)
  step B  monotonic audit      cumulative minimum over increasing budget

Chapter 5 claims the envelope is what stops a weak attack from being reported as
robustness. That claim is currently asserted rather than measured. This script
measures it, by recomputing the headline summaries (nAUC and critical epsilon)
under four curves and comparing them:

  FGSM only          what a single-step attack would have reported
  PGD only           what the standard iterative attack alone would have reported
  attack-set min     step A only, no monotone smoothing
  audited envelope   step A + step B, as published

Reads  reports/thesis_evidence/audited_envelope_seed.csv
Writes reports/thesis_evidence/envelope_contribution.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "reports" / "thesis_evidence"
EPS_GRID = np.array([0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, 4, 8, 16], dtype=float)
LOG_X = np.log10(EPS_GRID)
X_SPAN = float(LOG_X[-1] - LOG_X[0])
CURVES = ["fgsm", "pgd", "attack_set_min", "audited_envelope"]


def summarise(y: np.ndarray) -> tuple[float, float]:
    """nAUC and critical epsilon for one curve, matching build_thesis_evidence_package."""
    auc = float(np.trapezoid(y, x=LOG_X) / X_SPAN)
    below = np.flatnonzero(y <= 0.5)
    crit = float(EPS_GRID[below[0]]) if below.size else np.nan
    return auc, crit


def main() -> None:
    df = pd.read_csv(EVID / "audited_envelope_seed.csv").sort_values(
        ["dataset", "model", "seed", "epsilon_255"]
    )
    n_obs = len(df)

    # --- how often does each step bind, and by how much -----------------------
    a_binds = df["attack_set_min"] < df["pgd"] - 1e-12   # FGSM was the weaker of the two
    b_binds = df["audited_envelope"] < df["attack_set_min"] - 1e-12
    a_delta = (df["pgd"] - df["attack_set_min"])[a_binds]
    b_delta = (df["attack_set_min"] - df["audited_envelope"])[b_binds]

    print(f"Observations (seed x budget x run): {n_obs}\n")
    print("Step A - attack-set minimum, min(FGSM, PGD)")
    print(f"  binds (FGSM below PGD):   {a_binds.sum():4d} / {n_obs}  ({100*a_binds.mean():.1f}%)")
    if a_binds.any():
        print(f"  mean reduction vs PGD:    {a_delta.mean():.4f}   max {a_delta.max():.4f}")
    print("\nStep B - monotonic audit, cumulative minimum")
    print(f"  binds (audit lowered it): {b_binds.sum():4d} / {n_obs}  ({100*b_binds.mean():.1f}%)")
    if b_binds.any():
        print(f"  mean reduction vs step A: {b_delta.mean():.4f}   max {b_delta.max():.4f}")

    # how often is FGSM the weaker attack, i.e. how much work step A can do at all
    fgsm_lower = (df["fgsm"] < df["pgd"] - 1e-12).sum()
    print(f"\n  FGSM strictly below PGD in {fgsm_lower} of {n_obs} observations "
          f"({100*fgsm_lower/n_obs:.1f}%)")

    # --- headline summaries under each curve ---------------------------------
    rows = []
    for (ds, model, seed), g in df.groupby(["dataset", "model", "seed"]):
        g = g.sort_values("epsilon_255")
        rec = {"dataset": ds, "model": model, "seed": seed}
        for c in CURVES:
            auc, crit = summarise(g[c].to_numpy(dtype=float))
            rec[f"nauc_{c}"] = auc
            rec[f"eps_crit_{c}"] = crit
        rows.append(rec)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(EVID / "envelope_contribution.csv", index=False)

    print("\n\nHeadline nAUC by curve (mean over 15 runs x 3 seeds per dataset)")
    print(f"{'dataset':16s}" + "".join(f"{c:>20s}" for c in CURVES))
    for ds, g in per_run.groupby("dataset"):
        print(f"{ds:16s}" + "".join(f"{g[f'nauc_{c}'].mean():20.4f}" for c in CURVES))

    print("\nHeadline critical epsilon /255 by curve (mean, NaN = never fell to 0.5)")
    print(f"{'dataset':16s}" + "".join(f"{c:>20s}" for c in CURVES))
    for ds, g in per_run.groupby("dataset"):
        print(f"{ds:16s}" + "".join(f"{g[f'eps_crit_{c}'].mean():20.4f}" for c in CURVES))

    # --- the decisive comparison: envelope vs PGD alone -----------------------
    d_auc = (per_run["nauc_pgd"] - per_run["nauc_audited_envelope"]).abs()
    same_crit = (
        per_run["eps_crit_pgd"].fillna(-1) == per_run["eps_crit_audited_envelope"].fillna(-1)
    )
    print("\n\nDoes the envelope differ from simply reporting PGD?")
    print(f"  runs where nAUC differs by >0.001:      {(d_auc > 0.001).sum()} / {len(per_run)}")
    print(f"  runs with identical critical epsilon:   {same_crit.sum()} / {len(per_run)}")
    print(f"  mean |nAUC(PGD) - nAUC(envelope)|:      {d_auc.mean():.5f}")
    print(f"  max  |nAUC(PGD) - nAUC(envelope)|:      {d_auc.max():.5f}")

    d_fgsm = (per_run["nauc_fgsm"] - per_run["nauc_audited_envelope"]).abs()
    print("\nAnd against FGSM alone, for scale:")
    print(f"  mean |nAUC(FGSM) - nAUC(envelope)|:     {d_fgsm.mean():.5f}")
    print(f"  max  |nAUC(FGSM) - nAUC(envelope)|:     {d_fgsm.max():.5f}")

    print(f"\nWrote {EVID / 'envelope_contribution.csv'}")


if __name__ == "__main__":
    main()
