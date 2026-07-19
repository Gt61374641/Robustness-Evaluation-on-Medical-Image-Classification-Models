"""Audit experiment-result coverage and generated figure/table artifacts.

This is a lightweight integrity check for the current dissertation experiment
matrix. It separates true required gaps from optional extension cells, so the
audit can be used after AutoDL result merges without treating known paper-scope
choices as failures.

Usage:
  python scripts/audit_results.py
  python scripts/audit_results.py --strict-optional
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

DATASETS = ("chest_xray_pneumonia", "malaria", "oct2017")
MODELS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
SEEDS = ("seed42", "seed43", "seed44")
DEFENSES = ("PGD-AT", "TRADES", "MART")

# Cells that are part of the paper's current main evidence, not every possible
# defense/model/dataset combination.
REQUIRED_DEFENSE_CELLS = {
    ("chest_xray_pneumonia", "resnet18", "TRADES"): SEEDS,
    ("chest_xray_pneumonia", "resnet18", "MART"): SEEDS,
    ("chest_xray_pneumonia", "resnet18", "PGD-AT"): ("seed42", "seed43"),
    ("chest_xray_pneumonia", "resnet50", "PGD-AT"): SEEDS,
    ("chest_xray_pneumonia", "resnet50", "TRADES"): SEEDS,
    ("chest_xray_pneumonia", "resnet50", "MART"): SEEDS,
    ("chest_xray_pneumonia", "resnet152", "PGD-AT"): SEEDS,
    ("chest_xray_pneumonia", "resnet152", "TRADES"): SEEDS,
    ("chest_xray_pneumonia", "resnet152", "MART"): SEEDS,
    ("malaria", "resnet50", "PGD-AT"): SEEDS,
    ("malaria", "resnet50", "TRADES"): SEEDS,
    ("malaria", "resnet50", "MART"): SEEDS,
    ("malaria", "resnet101", "PGD-AT"): SEEDS,
    ("malaria", "resnet152", "PGD-AT"): SEEDS,
    ("malaria", "resnet152", "TRADES"): SEEDS,
    ("malaria", "resnet152", "MART"): SEEDS,
    ("oct2017", "resnet50", "PGD-AT"): SEEDS,
    ("oct2017", "resnet50", "TRADES"): SEEDS,
    ("oct2017", "resnet50", "MART"): SEEDS,
}

REQUIRED_PGD_AT_LADDER = {
    (ds, model): ("seed42",) for ds in DATASETS for model in MODELS
}
REQUIRED_RESCUE = {
    ("chest_xray_pneumonia", "resnet18"): ("seed42",),
    ("oct2017", "resnet152"): ("seed42",),
}

OPTIONAL_NOT_RUN = [
    ("oct2017", "resnet152", "TRADES", "method diagnostic not in current scope"),
    ("oct2017", "resnet152", "MART", "method diagnostic not in current scope"),
    ("malaria", "resnet18", "TRADES", "secondary dataset / collapsed PGD-AT point"),
    ("malaria", "resnet18", "MART", "secondary dataset / collapsed PGD-AT point"),
    ("malaria", "resnet34", "TRADES", "secondary dataset / collapsed PGD-AT point"),
    ("malaria", "resnet34", "MART", "secondary dataset / collapsed PGD-AT point"),
    ("malaria", "resnet101", "TRADES", "not part of three-method main evidence"),
    ("malaria", "resnet101", "MART", "not part of three-method main evidence"),
    ("oct2017", "resnet18", "TRADES", "secondary dataset / not part of method evidence"),
    ("oct2017", "resnet18", "MART", "secondary dataset / not part of method evidence"),
    ("oct2017", "resnet34", "TRADES", "secondary dataset / collapsed PGD-AT point"),
    ("oct2017", "resnet34", "MART", "secondary dataset / collapsed PGD-AT point"),
    ("oct2017", "resnet101", "TRADES", "secondary dataset / collapsed PGD-AT point"),
    ("oct2017", "resnet101", "MART", "secondary dataset / collapsed PGD-AT point"),
]

FIGURE_DATA = (
    "h1_pgd_curves.json",
    "h1_complexity_fixedeps.json",
    "defense_methods.json",
    "attack_methods.json",
    "at_ladder_h2.json",
    "at_rescue.json",
)
FIGURE_ARTIFACTS = (
    "figures/main/H1_pgd_across_datasets.png",
    "figures/main/H1_attack_budget.png",
    "figures/main/H1_complexity_ushape.png",
    "figures/main/defense_methods.png",
    "figures/main/attack_methods.png",
    "figures/at_ladder/H2_at_ladder_py.png",
    "figures/at_ladder/H2b_rescue_stability_py.png",
    "figures/paper_tables/table8_h2_at_ladder.csv",
    "figures/paper_tables/table9_at_rescue.csv",
    "figures/paper_tables/chest_xray_pneumonia/table5_defense_methods.csv",
    "figures/paper_tables/chest_xray_pneumonia/table6_attack_methods.csv",
)


@dataclass
class Issue:
    level: str
    area: str
    item: str
    detail: str
    path: str = ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eps255(key: str) -> float | None:
    match = re.search(r"eps=([0-9.]+)", key)
    if not match:
        return None
    return round(float(match.group(1)) * 255, 4)


def defense_json(ds: str, model: str, defense: str, seed: str) -> Path:
    return RESULTS / ds / model / f"defense_{defense}" / seed / "defense_results_max1024.json"


def clean_json(ds: str, model: str, seed: str) -> Path:
    return RESULTS / ds / model / "clean" / seed / "clean_results.json"


def robust_json(ds: str, model: str, seed: str, section: str) -> Path:
    return RESULTS / ds / model / "robustness" / seed / f"robustness_attacks_{section}_max1024.json"


def has_robust8(path: Path) -> bool:
    if not path.exists():
        return False
    data = read_json(path)
    for key, value in data.items():
        if not (key.startswith("PGD") and isinstance(value, dict)):
            continue
        if abs((eps255(key) or 0.0) - 8.0) < 0.5:
            return True
    return False


def defense_metrics(path: Path) -> tuple[float | None, float | None, bool | None]:
    if not path.exists():
        return None, None, None
    data = read_json(path)
    clean = data.get("clean_accuracy_defended")
    robust8 = None
    constant = False
    for key, value in data.items():
        if not (key.startswith("PGD") and isinstance(value, dict)):
            continue
        if abs((eps255(key) or 0.0) - 8.0) < 0.5:
            robust8 = value.get("robust_accuracy", {}).get("full_robust_accuracy")
        clean_dist = value.get("pred_distribution", {}).get("clean", {})
        fractions = [v.get("fraction", 0.0) for v in clean_dist.values() if isinstance(v, dict)]
        constant = constant or any(frac >= 0.99 for frac in fractions)
    collapsed = None if robust8 is None else bool(constant or robust8 < 0.02)
    return clean, robust8, collapsed


def expect_file(issues: list[Issue], level: str, area: str, item: str, path: Path):
    if not path.exists():
        issues.append(Issue(level, area, item, "missing file", rel(path)))


def audit_h1(issues: list[Issue]):
    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                expect_file(issues, "ERROR", "H1", f"{ds}/{model}/{seed}/clean", clean_json(ds, model, seed))
                fine = robust_json(ds, model, seed, "fine")
                main = robust_json(ds, model, seed, "main")
                if not fine.exists() and not main.exists():
                    issues.append(Issue("ERROR", "H1", f"{ds}/{model}/{seed}/robustness", "missing fine/main robustness json"))
                elif not (has_robust8(fine) or has_robust8(main)):
                    issues.append(Issue("ERROR", "H1", f"{ds}/{model}/{seed}/PGD@8", "robustness json exists but PGD@8 not found"))


def audit_defenses(issues: list[Issue], rows: list[dict]):
    for (ds, model), seeds in REQUIRED_PGD_AT_LADDER.items():
        for seed in seeds:
            expect_file(issues, "ERROR", "H2 ladder", f"{ds}/{model}/PGD-AT/{seed}", defense_json(ds, model, "PGD-AT", seed))

    for (ds, model, defense), seeds in REQUIRED_DEFENSE_CELLS.items():
        for seed in seeds:
            path = defense_json(ds, model, defense, seed)
            expect_file(issues, "ERROR", "Defense cells", f"{ds}/{model}/{defense}/{seed}", path)
            clean, robust8, collapsed = defense_metrics(path)
            if path.exists():
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "defense": defense,
                    "seed": seed,
                    "clean": clean,
                    "robust8": robust8,
                    "collapsed": collapsed,
                    "path": rel(path),
                })
                if robust8 is None:
                    issues.append(Issue("ERROR", "Defense cells", f"{ds}/{model}/{defense}/{seed}", "PGD@8 not found", rel(path)))
                elif collapsed:
                    issues.append(Issue("WARN", "Defense collapse", f"{ds}/{model}/{defense}/{seed}", f"collapsed or robust@8={robust8:.4f}", rel(path)))

    for ds, model, defense, reason in OPTIONAL_NOT_RUN:
        existing = [seed for seed in SEEDS if defense_json(ds, model, defense, seed).exists()]
        if not existing:
            issues.append(Issue("INFO", "Optional cells", f"{ds}/{model}/{defense}", reason))

    for (ds, model), seeds in REQUIRED_RESCUE.items():
        for seed in seeds:
            path = defense_json(ds, model, "PGD-AT-rescue", seed)
            expect_file(issues, "ERROR", "Rescue", f"{ds}/{model}/PGD-AT-rescue/{seed}", path)
            clean, robust8, collapsed = defense_metrics(path)
            if path.exists() and robust8 is None:
                issues.append(Issue("ERROR", "Rescue", f"{ds}/{model}/PGD-AT-rescue/{seed}", "PGD@8 not found", rel(path)))
            elif path.exists() and collapsed:
                issues.append(Issue("WARN", "Rescue collapse", f"{ds}/{model}/PGD-AT-rescue/{seed}", f"still collapsed or robust@8={robust8:.4f}", rel(path)))


def audit_generated_outputs(issues: list[Issue]):
    for name in FIGURE_DATA:
        path = FIGURES / "data" / name
        expect_file(issues, "ERROR", "Figure data", name, path)

    at_ladder = FIGURES / "data" / "at_ladder_h2.json"
    if at_ladder.exists():
        rows = read_json(at_ladder).get("rows", [])
        if len(rows) != len(DATASETS) * len(MODELS):
            issues.append(Issue("ERROR", "Figure data", "at_ladder_h2.json", f"expected 15 rows, got {len(rows)}", rel(at_ladder)))

    defense = FIGURES / "data" / "defense_methods.json"
    if defense.exists():
        rows = read_json(defense).get("rows", [])
        lookup = {(r.get("model"), r.get("method")): r for r in rows}
        for model in ("resnet18", "resnet50", "resnet152"):
            for method in ("TRADES", "MART"):
                rec = lookup.get((model, method))
                if not rec:
                    issues.append(Issue("ERROR", "Figure data", f"defense_methods {model}/{method}", "missing row", rel(defense)))
                elif rec.get("n_seeds") != 3:
                    issues.append(Issue("ERROR", "Figure data", f"defense_methods {model}/{method}", f"expected n_seeds=3, got {rec.get('n_seeds')}", rel(defense)))

    rescue = FIGURES / "data" / "at_rescue.json"
    if rescue.exists():
        rows = read_json(rescue).get("rows", [])
        keys = {(r.get("dataset"), r.get("model")) for r in rows}
        for key in REQUIRED_RESCUE:
            if key not in keys:
                issues.append(Issue("ERROR", "Figure data", f"at_rescue {key[0]}/{key[1]}", "missing row", rel(rescue)))

    for artifact in FIGURE_ARTIFACTS:
        expect_file(issues, "ERROR", "Generated artifacts", artifact, ROOT / artifact)


def write_reports(issues: list[Issue], rows: list[dict]):
    REPORTS.mkdir(exist_ok=True)
    payload = {
        "summary": summarize(issues),
        "issues": [asdict(issue) for issue in issues],
        "defense_rows": rows,
    }
    (REPORTS / "audit_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (REPORTS / "audit_results_issues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("level", "area", "item", "detail", "path"))
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))

    with (REPORTS / "audit_results_defenses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("dataset", "model", "defense", "seed", "clean", "robust8", "collapsed", "path"))
        writer.writeheader()
        writer.writerows(rows)


def summarize(issues: Iterable[Issue]) -> dict[str, int]:
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for issue in issues:
        counts[issue.level] = counts.get(issue.level, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-optional", action="store_true", help="treat optional missing method cells as errors")
    args = parser.parse_args()

    issues: list[Issue] = []
    defense_rows: list[dict] = []
    audit_h1(issues)
    audit_defenses(issues, defense_rows)
    audit_generated_outputs(issues)

    if args.strict_optional:
        for issue in issues:
            if issue.level == "INFO" and issue.area == "Optional cells":
                issue.level = "ERROR"

    write_reports(issues, defense_rows)
    counts = summarize(issues)

    print("Audit summary")
    print(f"  ERROR: {counts.get('ERROR', 0)}")
    print(f"  WARN : {counts.get('WARN', 0)}")
    print(f"  INFO : {counts.get('INFO', 0)}")
    print("  wrote: reports/audit_results.json")
    print("  wrote: reports/audit_results_issues.csv")
    print("  wrote: reports/audit_results_defenses.csv")

    for level in ("ERROR", "WARN", "INFO"):
        selected = [issue for issue in issues if issue.level == level]
        if not selected:
            continue
        print(f"\n{level}:")
        for issue in selected[:20]:
            suffix = f" ({issue.path})" if issue.path else ""
            print(f"  [{issue.area}] {issue.item}: {issue.detail}{suffix}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more; see reports/audit_results_issues.csv")

    return 1 if counts.get("ERROR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
