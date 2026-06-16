"""Generate paper-ready figures and summary tables from experiment results.

Usage:
    python scripts/generate_report.py --results-dir results/ --output-dir figures/
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization import (
    plot_eps_vs_accuracy,
    plot_robustness_heatmap,
    plot_per_class_asr,
)


def load_results(results_dir: Path) -> dict:
    """Recursively load all JSON result files."""
    all_results = {}
    for json_file in results_dir.rglob("*.json"):
        rel_path = json_file.relative_to(results_dir)
        with open(json_file) as f:
            all_results[str(rel_path)] = json.load(f)
    return all_results


def generate_summary_table(results_dir: Path, output_dir: Path):
    """Generate a CSV summary table of all experiments."""
    rows = []

    for robustness_file in results_dir.rglob("robustness_*.json"):
        if "_max" in robustness_file.stem or "_legacy" in robustness_file.stem:
            continue
        parts = robustness_file.relative_to(results_dir).parts
        if len(parts) >= 4:
            dataset, model = parts[0], parts[1]
        else:
            continue

        with open(robustness_file) as f:
            results = json.load(f)

        for attack_key, metrics in results.items():
            if attack_key.startswith("_") or "error" in metrics or "robust_accuracy" not in metrics:
                continue
            rows.append({
                "dataset": dataset,
                "model": model,
                "attack": attack_key,
                "robust_accuracy": metrics["robust_accuracy"]["robust_accuracy"],
                "asr": metrics["asr"],
                "accuracy_drop": metrics["accuracy_drop"]["accuracy_drop"],
                "clean_accuracy": metrics["accuracy_drop"]["clean_accuracy"],
            })

    if rows:
        df = pd.DataFrame(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary table saved to {csv_path}")
        print(df.to_string(index=False))
    else:
        print("No results found to summarize.")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures and tables")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating summary table...")
    generate_summary_table(results_dir, output_dir)

    # Generate per-dataset, per-model figures
    for robustness_file in results_dir.rglob("robustness_attacks_main.json"):
        if "_max" in robustness_file.stem or "_legacy" in robustness_file.stem:
            continue
        parts = robustness_file.relative_to(results_dir).parts
        if len(parts) >= 4:
            dataset, model = parts[0], parts[1]
        else:
            continue

        with open(robustness_file) as f:
            results = json.load(f)

        attack_results = {
            key: value for key, value in results.items()
            if not key.startswith("_") and isinstance(value, dict) and "robust_accuracy" in value
        }

        fig_dir = output_dir / dataset / model
        fig_dir.mkdir(parents=True, exist_ok=True)

        # eps vs accuracy curves
        for attack in ["FGSM", "PGD"]:
            plot_eps_vs_accuracy(
                attack_results, attack_name=attack,
                save_path=str(fig_dir / f"eps_vs_accuracy_{attack}.png"),
            )

        # Per-class ASR (use PGD at eps=8/255 as default)
        pgd_key = "PGD_eps=0.031372"
        if pgd_key in attack_results and "per_class" in attack_results[pgd_key]:
            plot_per_class_asr(
                attack_results[pgd_key]["per_class"],
                save_path=str(fig_dir / "per_class_asr_pgd.png"),
            )

        print(f"Figures generated for {dataset}/{model} -> {fig_dir}")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
