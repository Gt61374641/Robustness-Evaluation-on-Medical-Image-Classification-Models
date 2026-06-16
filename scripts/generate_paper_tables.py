"""Generate final paper tables from clean, robustness, and defense results."""

import argparse
import json
from pathlib import Path

import pandas as pd


REPRESENTATIVE_ATTACKS = [
    ("FGSM", "FGSM 8/255"),
    ("PGD", "PGD 8/255"),
    ("AutoPGD", "AutoPGD 8/255"),
    ("SquareAttack", "SquareAttack 8/255"),
    ("DeepFool", "DeepFool"),
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.{decimals}f}"


def fixed(value, decimals: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def save_table(raw: pd.DataFrame, formatted: pd.DataFrame, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{name}.csv"
    formatted_path = output_dir / f"{name}_formatted.csv"
    tex_path = output_dir / f"{name}.tex"

    raw.to_csv(raw_path, index=False)
    formatted.to_csv(formatted_path, index=False)
    formatted.to_latex(tex_path, index=False, escape=False)
    return [raw_path, formatted_path, tex_path]


def clean_overall_table(clean_summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame([
        {"metric": "Accuracy", "value": clean_summary["accuracy"]},
        {"metric": "Balanced accuracy", "value": clean_summary["balanced_accuracy"]},
        {"metric": "ROC AUC", "value": clean_summary["auc"]},
        {"metric": "ECE", "value": clean_summary["ece"]},
        {"metric": "Samples", "value": clean_summary["num_samples"]},
    ])
    formatted = raw.copy()
    formatted["value"] = [
        pct(raw.loc[0, "value"]),
        pct(raw.loc[1, "value"]),
        fixed(raw.loc[2, "value"]),
        pct(raw.loc[3, "value"]),
        str(int(raw.loc[4, "value"])),
    ]
    return raw, formatted


def clean_class_table(class_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = class_metrics[["class", "support", "class_accuracy", "precision", "recall", "f1"]].copy()
    formatted = raw.copy()
    for col in ["class_accuracy", "precision", "recall", "f1"]:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def representative_attack_table(attack_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for attack, label in REPRESENTATIVE_ATTACKS:
        subset = attack_df[(attack_df["attack"] == attack) & (attack_df["label"] == label)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append({
            "attack": label.replace("SquareAttack", "Square"),
            "clean_accuracy": row["clean_accuracy"],
            "robust_accuracy": row["robust_accuracy"],
            "asr": row["asr"],
            "accuracy_drop": row["accuracy_drop"],
            "ece_adv": row["ece_adv"],
        })
    raw = pd.DataFrame(rows)
    formatted = raw.copy()
    for col in ["clean_accuracy", "robust_accuracy", "asr", "accuracy_drop", "ece_adv"]:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def attack_sweep_table(attack_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["attack", "label", "robust_accuracy", "asr", "accuracy_drop", "ece_adv"]
    raw = attack_df[columns].copy()
    formatted = raw.copy()
    for col in ["robust_accuracy", "asr", "accuracy_drop", "ece_adv"]:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def defense_pgd8_table(defense_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = defense_df[defense_df["label"] == "PGD 8/255"].copy()
    asr_columns = sorted(col for col in raw.columns if col.startswith("asr_"))
    columns = [
        "method",
        "category",
        "clean_accuracy",
        "robust_accuracy",
        "asr",
        "accuracy_drop",
    ] + asr_columns
    raw = raw[columns].copy()
    formatted = raw.copy()
    for col in ["clean_accuracy", "robust_accuracy", "asr", "accuracy_drop", *asr_columns]:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def defense_deepfool_table(defense_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = defense_df[defense_df["attack"] == "DeepFool"].copy()
    columns = ["method", "category", "clean_accuracy", "robust_accuracy", "asr", "accuracy_drop"]
    raw = raw[columns].copy()
    formatted = raw.copy()
    for col in ["clean_accuracy", "robust_accuracy", "asr", "accuracy_drop"]:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def per_class_vulnerability_table(defense_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = defense_df[defense_df["label"] == "PGD 8/255"].copy()
    metric_columns = []
    for prefix in ("clean_accuracy_", "asr_", "robust_accuracy_"):
        metric_columns.extend(sorted(col for col in raw.columns if col.startswith(prefix)))
    columns = ["method"] + metric_columns
    raw = raw[columns].copy()
    formatted = raw.copy()
    for col in metric_columns:
        formatted[col] = formatted[col].map(pct)
    return raw, formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final dissertation paper tables")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "paper_tables")
    parser.add_argument("--dataset", default="chest_xray_pneumonia")
    parser.add_argument("--model", default="densenet121")
    args = parser.parse_args()

    clean_dir = args.figures_dir / "sci_clean" / args.dataset / args.model
    attack_dir = args.figures_dir / "sci" / args.dataset / args.model
    defense_dir = args.figures_dir / "sci_defense" / args.dataset / args.model
    out_dir = args.output_dir / args.dataset / args.model

    clean_summary = read_json(clean_dir / "clean_diagnostics_summary.json")
    class_metrics = pd.read_csv(clean_dir / "clean_classification_metrics.csv")
    attack_df = pd.read_csv(attack_dir / "sci_summary_metrics.csv")
    defense_df = pd.read_csv(defense_dir / "sci_defense_summary_metrics.csv")

    outputs = []
    for name, (raw, formatted) in {
        "table1_clean_overall": clean_overall_table(clean_summary),
        "table2_clean_class_metrics": clean_class_table(class_metrics),
        "table3_representative_attacks": representative_attack_table(attack_df),
        "table4_attack_epsilon_sweep": attack_sweep_table(attack_df),
        "table5_defense_pgd8_comparison": defense_pgd8_table(defense_df),
        "table6_defense_deepfool_comparison": defense_deepfool_table(defense_df),
        "table7_per_class_vulnerability_pgd8": per_class_vulnerability_table(defense_df),
    }.items():
        outputs.extend(save_table(raw, formatted, out_dir, name))

    print(f"Generated {len(outputs)} paper table files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
