"""Generate per-model configs for the ResNet complexity ladder + architecture comparison.

Reads each `configs/{dataset}_base.yaml` and writes `configs/{dataset}_{model}.yaml`
for every model, changing ONLY the model name. This guarantees the eps grid,
training block and class-balance settings are byte-identical across models — a
prerequisite for fair complexity AND architecture comparisons.

The architecture-comparison models (deit_small = ViT-S/16, convnext_tiny) are
generated for the PRIMARY dataset (chest) only.

Usage:
    python scripts/make_configs.py            # regenerate all 17 configs
"""

import re
import sys
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

BASES = [
    "chest_xray_pneumonia_base.yaml",
    "oct2017_base.yaml",
    "malaria_base.yaml",
]

RESNETS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# Non-ResNet architectures (params matched to ResNet-50) — primary dataset only.
EXTRA_MODELS_BY_BASE = {
    "chest_xray_pneumonia_base.yaml": ["deit_small", "convnext_tiny"],
}

# Matches the top-level (2-space) model name line, not the 4-space attack names.
_NAME_RE = re.compile(r'(?m)^(  name: )"[^"]*".*$')


def main():
    written = []
    for base_name in BASES:
        base_path = CONFIGS_DIR / base_name
        if not base_path.exists():
            sys.exit(f"Missing base config: {base_path}")
        dataset = base_name[: -len("_base.yaml")]
        text = base_path.read_text(encoding="utf-8")
        if not _NAME_RE.search(text):
            sys.exit(f"Could not find a top-level 'name:' line in {base_path}")

        models = RESNETS + EXTRA_MODELS_BY_BASE.get(base_name, [])
        for model in models:
            out_text = _NAME_RE.sub(rf'\1"{model}"', text, count=1)
            out_path = CONFIGS_DIR / f"{dataset}_{model}.yaml"
            out_path.write_text(out_text, encoding="utf-8")
            written.append(out_path.name)

    print(f"Generated {len(written)} configs:")
    for name in written:
        print(" ", name)


if __name__ == "__main__":
    main()
