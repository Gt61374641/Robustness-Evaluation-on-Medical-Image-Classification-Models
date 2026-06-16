"""Experiment reproducibility utilities.

Provides seed fixing, config snapshots, structured result directories,
and checkpoint naming to ensure every experiment is recoverable.
"""

import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def get_results_dir(
    base_dir: str,
    dataset: str,
    model: str,
    experiment: str = "clean",
    seed: int = 42,
) -> Path:
    """Create and return structured results directory.

    Structure: results/{dataset}/{model}/{experiment}/seed{N}/
    """
    results_dir = Path(base_dir) / dataset / model / experiment / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_checkpoint_path(
    base_dir: str,
    dataset: str,
    model: str,
    seed: int = 42,
    suffix: str = "",
) -> Path:
    """Get checkpoint file path.

    Naming: {dataset}_{model}_seed{N}{suffix}.pth
    e.g., chest_xray_pneumonia_densenet121_seed42.pth
         chest_xray_pneumonia_densenet121_seed42_pgd_at.pth
    """
    ckpt_dir = Path(base_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"{dataset}_{model}_seed{seed}{suffix_str}.pth"
    return ckpt_dir / filename


def save_config_snapshot(cfg: dict, results_dir: Path):
    """Save a complete copy of the config to the results directory."""
    snapshot = cfg.copy()
    snapshot["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)
    return config_path


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
