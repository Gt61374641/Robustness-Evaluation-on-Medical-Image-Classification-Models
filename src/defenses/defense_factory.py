"""Defense factory for creating ART defense instances.

IMPORTANT DISTINCTION:
- Main defenses (adversarial training): PGD-AT, TRADES
  These modify model parameters and can be claimed as "effective defenses" in papers.
- Baseline defenses (preprocessors): SpatialSmoothing, JpegCompression, FeatureSqueezing
  These are supplementary only. They risk gradient obfuscation and CANNOT be claimed
  as "effectively improving robustness" without adaptive attack verification.
"""

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import (
    AdversarialTrainerMadryPGD,
    AdversarialTrainerTRADESPyTorch,
)
from art.defences.preprocessor import (
    SpatialSmoothing,
    JpegCompression,
    FeatureSqueezing,
)


def create_defense_trainer(classifier: PyTorchClassifier, defense_cfg: dict):
    """Create an adversarial training defense (main defense).

    Args:
        classifier: ART PyTorchClassifier to be adversarially trained.
        defense_cfg: Defense config dict with 'name' and parameters.

    Returns:
        ART adversarial trainer instance.
    """
    name = defense_cfg["name"]

    if name == "PGD-AT":
        return AdversarialTrainerMadryPGD(
            classifier,
            nb_epochs=defense_cfg.get("nb_epochs", 20),
            eps=defense_cfg.get("eps", 8 / 255),
            eps_step=defense_cfg.get("eps_step", 2 / 255),
        )
    elif name == "TRADES":
        pgd_attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=defense_cfg.get("eps", 8 / 255),
            eps_step=defense_cfg.get("eps_step", 2 / 255),
            max_iter=defense_cfg.get("max_iter", 10),
            num_random_init=1,
        )
        return AdversarialTrainerTRADESPyTorch(
            classifier,
            attack=pgd_attack,
            beta=defense_cfg.get("beta", 6.0),
        )
    else:
        raise ValueError(f"Unknown defense trainer: {name}. Available: PGD-AT, TRADES")


def create_preprocessor_defense(defense_cfg: dict):
    """Create a preprocessor defense (baseline/supplementary only).

    WARNING: These defenses may cause gradient obfuscation. They can make
    non-adaptive attacks fail without truly improving robustness. Use only
    as baseline comparisons, not as main defense conclusions in the paper.

    Args:
        defense_cfg: Defense config dict.

    Returns:
        ART preprocessor defense instance.
    """
    name = defense_cfg["name"]

    if name == "SpatialSmoothing":
        return SpatialSmoothing(
            window_size=defense_cfg.get("window_size", 3),
        )
    elif name == "JpegCompression":
        return JpegCompression(
            clip_values=(0.0, 1.0),
            quality=defense_cfg.get("quality", 75),
            channels_first=True,
        )
    elif name == "FeatureSqueezing":
        return FeatureSqueezing(
            bit_depth=defense_cfg.get("bit_depth", 4),
            clip_values=(0.0, 1.0),
        )
    else:
        raise ValueError(
            f"Unknown preprocessor defense: {name}. "
            "Available: SpatialSmoothing, JpegCompression, FeatureSqueezing"
        )
