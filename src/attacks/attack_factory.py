"""Attack factory for creating ART evasion attack instances from config.

Attacks are split into:
- Main attacks (FGSM, PGD, CW/DeepFool): core experiments, must-do
- Extended attacks (AutoPGD, SquareAttack, HopSkipJump): bonus experiments
"""

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    DeepFool,
    AutoProjectedGradientDescent,
    SquareAttack,
    HopSkipJump,
)


ATTACK_REGISTRY = {
    "FGSM": FastGradientMethod,
    "PGD": ProjectedGradientDescent,
    "CW": CarliniL2Method,
    "DeepFool": DeepFool,
    "AutoPGD": AutoProjectedGradientDescent,
    "SquareAttack": SquareAttack,
    "HopSkipJump": HopSkipJump,
}


def create_attack(classifier: PyTorchClassifier, attack_cfg: dict, eps: float = None):
    """Create a single ART attack instance.

    Args:
        classifier: ART PyTorchClassifier wrapping the target model.
        attack_cfg: Attack config dict with 'name' and parameters.
        eps: Override eps value (for parameter sweeps).

    Returns:
        ART attack instance.
    """
    name = attack_cfg["name"]
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")

    attack_cls = ATTACK_REGISTRY[name]

    if name == "FGSM":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        return attack_cls(estimator=classifier, eps=attack_eps)

    elif name == "PGD":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        max_iter = attack_cfg.get("max_iter", 20)
        eps_step = attack_eps / 10  # standard heuristic
        return attack_cls(
            estimator=classifier,
            eps=attack_eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=1,
        )

    elif name == "CW":
        return attack_cls(
            classifier=classifier,
            confidence=attack_cfg.get("confidence", 0),
            max_iter=attack_cfg.get("max_iter", 100),
            learning_rate=attack_cfg.get("lr", 0.01),
        )

    elif name == "DeepFool":
        return attack_cls(
            classifier=classifier,
            max_iter=attack_cfg.get("max_iter", 50),
            epsilon=attack_cfg.get("epsilon", 1e-6),
        )

    elif name == "AutoPGD":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        return attack_cls(
            estimator=classifier,
            eps=attack_eps,
            eps_step=attack_eps / 10,
            max_iter=attack_cfg.get("max_iter", 100),
        )

    elif name == "SquareAttack":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        return attack_cls(
            estimator=classifier,
            eps=attack_eps,
            max_iter=attack_cfg.get("max_queries", 5000),
            norm="inf",
        )

    elif name == "HopSkipJump":
        return attack_cls(
            classifier=classifier,
            max_iter=attack_cfg.get("max_iter", 50),
            max_eval=attack_cfg.get("max_eval", 10000),
            init_eval=attack_cfg.get("init_eval", 100),
        )

    else:
        raise ValueError(f"No factory logic for attack: {name}")


def create_attacks_from_config(classifier: PyTorchClassifier, cfg: dict, section: str = "attacks_main"):
    """Create all attacks defined in a config section.

    For attacks with multiple eps values, creates one instance per eps.

    Args:
        classifier: ART PyTorchClassifier.
        cfg: Full config dict.
        section: Config section key ('attacks_main' or 'attacks_extended').

    Returns:
        List of (attack_name, eps_value, attack_instance) tuples.
    """
    attacks = []
    attack_configs = cfg.get(section, [])

    for attack_cfg in attack_configs:
        name = attack_cfg["name"]
        eps_values = attack_cfg.get("eps", None)

        if isinstance(eps_values, list):
            # Create one attack per eps value
            for eps_val in eps_values:
                attack = create_attack(classifier, attack_cfg, eps=eps_val)
                attacks.append((name, eps_val, attack))
        else:
            # Single eps or no eps (e.g., CW, DeepFool)
            eps_val = eps_values if eps_values is not None else None
            attack = create_attack(classifier, attack_cfg, eps=eps_val)
            attacks.append((name, eps_val, attack))

    return attacks
