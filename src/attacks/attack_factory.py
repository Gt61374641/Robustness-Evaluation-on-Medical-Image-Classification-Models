"""Attack factory for creating ART evasion attack instances from config.

Attacks are split into:
- Main attacks (FGSM, PGD, CW/DeepFool): core experiments, must-do
- Extended attacks (AutoPGD, SquareAttack, HopSkipJump): bonus experiments
"""

import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    DeepFool,
    AutoProjectedGradientDescent,
    AutoAttack,
    SquareAttack,
    HopSkipJump,
    UniversalPerturbation,
)


ATTACK_REGISTRY = {
    "FGSM": FastGradientMethod,
    "PGD": ProjectedGradientDescent,
    "CW": CarliniL2Method,
    "DeepFool": DeepFool,
    "AutoPGD": AutoProjectedGradientDescent,
    "AutoAttack": AutoAttack,  # strong ensemble (APGD-ce/dlr, FAB, Square) for defense eval
    "SquareAttack": SquareAttack,
    "HopSkipJump": HopSkipJump,
    "UAP": UniversalPerturbation,  # white-box, image-agnostic (one perturbation for all)
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
        eps_step = attack_cfg.get("eps_step", attack_eps / 10)  # standard heuristic
        return attack_cls(
            estimator=classifier,
            eps=attack_eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=attack_cfg.get("num_random_init", 1),
        )

    elif name == "CW":
        # ART's default batch_size is 1, which is unusably slow on 224x224 inputs.
        return attack_cls(
            classifier=classifier,
            confidence=attack_cfg.get("confidence", 0),
            max_iter=attack_cfg.get("max_iter", 100),
            learning_rate=attack_cfg.get("lr", 0.01),
            binary_search_steps=attack_cfg.get("binary_search_steps", 10),
            batch_size=attack_cfg.get("batch_size", 32),
        )

    elif name == "DeepFool":
        return attack_cls(
            classifier=classifier,
            max_iter=attack_cfg.get("max_iter", 50),
            epsilon=attack_cfg.get("epsilon", 1e-6),
            batch_size=attack_cfg.get("batch_size", 32),
        )

    elif name == "AutoPGD":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        return attack_cls(
            estimator=classifier,
            eps=attack_eps,
            eps_step=attack_eps / 10,
            max_iter=attack_cfg.get("max_iter", 100),
        )

    elif name == "AutoAttack":
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 8 / 255)
        eps_step = attack_cfg.get("eps_step", attack_eps / 10)
        batch_size = attack_cfg.get("batch_size", 32)
        kwargs = dict(
            estimator=classifier,
            norm="inf",
            eps=attack_eps,
            eps_step=eps_step,
            batch_size=batch_size,
        )
        # ART's default ensemble is APGD-CE + APGD-DLR + DeepFool + Square. The
        # DLR loss needs the 3rd-highest logit and is UNDEFINED for binary tasks
        # (crashes with "index -3 is out of bounds ... size 2"), so for <3 classes
        # we rebuild the ensemble without the DLR member. AutoAttack itself rejects
        # any candidate whose perturbation exceeds eps, so unbounded members
        # (DeepFool) stay within budget.
        nb = getattr(classifier, "nb_classes", None)
        if nb is not None and nb < 3:
            kwargs["attacks"] = [
                AutoProjectedGradientDescent(
                    estimator=classifier, norm="inf", eps=attack_eps,
                    eps_step=eps_step, max_iter=attack_cfg.get("max_iter", 100),
                    batch_size=batch_size, loss_type="cross_entropy",
                ),
                DeepFool(classifier=classifier, max_iter=50, batch_size=batch_size),
                SquareAttack(
                    estimator=classifier, norm="inf", eps=attack_eps,
                    max_iter=attack_cfg.get("max_queries", 1000), batch_size=batch_size,
                ),
            ]
        return attack_cls(**kwargs)

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

    elif name == "UAP":
        # Universal Adversarial Perturbation: a SINGLE image-agnostic perturbation
        # is fitted on the supplied images (via a base per-image attack) and then
        # applied to all of them. White-box (the base attack uses gradients).
        attack_eps = eps if eps is not None else attack_cfg.get("eps", 16 / 255)
        attacker = attack_cfg.get("attacker", "fgsm")
        # The base per-image step MUST be small relative to the projection budget
        # (attack_eps). ART overwrites then projects the universal noise:
        #   noise = projection(noise + step*sign(grad), eps).
        # With step == eps the noise is dominated by a single image's full-budget
        # perturbation and never accumulates across images -> fooling rate ~0
        # (and worse as eps grows). A small step (eps/5) lets the universal
        # perturbation build up over many images. Verified: ASR 0.00 -> 0.85+.
        base_step = attack_cfg.get("attacker_step", attack_eps / 5.0)
        attacker_params = attack_cfg.get("attacker_params", {"eps": base_step})
        return attack_cls(
            classifier=classifier,
            attacker=attacker,
            attacker_params=attacker_params,
            delta=attack_cfg.get("delta", 0.2),
            max_iter=attack_cfg.get("max_iter", 20),
            eps=attack_eps,
            norm=np.inf,
            batch_size=attack_cfg.get("batch_size", 32),
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


def create_defense_eval_attacks(classifier: PyTorchClassifier, cfg: dict):
    """Build the STRONG attacks used to evaluate a defended (adversarially trained)
    model, from the ``defense_eval`` config section. Using a strong protocol
    (PGD-50 with random restarts + AutoAttack) avoids overestimating robustness.

    Returns a list of (label, eps_value, attack_instance) tuples.
    """
    eval_cfg = cfg.get("defense_eval", {})
    attacks = []

    pgd_cfg = eval_cfg.get("pgd_eval")
    if pgd_cfg:
        base = {
            "name": "PGD",
            "max_iter": pgd_cfg.get("max_iter", 50),
            "num_random_init": pgd_cfg.get("num_random_init", 5),
        }
        for eps_val in pgd_cfg.get("eps", []):
            attacks.append((f"PGD{base['max_iter']}-{base['num_random_init']}restart",
                            eps_val, create_attack(classifier, base, eps=eps_val)))

    aa_cfg = eval_cfg.get("autoattack")
    # For <3 classes we keep skipping AutoAttack in the DEFENSE eval, even though
    # create_attack() now supports a binary-safe reduced ensemble (used by the
    # standard-model attacks_extra comparison). Reason: all existing defended-model
    # results were produced under the PGD-50+restarts-only protocol; adding
    # AutoAttack for new runs only would silently mix two protocols in the same
    # comparison tables. Backfill ALL defended models first if you enable this.
    nb = getattr(classifier, "nb_classes", None)
    if aa_cfg and nb is not None and nb < 3:
        print(f"[defense_eval] Skipping AutoAttack for binary task ({nb} classes): "
              "PGD-50 + restarts is the strong eval protocol all defended models share.")
        aa_cfg = None
    if aa_cfg:
        for eps_val in aa_cfg.get("eps", []):
            attacks.append(("AutoAttack", eps_val,
                            create_attack(classifier, {"name": "AutoAttack"}, eps=eps_val)))

    return attacks
