"""Robustness evaluation metrics.

Core metrics:
- Robust Accuracy: accuracy on adversarial samples, computed ONLY on samples
  that were correctly classified on clean data (avoids polluting interpretation).
- Attack Success Rate (ASR): fraction of originally-correct samples that become
  misclassified after attack.
- Accuracy Drop: clean accuracy minus adversarial accuracy (on full test set).

Calibration metrics:
- ECE (Expected Calibration Error): measures how well model confidence matches
  actual accuracy. In medical imaging, "high confidence wrong predictions"
  is itself a meaningful finding.
- Confidence statistics: distributions for clean vs adversarial, correct vs wrong.
"""

import numpy as np
from collections import defaultdict


def compute_clean_metrics(
    preds: np.ndarray,
    probs: np.ndarray,
    true_labels: np.ndarray,
    class_names: list = None,
) -> dict:
    """Full clean-baseline classification metrics for medical image evaluation.

    Reports accuracy, balanced accuracy, ROC-AUC, per-class & macro/weighted
    precision/recall/F1, and the confusion matrix. ``probs`` is the (N, C) matrix
    of softmax probabilities (needed for ROC-AUC).
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    preds = np.asarray(preds)
    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels)
    n_classes = probs.shape[1]
    labels_idx = list(range(n_classes))
    if class_names is None:
        class_names = [str(i) for i in labels_idx]

    accuracy = float((preds == true_labels).mean())
    balanced_acc = float(balanced_accuracy_score(true_labels, preds))

    prec, rec, f1, support = precision_recall_fscore_support(
        true_labels, preds, labels=labels_idx, average=None, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels_idx
    }
    macro = precision_recall_fscore_support(
        true_labels, preds, labels=labels_idx, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        true_labels, preds, labels=labels_idx, average="weighted", zero_division=0
    )

    # ROC-AUC: positive-class prob for binary, one-vs-rest macro for multiclass.
    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(true_labels, probs[:, 1]))
        else:
            roc_auc = float(roc_auc_score(
                true_labels, probs, multi_class="ovr", average="macro", labels=labels_idx
            ))
    except ValueError:
        roc_auc = None  # e.g. a class missing from this split

    cm = confusion_matrix(true_labels, preds, labels=labels_idx)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def compute_robust_accuracy(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Compute both full and conditional robust accuracy.

    Let ``still_correct`` = #(clean-correct AND still-correct-after-attack).

    - **Full robust accuracy**        = still_correct / N (all test samples).
      This is the y-axis of the paper's accuracy-vs-epsilon curves.
    - **Conditional robust accuracy**  = still_correct / #(clean-correct).
      Robustness among samples the model originally got right; ASR = 1 - this.

    Args:
        clean_preds: Model predictions on clean data (N,).
        adv_preds: Model predictions on adversarial data (N,).
        true_labels: Ground truth labels (N,).

    Returns:
        Dict with full_robust_accuracy, conditional_robust_accuracy, counts, and a
        backward-compatible ``robust_accuracy`` alias (== conditional).
    """
    n_total = len(true_labels)
    originally_correct = clean_preds == true_labels
    num_originally_correct = int(originally_correct.sum())

    num_still_correct = int(
        (adv_preds[originally_correct] == true_labels[originally_correct]).sum()
    ) if num_originally_correct > 0 else 0

    conditional = (num_still_correct / num_originally_correct) if num_originally_correct > 0 else 0.0
    full = (num_still_correct / n_total) if n_total > 0 else 0.0

    return {
        "full_robust_accuracy": float(full),
        "conditional_robust_accuracy": float(conditional),
        "robust_accuracy": float(conditional),  # backward-compatible alias
        "num_total": int(n_total),
        "num_originally_correct": num_originally_correct,
        "num_still_correct": num_still_correct,
    }


def bootstrap_robust_accuracy(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for full & conditional robust accuracy.

    Resamples test samples with replacement to give error bands on the
    complexity-vs-epsilon curves (a single-seed substitute for the paper's
    10-run mean +/- std).
    """
    clean_preds = np.asarray(clean_preds)
    adv_preds = np.asarray(adv_preds)
    true_labels = np.asarray(true_labels)
    n = len(true_labels)
    clean_correct = clean_preds == true_labels
    both_correct = clean_correct & (adv_preds == true_labels)

    rng = np.random.default_rng(seed)
    full, cond = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        bc = both_correct[s]
        denom = clean_correct[s].sum()
        full[b] = bc.mean()
        cond[b] = bc.sum() / denom if denom > 0 else 0.0

    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "full_ci_low": float(np.percentile(full, lo)),
        "full_ci_high": float(np.percentile(full, hi)),
        "conditional_ci_low": float(np.percentile(cond, lo)),
        "conditional_ci_high": float(np.percentile(cond, hi)),
        "n_boot": int(n_boot),
        "ci": ci,
    }


def compute_attack_success_rate(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute Attack Success Rate (ASR).

    ASR = fraction of originally-correct samples misclassified after attack
        = 1 - conditional robust accuracy.
    """
    result = compute_robust_accuracy(clean_preds, adv_preds, true_labels)
    return 1.0 - result["conditional_robust_accuracy"]


def compute_accuracy_drop(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Compute accuracy drop on the full test set."""
    clean_acc = (clean_preds == true_labels).mean()
    adv_acc = (adv_preds == true_labels).mean()
    return {
        "clean_accuracy": float(clean_acc),
        "adversarial_accuracy": float(adv_acc),
        "accuracy_drop": float(clean_acc - adv_acc),
    }


def compute_per_class_robustness(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
    class_names: list = None,
) -> dict:
    """Compute per-class ASR to identify which disease categories are more vulnerable.

    Returns:
        Dict mapping class_name -> {clean_acc, robust_acc, asr, count}.
    """
    classes = sorted(np.unique(true_labels))
    results = {}

    for cls in classes:
        mask = true_labels == cls
        cls_clean = clean_preds[mask]
        cls_adv = adv_preds[mask]
        cls_labels = true_labels[mask]

        clean_correct = cls_clean == cls_labels
        num_correct = clean_correct.sum()

        if num_correct > 0:
            still_correct = (cls_adv[clean_correct] == cls_labels[clean_correct]).sum()
            robust_acc = float(still_correct / num_correct)
        else:
            robust_acc = 0.0

        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        results[name] = {
            "clean_accuracy": float(clean_correct.mean()),
            "robust_accuracy": robust_acc,
            "asr": 1.0 - robust_acc,
            "count": int(mask.sum()),
        }

    return results


def compute_pred_distribution(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
    class_names: list = None,
) -> dict:
    """Predicted-class distribution on clean vs adversarial inputs.

    Diagnostic for the large-epsilon degeneration: when an attack pushes a model
    to collapse onto the majority class, ``full_robust_accuracy`` stays
    artificially high (it just tracks that class's base rate) even though the
    model has stopped discriminating. Comparing the clean and adversarial
    distributions to the true-label distribution makes that collapse explicit.

    Returns counts and fractions per class for clean preds, adversarial preds,
    and the ground truth, plus a ``collapse`` summary (the most-predicted
    adversarial class and its fraction).
    """
    clean_preds = np.asarray(clean_preds)
    adv_preds = np.asarray(adv_preds)
    true_labels = np.asarray(true_labels)

    n_classes = int(max(true_labels.max(initial=-1),
                        clean_preds.max(initial=-1),
                        adv_preds.max(initial=-1))) + 1
    if class_names is None or len(class_names) < n_classes:
        class_names = [str(i) for i in range(n_classes)]

    n = len(true_labels)

    def _dist(arr):
        counts = np.bincount(arr, minlength=n_classes)
        return {
            class_names[i]: {"count": int(counts[i]),
                             "fraction": float(counts[i] / n) if n else 0.0}
            for i in range(n_classes)
        }

    adv_counts = np.bincount(adv_preds, minlength=n_classes)
    top_cls = int(adv_counts.argmax())

    return {
        "true": _dist(true_labels),
        "clean": _dist(clean_preds),
        "adv": _dist(adv_preds),
        "collapse": {
            "adv_majority_class": class_names[top_cls],
            "adv_majority_fraction": float(adv_counts[top_cls] / n) if n else 0.0,
        },
    }


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well model confidence aligns with actual accuracy.
    High ECE under attack means the model is confidently wrong — a critical
    finding for medical imaging safety.

    Args:
        confidences: Maximum softmax probabilities (N,).
        predictions: Predicted class labels (N,).
        true_labels: Ground truth labels (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dict with ece, bin_accuracies, bin_confidences, bin_counts.
    """
    confidences = np.asarray(confidences)
    if np.any(confidences < -1e-6) or np.any(confidences > 1.0 + 1e-6):
        raise ValueError(
            "ECE expects confidence values in [0, 1]. "
            "Convert model logits to softmax probabilities before calling compute_ece."
        )
    confidences = np.clip(confidences, 0.0, 1.0)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum()
        bin_counts.append(int(count))

        if count > 0:
            bin_acc = (predictions[in_bin] == true_labels[in_bin]).mean()
            bin_conf = confidences[in_bin].mean()
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Weighted ECE
    total = len(confidences)
    ece = sum(
        (count / total) * abs(acc - conf)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        if count > 0
    )

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def compute_confidence_stats(
    clean_confidences: np.ndarray,
    clean_preds: np.ndarray,
    adv_confidences: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Compute confidence distribution statistics for clean vs adversarial.

    Separates correct and incorrect predictions for more informative analysis.
    """
    # Clean
    clean_correct_mask = clean_preds == true_labels
    clean_conf_correct = clean_confidences[clean_correct_mask]
    clean_conf_wrong = clean_confidences[~clean_correct_mask]

    # Adversarial
    adv_correct_mask = adv_preds == true_labels
    adv_conf_correct = adv_confidences[adv_correct_mask]
    adv_conf_wrong = adv_confidences[~adv_correct_mask]

    def _stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "count": int(len(arr)),
        }

    return {
        "clean_correct": _stats(clean_conf_correct),
        "clean_wrong": _stats(clean_conf_wrong),
        "adv_correct": _stats(adv_conf_correct),
        "adv_wrong": _stats(adv_conf_wrong),
    }


def evaluate_robustness(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
    clean_confidences: np.ndarray = None,
    adv_confidences: np.ndarray = None,
    class_names: list = None,
    bootstrap: bool = False,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict:
    """Run all robustness evaluation metrics.

    Args:
        clean_preds: Predictions on clean data.
        adv_preds: Predictions on adversarial data.
        true_labels: Ground truth labels.
        clean_confidences: Max softmax probs on clean data (optional, for ECE).
        adv_confidences: Max softmax probs on adversarial data (optional, for ECE).
        class_names: List of class names for per-class analysis.

    Returns:
        Comprehensive dict of all metrics.
    """
    results = {}

    # Core metrics
    results["robust_accuracy"] = compute_robust_accuracy(clean_preds, adv_preds, true_labels)
    if bootstrap:
        results["robust_accuracy"].update(
            bootstrap_robust_accuracy(clean_preds, adv_preds, true_labels,
                                      n_boot=n_boot, seed=bootstrap_seed)
        )
    results["asr"] = compute_attack_success_rate(clean_preds, adv_preds, true_labels)
    results["accuracy_drop"] = compute_accuracy_drop(clean_preds, adv_preds, true_labels)
    results["per_class"] = compute_per_class_robustness(clean_preds, adv_preds, true_labels, class_names)
    results["pred_distribution"] = compute_pred_distribution(clean_preds, adv_preds, true_labels, class_names)

    # Calibration metrics (if confidences provided)
    if clean_confidences is not None and adv_confidences is not None:
        results["ece_clean"] = compute_ece(clean_confidences, clean_preds, true_labels)
        results["ece_adv"] = compute_ece(adv_confidences, adv_preds, true_labels)
        results["confidence_stats"] = compute_confidence_stats(
            clean_confidences, clean_preds, adv_confidences, adv_preds, true_labels
        )

    return results
