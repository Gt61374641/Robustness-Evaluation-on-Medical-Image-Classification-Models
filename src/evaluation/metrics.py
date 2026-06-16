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


def compute_robust_accuracy(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Compute robust accuracy on originally correct samples only.

    Args:
        clean_preds: Model predictions on clean data (N,).
        adv_preds: Model predictions on adversarial data (N,).
        true_labels: Ground truth labels (N,).

    Returns:
        Dict with robust_accuracy, num_originally_correct, num_still_correct.
    """
    originally_correct = clean_preds == true_labels
    num_originally_correct = originally_correct.sum()

    if num_originally_correct == 0:
        return {
            "robust_accuracy": 0.0,
            "num_originally_correct": 0,
            "num_still_correct": 0,
        }

    still_correct = (adv_preds[originally_correct] == true_labels[originally_correct])
    num_still_correct = still_correct.sum()

    return {
        "robust_accuracy": float(num_still_correct / num_originally_correct),
        "num_originally_correct": int(num_originally_correct),
        "num_still_correct": int(num_still_correct),
    }


def compute_attack_success_rate(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute Attack Success Rate (ASR).

    ASR = fraction of originally-correct samples that are misclassified after attack.
    ASR = 1 - robust_accuracy (on originally correct samples).
    """
    result = compute_robust_accuracy(clean_preds, adv_preds, true_labels)
    return 1.0 - result["robust_accuracy"]


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
    results["asr"] = compute_attack_success_rate(clean_preds, adv_preds, true_labels)
    results["accuracy_drop"] = compute_accuracy_drop(clean_preds, adv_preds, true_labels)
    results["per_class"] = compute_per_class_robustness(clean_preds, adv_preds, true_labels, class_names)

    # Calibration metrics (if confidences provided)
    if clean_confidences is not None and adv_confidences is not None:
        results["ece_clean"] = compute_ece(clean_confidences, clean_preds, true_labels)
        results["ece_adv"] = compute_ece(adv_confidences, adv_preds, true_labels)
        results["confidence_stats"] = compute_confidence_stats(
            clean_confidences, clean_preds, adv_confidences, adv_preds, true_labels
        )

    return results
