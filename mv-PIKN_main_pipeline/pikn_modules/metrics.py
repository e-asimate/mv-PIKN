"""Metrics for the mv-PIKN training pipeline."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
except Exception:
    accuracy_score = balanced_accuracy_score = confusion_matrix = f1_score = None
    precision_score = recall_score = roc_auc_score = average_precision_score = None

from .utils import (safe_float, status)


def binary_brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - prob) ** 2))


def binary_ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    inds = np.digitize(prob, bins) - 1
    ece = 0.0
    for i in range(len(bins) - 1):
        m = inds == i
        if np.any(m):
            acc = float(y_true[m].mean())
            conf = float(prob[m].mean())
            ece += float(m.mean()) * abs(acc - conf)
    return float(ece)


def safe_binary_roc_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    y_true = np.asarray(y_true).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return float("nan")


def safe_binary_average_precision(y_true: np.ndarray, prob: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    y_true = np.asarray(y_true).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, prob))
    except Exception:
        return float("nan")


def derived_probabilities(logits: Dict[str, torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    # Softened with temperature to prevent collapse into preictal trap
    p_ictal = torch.sigmoid(logits["ictal_logit"] / temperature)
    p_risk = torch.sigmoid(logits["risk_logit"] / temperature)
    p_preictal = (1.0 - p_ictal) * p_risk
    p_interictal = (1.0 - p_ictal) * (1.0 - p_risk)
    probs = torch.stack([p_interictal, p_preictal, p_ictal], dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probs


def threshold_predict(
    logits: Dict[str, torch.Tensor],
    preictal_thresh: float = 0.35,
    ictal_thresh: float = 0.45,
) -> torch.Tensor:
    probs = derived_probabilities(logits)
    preds = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
    preds[probs[:, 2] >= ictal_thresh] = 2
    preds[(probs[:, 1] >= preictal_thresh) & (preds != 2)] = 1
    return preds


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    class_names: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    if confusion_matrix is None or len(y_true) == 0:
        return None
    labels_present = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    names = class_names if class_names else [str(c) for c in labels_present]
    
    # Format compactly
    lines = [f"┌─ [CM] {split_name}"]
    lines.append("│ " + " " * 11 + "".join(f"{n:>12}" for n in names))
    for i, row in enumerate(cm):
        lines.append(f"│ True {names[i]:<7} " + "".join(f"{int(v):>12}" for v in row))
    status("\n".join(lines))
    return cm


def compute_monitor(val_metrics: Dict[str, Any]) -> float:
    bal_acc = safe_float(val_metrics.get("balanced_accuracy", float("nan")))
    risk_auroc = safe_float(val_metrics.get("risk_auroc", float("nan")))
    if np.isfinite(risk_auroc):
        return 0.5 * bal_acc + 0.5 * risk_auroc
    return bal_acc
