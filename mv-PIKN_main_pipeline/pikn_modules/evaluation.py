"""Evaluation for the mv-PIKN training pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from .config import (TrainConfig)
from .data import (move_batch_to_device, unwrap_batch)
from .diagnostics import (_compute_logit_stats, reconstruction_sanity_check)
from .losses import (build_transition_targets, compute_losses)
from .metrics import (binary_brier_score, binary_ece, derived_probabilities, log_confusion_matrix, safe_binary_average_precision, safe_binary_roc_auc, threshold_predict)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    ictal_pos_weight: Optional[torch.Tensor] = None,
    epoch: int = -1,
    split_name: str = "val",
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    logit_stats_agg: Dict[str, float] = defaultdict(float)
    recon_sanity_done = False
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    sample_rows: List[Dict[str, Any]] = []
    latent_cache: Dict[str, List[torch.Tensor]] = defaultdict(list)
    risk_prob_all: List[np.ndarray] = []
    ictal_prob_all: List[np.ndarray] = []
    risk_target_all: List[np.ndarray] = []
    ictal_target_all: List[np.ndarray] = []
    prob_class_all: List[np.ndarray] = []

    pbar = tqdm(
        total=len(loader) if hasattr(loader, "__len__") else None,
        desc=f"{split_name} e{epoch:03d}" if epoch >= 0 else split_name,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
        smoothing=0.05,
    )

    for bi, batch in enumerate(loader, start=1):
        batch = unwrap_batch(batch)
        batch = move_batch_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
            if amp_enabled else nullcontext()
        )
        with autocast_ctx:
            outputs = model(batch)
            loss, metrics = compute_losses(batch, outputs, cfg, model, ictal_pos_weight=ictal_pos_weight)

        totals["loss"] += float(loss.detach().item())
        for k, v in metrics.items():
            totals[k] += float(v)

        preds = threshold_predict(outputs, preictal_thresh=cfg.preictal_thresh, ictal_thresh=cfg.ictal_thresh)
        probs = derived_probabilities(outputs).detach().float().cpu()
        risk_target, ictal_target = build_transition_targets(batch, cfg)

        y_true.append(batch["label"].detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())
        prob_class_all.append(probs.numpy())
        risk_prob_all.append(torch.sigmoid(outputs["risk_logit"]).detach().cpu().numpy())
        ictal_prob_all.append(torch.sigmoid(outputs["ictal_logit"]).detach().cpu().numpy())
        risk_target_all.append(risk_target.detach().cpu().numpy())
        ictal_target_all.append(ictal_target.detach().cpu().numpy())
        n_batches += 1

        for k, v in _compute_logit_stats(outputs).items():
            logit_stats_agg[k] += v

        if not recon_sanity_done and (epoch == 1 or epoch % 5 == 0 or epoch == -1):
            reconstruction_sanity_check(outputs, epoch=epoch, split_name=split_name)
            recon_sanity_done = True

        if cfg.save_predictions:
            labels_cpu = batch["label"].detach().cpu().numpy().tolist()
            pred_cpu = preds.detach().cpu().numpy().tolist()
            p_cpu = probs.numpy()
            risk_prob_list = torch.sigmoid(outputs["risk_logit"]).detach().cpu().numpy().tolist()
            ictal_prob_list = torch.sigmoid(outputs["ictal_logit"]).detach().cpu().numpy().tolist()
            r_target = risk_target.detach().cpu().numpy().tolist()
            i_target = ictal_target.detach().cpu().numpy().tolist()
            for i in range(len(labels_cpu)):
                sample_rows.append({
                    "batch": bi,
                    "true_label": int(labels_cpu[i]),
                    "pred_label": int(pred_cpu[i]),
                    "true_any_seizure": int(labels_cpu[i] > 0),
                    "true_ictal": int(labels_cpu[i] == 2),
                    "p_interictal": float(p_cpu[i, 0]),
                    "p_preictal": float(p_cpu[i, 1]),
                    "p_ictal": float(p_cpu[i, 2]),
                    "risk_prob": float(risk_prob_list[i]),
                    "ictal_prob": float(ictal_prob_list[i]),
                    "risk_target": float(r_target[i]),
                    "ictal_target": float(i_target[i]),
                })

        latent_cache["fused"].append(outputs["fused"].detach().cpu())
        latent_cache["z_raw_t"].append(outputs["z_raw_t"].detach().cpu())
        latent_cache["risk_logit"].append(outputs["risk_logit"].detach().cpu())
        latent_cache["ictal_logit"].append(outputs["ictal_logit"].detach().cpu())

        pbar.update(1)

    pbar.close()

    out: Dict[str, Any] = {k: v / max(1, n_batches) for k, v in totals.items()}
    for k, v in logit_stats_agg.items():
        out[f"diag_{k}"] = v / max(1, n_batches)

    yt = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    yp = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)

    if prob_class_all:
        all_probs = np.concatenate(prob_class_all, axis=0) 
        out["diag_prob_interictal_mean"] = float(all_probs[:, 0].mean())
        out["diag_prob_preictal_mean"] = float(all_probs[:, 1].mean())
        out["diag_prob_ictal_mean"] = float(all_probs[:, 2].mean())
    else:
        out["diag_prob_interictal_mean"] = float("nan")
        out["diag_prob_preictal_mean"] = float("nan")
        out["diag_prob_ictal_mean"] = float("nan")

    if len(yt) > 0 and accuracy_score is not None:
        out["accuracy"] = float(accuracy_score(yt, yp))
        out["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
        out["macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        out["macro_precision"] = float(precision_score(yt, yp, average="macro", zero_division=0))
        out["macro_recall"] = float(recall_score(yt, yp, average="macro", zero_division=0))
        per_cls_rec = recall_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2])
        per_cls_prec = precision_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2])
        names = ["interictal", "preictal", "ictal"]
        for i, name in enumerate(names):
            out[f"recall_{name}"] = float(per_cls_rec[i])
            out[f"precision_{name}"] = float(per_cls_prec[i])
            
        log_confusion_matrix(yt, yp, f"{split_name} epoch={epoch}", class_names=["Interictal", "Preictal", "Ictal"])

        risk_prob_np = np.concatenate(risk_prob_all) if risk_prob_all else np.array([], dtype=np.float32)
        ictal_prob_np = np.concatenate(ictal_prob_all) if ictal_prob_all else np.array([], dtype=np.float32)
        risk_target_np = np.concatenate(risk_target_all) if risk_target_all else np.array([], dtype=np.float32)
        ictal_target_np = np.concatenate(ictal_target_all) if ictal_target_all else np.array([], dtype=np.float32)
        any_true = (yt > 0).astype(np.int32)
        ictal_true = (yt == 2).astype(np.int32)
        out["risk_target_mean"] = float(risk_target_np.mean()) if risk_target_np.size else float("nan")
        out["ictal_target_mean"] = float(ictal_target_np.mean()) if ictal_target_np.size else float("nan")
        out["risk_brier"] = binary_brier_score(any_true, risk_prob_np)
        out["risk_ece"] = binary_ece(any_true, risk_prob_np, n_bins=cfg.calibration_bins)
        out["risk_auroc"] = safe_binary_roc_auc(any_true, risk_prob_np)
        out["risk_auprc"] = safe_binary_average_precision(any_true, risk_prob_np)
        out["ictal_brier"] = binary_brier_score(ictal_true, ictal_prob_np)
        out["ictal_ece"] = binary_ece(ictal_true, ictal_prob_np, n_bins=cfg.calibration_bins)
        out["ictal_auroc"] = safe_binary_roc_auc(ictal_true, ictal_prob_np)
        out["ictal_auprc"] = safe_binary_average_precision(ictal_true, ictal_prob_np)

    if sample_rows:
        out["sample_rows"] = sample_rows  

    latent_bundle = {
        k: torch.cat(v, dim=0) if len(v) else torch.empty(0)
        for k, v in latent_cache.items()
    }
    return out, latent_bundle


def extract_predictions(eval_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = eval_out.get("sample_rows", [])
    return rows if isinstance(rows, list) else []
