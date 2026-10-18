"""Diagnostics for the mv-PIKN training pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



from .model import (TransitionPIKN)
from .utils import (status)


def zero_nan_grads(model: nn.Module) -> int:
    count = 0
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            param.grad.zero_()
            count += 1
    return count


def compute_grad_norms(model: nn.Module) -> Dict[str, float]:
    total_sq = 0.0
    param_max = 0.0
    component_sq: Dict[str, float] = defaultdict(float)
    component_map = {
        "raw_encoder": "encoder",
        "spec_encoder": "encoder",
        "proj_": "encoder",
        "fusion": "classifier",
        "risk_head": "classifier",
        "ictal_head": "classifier",
        "koopman": "physics",
        "wc_cell": "physics",
        "decoder": "decoder",
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = float(param.grad.detach().norm(2))
        total_sq += g * g
        param_max = max(param_max, g)
        matched = False
        for prefix, comp in component_map.items():
            if name.startswith(prefix):
                component_sq[comp] += g * g
                matched = True
                break
        if not matched:
            component_sq["other"] += g * g
    out = {"grad_norm_total": math.sqrt(total_sq), "grad_norm_max_param": param_max}
    for comp, sq in component_sq.items():
        out[f"grad_norm_{comp}"] = math.sqrt(sq)
    return out


@torch.no_grad()
def reconstruction_sanity_check(outputs: Dict[str, torch.Tensor], epoch: int, split_name: str = "val") -> None:
    z_recon = outputs["z_recon"].detach().float().cpu()
    z_enc = outputs["z_raw_t_enc"].detach().float().cpu()
    mse = float(F.mse_loss(z_recon, z_enc))
    cos = float(F.cosine_similarity(z_recon, z_enc, dim=-1).mean())

    z_koop = outputs["z_koopman_pred"].detach().float().cpu()
    z_tdt = ((outputs["z_raw_tdt"] + outputs["z_spec_tdt"]) / 2.0).detach().float().cpu()
    koop_mse = float(F.mse_loss(z_koop, z_tdt))
    koop_cos = float(F.cosine_similarity(z_koop, z_tdt, dim=-1).mean())

    status(f"[recon e{epoch:03d} {split_name}] MSE={mse:.4f} (Cos={cos:.4f}) | Koop MSE={koop_mse:.4f} (Cos={koop_cos:.4f})")


def _compute_logit_stats(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("risk_logit", "ictal_logit"):
        t = outputs.get(key)
        if t is None:
            out[f"{key}_mean"] = float("nan")
        else:
            out[f"{key}_mean"] = float(t.detach().float().mean().item())
    return out


def clip_physics_grads(model: TransitionPIKN, max_norm: float) -> None:
    physics_params = list(model.koopman.parameters()) + list(model.wc_cell.parameters())
    nn.utils.clip_grad_norm_(physics_params, max_norm)


def clip_encoder_grads(model: TransitionPIKN, max_norm: float) -> None:
    encoder_params = (
        list(model.raw_encoder.parameters())
        + list(model.spec_encoder.parameters())
        + list(model.proj_raw_past.parameters())
        + list(model.proj_raw_t.parameters())
        + list(model.proj_raw_tdt.parameters())
        + list(model.proj_spec_t.parameters())
        + list(model.proj_spec_tdt.parameters())
    )
    if model.raw_past_encoder is not model.raw_encoder:
        encoder_params += list(model.raw_past_encoder.parameters())
    nn.utils.clip_grad_norm_(encoder_params, max_norm)


def clip_classifier_grads(model: TransitionPIKN, max_norm: float) -> None:
    classifier_params = (
        list(model.fusion.parameters())
        + list(model.risk_head.parameters())
        + list(model.ictal_head.parameters())
    )
    nn.utils.clip_grad_norm_(classifier_params, max_norm)
