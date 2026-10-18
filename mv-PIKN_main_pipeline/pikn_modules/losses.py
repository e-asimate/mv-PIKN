"""Losses for the mv-PIKN training pipeline."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import torch
import torch.nn.functional as F



from .config import (TrainConfig)
from .model import (TransitionPIKN)
from .utils import (maybe_get_float)


def build_transition_targets(
    batch: Dict[str, Any], cfg: TrainConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = batch["label"].float()
    ictal_target = (labels == 2).float()

    tts = maybe_get_float(batch, "time_to_seizure_min", "tts_min", "time_to_seizure", "tau_min")
    if tts is not None:
        t = tts.view(-1)
        is_sentinel = t >= cfg.tts_sentinel_floor
        sentinel_frac = is_sentinel.float().mean().item()

        if sentinel_frac > 0.9:
            # Dynamic Fallback: If almost everything is a sentinel mask, 
            # the TTS supervision is mostly absent. Fall back to label proxy.
            risk_target = (labels / 2.0).clamp(0.0, 1.0)
        else:
            x = (cfg.transition_horizon_min - t) / max(cfg.transition_softness_min, 1e-6)
            risk_target = torch.sigmoid(x).clamp(0.0, 1.0)
            risk_target = torch.where(ictal_target > 0.5, torch.ones_like(risk_target), risk_target)
            risk_target = torch.where(is_sentinel, torch.zeros_like(risk_target), risk_target)
    else:
        # Fallback when TTS not available: use label ordering as soft proxy.
        risk_target = (labels / 2.0).clamp(0.0, 1.0)

    return risk_target, ictal_target


def binary_bce_with_soft_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focus: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if focus is not None:
        loss = loss * focus
    return loss.mean()


def compute_losses(
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    model: TransitionPIKN,
    ictal_pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    risk_target, ictal_target = build_transition_targets(batch, cfg)
    risk_focus = 1.0 + cfg.risk_focus * risk_target.detach()

    risk_loss = binary_bce_with_soft_targets(outputs["risk_logit"], risk_target, focus=risk_focus)
    ictal_loss = F.binary_cross_entropy_with_logits(
        outputs["ictal_logit"],
        ictal_target,
        pos_weight=ictal_pos_weight.to(outputs["ictal_logit"].device) if ictal_pos_weight is not None else None,
    )

    recon_loss = F.mse_loss(outputs["z_recon"], outputs["z_raw_t_enc"].detach())
    z_tdt_target = (outputs["z_raw_tdt"] + outputs["z_spec_tdt"]) / 2.0
    pred_loss = F.mse_loss(outputs["z_koopman_pred"], z_tdt_target.detach())
    physics_loss = F.mse_loss(outputs["z_wc_pred"], z_tdt_target.detach())
    spectral_align = F.mse_loss(outputs["z_raw_t"], outputs["z_spec_t"].detach())
    latent_l2 = (outputs["z_raw_t"].pow(2).mean() + outputs["z_spec_t"].pow(2).mean()) / 2.0
    stability = model.koopman.stability_loss(outputs["z_raw_t"])

    total_loss = (
        cfg.risk_weight * risk_loss
        + cfg.ictal_weight * ictal_loss
        + cfg.recon_weight * recon_loss
        + cfg.pred_weight * pred_loss
        + cfg.physics_weight * physics_loss
        + cfg.spectral_align_weight * spectral_align
        + cfg.latent_l2_weight * latent_l2
        + cfg.stability_weight * stability
    )

    metrics = {
        "risk_loss": float(risk_loss.detach().item()),
        "ictal_loss": float(ictal_loss.detach().item()),
        "recon_loss": float(recon_loss.detach().item()),
        "pred_loss": float(pred_loss.detach().item()),
        "physics_loss": float(physics_loss.detach().item()),
        "spectral_align": float(spectral_align.detach().item()),
        "latent_l2": float(latent_l2.detach().item()),
        "stability": float(stability.detach().item()),
        "risk_target_mean": float(risk_target.detach().mean().item()),
        "ictal_target_mean": float(ictal_target.detach().mean().item()),
        "risk_focus_mean": float(risk_focus.detach().mean().item()),
    }
    return total_loss, metrics
