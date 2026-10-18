"""Checkpoints for the mv-PIKN training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import random
import numpy as np
import torch
import torch.nn as nn



from .config import (TrainConfig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    cfg: TrainConfig,
    batch_idx: int = 0,
    global_step: int = 0,
    samples_seen_total: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "samples_seen_total": samples_seen_total,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    rng = ckpt.get("rng_state", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    return ckpt
