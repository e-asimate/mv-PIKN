"""Config for the mv-PIKN training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import platform
import torch



from .utils import (status)


@dataclass
class TrainConfig:
    processed_dir: Path
    output_dir: Path

    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    max_loaded_shards: int = 1
    log_every_batches: int = 50

    balanced_sampling: bool = True
    checkpoint_every_samples: int = 0
    resume: Optional[str] = None

    latent_dim: int = 32
    hidden_dim: int = 128
    spectral_hidden_dim: int = 128
    temp_dropout: float = 0.1
    fusion_dropout: float = 0.1
    decoder_dropout: float = 0.1

    risk_weight: float = 1.0
    ictal_weight: float = 1.0            # Reduced from 3.0
    recon_weight: float = 0.05
    pred_weight: float = 0.5
    physics_weight: float = 0.2
    spectral_align_weight: float = 0.0
    latent_l2_weight: float = 0.01
    stability_weight: float = 0.05
    risk_focus: float = 0.1              # Reduced from 0.5

    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    physics_grad_clip: float = 0.5
    encoder_grad_clip: float = 1.0       
    classifier_grad_clip: float = 2.0    

    compile_model: bool = False
    amp: bool = False
    deterministic: bool = False

    class_weight_power: float = 0.5
    preictal_thresh: float = 0.35
    ictal_thresh: float = 0.45           
    calibration_bins: int = 15

    dt: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stop_patience: int = 7
    freeze_ictal_bias_epochs: int = 3
    transition_horizon_min: float = 60.0
    transition_softness_min: float = 10.0
    tts_sentinel_floor: float = 990.0    

    save_predictions: bool = True
    save_plots: bool = True
    latent_plot_max_points: int = 6000
    report_split: str = "val"
    use_umap: bool = True

    overwrite: bool = False

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        self.predictions_dir = self.output_dir / "predictions"
        for p in (self.output_dir, self.checkpoints_dir, self.logs_dir,
                  self.plots_dir, self.predictions_dir):
            p.mkdir(parents=True, exist_ok=True)

        if platform.system().lower().startswith("win") and self.num_workers > 0:
            status("[warn] Windows detected; forcing num_workers=0")
            self.num_workers = 0
        if self.num_workers <= 0:
            self.persistent_workers = False
        self.pin_memory = bool(self.pin_memory and torch.cuda.is_available())
        if self.amp and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                if any(x in gpu_name for x in ("1060", "1050", "Pascal")):
                    status(f"[warn] AMP disabled on Pascal-style GPU: {gpu_name}")
                    self.amp = False
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("processed_dir", "output_dir", "checkpoints_dir",
                    "logs_dir", "plots_dir", "predictions_dir"):
            d[key] = str(getattr(self, key))
        return d
