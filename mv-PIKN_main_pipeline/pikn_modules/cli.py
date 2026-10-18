"""Cli for the mv-PIKN training pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import torch



from .config import (TrainConfig)
from .training import (train)
from .utils import (status)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHB-MIT EEG transition PIKN training (v12)")
    p.add_argument("--processed-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--max-loaded-shards", type=int, default=1)
    p.add_argument("--log-every-batches", type=int, default=50)
    p.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-every-samples", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--spectral-hidden-dim", type=int, default=128)
    p.add_argument("--temp-dropout", type=float, default=0.1)
    p.add_argument("--fusion-dropout", type=float, default=0.1)
    p.add_argument("--decoder-dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--physics-grad-clip", type=float, default=0.5)
    p.add_argument("--encoder-grad-clip", type=float, default=1.0)           
    p.add_argument("--classifier-grad-clip", type=float, default=2.0)        
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--risk-weight", type=float, default=1.0)
    p.add_argument("--ictal-weight", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=0.05)
    p.add_argument("--pred-weight", type=float, default=0.5)
    p.add_argument("--physics-weight", type=float, default=0.2)
    p.add_argument("--spectral-align-weight", type=float, default=0.0)
    p.add_argument("--latent-l2-weight", type=float, default=0.01)
    p.add_argument("--stability-weight", type=float, default=0.05)
    p.add_argument("--risk-focus", type=float, default=0.1)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--preictal-thresh", type=float, default=0.35)
    p.add_argument("--ictal-thresh", type=float, default=0.45)           
    p.add_argument("--calibration-bins", type=int, default=15)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--early-stop-patience", type=int, default=7)
    p.add_argument("--freeze-ictal-bias-epochs", type=int, default=3)
    p.add_argument("--transition-horizon-min", type=float, default=60.0)
    p.add_argument("--transition-softness-min", type=float, default=10.0)
    p.add_argument("--tts-sentinel-floor", type=float, default=990.0)    
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent-plot-max-points", type=int, default=6000)
    p.add_argument("--report-split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--use-umap", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        max_loaded_shards=args.max_loaded_shards,
        log_every_batches=args.log_every_batches,
        balanced_sampling=args.balanced_sampling,
        checkpoint_every_samples=args.checkpoint_every_samples,
        resume=args.resume,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        spectral_hidden_dim=args.spectral_hidden_dim,
        temp_dropout=args.temp_dropout,
        fusion_dropout=args.fusion_dropout,
        decoder_dropout=args.decoder_dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        physics_grad_clip=args.physics_grad_clip,
        encoder_grad_clip=args.encoder_grad_clip,
        classifier_grad_clip=args.classifier_grad_clip,
        compile_model=args.compile_model,
        amp=args.amp,
        deterministic=args.deterministic,
        risk_weight=args.risk_weight,
        ictal_weight=args.ictal_weight,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        physics_weight=args.physics_weight,
        spectral_align_weight=args.spectral_align_weight,
        latent_l2_weight=args.latent_l2_weight,
        stability_weight=args.stability_weight,
        risk_focus=args.risk_focus,
        class_weight_power=args.class_weight_power,
        preictal_thresh=args.preictal_thresh,
        ictal_thresh=args.ictal_thresh,
        calibration_bins=args.calibration_bins,
        dt=args.dt,
        seed=args.seed,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        freeze_ictal_bias_epochs=args.freeze_ictal_bias_epochs,
        transition_horizon_min=args.transition_horizon_min,
        transition_softness_min=args.transition_softness_min,
        tts_sentinel_floor=args.tts_sentinel_floor,
        save_predictions=args.save_predictions,
        save_plots=args.save_plots,
        latent_plot_max_points=args.latent_plot_max_points,
        report_split=args.report_split,
        use_umap=args.use_umap,
    )
    status(f"[config] {json.dumps(cfg.to_dict(), indent=2, sort_keys=True)}")
    train(cfg)
