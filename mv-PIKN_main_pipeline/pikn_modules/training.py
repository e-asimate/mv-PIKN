"""Training for the mv-PIKN training pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import math
import time
import numpy as np
import torch
import torch.nn as nn

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

from .checkpoints import (load_checkpoint, save_checkpoint)
from .config import (TrainConfig)
from .data import (build_loaders, check_first_batch, infer_model_shapes_from_batch, move_batch_to_device, unwrap_batch)
from .diagnostics import (clip_classifier_grads, clip_encoder_grads, clip_physics_grads, compute_grad_norms, zero_nan_grads)
from .evaluation import (evaluate, extract_predictions)
from .losses import (compute_losses)
from .metrics import (compute_monitor)
from .model import (TransitionPIKN)
from .utils import (configure_torch_runtime, format_seconds, safe_float, save_csv, save_json, section, seed_everything, status)
from .visualization import (plot_calibration, plot_confusion_matrix, plot_histogram_by_class, plot_latent_projection, save_history_plots)


def train(cfg: TrainConfig) -> Dict[str, Any]:
    section("Train")
    status("[train] starting transition PIKN v12")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device)
    configure_torch_runtime(use_cuda=(device.type == "cuda"), use_tf32=True)

    train_loader, val_loader, test_loader, class_weights_cpu, class_counts = build_loaders(cfg)
    class_weights = class_weights_cpu.to(device)  

    raw_weight = float((class_counts[0] + class_counts[1]) / max(class_counts[2], 1))
    capped_weight = min(raw_weight, 20.0)
    ictal_pos_weight = torch.tensor(capped_weight, dtype=torch.float32, device=device)

    first_batch = unwrap_batch(next(iter(train_loader)))
    check_first_batch(first_batch)
    raw_past_channels, raw_channels, seq_len, spec_in_dim = infer_model_shapes_from_batch(first_batch)

    model = TransitionPIKN(
        raw_past_channels=raw_past_channels,
        raw_channels=raw_channels,
        seq_len=seq_len,
        spec_in_dim=spec_in_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        spectral_hidden_dim=cfg.spectral_hidden_dim,
        temp_dropout=cfg.temp_dropout,
        fusion_dropout=cfg.fusion_dropout,
        decoder_dropout=cfg.decoder_dropout,
        dt=cfg.dt,
    ).to(device)

    total_samples = class_counts.sum()
    true_prior = np.clip(class_counts / max(total_samples, 1), 1e-6, 1.0)
    risk_prior = float(np.clip((class_counts[1] + class_counts[2]) / max(total_samples, 1), 1e-6, 1 - 1e-6))
    ictal_prior = float(np.clip(class_counts[2] / max(total_samples, 1), 1e-6, 1 - 1e-6))

    with torch.no_grad():
        risk_bias = math.log(risk_prior / (1.0 - risk_prior))
        risk_bias = max(min(risk_bias, 2.0), -2.0)
        model.risk_head[-1].bias.fill_(risk_bias)

        ictal_bias = math.log(ictal_prior / (1.0 - ictal_prior))
        ictal_bias = max(ictal_bias, -3.0)
        model.ictal_head[-1].bias.fill_(ictal_bias)

    status(f"[train] class counts: {class_counts.tolist()}")
    status(f"[train] risk bias init={risk_bias:.4f}, ictal bias init={ictal_bias:.4f}")

    if cfg.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            status("[train] torch.compile enabled")
        except Exception as e:
            status(f"[warn] torch.compile failed: {e}")

    optimizer_kwargs: Dict[str, Any] = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    if cfg.freeze_ictal_bias_epochs > 0:
        model.ictal_head[-1].bias.requires_grad = False
        
    best_val = -float("inf")
    best_epoch = -1
    patience_left = cfg.early_stop_patience
    history: List[Dict[str, Any]] = []
    global_step = 0
    samples_seen_total = 0
    start_epoch = 1
    start_batch_in_epoch = 0

    if cfg.resume:
        ckpt = load_checkpoint(Path(cfg.resume), model, optimizer, scaler)
        start_epoch = int(ckpt.get("epoch", 1))
        start_batch_in_epoch = int(ckpt.get("batch_idx", 0))
        global_step = int(ckpt.get("global_step", 0))
        samples_seen_total = int(ckpt.get("samples_seen_total", 0))
        best_val = float(ckpt.get("best_val", best_val))

    total_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        if cfg.freeze_ictal_bias_epochs > 0 and epoch == cfg.freeze_ictal_bias_epochs + 1:
            if not model.ictal_head[-1].bias.requires_grad:
                model.ictal_head[-1].bias.requires_grad = True
                
        model.train()
        running: Dict[str, float] = defaultdict(float)
        batch_count = 0
        epoch_label_counts = np.zeros(3, dtype=np.int64)
        last_checkpoint_sample = samples_seen_total
        epoch_grad_norms_agg: Dict[str, float] = defaultdict(float)
        epoch_grad_norm_max = 0.0
        epoch_nan_grad_count = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        it = iter(train_loader)
        if epoch == start_epoch and start_batch_in_epoch > 0:
            for _ in range(start_batch_in_epoch):
                try:
                    next(it)
                except StopIteration:
                    break

        train_total = len(train_loader) if hasattr(train_loader, "__len__") else None
        pbar = tqdm(
            total=train_total,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            smoothing=0.05,
        )

        for batch_idx, batch in enumerate(
            it, start=1 + (start_batch_in_epoch if epoch == start_epoch else 0)
        ):
            batch = unwrap_batch(batch)
            batch = move_batch_to_device(batch, device)

            batch_samples = int(batch["label"].shape[0])
            samples_seen_total += batch_samples
            global_step += 1
            epoch_label_counts += np.bincount(batch["label"].detach().cpu().numpy(), minlength=3)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                outputs = model(batch)
                loss, metrics = compute_losses(batch, outputs, cfg, model, ictal_pos_weight=ictal_pos_weight)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_idx}: {float(loss)}"
                )

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                epoch_nan_grad_count += zero_nan_grads(model)
                clip_physics_grads(model, cfg.physics_grad_clip)
                clip_encoder_grads(model, cfg.encoder_grad_clip)
                clip_classifier_grads(model, cfg.classifier_grad_clip)
                grad_norms = compute_grad_norms(model)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, foreach=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                epoch_nan_grad_count += zero_nan_grads(model)
                clip_physics_grads(model, cfg.physics_grad_clip)
                clip_encoder_grads(model, cfg.encoder_grad_clip)
                clip_classifier_grads(model, cfg.classifier_grad_clip)
                grad_norms = compute_grad_norms(model)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, foreach=True)
                optimizer.step()

            for k, v in grad_norms.items():
                epoch_grad_norms_agg[k] += v
            epoch_grad_norm_max = max(epoch_grad_norm_max, grad_norms.get("grad_norm_max_param", 0.0))

            for k, v in metrics.items():
                running[k] += float(v)
            running["loss"] += float(loss.detach().item())
            batch_count += 1
            pbar.update(1)

            if cfg.checkpoint_every_samples > 0 and (
                samples_seen_total - last_checkpoint_sample
            ) >= cfg.checkpoint_every_samples:
                save_checkpoint(
                    cfg.checkpoints_dir / "resume.pt",
                    model, optimizer, epoch, best_val, cfg,
                    batch_idx=batch_idx, global_step=global_step,
                    samples_seen_total=samples_seen_total, scaler=scaler,
                )
                last_checkpoint_sample = samples_seen_total

        pbar.close()

        train_metrics = {k: v / max(1, batch_count) for k, v in running.items()}
        epoch_seconds = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - total_start
        avg_grad_norms = {k: v / max(1, batch_count) for k, v in epoch_grad_norms_agg.items()}

        seen_total = max(1, int(epoch_label_counts.sum()))
        seen_frac = epoch_label_counts / seen_total

        if val_loader is not None:
            val_metrics, val_latents = evaluate(
                model, val_loader, cfg, device,
                ictal_pos_weight=ictal_pos_weight, epoch=epoch, split_name="val"
            )
            monitor = compute_monitor(val_metrics)
        else:
            val_metrics = {k: float("nan") for k in train_metrics}
            val_latents = {}
            monitor = float(train_metrics.get("loss", 0.0)) * -1.0

        scheduler.step()
        improved = monitor > best_val

        summary_lines = [
            f"\n┌─ [EPOCH SUMMARY] Epoch {epoch:03d} ─",
            f"│ Train Loss: {train_metrics['loss']:.4f} (Risk: {train_metrics['risk_loss']:.4f}, Ictal: {train_metrics['ictal_loss']:.4f})"
        ]
        if val_loader:
            summary_lines.append(f"│ Val Loss:   {val_metrics['loss']:.4f} | Monitor: {monitor:.4f} {'(NEW BEST)' if improved else ''}")
            summary_lines.append(f"│ Val BalAcc: {val_metrics.get('balanced_accuracy', float('nan')):.4f} | Risk AUROC: {val_metrics.get('risk_auroc', float('nan')):.4f} | Ictal AUROC: {val_metrics.get('ictal_auroc', float('nan')):.4f}")
            summary_lines.append(f"│ Prob Means: Inter={val_metrics.get('diag_prob_interictal_mean', float('nan')):.4f}, Pre={val_metrics.get('diag_prob_preictal_mean', float('nan')):.4f}, Ict={val_metrics.get('diag_prob_ictal_mean', float('nan')):.4f}")
        
        summary_lines.append(f"│ Grads:      Max={epoch_grad_norm_max:.2f}, NaNs={epoch_nan_grad_count} | Tot={avg_grad_norms.get('grad_norm_total', 0):.2f}, Enc={avg_grad_norms.get('grad_norm_encoder', 0):.2f}, Cls={avg_grad_norms.get('grad_norm_classifier', 0):.2f}")
        summary_lines.append(f"│ Imbalance:  Int={seen_frac[0]:.2f}, Pre={seen_frac[1]:.2f}, Ict={seen_frac[2]:.2f} (Target Means: Risk={train_metrics.get('risk_target_mean', 0):.2f}, Ictal={train_metrics.get('ictal_target_mean', 0):.2f})")
        summary_lines.append(f"└─ Time: {format_seconds(epoch_seconds)} (Total Elapsed: {format_seconds(total_elapsed)})\n")
        
        status("\n".join(summary_lines))

        if improved:
            best_val = float(monitor)
            best_epoch = epoch
            patience_left = cfg.early_stop_patience
            save_checkpoint(
                cfg.checkpoints_dir / "best.pt",
                model, optimizer, epoch, best_val, cfg,
                global_step=global_step, samples_seen_total=samples_seen_total, scaler=scaler,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                status(f"[early-stop] stopping early (best monitor={best_val:.4f} @ epoch {best_epoch})")
                history.append({
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"diag_{k}": v for k, v in avg_grad_norms.items()},
                    "diag_grad_norm_param_max": epoch_grad_norm_max,
                    "diag_nan_grad_events": epoch_nan_grad_count,
                    "elapsed_seconds": epoch_seconds,
                    "samples_seen_total": samples_seen_total,
                })
                break

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"diag_{k}": v for k, v in avg_grad_norms.items()},
            "diag_grad_norm_param_max": epoch_grad_norm_max,
            "diag_nan_grad_events": epoch_nan_grad_count,
            "elapsed_seconds": epoch_seconds,
            "samples_seen_total": samples_seen_total,
        })
        save_json(
            cfg.logs_dir / "history.json",
            {"history": history, "best_val": best_val, "best_epoch": best_epoch}
        )
        save_csv(cfg.logs_dir / "history.csv", history)
        save_checkpoint(
            cfg.checkpoints_dir / "last.pt",
            model, optimizer, epoch, best_val, cfg,
            global_step=global_step, samples_seen_total=samples_seen_total, scaler=scaler,
        )

    # Final report
    status("[report] generating final artifacts")
    best_ckpt = cfg.checkpoints_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)

    report_split = cfg.report_split.lower().strip()
    report_loader = (
        val_loader if report_split == "val"
        else test_loader if report_split == "test"
        else val_loader
    )
    report_metrics: Dict[str, float] = {}
    report_latents: Dict[str, torch.Tensor] = {}
    if report_loader is not None:
        report_metrics, report_latents = evaluate(
            model, report_loader, cfg, device,
            ictal_pos_weight=ictal_pos_weight, epoch=-1, split_name=report_split
        )

    report_rows = extract_predictions(report_metrics)
    if cfg.save_predictions and report_rows:
        save_csv(cfg.predictions_dir / f"{report_split}_predictions.csv", report_rows)

    if cfg.save_plots:
        save_history_plots(history, cfg.plots_dir)
        if report_loader is not None and report_rows:
            labels_arr = np.asarray([int(r["true_label"]) for r in report_rows], dtype=np.int64)
            preds_arr = np.asarray([int(r["pred_label"]) for r in report_rows], dtype=np.int64)
            cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1, 2]) if confusion_matrix is not None else None
            if cm is not None:
                plot_confusion_matrix(
                    cm, ["Interictal", "Preictal", "Ictal"],
                    f"{report_split.title()} confusion matrix",
                    cfg.plots_dir / f"{report_split}_confusion_matrix",
                )
            risk_prob_arr = np.asarray([float(r["risk_prob"]) for r in report_rows], dtype=np.float32)
            ictal_prob_arr = np.asarray([float(r["ictal_prob"]) for r in report_rows], dtype=np.float32)
            plot_histogram_by_class(
                risk_prob_arr, labels_arr, bins=30,
                title=f"{report_split.title()} risk probability by class", xlabel="Risk probability",
                out_base=cfg.plots_dir / f"{report_split}_risk_hist",
            )
            plot_histogram_by_class(
                ictal_prob_arr, labels_arr, bins=30,
                title=f"{report_split.title()} ictal probability by class", xlabel="Ictal probability",
                out_base=cfg.plots_dir / f"{report_split}_ictal_hist",
            )
            plot_calibration(
                (labels_arr == 2).astype(np.int32), ictal_prob_arr,
                f"{report_split.title()} ictal calibration",
                cfg.plots_dir / f"{report_split}_ictal_calibration",
                n_bins=cfg.calibration_bins,
            )
            plot_calibration(
                (labels_arr > 0).astype(np.int32), risk_prob_arr,
                f"{report_split.title()} risk calibration",
                cfg.plots_dir / f"{report_split}_risk_calibration",
                n_bins=cfg.calibration_bins,
            )
            if "fused" in report_latents and report_latents["fused"].numel() > 0:
                plot_latent_projection(
                    report_latents["fused"], labels_arr,
                    f"{report_split.title()} fused latent projection",
                    cfg.plots_dir / f"{report_split}_latent_projection",
                    max_points=cfg.latent_plot_max_points,
                )

    result = {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "report_metrics": {
            k: safe_float(v) for k, v in report_metrics.items()
            if not isinstance(v, (list, dict))
        },
        "samples_seen_total": samples_seen_total,
        "thresholds": {
            "preictal": cfg.preictal_thresh,
            "ictal": cfg.ictal_thresh,
        },
    }
    save_json(cfg.logs_dir / "result.json", result)
    status("[done] training complete")
    return result
