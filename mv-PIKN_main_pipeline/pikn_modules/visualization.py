"""Visualization for the mv-PIKN training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    from umap import UMAP
except Exception:
    UMAP = None

from .utils import (safe_float)


def plot_line_series(
    x: List[int],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_base: Path,
) -> None:
    if not x:
        return
    fig = plt.figure(figsize=(10, 5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    for name, y in series.items():
        ax.plot(x, y, marker="o", linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_base: Path) -> None:
    if cm is None:
        return
    fig = plt.figure(figsize=(6.5, 5.5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_histogram_by_class(
    values: np.ndarray,
    labels: np.ndarray,
    bins: int,
    title: str,
    xlabel: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    for cls, name in [(0, "Interictal"), (1, "Preictal"), (2, "Ictal")]:
        mask = labels == cls
        if np.any(mask):
            ax.hist(values[mask], bins=bins, alpha=0.45, density=True, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, prob: np.ndarray, title: str, out_base: Path, n_bins: int = 15
) -> None:
    if len(y_true) == 0:
        return
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    inds = np.digitize(prob, bins) - 1
    for i in range(n_bins):
        m = inds == i
        if np.any(m):
            acc[i] = y_true[m].mean()
            conf[i] = prob[m].mean()
    fig = plt.figure(figsize=(6.5, 5.5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(conf, acc, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_latent_projection(
    latent: torch.Tensor,
    labels: np.ndarray,
    title: str,
    out_base: Path,
    max_points: int = 6000,
) -> None:
    if latent is None or latent.numel() == 0:
        return
    x = latent.detach().cpu().numpy()
    y = labels.astype(np.int64)
    if len(x) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    proj = None
    if UMAP is not None:
        try:
            n_neighbors = min(30, max(5, len(x) - 1))
            proj = UMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=0.15,
                metric="euclidean", random_state=42,
            ).fit_transform(x)
        except Exception:
            proj = None
    if proj is None:
        if PCA is not None:
            proj = PCA(n_components=3, random_state=42).fit_transform(x)
        else:
            proj = x[:, :3] if x.shape[1] >= 3 else np.pad(
                x, ((0, 0), (0, 3 - x.shape[1])), constant_values=0.0
            )
            
    fig = plt.figure(figsize=(8, 7), dpi=180)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:red"}
    names = {0: "Interictal", 1: "Preictal", 2: "Ictal"}
    for cls in (0, 1, 2):
        m = y == cls
        if np.any(m):
            ax.scatter(proj[m, 0], proj[m, 1], proj[m, 2], s=6, alpha=0.45, label=names[cls], color=cmap[cls])
            
    ax.set_title(title, y=1.02)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def save_history_plots(history: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not history:
        return
    epochs = [int(h["epoch"]) for h in history]

    plot_line_series(
        epochs,
        {
            "balanced_accuracy": [safe_float(h.get("val_balanced_accuracy")) for h in history],
            "macro_f1": [safe_float(h.get("val_macro_f1")) for h in history],
            "risk_auroc": [safe_float(h.get("val_risk_auroc")) for h in history],
            "ictal_auroc": [safe_float(h.get("val_ictal_auroc")) for h in history],
            "loss": [safe_float(h.get("val_loss")) for h in history],
        },
        "Validation performance",
        "Metric value",
        plots_dir / "val_performance",
    )
    plot_line_series(
        epochs,
        {
            "risk_loss": [safe_float(h.get("train_risk_loss")) for h in history],
            "ictal_loss": [safe_float(h.get("train_ictal_loss")) for h in history],
            "latent_l2": [safe_float(h.get("train_latent_l2")) for h in history],
            "stability": [safe_float(h.get("train_stability")) for h in history],
        },
        "Training auxiliary losses",
        "Metric value",
        plots_dir / "train_aux_losses",
    )
    plot_line_series(
        epochs,
        {
            "interictal": [safe_float(h.get("val_diag_prob_interictal_mean")) for h in history],
            "preictal": [safe_float(h.get("val_diag_prob_preictal_mean")) for h in history],
            "ictal": [safe_float(h.get("val_diag_prob_ictal_mean")) for h in history],
        },
        "Mean predicted probabilities",
        "Probability",
        plots_dir / "probabilities",
    )
    plot_line_series(
        epochs,
        {
            "latent_l2": [safe_float(h.get("val_latent_l2")) for h in history],
            "spectral_align": [safe_float(h.get("val_spectral_align")) for h in history],
        },
        "Latent and alignment diagnostics",
        "Value",
        plots_dir / "latent_diagnostics",
    )
    plot_line_series(
        epochs,
        {
            "risk_ece": [safe_float(h.get("val_risk_ece")) for h in history],
            "ictal_ece": [safe_float(h.get("val_ictal_ece")) for h in history],
            "risk_brier": [safe_float(h.get("val_risk_brier")) for h in history],
            "ictal_brier": [safe_float(h.get("val_ictal_brier")) for h in history],
        },
        "Calibration diagnostics",
        "Metric value",
        plots_dir / "calibration",
    )
