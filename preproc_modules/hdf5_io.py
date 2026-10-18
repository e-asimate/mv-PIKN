"""Hdf5 Io for CHB-MIT EEG preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from .config import PreprocessConfig, ReferenceShapes, ResolvedKeys
from .helpers import check_finite, ensure_1d_labels, infer_n_samples, resolve_keys



def bootstrap_reference(fp: Path, cfg: PreprocessConfig) -> Tuple[ReferenceShapes, ResolvedKeys, int, Dict[str, Tuple[int, ...]], Dict[int, int]]:
    """Inspect the first file to lock in reference shapes and a sample-size estimate."""

    with h5py.File(fp, "r") as h5:
        keys = resolve_keys(h5)
        required = {
            "raw_past": keys.raw_past,
            "raw_t": keys.raw_t,
            "raw_tdt": keys.raw_tdt,
            "bandpower_t": keys.bandpower_t,
            "bandpower_tdt": keys.bandpower_tdt,
            "loge_t": keys.loge_t,
            "loge_tdt": keys.loge_tdt,
            "label": keys.label,
        }
        missing = [name for name, key in required.items() if key is None]
        if missing:
            raise ValueError(f"{fp.name} is missing required datasets: {missing}")

        n = infer_n_samples(h5, keys)
        sample_0 = {
            "raw_past": np.asarray(h5[keys.raw_past][0]),
            "raw_t": np.asarray(h5[keys.raw_t][0]),
            "raw_tdt": np.asarray(h5[keys.raw_tdt][0]),
            "bandpower_t": np.asarray(h5[keys.bandpower_t][0]),
            "bandpower_tdt": np.asarray(h5[keys.bandpower_tdt][0]),
            "loge_t": np.asarray(h5[keys.loge_t][0]),
            "loge_tdt": np.asarray(h5[keys.loge_tdt][0]),
        }

        if cfg.check_finite:
            for name, arr in sample_0.items():
                check_finite(arr, name, fp)

        ref_shapes = ReferenceShapes(
            raw_past=tuple(sample_0["raw_past"].shape),
            raw_t=tuple(sample_0["raw_t"].shape),
            raw_tdt=tuple(sample_0["raw_tdt"].shape),
            bandpower_t=tuple(sample_0["bandpower_t"].shape),
            bandpower_tdt=tuple(sample_0["bandpower_tdt"].shape),
            loge_t=tuple(sample_0["loge_t"].shape),
            loge_tdt=tuple(sample_0["loge_tdt"].shape),
        )

        label_counts: Dict[int, int] = {}
        labels0 = ensure_1d_labels(np.asarray(h5[keys.label][0: min(n, 8192)]))
        vals, counts = np.unique(labels0.astype(np.int64), return_counts=True)
        for k, v in zip(vals.tolist(), counts.tolist()):
            label_counts[int(k)] = int(v)

        return ref_shapes, keys, n, {k: tuple(v.shape) for k, v in sample_0.items()}, label_counts


def _dtype_nbytes(dtype: np.dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def estimate_sample_bytes(ref_shapes: ReferenceShapes, cfg: PreprocessConfig) -> int:
    """Estimate bytes per sample after preprocessing for shard size selection."""

    total = 0
    if cfg.save_raw:
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.raw_past))
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.raw_t))
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.raw_tdt))

    if cfg.save_separate_features:
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.bandpower_t))
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.bandpower_tdt))
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.loge_t))
        total += _dtype_nbytes(np.float32) * int(np.prod(ref_shapes.loge_tdt))
    elif cfg.save_combined_spec:
        # Combined spec is typically bandpower + log-energy concatenated along the last axis.
        bp = ref_shapes.bandpower_t
        le = ref_shapes.loge_t
        bpdt = ref_shapes.bandpower_tdt
        ledt = ref_shapes.loge_tdt
        if len(bp) == 2:
            total += _dtype_nbytes(np.float32) * int(bp[0] * bp[1] * 2)
            total += _dtype_nbytes(np.float32) * int(bpdt[0] * bpdt[1] * 2)
        else:
            total += _dtype_nbytes(np.float32) * int(np.prod(bp))
            total += _dtype_nbytes(np.float32) * int(np.prod(bpdt))
        # If shapes are 2D this double-counts through the explicit concatenation rule above.
        # For 3D arrays the concatenation is not used.
        _ = le, ledt

    if cfg.save_mask:
        # Stored as float32 for easiest training-time use.
        # Default mask shape is inferred later from raw_t or bandpower shape.
        total += _dtype_nbytes(np.float32) * int(ref_shapes.raw_t[0])

    if cfg.save_patient_idx:
        total += _dtype_nbytes(np.int64)

    total += _dtype_nbytes(np.int64)  # label
    return max(total, 1)


def choose_samples_per_shard(ref_shapes: ReferenceShapes, cfg: PreprocessConfig) -> int:
    if cfg.samples_per_shard is not None:
        return max(1, int(cfg.samples_per_shard))
    if cfg.target_shard_mb is None:
        return 256

    sample_bytes = estimate_sample_bytes(ref_shapes, cfg)
    target_bytes = int(float(cfg.target_shard_mb) * 1024 * 1024)
    n = max(1, target_bytes // sample_bytes)
    # Clamp to avoid pathological shard sizes.
    return int(min(max(n, 16), 8192))


def read_chunk(h5: h5py.File, key: Optional[str], start: int, end: int) -> Optional[np.ndarray]:
    if key is None or key not in h5:
        return None
    return np.asarray(h5[key][start:end])


def ensure_mask(
    mask: Optional[np.ndarray],
    fallback_width: int,
    n: int,
    save_mask: bool,
) -> Optional[np.ndarray]:
    if not save_mask and mask is None:
        return None
    if mask is not None:
        return np.asarray(mask)
    if save_mask:
        return np.ones((n, fallback_width), dtype=np.float32)
    return None
