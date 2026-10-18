"""Helpers for CHB-MIT EEG preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from .config import (
    BANDPOWER_T_ALIASES, BANDPOWER_TDT_ALIASES, LABEL_ALIASES,
    LOGE_T_ALIASES, LOGE_TDT_ALIASES, MASK_ALIASES,
    RAW_PAST_ALIASES, RAW_T_ALIASES, RAW_TDT_ALIASES, ResolvedKeys,
)



def discover_h5_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".h5", ".hdf5"}])
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {input_dir}")
    return files


def first_existing_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    key_set = set(keys)
    for cand in candidates:
        if cand in key_set:
            return cand
    return None


def resolve_keys(h5: h5py.File) -> ResolvedKeys:
    keys = list(h5.keys())
    return ResolvedKeys(
        raw_past=first_existing_key(keys, RAW_PAST_ALIASES),
        raw_t=first_existing_key(keys, RAW_T_ALIASES),
        raw_tdt=first_existing_key(keys, RAW_TDT_ALIASES),
        bandpower_t=first_existing_key(keys, BANDPOWER_T_ALIASES),
        bandpower_tdt=first_existing_key(keys, BANDPOWER_TDT_ALIASES),
        loge_t=first_existing_key(keys, LOGE_T_ALIASES),
        loge_tdt=first_existing_key(keys, LOGE_TDT_ALIASES),
        label=first_existing_key(keys, LABEL_ALIASES),
        mask=first_existing_key(keys, MASK_ALIASES),
    )


def infer_n_samples(h5: h5py.File, keys: ResolvedKeys) -> int:
    preferred = [keys.raw_past, keys.raw_t, keys.bandpower_t, keys.loge_t, keys.label]
    for key in preferred:
        if key is not None and key in h5 and isinstance(h5[key], h5py.Dataset):
            ds = h5[key]
            if ds.ndim >= 1:
                return int(ds.shape[0])
    for key in h5.keys():
        obj = h5[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            return int(obj.shape[0])
    raise ValueError("Could not infer sample count from file.")


def ensure_1d_labels(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != 1:
            raise ValueError(f"Label array must be 1D or Nx1, got shape {arr.shape}")
        arr = arr[:, 0]
    return arr


def as_float32_contiguous(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def as_int64_contiguous(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int64)


def check_finite(arr: np.ndarray, name: str, source: Path) -> None:
    if not np.isfinite(arr).all():
        bad = np.size(arr) - int(np.isfinite(arr).sum())
        raise ValueError(f"Non-finite values detected in {source.name} dataset '{name}' ({bad} bad values)")


def pad_or_truncate_axis(arr: np.ndarray, axis: int, target: int, pad_value: float = 0.0) -> np.ndarray:
    current = arr.shape[axis]
    if current == target:
        return arr
    if current > target:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(0, target)
        return arr[tuple(slc)]

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=pad_value)


def pad_or_truncate_batch(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate a batch tensor with shape [N, C, T] or [N, C, F]."""

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D batch tensor [N, C, L], got shape {arr.shape}")
    arr = pad_or_truncate_axis(arr, 1, target_shape[0], pad_value=pad_value)
    arr = pad_or_truncate_axis(arr, 2, target_shape[1], pad_value=pad_value)
    return arr


def standardize_optional_batch(
    arr: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    pad_value: float = 0.0,
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return pad_or_truncate_batch(arr, target_shape=target_shape, pad_value=pad_value)


def to_batch_array(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    pad_truncate: bool,
    pad_value: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {arr.shape}")

    if pad_truncate:
        arr = pad_or_truncate_batch(arr, target_shape=target_shape, pad_value=pad_value)

    if squeeze:
        arr = arr[0]
    return arr


def compose_spec_features(
    bandpower: Optional[np.ndarray],
    log_energy: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if bandpower is None and log_energy is None:
        return None
    if bandpower is None:
        return np.asarray(log_energy)
    if log_energy is None:
        return np.asarray(bandpower)

    bp = np.asarray(bandpower)
    le = np.asarray(log_energy)

    if bp.ndim == 2:
        bp = bp[..., None]
    if le.ndim == 2:
        le = le[..., None]
    if bp.shape[:2] != le.shape[:2]:
        raise ValueError(f"Cannot compose spec features with mismatched shapes {bp.shape} vs {le.shape}")
    return np.concatenate([bp, le], axis=-1)


def build_patient_idx_tensor(n: int, patient_idx: int) -> torch.Tensor:
    return torch.full((n,), int(patient_idx), dtype=torch.long)
