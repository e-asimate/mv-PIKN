"""Processor for CHB-MIT EEG preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch

from .config import FileSummary, PreprocessConfig, ReferenceShapes, ShardSummary
from .helpers import (
    as_float32_contiguous, as_int64_contiguous, build_patient_idx_tensor,
    check_finite, compose_spec_features, ensure_1d_labels, infer_n_samples,
    pad_or_truncate_axis, resolve_keys, to_batch_array,
)
from .hdf5_io import read_chunk



def validate_or_pad(
    arr: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    pad_truncate: bool,
    name: str,
    source: Path,
    check_finite_flag: bool,
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if check_finite_flag:
        check_finite(arr, name, source)
    return to_batch_array(arr, target_shape=target_shape, pad_truncate=pad_truncate)


def _expected_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected 2D per-sample shape, got {shape}")
    return int(shape[0]), int(shape[1])


def process_file(
    fp: Path,
    cfg: PreprocessConfig,
    patient_idx: int,
    global_index_start: int,
    ref: ReferenceShapes,
    samples_per_shard: int,
    index_writer: csv.DictWriter,
) -> Tuple[List[ShardSummary], FileSummary, int, Dict[int, int]]:
    shard_dir = cfg.output_dir / "shards"
    shard_summaries: List[ShardSummary] = []
    file_label_counts: Dict[int, int] = {}

    patient_name = fp.stem.replace("_koopman", "")
    shard_count = 0
    next_global = global_index_start

    with h5py.File(fp, "r") as h5:
        keys = resolve_keys(h5)
        n = infer_n_samples(h5, keys)

        raw_past_shape = tuple(np.asarray(h5[keys.raw_past][0]).shape)
        raw_t_shape = tuple(np.asarray(h5[keys.raw_t][0]).shape)
        raw_tdt_shape = tuple(np.asarray(h5[keys.raw_tdt][0]).shape)
        bp_t_shape = tuple(np.asarray(h5[keys.bandpower_t][0]).shape)
        bp_tdt_shape = tuple(np.asarray(h5[keys.bandpower_tdt][0]).shape)
        le_t_shape = tuple(np.asarray(h5[keys.loge_t][0]).shape)
        le_tdt_shape = tuple(np.asarray(h5[keys.loge_tdt][0]).shape)

        if cfg.pad_truncate is False:
            got = {
                "raw_past": raw_past_shape,
                "raw_t": raw_t_shape,
                "raw_tdt": raw_tdt_shape,
                "bandpower_t": bp_t_shape,
                "bandpower_tdt": bp_tdt_shape,
                "loge_t": le_t_shape,
                "loge_tdt": le_tdt_shape,
            }
            expected = {
                "raw_past": ref.raw_past,
                "raw_t": ref.raw_t,
                "raw_tdt": ref.raw_tdt,
                "bandpower_t": ref.bandpower_t,
                "bandpower_tdt": ref.bandpower_tdt,
                "loge_t": ref.loge_t,
                "loge_tdt": ref.loge_tdt,
            }
            for name, shape in got.items():
                if tuple(shape) != tuple(expected[name]):
                    raise ValueError(
                        f"Shape mismatch in {fp.name} for {name}: got {shape}, expected {expected[name]}. "
                        f"Enable --pad-truncate to standardize shapes offline."
                    )

        for start in range(0, n, samples_per_shard):
            end = min(start + samples_per_shard, n)
            chunk_n = end - start

            raw_past = read_chunk(h5, keys.raw_past, start, end)
            raw_t = read_chunk(h5, keys.raw_t, start, end)
            raw_tdt = read_chunk(h5, keys.raw_tdt, start, end)
            bandpower_t = read_chunk(h5, keys.bandpower_t, start, end)
            bandpower_tdt = read_chunk(h5, keys.bandpower_tdt, start, end)
            loge_t = read_chunk(h5, keys.loge_t, start, end)
            loge_tdt = read_chunk(h5, keys.loge_tdt, start, end)
            labels = read_chunk(h5, keys.label, start, end)
            mask = read_chunk(h5, keys.mask, start, end)

            if labels is None:
                raise ValueError(f"Missing labels in {fp.name}")

            labels = as_int64_contiguous(ensure_1d_labels(labels))
            if cfg.check_finite:
                # Labels are already int64; no finite scan needed.
                pass

            if cfg.pad_truncate:
                raw_past = validate_or_pad(raw_past, _expected_shape(ref.raw_past), True, "raw_past", fp, cfg.check_finite)
                raw_t = validate_or_pad(raw_t, _expected_shape(ref.raw_t), True, "raw_t", fp, cfg.check_finite)
                raw_tdt = validate_or_pad(raw_tdt, _expected_shape(ref.raw_tdt), True, "raw_tdt", fp, cfg.check_finite)
                bandpower_t = validate_or_pad(bandpower_t, _expected_shape(ref.bandpower_t), True, "bandpower_t", fp, cfg.check_finite)
                bandpower_tdt = validate_or_pad(bandpower_tdt, _expected_shape(ref.bandpower_tdt), True, "bandpower_tdt", fp, cfg.check_finite)
                loge_t = validate_or_pad(loge_t, _expected_shape(ref.loge_t), True, "loge_t", fp, cfg.check_finite)
                loge_tdt = validate_or_pad(loge_tdt, _expected_shape(ref.loge_tdt), True, "loge_tdt", fp, cfg.check_finite)
                if mask is not None:
                    mask = np.asarray(mask)

            if cfg.check_finite and not cfg.pad_truncate:
                for name, arr in (
                    ("raw_past", raw_past),
                    ("raw_t", raw_t),
                    ("raw_tdt", raw_tdt),
                    ("bandpower_t", bandpower_t),
                    ("bandpower_tdt", bandpower_tdt),
                    ("loge_t", loge_t),
                    ("loge_tdt", loge_tdt),
                ):
                    if arr is not None:
                        check_finite(np.asarray(arr), name, fp)

            # Standardize masks if requested.
            if mask is not None:
                mask = np.asarray(mask)
                if mask.ndim == 1:
                    mask = mask[None, ...]
                if mask.ndim == 2 and cfg.pad_truncate:
                    mask = pad_or_truncate_axis(mask, 1, _expected_shape(ref.raw_t)[0])
                mask = np.ascontiguousarray(mask, dtype=np.float32)
            elif cfg.save_mask and cfg.infer_mask_if_missing:
                mask = np.ones((chunk_n, _expected_shape(ref.raw_t)[0]), dtype=np.float32)

            # Build combined spectral tensors only if requested.
            spec_t = None
            spec_tdt = None
            if cfg.save_combined_spec:
                spec_t = compose_spec_features(bandpower_t, loge_t)
                spec_tdt = compose_spec_features(bandpower_tdt, loge_tdt)
                if cfg.pad_truncate and spec_t is not None and spec_t.ndim == 3 and spec_t.shape[1:] != _expected_shape(ref.bandpower_t):
                    # This case is only relevant if the source spec came in as a 2D batch per sample.
                    pass

            shard_name = f"{fp.stem}__part_{shard_count:05d}.pt"
            shard_path = shard_dir / shard_name
            if shard_path.exists() and not cfg.overwrite:
                raise FileExistsError(f"Shard already exists: {shard_path}. Use --overwrite to replace it.")

            payload: Dict[str, Any] = {
                "label": torch.from_numpy(labels),
                "source_file": str(fp),
                "patient_name": patient_name,
                "chunk_start": int(start),
                "chunk_end": int(end),
            }

            if cfg.save_raw:
                payload["raw_past"] = torch.from_numpy(as_float32_contiguous(raw_past))
                payload["raw_t"] = torch.from_numpy(as_float32_contiguous(raw_t))
                payload["raw_tdt"] = torch.from_numpy(as_float32_contiguous(raw_tdt))

            if cfg.save_separate_features:
                payload["bandpower_t"] = torch.from_numpy(as_float32_contiguous(bandpower_t))
                payload["bandpower_tdt"] = torch.from_numpy(as_float32_contiguous(bandpower_tdt))
                payload["log_energy_t"] = torch.from_numpy(as_float32_contiguous(loge_t))
                payload["log_energy_tdt"] = torch.from_numpy(as_float32_contiguous(loge_tdt))
            elif cfg.save_combined_spec:
                payload["spec_t"] = torch.from_numpy(as_float32_contiguous(spec_t))
                payload["spec_tdt"] = torch.from_numpy(as_float32_contiguous(spec_tdt))

            if cfg.save_patient_idx:
                payload["patient_idx"] = build_patient_idx_tensor(chunk_n, patient_idx)

            if cfg.save_mask and mask is not None:
                payload["mask"] = torch.from_numpy(as_float32_contiguous(mask))

            torch.save(payload, shard_path)

            labels_np = labels.astype(np.int64, copy=False)
            vals, counts = np.unique(labels_np, return_counts=True)
            shard_label_counts = {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}
            for lab, count in shard_label_counts.items():
                file_label_counts[lab] = file_label_counts.get(lab, 0) + count

            for j in range(chunk_n):
                global_idx = next_global + j
                index_writer.writerow(
                    {
                        "global_idx": global_idx,
                        "source_file": str(fp),
                        "shard_file": str(shard_path),
                        "local_idx": j,
                        "patient_name": patient_name,
                        "patient_idx": patient_idx,
                        "label": int(labels_np[j]),
                    }
                )
            next_global += chunk_n

            shard_summaries.append(
                ShardSummary(
                    shard_file=str(shard_path),
                    source_file=str(fp),
                    start_idx=int(start),
                    end_idx=int(end),
                    n_samples=int(chunk_n),
                    patient_name=patient_name,
                    patient_idx=patient_idx,
                    label_counts=shard_label_counts,
                )
            )
            shard_count += 1

    file_summary = FileSummary(
        source_file=str(fp),
        patient_name=patient_name,
        patient_idx=patient_idx,
        n_samples=int(n),
        shard_count=int(shard_count),
        raw_past_shape=raw_past_shape,
        raw_t_shape=raw_t_shape,
        raw_tdt_shape=raw_tdt_shape,
        bandpower_t_shape=bp_t_shape,
        bandpower_tdt_shape=bp_tdt_shape,
        loge_t_shape=le_t_shape,
        loge_tdt_shape=le_tdt_shape,
        label_counts={int(k): int(v) for k, v in sorted(file_label_counts.items())},
    )

    return shard_summaries, file_summary, next_global, file_label_counts
