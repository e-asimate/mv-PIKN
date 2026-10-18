"""Modular CHB-MIT EEG preprocessing pipeline."""

from .config import (
    BANDPOWER_T_ALIASES, BANDPOWER_TDT_ALIASES, LABEL_ALIASES,
    LOGE_T_ALIASES, LOGE_TDT_ALIASES, MASK_ALIASES,
    RAW_PAST_ALIASES, RAW_T_ALIASES, RAW_TDT_ALIASES,
    FileSummary, FlushingStreamHandler, PreprocessConfig, ReferenceShapes,
    ResolvedKeys, ShardSummary, logger,
)
from .helpers import (
    as_float32_contiguous, as_int64_contiguous, build_patient_idx_tensor,
    check_finite, compose_spec_features, discover_h5_files, ensure_1d_labels,
    first_existing_key, infer_n_samples, pad_or_truncate_axis,
    pad_or_truncate_batch, resolve_keys, standardize_optional_batch, to_batch_array,
)
from .hdf5_io import (
    bootstrap_reference, choose_samples_per_shard, ensure_mask,
    estimate_sample_bytes, read_chunk,
)
from .processor import process_file, validate_or_pad
from .pipeline import append_csv_row, preprocess, write_csv
from .cli import build_argparser, main

__all__ = [name for name in globals() if not name.startswith('_')]
