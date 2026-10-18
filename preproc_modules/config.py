"""Config for CHB-MIT EEG preprocessing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[FlushingStreamHandler()],
)
logger = logging.getLogger("eeg_preprocess_v2")



RAW_PAST_ALIASES = ("X_past", "raw_past", "x_past")


RAW_T_ALIASES = ("X_t", "raw_t", "x_t")


RAW_TDT_ALIASES = ("X_t_dt", "X_tdt", "raw_tdt", "raw_next", "x_tdt", "x_t_dt")


BANDPOWER_T_ALIASES = ("bandpower_t", "spec_t", "spectral_t", "obs_t")


BANDPOWER_TDT_ALIASES = ("bandpower_tdt", "spec_tdt", "spec_t_dt", "spectral_tdt", "obs_tdt")


LOGE_T_ALIASES = ("log_energy_t", "loge_t", "energy_t")


LOGE_TDT_ALIASES = ("log_energy_tdt", "log_energy_t_dt", "loge_tdt", "energy_tdt")


LABEL_ALIASES = ("y_label", "label", "y", "target")


MASK_ALIASES = ("mask_t", "channel_mask", "mask")


class PreprocessConfig:
    input_dir: Path
    output_dir: Path
    samples_per_shard: Optional[int] = None
    target_shard_mb: Optional[float] = 384.0
    pad_truncate: bool = True
    check_finite: bool = False
    overwrite: bool = False
    save_raw: bool = False
    save_separate_features: bool = False
    save_combined_spec: bool = True
    save_mask: bool = False
    save_patient_idx: bool = True
    infer_mask_if_missing: bool = True

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "shards").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)


class ResolvedKeys:
    raw_past: Optional[str]
    raw_t: Optional[str]
    raw_tdt: Optional[str]
    bandpower_t: Optional[str]
    bandpower_tdt: Optional[str]
    loge_t: Optional[str]
    loge_tdt: Optional[str]
    label: Optional[str]
    mask: Optional[str]


class FileSummary:
    source_file: str
    patient_name: str
    patient_idx: int
    n_samples: int
    shard_count: int
    raw_past_shape: Tuple[int, ...]
    raw_t_shape: Tuple[int, ...]
    raw_tdt_shape: Tuple[int, ...]
    bandpower_t_shape: Tuple[int, ...]
    bandpower_tdt_shape: Tuple[int, ...]
    loge_t_shape: Tuple[int, ...]
    loge_tdt_shape: Tuple[int, ...]
    label_counts: Dict[int, int]


class ShardSummary:
    shard_file: str
    source_file: str
    start_idx: int
    end_idx: int
    n_samples: int
    patient_name: str
    patient_idx: int
    label_counts: Dict[int, int]


class ReferenceShapes:
    raw_past: Tuple[int, ...]
    raw_t: Tuple[int, ...]
    raw_tdt: Tuple[int, ...]
    bandpower_t: Tuple[int, ...]
    bandpower_tdt: Tuple[int, ...]
    loge_t: Tuple[int, ...]
    loge_tdt: Tuple[int, ...]
