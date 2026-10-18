"""Cli for CHB-MIT EEG preprocessing."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .config import PreprocessConfig, logger
from .pipeline import preprocess



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess CHB-MIT EEG HDF5 files into small PyTorch shards")
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing raw .h5/.hdf5 files")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write processed shards")
    p.add_argument(
        "--samples-per-shard",
        type=int,
        default=None,
        help="Explicit number of samples to pack into each shard. Overrides --target-shard-mb.",
    )
    p.add_argument(
        "--target-shard-mb",
        type=float,
        default=384.0,
        help="Approximate shard size in MB when samples-per-shard is not provided.",
    )
    p.add_argument(
        "--pad-truncate",
        action="store_true",
        default=True,
        help="Pad/truncate tensors to the first file's shapes. This is the default behavior.",
    )
    p.add_argument(
        "--no-pad-truncate",
        dest="pad_truncate",
        action="store_false",
        help="Disable shape standardization and fail on mismatches.",
    )
    p.add_argument(
        "--check-finite",
        action="store_true",
        default=False,
        help="Validate tensors for NaN/Inf values during preprocessing.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files if they already exist.",
    )
    p.add_argument(
        "--save-raw",
        action="store_true",
        help="Also store raw waveform tensors in each shard.",
    )
    p.add_argument(
        "--save-separate-features",
        action="store_true",
        help="Store bandpower and log-energy tensors separately in addition to or instead of combined spec.",
    )
    p.add_argument(
        "--no-combined-spec",
        action="store_true",
        help="Do not store combined spec_t / spec_tdt tensors.",
    )
    p.add_argument(
        "--save-mask",
        action="store_true",
        help="Store channel masks in each shard.",
    )
    p.add_argument(
        "--no-infer-mask",
        dest="infer_mask_if_missing",
        action="store_false",
        help="Do not create all-ones masks when a mask is missing.",
    )
    p.add_argument(
        "--no-patient-idx",
        dest="save_patient_idx",
        action="store_false",
        help="Do not store patient_idx tensors in shards.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = PreprocessConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        samples_per_shard=args.samples_per_shard,
        target_shard_mb=args.target_shard_mb,
        pad_truncate=bool(args.pad_truncate),
        check_finite=bool(args.check_finite),
        overwrite=bool(args.overwrite),
        save_raw=bool(args.save_raw),
        save_separate_features=bool(args.save_separate_features),
        save_combined_spec=not bool(args.no_combined_spec),
        save_mask=bool(args.save_mask),
        save_patient_idx=bool(args.save_patient_idx),
        infer_mask_if_missing=bool(args.infer_mask_if_missing),
    )

    logger.info("Config: %s", json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))
    preprocess(cfg)
