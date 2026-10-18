"""Pipeline for CHB-MIT EEG preprocessing."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import FileSummary, PreprocessConfig, ShardSummary, logger
from .helpers import discover_h5_files
from .hdf5_io import bootstrap_reference, choose_samples_per_shard
from .processor import process_file



def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(writer: csv.DictWriter, row: Dict[str, Any]) -> None:
    writer.writerow(row)


def preprocess(cfg: PreprocessConfig) -> Dict[str, Any]:
    files = discover_h5_files(cfg.input_dir)
    logger.info("Found %d HDF5 file(s)", len(files))

    first_ref, _, _, _, _ = bootstrap_reference(files[0], cfg)
    samples_per_shard = choose_samples_per_shard(first_ref, cfg)
    logger.info("Using %d sample(s) per shard", samples_per_shard)

    patient_name_to_idx: Dict[str, int] = {}
    dataset_label_counts: Dict[int, int] = {}
    file_summaries: List[FileSummary] = []
    shard_summaries: List[ShardSummary] = []
    total_samples = 0
    total_shards = 0

    index_csv_path = cfg.output_dir / "index.csv"
    shards_csv_path = cfg.output_dir / "shards.csv"
    files_csv_path = cfg.output_dir / "files.csv"
    manifest_path = cfg.output_dir / "manifest.json"
    summary_path = cfg.output_dir / "logs" / "summary.json"

    index_fieldnames = [
        "global_idx",
        "source_file",
        "shard_file",
        "local_idx",
        "patient_name",
        "patient_idx",
        "label",
    ]
    shard_fieldnames = [
        "shard_file",
        "source_file",
        "start_idx",
        "end_idx",
        "n_samples",
        "patient_name",
        "patient_idx",
        "label_counts",
    ]
    file_fieldnames = [
        "source_file",
        "patient_name",
        "patient_idx",
        "n_samples",
        "shard_count",
        "raw_past_shape",
        "raw_t_shape",
        "raw_tdt_shape",
        "bandpower_t_shape",
        "bandpower_tdt_shape",
        "loge_t_shape",
        "loge_tdt_shape",
        "label_counts",
    ]

    with index_csv_path.open("w", newline="", encoding="utf-8") as index_f:
        index_writer = csv.DictWriter(index_f, fieldnames=index_fieldnames)
        index_writer.writeheader()

        for fp in files:
            patient_name = fp.stem.replace("_koopman", "")
            if patient_name not in patient_name_to_idx:
                patient_name_to_idx[patient_name] = len(patient_name_to_idx)
            patient_idx = patient_name_to_idx[patient_name]

            logger.info("Processing %s", fp.name)
            shard_summaries_fp, file_summary, total_samples, file_label_counts = process_file(
                fp=fp,
                cfg=cfg,
                patient_idx=patient_idx,
                global_index_start=total_samples,
                ref=first_ref,
                samples_per_shard=samples_per_shard,
                index_writer=index_writer,
            )

            shard_summaries.extend(shard_summaries_fp)
            file_summaries.append(file_summary)
            total_shards += len(shard_summaries_fp)

            for lab, count in file_label_counts.items():
                dataset_label_counts[lab] = dataset_label_counts.get(lab, 0) + int(count)

            logger.info(
                "Done %s | samples=%d | shards=%d",
                fp.name,
                file_summary.n_samples,
                file_summary.shard_count,
            )

    # Write shard/file metadata to separate CSVs so manifest stays light.
    write_csv(shards_csv_path, [asdict(x) for x in shard_summaries], shard_fieldnames)
    write_csv(files_csv_path, [asdict(x) for x in file_summaries], file_fieldnames)

    manifest: Dict[str, Any] = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "pad_truncate": cfg.pad_truncate,
        "check_finite": cfg.check_finite,
        "save_raw": cfg.save_raw,
        "save_separate_features": cfg.save_separate_features,
        "save_combined_spec": cfg.save_combined_spec,
        "save_mask": cfg.save_mask,
        "save_patient_idx": cfg.save_patient_idx,
        "infer_mask_if_missing": cfg.infer_mask_if_missing,
        "samples_per_shard": samples_per_shard,
        "n_files": len(files),
        "n_shards": total_shards,
        "n_samples": total_samples,
        "patient_count": len(patient_name_to_idx),
        "label_counts": {str(k): int(v) for k, v in sorted(dataset_label_counts.items())},
        "reference_shapes": asdict(first_ref),
        "index_csv": str(index_csv_path),
        "shards_csv": str(shards_csv_path),
        "files_csv": str(files_csv_path),
        "files": [asdict(fs) for fs in file_summaries],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path.write_text(
        json.dumps(
            {
                "n_files": len(files),
                "n_shards": total_shards,
                "n_samples": total_samples,
                "patient_count": len(patient_name_to_idx),
                "label_counts": {str(k): int(v) for k, v in sorted(dataset_label_counts.items())},
                "samples_per_shard": samples_per_shard,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Preprocessing complete.")
    logger.info("Samples: %d | Shards: %d | Patients: %d", total_samples, total_shards, len(patient_name_to_idx))
    logger.info("Label counts: %s", json.dumps({str(k): int(v) for k, v in sorted(dataset_label_counts.items())}))
    logger.info("Index CSV: %s", index_csv_path)
    logger.info("Manifest: %s", manifest_path)

    return manifest
