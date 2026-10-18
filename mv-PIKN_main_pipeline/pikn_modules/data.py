"""Data for the mv-PIKN training pipeline."""

from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info



from .config import (TrainConfig)
from .utils import (as_float_tensor, as_long_tensor, infer_patient_id, read_csv_rows, section, status, to_int)


def _build_spectral_tensor(shard: Dict[str, Any], indices: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if "spec_t" in shard and "spec_tdt" in shard:
        return (
            torch.as_tensor(shard["spec_t"][indices], dtype=torch.float32),
            torch.as_tensor(shard["spec_tdt"][indices], dtype=torch.float32),
        )
    if "combined_spec_t" in shard and "combined_spec_tdt" in shard:
        return (
            torch.as_tensor(shard["combined_spec_t"][indices], dtype=torch.float32),
            torch.as_tensor(shard["combined_spec_tdt"][indices], dtype=torch.float32),
        )
    if (
        "bandpower_t" in shard and "bandpower_tdt" in shard
        and "loge_t" in shard and "loge_tdt" in shard
    ):
        bp_t = torch.as_tensor(shard["bandpower_t"][indices], dtype=torch.float32)
        bp_tdt = torch.as_tensor(shard["bandpower_tdt"][indices], dtype=torch.float32)
        loge_t = torch.as_tensor(shard["loge_t"][indices], dtype=torch.float32)
        loge_tdt = torch.as_tensor(shard["loge_tdt"][indices], dtype=torch.float32)
        return torch.cat([bp_t, loge_t], dim=-1), torch.cat([bp_tdt, loge_tdt], dim=-1)

    raise KeyError(f"Cannot construct spectral tensor. Available keys: {list(shard.keys())}")


def split_by_patient(
    rows: List[Dict[str, str]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    for row in rows:
        row["_pid"] = infer_patient_id(row)

    unique_patients = sorted(set(r["_pid"] for r in rows))
    if not unique_patients:
        raise ValueError("Could not infer any patient IDs")

    rng = np.random.default_rng(seed)
    patients = list(unique_patients)
    rng.shuffle(patients)

    n = len(patients)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(0, int(round(val_frac * n))) if val_frac > 0 else 0
    if n_train + n_val >= n:
        if n >= 3:
            n_val = max(1, min(n - 2, n_val))
            n_train = max(1, min(n - n_val - 1, n_train))
        else:
            n_val = max(0, n - n_train - 1)
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and n >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:n_train + n_val + n_test])

    train_rows = [r for r in rows if r["_pid"] in train_patients]
    val_rows = [r for r in rows if r["_pid"] in val_patients]
    test_rows = [r for r in rows if r["_pid"] in test_patients]

    status(f"[split] patients train={sorted(train_patients)} val={sorted(val_patients)} test={sorted(test_patients)}")
    status(f"[split] samples train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    return train_rows, val_rows, test_rows


def compute_class_weights(counts: np.ndarray, power: float = 0.5) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = 1.0 / np.power(counts, float(power))
    weights = weights / weights[0]
    return torch.tensor(weights, dtype=torch.float32)


def resolve_path(path_str: str, processed_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = processed_dir / p
    if candidate.exists():
        return candidate
    return p


def load_shard(path: Path) -> Dict[str, Any]:
    shard = torch.load(path, map_location="cpu")
    if not isinstance(shard, dict):
        raise TypeError(f"Shard {path.name} did not load as a dict")
    return shard


class ShardBatchIterableDataset(IterableDataset):
    def __init__(
        self,
        processed_dir: Path,
        rows: List[Dict[str, str]],
        batch_size: int,
        max_loaded_shards: int = 1,
        shuffle_shards: bool = True,
        seed: int = 42,
        balanced_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.rows = list(rows)
        self.batch_size = max(1, int(batch_size))
        self.max_loaded_shards = max(1, int(max_loaded_shards))
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.balanced_sampling = bool(balanced_sampling)

        self.index_csv = self.processed_dir / "index.csv"
        self.shards_dir = self.processed_dir / "shards"
        
        self.labels = np.asarray([to_int(r["label"]) for r in self.rows], dtype=np.int64)
        self.patient_indices = np.asarray([to_int(r["patient_idx"]) for r in self.rows], dtype=np.int64)
        self.shard_to_rows: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            self.shard_to_rows[str(r["shard_file"])].append(i)
        self.shard_files_unique = sorted(self.shard_to_rows.keys())

    def __len__(self) -> int:
        total = 0
        for _, row_idxs in self.shard_to_rows.items():
            if self.shuffle_shards and self.balanced_sampling:
                labels = self.labels[row_idxs]
                n0 = int(np.sum(labels == 0))
                majority_per_batch = max(1, int(self.batch_size * 0.5))
                total += max(1, math.ceil(n0 / majority_per_batch))
            else:
                total += max(1, math.ceil(len(row_idxs) / self.batch_size))
        return max(1, total)

    def _iter_worker_shards(self) -> List[str]:
        shard_paths = list(self.shard_files_unique)
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + torch.initial_seed() % 2**16)
            rng.shuffle(shard_paths)
        info = get_worker_info()
        if info is None:
            return shard_paths
        return shard_paths[info.id :: info.num_workers]

    def _make_batch_from_indices(
        self, shard: Dict[str, Any], indices: np.ndarray, shard_path: Path
    ) -> Optional[Dict[str, Any]]:
        if len(indices) == 0:
            return None
        spec_t, spec_tdt = _build_spectral_tensor(shard, indices)
        out: Dict[str, Any] = {
            "raw_past": shard["raw_past"][indices],
            "raw_t": shard["raw_t"][indices],
            "raw_tdt": shard["raw_tdt"][indices],
            "spec_t": spec_t,
            "spec_tdt": spec_tdt,
            "label": shard["label"][indices],
            "patient_idx": shard["patient_idx"][indices],
        }
        if "mask" in shard:
            out["mask"] = shard["mask"][indices]
        if "time_to_seizure_min" in shard:
            out["time_to_seizure_min"] = shard["time_to_seizure_min"][indices]
        return out

    def _finalise_batch(
        self, chunks: List[Dict[str, Any]], shard: Dict[str, Any],
        shard_path: Path, rng: np.random.Generator
    ) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for k in chunks[0].keys():
            batch[k] = torch.cat([c[k] for c in chunks if c is not None], dim=0)
        perm = rng.permutation(batch["label"].shape[0])
        for k in list(batch.keys()):
            batch[k] = batch[k][perm]
        batch["source_file"] = shard.get("source_file", str(shard_path))
        batch["shard_file"] = str(shard_path)
        batch["raw_past"] = as_float_tensor(batch["raw_past"])
        batch["raw_t"] = as_float_tensor(batch["raw_t"])
        batch["raw_tdt"] = as_float_tensor(batch["raw_tdt"])
        batch["spec_t"] = as_float_tensor(batch["spec_t"])
        batch["spec_tdt"] = as_float_tensor(batch["spec_tdt"])
        batch["label"] = as_long_tensor(batch["label"]).view(-1)
        batch["patient_idx"] = as_long_tensor(batch["patient_idx"]).view(-1)
        if "mask" in batch:
            batch["mask"] = as_float_tensor(batch["mask"])
        if "time_to_seizure_min" in batch:
            batch["time_to_seizure_min"] = as_float_tensor(batch["time_to_seizure_min"]).view(-1)
        return batch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        minority_cache: Dict[int, Optional[Dict[str, Any]]] = {1: None, 2: None}
        max_cache = 1000
        shard_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        for shard_name in self._iter_worker_shards():
            shard_path = resolve_path(shard_name, self.processed_dir)
            shard_key = str(shard_path)

            if shard_key not in shard_cache:
                shard = load_shard(shard_path)
                shard_cache[shard_key] = shard
                while len(shard_cache) > self.max_loaded_shards:
                    _, evicted = shard_cache.popitem(last=False)
                    del evicted
            else:
                shard = shard_cache[shard_key]
                shard_cache.move_to_end(shard_key)

            raw_labels = shard["label"]
            labels = (
                raw_labels.detach().cpu().numpy().reshape(-1)
                if torch.is_tensor(raw_labels)
                else np.asarray(raw_labels).reshape(-1)
            )
            idx0 = np.where(labels == 0)[0]
            idx1 = np.where(labels == 1)[0]
            idx2 = np.where(labels == 2)[0]

            rng = np.random.default_rng(self.seed + (abs(hash(str(shard_path))) % 2**16))
            if self.shuffle_shards:
                rng.shuffle(idx0)
                rng.shuffle(idx1)
                rng.shuffle(idx2)

            if self.shuffle_shards and self.balanced_sampling:
                for cls_idx, idx_cls in ((1, idx1), (2, idx2)):
                    t = self._make_batch_from_indices(shard, idx_cls, shard_path)
                    if t is not None:
                        if minority_cache[cls_idx] is None:
                            minority_cache[cls_idx] = t
                        else:
                            for k in t.keys():
                                if k in minority_cache[cls_idx]:
                                    minority_cache[cls_idx][k] = torch.cat(
                                        [minority_cache[cls_idx][k], t[k]], dim=0
                                    )[-max_cache:]

                target1 = int(self.batch_size * 0.25)
                target2 = int(self.batch_size * 0.25)
                target0 = self.batch_size - target1 - target2

                c0 = c1 = c2 = 0
                while c0 < len(idx0):
                    chunks: List[Dict[str, Any]] = []

                    take0 = min(target0, len(idx0) - c0)
                    if take0 > 0:
                        chunk = self._make_batch_from_indices(shard, idx0[c0:c0 + take0], shard_path)
                        if chunk is not None:
                            chunks.append(chunk)
                        c0 += take0

                    for cls_idx, cursor, idx_cls, target in [
                        (1, c1, idx1, target1),
                        (2, c2, idx2, target2),
                    ]:
                        avail = len(idx_cls) - cursor
                        take = min(target, avail)
                        if take > 0:
                            chunk = self._make_batch_from_indices(
                                shard, idx_cls[cursor:cursor + take], shard_path
                            )
                            if chunk is not None:
                                chunks.append(chunk)
                            if cls_idx == 1:
                                c1 += take
                            else:
                                c2 += take
                        needed = target - take
                        if needed > 0 and minority_cache[cls_idx] is not None:
                            cache = minority_cache[cls_idx]
                            c_idx = rng.choice(cache["label"].shape[0], size=needed, replace=True)
                            chunks.append({k: cache[k][c_idx] for k in cache.keys()})
                        elif needed > 0:
                            pad_take = min(needed, len(idx0) - c0)
                            if pad_take > 0:
                                chunk = self._make_batch_from_indices(
                                    shard, idx0[c0:c0 + pad_take], shard_path
                                )
                                if chunk is not None:
                                    chunks.append(chunk)
                                c0 += pad_take

                    if not chunks:
                        continue

                    yield self._finalise_batch(chunks, shard, shard_path, rng)

            else:
                all_idx = np.concatenate([idx0, idx1, idx2])
                all_idx.sort()
                for start in range(0, len(all_idx), self.batch_size):
                    chunk_idx = all_idx[start:start + self.batch_size]
                    spec_t, spec_tdt = _build_spectral_tensor(shard, chunk_idx)
                    batch: Dict[str, Any] = {
                        "raw_past": as_float_tensor(shard["raw_past"][chunk_idx]),
                        "raw_t": as_float_tensor(shard["raw_t"][chunk_idx]),
                        "raw_tdt": as_float_tensor(shard["raw_tdt"][chunk_idx]),
                        "spec_t": as_float_tensor(spec_t),
                        "spec_tdt": as_float_tensor(spec_tdt),
                        "label": as_long_tensor(shard["label"][chunk_idx]).view(-1),
                        "patient_idx": as_long_tensor(shard["patient_idx"][chunk_idx]).view(-1),
                        "source_file": shard.get("source_file", str(shard_path)),
                        "shard_file": str(shard_path),
                        "sample_idx": chunk_idx.tolist(),
                    }
                    if "mask" in shard:
                        batch["mask"] = as_float_tensor(shard["mask"][chunk_idx])
                    if "time_to_seizure_min" in shard:
                        batch["time_to_seizure_min"] = as_float_tensor(
                            shard["time_to_seizure_min"][chunk_idx]
                        ).view(-1)
                    yield batch


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ("raw_past", "raw_t", "raw_tdt", "spec_t", "spec_tdt",
                "label", "patient_idx", "time_to_seizure_min"):
        if torch.is_tensor(out.get(key)):
            out[key] = out[key].to(device, non_blocking=True)
    if torch.is_tensor(out.get("mask")):
        out["mask"] = out["mask"].to(device, non_blocking=True)
    return out


def build_loaders(
    cfg: TrainConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], torch.Tensor, np.ndarray]:
    section("Data")
    index_csv = cfg.processed_dir / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing index.csv: {index_csv}")
    rows = read_csv_rows(index_csv)

    train_rows, val_rows, test_rows = split_by_patient(
        rows, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )
    train_labels = np.asarray([to_int(r["label"]) for r in train_rows], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=3)

    if cfg.balanced_sampling:
        class_weights = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)
    else:
        class_weights = compute_class_weights(class_counts, power=cfg.class_weight_power)

    def make_ds(r: List[Dict[str, str]], shuffle: bool) -> ShardBatchIterableDataset:
        return ShardBatchIterableDataset(
            processed_dir=cfg.processed_dir,
            rows=r,
            batch_size=cfg.batch_size,
            max_loaded_shards=cfg.max_loaded_shards,
            shuffle_shards=shuffle,
            seed=cfg.seed,
            balanced_sampling=cfg.balanced_sampling and shuffle,
        )

    train_ds = make_ds(train_rows, True)
    val_ds = make_ds(val_rows, False) if val_rows else None
    test_ds = make_ds(test_rows, False) if test_rows else None

    loader_kwargs: Dict[str, Any] = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(cfg.prefetch_factor))

    train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, **loader_kwargs) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, **loader_kwargs) if test_ds else None
    return train_loader, val_loader, test_loader, class_weights, class_counts


def unwrap_batch(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, (list, tuple)) and len(batch) == 1:
        return batch[0]
    return batch


def infer_model_shapes_from_batch(batch: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(batch["raw_past"].shape[1]),
        int(batch["raw_t"].shape[1]),
        int(batch["raw_t"].shape[2]),
        int(batch["spec_t"].reshape(batch["spec_t"].shape[0], -1).shape[-1]),
    )


def check_first_batch(batch: Dict[str, Any]) -> None:
    required = ["raw_past", "raw_t", "raw_tdt", "spec_t", "spec_tdt", "label", "patient_idx"]
    for k in required:
        if k not in batch:
            raise KeyError(f"Missing batch key: {k}")
