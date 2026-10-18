"""Utils for the mv-PIKN training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import csv
import json
import logging
import random
import re
import sys
import numpy as np
import torch






class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stdout)],
)


logger = logging.getLogger("pikn_v12")


def status(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def section(title: str) -> None:
    bar = "─" * max(12, 78 - len(title))
    status(f"\n┌ {title} {bar}")


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def configure_torch_runtime(use_cuda: bool, use_tf32: bool = True) -> None:
    if use_cuda and torch.cuda.is_available() and use_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m {s:02d}s"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def to_int(x: Any, default: Optional[int] = None) -> int:
    try:
        return int(x)
    except Exception:
        if default is None:
            raise
        return default


def as_float_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float().contiguous()
    return torch.as_tensor(x, dtype=torch.float32).contiguous()


def as_long_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.long().contiguous()
    return torch.as_tensor(x, dtype=torch.long).contiguous()


def infer_patient_id(row: Dict[str, str]) -> str:
    pid = str(row.get("patient_id", "")).strip()
    if pid and pid.lower() != "nan":
        return pid
    src = str(row.get("source_file", "")).strip()
    if src:
        m = re.match(r"(chb\d+[a-z]?)", src, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        stem = Path(src).stem
        m2 = re.match(r"([a-zA-Z]+\d+)", stem)
        if m2:
            return m2.group(1).lower()
    return "unknown"


def maybe_get_float(batch: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for k in keys:
        if k in batch and batch[k] is not None:
            v = batch[k]
            return v.float() if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)
    return None
