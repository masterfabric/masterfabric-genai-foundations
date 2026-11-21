"""Device selection, seeding and small helpers shared by every experiment."""
from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def env_info(dev: torch.device) -> dict:
    return {"torch": torch.__version__, "device": dev.type, "machine": platform.machine(), "python": platform.python_version()}


def save(name: str, obj: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False))
