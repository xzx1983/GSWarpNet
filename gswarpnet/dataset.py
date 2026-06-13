"""SRadar dataset (SHMU Malý Javorník) for radar echo nowcasting.

Reads from a pre-built float16 memmap shared across all DataLoader workers.

Preprocessing (run once before training):
    python scripts/prepare_data.py --hdf5-dir <HDF5_ROOT> --cache-dir <CACHE_DIR>

This produces:
    <CACHE_DIR>/frames.bin       float16 memmap  [N, 336, 336]
    <CACHE_DIR>/frame_idx.json   basename → row index
    <CACHE_DIR>/train.json       list of {inputs, targets} dicts
    <CACHE_DIR>/val.json
    <CACHE_DIR>/test.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Module-level shared memmap (opened once per process)
_MMAP: Optional[np.ndarray] = None
_FRAME_IDX: Dict[str, int]  = {}
_MMAP_PATH: str              = ""


def _open_memmap(cache_dir: str) -> None:
    """Open the shared memmap (idempotent after first call)."""
    global _MMAP, _FRAME_IDX, _MMAP_PATH
    bin_path = str(Path(cache_dir) / "frames.bin")
    idx_path = str(Path(cache_dir) / "frame_idx.json")

    if _MMAP is not None and _MMAP_PATH == bin_path:
        return

    if not Path(bin_path).exists():
        raise FileNotFoundError(
            f"Memmap not found: {bin_path}\n"
            "Run: python scripts/prepare_data.py --hdf5-dir <HDF5_ROOT> --cache-dir <CACHE_DIR>"
        )

    with open(idx_path) as f:
        _FRAME_IDX = json.load(f)

    N = len(_FRAME_IDX)
    _MMAP = np.memmap(bin_path, dtype="float16", mode="r", shape=(N, 336, 336))
    _MMAP_PATH = bin_path
    size_gb = N * 336 * 336 * 2 / 1e9
    logger.info(f"Opened SRadar memmap: {N} frames ({size_gb:.1f} GB float16)")


def _get_frame(path: str) -> np.ndarray:
    """Load one float32 [336, 336] frame from the shared memmap."""
    key = Path(path).name
    row = _FRAME_IDX[key]
    return _MMAP[row].astype(np.float32)


class SRadarDataset(Dataset):
    """SRadar SHMU dataset for T_in → T_out radar echo nowcasting.

    Args:
        cache_dir:  Directory containing frames.bin and split JSON files.
        split:      One of "train", "val", "test".
        transform:  Optional callable(inputs, targets) → (inputs, targets).

    Returns per item:
        inputs:  float32 tensor [T_in,  336, 336] in [0, 1]
        targets: float32 tensor [T_out, 336, 336] in [0, 1]
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.split     = split
        self.transform = transform

        _open_memmap(cache_dir)

        idx_path = Path(cache_dir) / f"{split}.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"Split index not found: {idx_path}")
        with open(idx_path) as f:
            self.sequences: List[dict] = json.load(f)

        logger.info(f"[{split}] {len(self.sequences)} SRadar sequences loaded")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq     = self.sequences[idx]
        inputs  = np.stack([_get_frame(p) for p in seq["inputs"]],  axis=0)
        targets = np.stack([_get_frame(p) for p in seq["targets"]], axis=0)
        x = torch.from_numpy(inputs)
        y = torch.from_numpy(targets)
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y
