"""Preprocess SRadar HDF5 files into a shared float16 memmap and build split indexes.

This script must be run once before training or evaluation.

Steps:
  1. Scan all *.h5 files in --hdf5-dir
  2. Convert to float16 memmap: clip(raw*0.5-31.5, 0, DBZ_MAX) / DBZ_MAX
  3. Read timestamps from metadata.h5 to identify consecutive 5-min runs
  4. Build train / val / test JSON indexes (chronological split)

Outputs in --cache-dir:
  frames.bin       float16 memmap  [N, 336, 336]
  frame_idx.json   {filename: row_index}
  train.json / val.json / test.json

Usage:
  python scripts/prepare_data.py --hdf5-dir /path/to/SRadar/mj_uint8 \\
                                  --metadata /path/to/SRadar/metadata.h5 \\
                                  --cache-dir ./data/sradar_cache
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

T_IN          = 4
T_OUT         = 12
SEQ_LEN       = T_IN + T_OUT
DBZ_MAX       = 60.0
CROP          = slice(2, 338)        # 340 → 336
VAL_FRAC      = 0.15
TEST_BOUNDARY = datetime(2018, 9, 2)


def build_memmap(hdf5_dir: Path, cache_dir: Path) -> None:
    """Step 1 & 2: build frames.bin and frame_idx.json."""
    files = sorted(hdf5_dir.glob("*.h5"))
    N = len(files)
    if N == 0:
        raise FileNotFoundError(f"No *.h5 files found in {hdf5_dir}")
    logger.info(f"Total frames: {N}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir / "frames.bin"
    idx_path = cache_dir / "frame_idx.json"

    size_gb = N * 336 * 336 * 2 / 1e9
    logger.info(f"Allocating {bin_path} ({size_gb:.1f} GB) …")
    mmap = np.memmap(str(bin_path), dtype="float16", mode="w+", shape=(N, 336, 336))
    frame_idx: dict = {}

    for i, path in enumerate(tqdm(files, desc="Converting")):
        with h5py.File(str(path), "r") as f:
            raw = f["dBZ"]["data"]["data"][()].astype(np.float32)
        dbz  = raw * 0.5 - 31.5
        dbz  = dbz[CROP, CROP]
        norm = np.clip(dbz, 0.0, DBZ_MAX) / DBZ_MAX
        mmap[i] = norm.astype(np.float16)
        frame_idx[path.name] = i

    mmap.flush()
    logger.info("Memmap written.")

    with open(idx_path, "w") as f:
        json.dump(frame_idx, f)
    logger.info(f"Frame index → {idx_path}")


def build_indexes(hdf5_dir: Path, metadata_h5: Path, cache_dir: Path) -> None:
    """Step 3 & 4: build train/val/test JSON split indexes."""
    files = sorted(hdf5_dir.glob("*.h5"))
    file_paths = [str(p) for p in files]

    with h5py.File(str(metadata_h5), "r") as f:
        timestamps = [ts.decode() for ts in f["timestamps"][()]]

    if len(timestamps) != len(files):
        raise ValueError(
            f"File count ({len(files)}) != timestamp count ({len(timestamps)}). "
            "Check that --hdf5-dir and --metadata match the same dataset."
        )

    def parse_ts(s: str) -> datetime:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")

    dts = [parse_ts(t) for t in timestamps]
    logger.info(f"Building sequences (T_in={T_IN}, T_out={T_OUT}) …")

    # Find consecutive 5-min runs
    runs: list = []
    run_start = 0
    for i in range(1, len(dts)):
        gap_min = (dts[i] - dts[i - 1]).total_seconds() / 60
        if abs(gap_min - 5.0) > 0.5:
            if i - run_start >= SEQ_LEN:
                runs.append((run_start, i - 1))
            run_start = i
    if len(dts) - run_start >= SEQ_LEN:
        runs.append((run_start, len(dts) - 1))
    logger.info(f"Consecutive runs found: {len(runs)}")

    all_seqs: list = []
    for (r_start, r_end) in runs:
        for i in range(r_start, r_end - SEQ_LEN + 2):
            all_seqs.append({
                "inputs":   [file_paths[j] for j in range(i, i + T_IN)],
                "targets":  [file_paths[j] for j in range(i + T_IN, i + SEQ_LEN)],
                "_first_ts": dts[i],
            })
    logger.info(f"Total sequences: {len(all_seqs)}")

    pre_test  = [s for s in all_seqs if s["_first_ts"] < TEST_BOUNDARY]
    test_seqs = [s for s in all_seqs if s["_first_ts"] >= TEST_BOUNDARY]
    val_start = int(len(pre_test) * (1 - VAL_FRAC))
    train_seqs = pre_test[:val_start]
    val_seqs   = pre_test[val_start:]

    logger.info(f"Train={len(train_seqs)}  Val={len(val_seqs)}  Test={len(test_seqs)}")

    def strip(seqs: list) -> list:
        return [{"inputs": s["inputs"], "targets": s["targets"]} for s in seqs]

    for name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        out = cache_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(strip(seqs), f)
        logger.info(f"  {out} ({len(seqs)} sequences)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SRadar dataset")
    parser.add_argument("--hdf5-dir",  required=True,
                        help="Directory containing SRadar *.h5 frames (mj_uint8)")
    parser.add_argument("--metadata",  required=True,
                        help="Path to metadata.h5 containing timestamps dataset")
    parser.add_argument("--cache-dir", required=True,
                        help="Output directory for memmap and index files")
    args = parser.parse_args()

    hdf5_dir   = Path(args.hdf5_dir)
    metadata   = Path(args.metadata)
    cache_dir  = Path(args.cache_dir)

    build_memmap(hdf5_dir, cache_dir)
    build_indexes(hdf5_dir, metadata, cache_dir)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
