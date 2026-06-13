"""Evaluate GSWarpNet (base + DBZ fine-tune), Persistence, and OpticalFlow baselines
on the SRadar test set.

Usage:
  python scripts/evaluate.py \\
      --cache-dir      ./data/sradar_cache \\
      --base-ckpt      ./checkpoints/gswarpnet_base/best_model.pt \\
      --dbz-ckpt       ./checkpoints/gswarpnet_dbz/best_model.pt \\
      --output-dir     ./outputs/results

Options:
  --base-ckpt   Base GSWarpNet checkpoint (optional; skip if not available)
  --dbz-ckpt    DBZ fine-tuned checkpoint (optional)
  --no-optical  Skip optical flow baseline (requires opencv)
  --batch-size  Batch size (default 8)
  --device      Device string (default: auto-detect)
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gswarpnet import (
    GSWarpNet, SRadarDataset,
    PersistenceBaseline, OpticalFlowBaseline,
    compute_all_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DBZ_MAX = 60.0
T_IN    = 4
T_OUT   = 12


def _agg(batch_metrics: list) -> dict:
    agg = defaultdict(list)
    for m in batch_metrics:
        for k, v in m.items():
            if v == v:  # skip NaN
                agg[k].append(v)
    return {k: float(np.mean(vs)) for k, vs in agg.items()}


def _metrics(preds: torch.Tensor, targets: torch.Tensor, device: str) -> dict:
    return compute_all_metrics(
        preds.to(device), targets.to(device),
        dbz_max=DBZ_MAX, frame_interval_min=5,
    )


def _summarise(name: str, m: dict) -> str:
    fmt = lambda k: f"{m.get(k, float('nan')):.4f}"
    return (
        f"{name:<24}  "
        f"CSI@20={fmt('CSI20_mean')}  CSI@35={fmt('CSI35_mean')}  "
        f"CSI@45={fmt('CSI45_mean')}  MAE={fmt('MAE_mean')}  "
        f"SSIM={fmt('SSIM_mean')}  GMS={fmt('GMS_mean')}  SRI={fmt('SRI_mean')}"
    )


def evaluate_persistence(loader, device) -> dict:
    logger.info("Evaluating Persistence …")
    all_m = []
    for inputs, targets in tqdm(loader, desc="Persistence"):
        last  = inputs[:, -1:]
        preds = last.expand(-1, T_OUT, -1, -1)
        all_m.append(_metrics(preds, targets, device))
    return _agg(all_m)


def evaluate_optical_flow(loader, device) -> dict:
    logger.info("Evaluating OpticalFlow …")
    of = OpticalFlowBaseline(t_out=T_OUT)
    all_m = []
    for inputs, targets in tqdm(loader, desc="OpticalFlow"):
        preds = of(inputs).clamp(0, 1)
        all_m.append(_metrics(preds, targets, device))
    return _agg(all_m)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: str, name: str) -> dict:
    logger.info(f"Evaluating {name} …")
    model.eval()
    all_m = []
    for inputs, targets in tqdm(loader, desc=name):
        inputs = inputs.to(device)
        preds  = model(inputs).clamp(0, 1)
        all_m.append(_metrics(preds, targets, device))
    return _agg(all_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate nowcasting models on SRadar test set")
    parser.add_argument("--cache-dir",   required=True)
    parser.add_argument("--base-ckpt",   default=None,
                        help="GSWarpNet base checkpoint (optional)")
    parser.add_argument("--dbz-ckpt",    default=None,
                        help="GSWarpNet+DBZ checkpoint (optional)")
    parser.add_argument("--output-dir",  default="./outputs/results")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--no-optical",  action="store_true",
                        help="Skip optical flow baseline")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = SRadarDataset(args.cache_dir, split="test")
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)
    logger.info(f"Test sequences: {len(test_ds)}  Device: {device}")

    results: dict = {}

    results["Persistence"] = evaluate_persistence(loader, device)
    logger.info(_summarise("Persistence", results["Persistence"]))

    if not args.no_optical:
        try:
            results["OpticalFlow"] = evaluate_optical_flow(loader, device)
            logger.info(_summarise("OpticalFlow", results["OpticalFlow"]))
        except ImportError as e:
            logger.warning(f"Skipping OpticalFlow: {e}")

    if args.base_ckpt and Path(args.base_ckpt).exists():
        model = GSWarpNet.from_checkpoint(args.base_ckpt, device=device)
        results["GSWarpNet_base"] = evaluate_model(model, loader, device, "GSWarpNet_base")
        logger.info(_summarise("GSWarpNet_base", results["GSWarpNet_base"]))
        del model

    if args.dbz_ckpt and Path(args.dbz_ckpt).exists():
        model = GSWarpNet.from_checkpoint(args.dbz_ckpt, device=device)
        results["GSWarpNet_DBZ"] = evaluate_model(model, loader, device, "GSWarpNet_DBZ")
        logger.info(_summarise("GSWarpNet_DBZ", results["GSWarpNet_DBZ"]))
        del model

    out_path = out_dir / "sradar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_path}")

    print("\n" + "=" * 100)
    print("SRadar Test Set Results (mean over 12 lead times)")
    print("=" * 100)
    for name, m in results.items():
        print(_summarise(name, m))
    print("=" * 100)


if __name__ == "__main__":
    main()
