"""Train GSWarpNet from scratch on the SRadar dataset.

Usage:
  python scripts/train.py --cache-dir ./data/sradar_cache \\
                           --ckpt-dir  ./checkpoints/gswarpnet_base

Options:
  --resume           Resume from checkpoint_latest.pt
  --epochs N         Max epochs (default 80)
  --batch-size N     Batch size (default 8)
  --lr FLOAT         Learning rate (default 1e-4)
  --device STR       Device string (default: auto-detect cuda)
  --seed N           Random seed (default 42)
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from gswarpnet import GSWarpNet, SRadarDataset, RadarSequenceAugment
from gswarpnet.loss import CompositeLossV5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs) -> float:
    model.train()
    total = 0.0
    for inputs, targets in tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output, flow, warp_seq = model(inputs, return_intermediates=True)
        loss, _ = criterion(output, flow, warp_seq, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()
    total = 0.0
    agg: dict = {}
    for inputs, targets in tqdm(loader, desc=f"Val   {epoch+1}/{epochs}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        output, flow, warp_seq = model(inputs, return_intermediates=True)
        loss, comps = criterion(output, flow, warp_seq, targets)
        total += loss.item()
        for k, v in comps.items():
            agg[k] = agg.get(k, 0.0) + v
    n = len(loader)
    return total / n, {k: v / n for k, v in agg.items()}


def save_ckpt(model, optimizer, scheduler, epoch, val_loss, best_val, ckpt_dir, is_best) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val,
        "config": {"model": {"t_in": model.t_in, "t_out": model.t_out}},
    }
    torch.save(state, ckpt_dir / "checkpoint_latest.pt")
    if is_best:
        torch.save(state, ckpt_dir / "best_model.pt")
        logger.info(f"  → New best saved (val={val_loss:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GSWarpNet on SRadar")
    parser.add_argument("--cache-dir",  required=True, help="Preprocessed data directory")
    parser.add_argument("--ckpt-dir",   default="./checkpoints/gswarpnet_base")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    ckpt_dir   = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    aug      = RadarSequenceAugment()
    train_ds = SRadarDataset(args.cache_dir, split="train", transform=aug)
    val_ds   = SRadarDataset(args.cache_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")

    model = GSWarpNet(t_in=4, t_out=12, base_channels=64, kernels_per_layer=2).to(device)
    criterion = CompositeLossV5(
        alpha=0.50, beta=0.25, gamma=0.15, delta=0.10,
        lambda_warp=0.30, lambda_pi=0.05, dbz_max=60.0,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val    = float("inf")
    no_improve  = 0
    start_epoch = 0

    if args.resume and (ckpt_dir / "checkpoint_latest.pt").exists():
        ckpt = torch.load(ckpt_dir / "checkpoint_latest.pt",
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val_loss"]
        logger.info(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, comps = val_epoch(model, val_loader, criterion, device, epoch, args.epochs)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | lr={lr_now:.2e} | "
            + " ".join(f"{k}={v:.4f}" for k, v in comps.items())
        )

        is_best = val_loss < best_val
        if is_best:
            best_val   = val_loss
            no_improve = 0
        else:
            no_improve += 1

        save_ckpt(model, optimizer, scheduler, epoch, val_loss, best_val, ckpt_dir, is_best)

        if no_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    logger.info(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
