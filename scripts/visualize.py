"""Generate qualitative figures and absolute error maps for the SRadar dataset.

Produces two figure types for each specified test-set case index:
  1. Qualitative grid: input history | ground truth | model predictions
  2. Absolute error map: |pred - target| in dBZ for each model

Usage:
  python scripts/visualize.py \\
      --cache-dir   ./data/sradar_cache \\
      --base-ckpt   ./checkpoints/gswarpnet_base/best_model.pt \\
      --dbz-ckpt    ./checkpoints/gswarpnet_dbz/best_model.pt \\
      --output-dir  ./outputs/figures \\
      --cases 200 7845

Options:
  --lead-idxs 1 5 8 11   0-indexed lead times to show (default: 1 5 8 11 → +10/30/45/60 min)
  --no-optical           Skip optical flow (requires opencv)
  --dpi N                Figure DPI (default 150)
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from gswarpnet import (
    GSWarpNet, SRadarDataset,
    PersistenceBaseline, OpticalFlowBaseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DBZ_MAX  = 60.0
CMAP_DBZ = "NWS_reflectivity" if "NWS_reflectivity" in plt.colormaps else "turbo"
CMAP_ERR = "hot_r"


def _dbz(x: torch.Tensor) -> np.ndarray:
    """[H, W] tensor → dBZ float32 numpy array."""
    return (x.cpu().float().numpy() * DBZ_MAX)


def _load_gswarp(path: str, device: str) -> torch.nn.Module:
    return GSWarpNet.from_checkpoint(path, device=device)


def _get_predictions(inputs: torch.Tensor, targets: torch.Tensor,
                     models: dict, device: str, no_optical: bool) -> dict:
    """Run all models and return {name: [T_out, H, W] tensor} predictions."""
    preds: dict = {}

    preds["Ground truth"] = targets[0]

    last = inputs[0, -1:].unsqueeze(0)
    preds["Persistence"] = last.expand(1, targets.shape[1], -1, -1)[0]

    if not no_optical:
        try:
            of = OpticalFlowBaseline(t_out=targets.shape[1])
            preds["OpticalFlow"] = of(inputs)[0].cpu()
        except ImportError:
            logger.warning("OpticalFlow skipped (opencv not installed)")

    for name, model in models.items():
        with torch.no_grad():
            out = model(inputs.to(device)).clamp(0, 1).cpu()
        preds[name] = out[0]

    return preds


def _plot_qualitative(inputs, preds, lead_idxs, lead_labels, case_idx, out_dir, dpi) -> None:
    """Plot input strip + prediction grid."""
    T_in = inputs.shape[1]
    model_names = list(preds.keys())
    n_models = len(model_names)
    n_leads  = len(lead_idxs)

    fig_w = (T_in + n_leads) * 1.3 + 0.5
    fig_h = (n_models + 1) * 1.3 + 0.5
    fig, axes = plt.subplots(n_models + 1, T_in + n_leads,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})

    vmin, vmax = 0.0, 60.0

    def _show(ax, arr, title="", cmap=CMAP_DBZ):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=7, pad=2)
        return im

    # Row 0: input frames
    for t in range(T_in):
        _show(axes[0, t], _dbz(inputs[0, t]), title=f"Input t-{T_in - t}" if t == 0 else "")
    for j, lt in enumerate(lead_labels):
        axes[0, T_in + j].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=7, labelpad=2)

    # Remaining rows: models
    for row, name in enumerate(model_names, start=1):
        for t in range(T_in):
            axes[row, t].axis("off")
        for j, (lt_idx, lt_lbl) in enumerate(zip(lead_idxs, lead_labels)):
            arr = _dbz(preds[name][lt_idx])
            title = lt_lbl if row == 1 else ""
            _show(axes[row, T_in + j], arr, title=title)
        axes[row, T_in].set_ylabel(name, fontsize=7, labelpad=2)

    plt.suptitle(f"SRadar case {case_idx}", fontsize=9, y=1.01)
    for fmt in ("png", "pdf"):
        out = out_dir / f"qualitative_case{case_idx}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        logger.info(f"  Saved {out}")
    plt.close(fig)


def _plot_error_maps(preds, lead_idxs, lead_labels, case_idx, out_dir, dpi) -> None:
    """Plot |pred - target| error maps in dBZ."""
    gt = preds["Ground truth"]
    model_names = [n for n in preds if n != "Ground truth"]
    n_models = len(model_names)
    n_leads  = len(lead_idxs)

    fig, axes = plt.subplots(n_models, n_leads,
                             figsize=(n_leads * 1.8, n_models * 1.5),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    if n_models == 1:
        axes = axes[np.newaxis, :]

    emax = 30.0

    for row, name in enumerate(model_names):
        axes[row, 0].set_ylabel(name, fontsize=7, labelpad=2)
        for j, (lt_idx, lt_lbl) in enumerate(zip(lead_idxs, lead_labels)):
            err = np.abs(_dbz(preds[name][lt_idx]) - _dbz(gt[lt_idx]))
            mae = err.mean()
            im  = axes[row, j].imshow(err, vmin=0, vmax=emax, cmap=CMAP_ERR,
                                      interpolation="nearest")
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            if row == 0:
                axes[row, j].set_title(lt_lbl, fontsize=7, pad=2)
            axes[row, j].text(0.02, 0.98, f"MAE={mae:.1f}", transform=axes[row, j].transAxes,
                              fontsize=6, va="top", color="white")

    plt.suptitle(f"Absolute Error Maps — SRadar case {case_idx}", fontsize=9, y=1.01)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Error (dBZ)")

    for fmt in ("png", "pdf"):
        out = out_dir / f"error_maps_case{case_idx}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        logger.info(f"  Saved {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative figures for SRadar")
    parser.add_argument("--cache-dir",   required=True)
    parser.add_argument("--base-ckpt",   default=None)
    parser.add_argument("--dbz-ckpt",    default=None)
    parser.add_argument("--output-dir",  default="./outputs/figures")
    parser.add_argument("--cases",       type=int, nargs="+", default=[200, 7845])
    parser.add_argument("--lead-idxs",   type=int, nargs="+", default=[1, 5, 8, 11])
    parser.add_argument("--no-optical",  action="store_true")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--dpi",         type=int, default=150)
    args = parser.parse_args()

    device  = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lead_labels = [f"+{(i + 1) * 5} min" for i in args.lead_idxs]

    # Load models
    models: dict = {}
    if args.base_ckpt and Path(args.base_ckpt).exists():
        models["GSWarpNet(base)"] = _load_gswarp(args.base_ckpt, device)
    if args.dbz_ckpt and Path(args.dbz_ckpt).exists():
        models["GSWarpNet+DBZ"]   = _load_gswarp(args.dbz_ckpt, device)

    # Load test dataset
    test_ds = SRadarDataset(args.cache_dir, split="test")

    for case_idx in args.cases:
        if case_idx >= len(test_ds):
            logger.warning(f"Case {case_idx} out of range ({len(test_ds)} test sequences)")
            continue

        inputs, targets = test_ds[case_idx]
        inputs  = inputs.unsqueeze(0)    # [1, T_in, H, W]
        targets = targets.unsqueeze(0)   # [1, T_out, H, W]

        logger.info(f"Processing case {case_idx} …")
        preds = _get_predictions(inputs, targets, models, device, args.no_optical)

        _plot_qualitative(inputs, preds, args.lead_idxs, lead_labels, case_idx, out_dir, args.dpi)
        _plot_error_maps(preds, args.lead_idxs, lead_labels, case_idx, out_dir, args.dpi)

    logger.info("Done.")


if __name__ == "__main__":
    main()
