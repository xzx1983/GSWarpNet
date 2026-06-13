"""Evaluation metrics for radar echo nowcasting.

Public API:
    compute_gms(pred, target)           → float   (novel metric)
    compute_sri(pred, target)           → float   (novel metric)
    compute_all_metrics(preds, targets) → Dict[str, float]

GMS (Gradient Map Similarity):
    Mean(2|∇ŷ||∇y| / (|∇ŷ|² + |∇y|² + ε))  over all T_out frames.
    Captures storm boundary sharpness. Novel metric: no prior literature.

SRI (Structural Reflectivity Index):
    IoU of binary Canny-threshold edge masks, mean over T_out frames.
    Captures storm outline fidelity. Novel metric: no prior literature.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List


DBZ_MAX = 60.0                   # SRadar normalisation ceiling
CSI_THRESHOLDS = [20.0, 35.0, 45.0]
EPS = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """[B, H, W] → gradient magnitude [B, H, W] via fixed Sobel kernels."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    f = x.unsqueeze(1)
    gx = F.conv2d(f, kx, padding=1)
    gy = F.conv2d(f, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + EPS).squeeze(1)


def _ssim_scalar(p: torch.Tensor, t: torch.Tensor, window_size: int = 11) -> float:
    """[B, H, W] × [B, H, W] → scalar SSIM."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=p.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    win = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).view(1, 1, window_size, window_size)
    pad = window_size // 2
    p = p.unsqueeze(1); t = t.unsqueeze(1)
    mu1 = F.conv2d(p, win, padding=pad); mu2 = F.conv2d(t, win, padding=pad)
    s1  = F.conv2d(p * p, win, padding=pad) - mu1 ** 2
    s2  = F.conv2d(t * t, win, padding=pad) - mu2 ** 2
    s12 = F.conv2d(p * t, win, padding=pad) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


def _contingency(pred_dbz: torch.Tensor, tgt_dbz: torch.Tensor,
                 threshold: float) -> Dict[str, float]:
    p = (pred_dbz >= threshold)
    t = (tgt_dbz  >= threshold)
    h  = (p &  t).float().sum().item()
    m  = (~p &  t).float().sum().item()
    fa = (p & ~t).float().sum().item()
    cn = (~p & ~t).float().sum().item()
    return {"h": h, "m": m, "fa": fa, "cn": cn}


def _csi(h: float, m: float, fa: float, **_) -> float:
    denom = h + m + fa
    return h / denom if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Public standalone metric functions
# ---------------------------------------------------------------------------

def compute_gms(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> float:
    """Gradient Map Similarity (GMS) — novel structure metric.

    Args:
        pred:   [B, T, H, W] or [T, H, W] normalised reflectivity in [0, 1].
        target: same shape as pred.
        eps:    numerical stability constant.

    Returns:
        Mean GMS across all frames, scalar float in (0, 1].
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, T, H, W = pred.shape
    flat_p = pred.clamp(0, 1).reshape(B * T, H, W)
    flat_t = target.clamp(0, 1).reshape(B * T, H, W)
    gm_p = _sobel_mag(flat_p)
    gm_t = _sobel_mag(flat_t)
    gms = (2.0 * gm_p * gm_t + eps) / (gm_p ** 2 + gm_t ** 2 + eps)
    return gms.mean().item()


def compute_sri(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_thresh_dbz: float = 10.0,
    dbz_max: float = DBZ_MAX,
) -> float:
    """Structural Reflectivity Index (SRI) — novel structure metric.

    IoU of binary Sobel-edge masks over all frames.

    Args:
        pred:             [B, T, H, W] or [T, H, W] normalised in [0, 1].
        target:           same shape as pred.
        edge_thresh_dbz:  Sobel magnitude threshold in dBZ (default 10.0).
        dbz_max:          normalisation ceiling (default 60.0 for SRadar).

    Returns:
        Mean IoU over all frames, scalar float in [0, 1].
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, T, H, W = pred.shape
    flat_p = pred.clamp(0, 1).reshape(B * T, H, W)
    flat_t = target.clamp(0, 1).reshape(B * T, H, W)
    gm_p = _sobel_mag(flat_p)
    gm_t = _sobel_mag(flat_t)
    thresh_norm = edge_thresh_dbz / dbz_max
    edge_p = (gm_p > thresh_norm)
    edge_t = (gm_t > thresh_norm)
    inter = (edge_p & edge_t).float().sum().item()
    union = (edge_p | edge_t).float().sum().item()
    return inter / union if union > 0 else float("nan")


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    dbz_max: float = DBZ_MAX,
    frame_interval_min: int = 5,
) -> Dict[str, float]:
    """Compute all evaluation metrics used in the paper.

    Args:
        preds:              [B, T_out, H, W] normalised in [0, 1].
        targets:            [B, T_out, H, W] normalised in [0, 1].
        dbz_max:            normalisation ceiling (default 60.0 for SRadar).
        frame_interval_min: minutes between frames (5 for SRadar, 1 for X-band).

    Returns:
        Flat dict with per-lead-time keys (e.g. 'CSI35_30min') and
        mean-across-lead-times keys (e.g. 'CSI35_mean').
    """
    B, T, H, W = preds.shape
    pred_dbz = preds.clamp(0, 1) * dbz_max
    tgt_dbz  = targets.clamp(0, 1) * dbz_max
    edge_thresh = 10.0 / dbz_max
    results: Dict[str, float] = {}

    lead_times = [frame_interval_min * (i + 1) for i in range(T)]

    for t_idx, lt in enumerate(lead_times):
        p_t  = pred_dbz[:, t_idx]
        tg_t = tgt_dbz[:, t_idx]
        pn_t = preds[:, t_idx].clamp(0, 1)
        tn_t = targets[:, t_idx].clamp(0, 1)
        tag  = f"_{lt}min"

        for thr in CSI_THRESHOLDS:
            cnt = _contingency(p_t, tg_t, thr)
            results[f"CSI{int(thr)}{tag}"] = _csi(**cnt)

        results[f"MAE{tag}"]  = (p_t - tg_t).abs().mean().item()
        results[f"SSIM{tag}"] = _ssim_scalar(pn_t, tn_t)

        gm_p = _sobel_mag(pn_t)
        gm_t = _sobel_mag(tn_t)
        gms  = (2 * gm_p * gm_t + EPS) / (gm_p ** 2 + gm_t ** 2 + EPS)
        results[f"GMS{tag}"] = gms.mean().item()

        edge_p = (gm_p > edge_thresh)
        edge_t = (gm_t > edge_thresh)
        inter  = (edge_p & edge_t).float().sum().item()
        union  = (edge_p | edge_t).float().sum().item()
        results[f"SRI{tag}"] = inter / union if union > 0 else float("nan")

    for prefix in ["CSI20", "CSI35", "CSI45", "MAE", "SSIM", "GMS", "SRI"]:
        vals = [v for k, v in results.items()
                if k.startswith(prefix + "_") and not (v != v)]
        if vals:
            results[f"{prefix}_mean"] = float(np.nanmean(vals))

    return results
