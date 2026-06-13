"""Composite losses for GSWarpNet training.

CompositeLossV5 = task loss + warp supervision + flow divergence regularisation.

Fixed weight schedule (do not modify):
  alpha=0.50  MSE
  beta=0.25   GMS
  gamma=0.15  SSIM
  delta=0.10  Edge
  lambda_warp=0.30  direct warp supervision
  lambda_pi=0.05    flow divergence regularisation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Building-block losses
# ---------------------------------------------------------------------------

class GradientMagnitudeSimilarityLoss(nn.Module):
    """L_GMS: 1 − mean(2|∇ŷ||∇y| / (|∇ŷ|² + |∇y|² + ε))."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        flat = x.clamp(0, 1).reshape(B * T, 1, H, W)
        gx = F.conv2d(flat, self.sobel_x, padding=1)
        gy = F.conv2d(flat, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + self.eps).reshape(B, T, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gm_p = self._grad_mag(pred)
        gm_t = self._grad_mag(target)
        gms = (2.0 * gm_p * gm_t + self.eps) / (gm_p ** 2 + gm_t ** 2 + self.eps)
        return 1.0 - gms.mean()


class SSIMLoss(nn.Module):
    """L_SSIM: 1 − SSIM(pred, target), single-scale window 11×11."""

    def __init__(self, window_size: int = 11,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> None:
        super().__init__()
        self.ws = window_size
        self.C1 = C1
        self.C2 = C2
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, H, W = pred.shape
        p = pred.clamp(0, 1).reshape(B * T, 1, H, W)
        t = target.clamp(0, 1).reshape(B * T, 1, H, W)
        pad = self.ws // 2
        mu1 = F.conv2d(p, self.window, padding=pad)
        mu2 = F.conv2d(t, self.window, padding=pad)
        mu1sq = mu1 ** 2; mu2sq = mu2 ** 2; mu12 = mu1 * mu2
        s1  = F.conv2d(p * p, self.window, padding=pad) - mu1sq
        s2  = F.conv2d(t * t, self.window, padding=pad) - mu2sq
        s12 = F.conv2d(p * t, self.window, padding=pad) - mu12
        num = (2 * mu12 + self.C1) * (2 * s12 + self.C2)
        den = (mu1sq + mu2sq + self.C1) * (s1 + s2 + self.C2)
        return 1.0 - (num / den).mean()


class EdgeLoss(nn.Module):
    """L_edge: soft BCE between Sobel-thresholded edge maps."""

    def __init__(self, threshold: float = 10.0 / 70.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        flat = x.clamp(0, 1).reshape(B * T, 1, H, W)
        gx = F.conv2d(flat, self.sobel_x, padding=1)
        gy = F.conv2d(flat, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        soft_edge = torch.sigmoid((mag - self.threshold) / (self.threshold + self.eps))
        return soft_edge.reshape(B, T, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(self._edge_map(pred), self._edge_map(target))


class FlowDivergenceLoss(nn.Module):
    """L_pi = mean|∇·u|, physics-informed incompressibility constraint.

    Reference: LUPIN (Pavlík et al., 2025).
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        du_dx = flow[:, :, 0, :, 1:] - flow[:, :, 0, :, :-1]
        dv_dy = flow[:, :, 1, 1:, :] - flow[:, :, 1, :-1, :]
        h = min(du_dx.shape[2], dv_dy.shape[2])
        w = min(du_dx.shape[3], dv_dy.shape[3])
        return (du_dx[:, :, :h, :w] + dv_dy[:, :, :h, :w]).abs().mean()


class WarpSupervisionLoss(nn.Module):
    """L_warp: intensity-weighted MSE(warp_seq, target).

    Provides direct gradient signal to FlowEstimator when gradient stop is used.
    """

    def __init__(self, dbz_max: float = 60.0) -> None:
        super().__init__()
        self._heavy_rain_thresh = 35.0 / dbz_max

    def forward(self, warp_seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        T = warp_seq.shape[1]
        w_t = torch.linspace(1.0, 2.0, T, device=warp_seq.device).view(1, T, 1, 1)
        w_i = 1.0 + 3.0 * (target >= self._heavy_rain_thresh).float()
        return (w_t * w_i * (warp_seq - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Composite losses
# ---------------------------------------------------------------------------

class _TaskLoss(nn.Module):
    """Internal: alpha*MSE + beta*GMS + gamma*SSIM + delta*Edge."""

    def __init__(
        self,
        alpha: float = 0.50,
        beta:  float = 0.25,
        gamma: float = 0.15,
        delta: float = 0.10,
        dbz_max: float = 60.0,
        intensity_power: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.intensity_power = intensity_power
        self._heavy_rain_thresh = 35.0 / dbz_max
        self.gms_loss  = GradientMagnitudeSimilarityLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss(threshold=10.0 / dbz_max)

    def _temporal_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        T = pred.shape[1]
        w_t = torch.linspace(1.0, 2.0, T, device=pred.device).view(1, T, 1, 1)
        w_i = 1.0 + 3.0 * (target >= self._heavy_rain_thresh).float()
        if self.intensity_power > 0.0:
            w_i = w_i * (1.0 + self.intensity_power * target.pow(2))
        return (w_t * w_i * (pred - target) ** 2).mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_mse  = self._temporal_mse(pred, target)
        l_gms  = self.gms_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        total  = self.alpha * l_mse + self.beta * l_gms + self.gamma * l_ssim + self.delta * l_edge
        return total, {
            "mse": l_mse.item(), "gms": l_gms.item(),
            "ssim": l_ssim.item(), "edge": l_edge.item(), "total": total.item(),
        }


class CompositeLossV5(nn.Module):
    """Full training loss for GSWarpNet.

    L_total = L_task(output, target)
            + lambda_warp * L_warp(warp_seq, target)
            + lambda_pi   * L_pi(flow)

    Args:
        alpha, beta, gamma, delta: task loss weights (default: 0.50/0.25/0.15/0.10).
        lambda_warp: warp supervision weight (default 0.30).
        lambda_pi:   divergence regularisation weight (default 0.05).
        dbz_max:     dataset normalisation ceiling (default 60.0 for SRadar).
        intensity_power: extra quadratic intensity boost (set >0 for DBZ fine-tuning).
    """

    def __init__(
        self,
        alpha: float = 0.50,
        beta:  float = 0.25,
        gamma: float = 0.15,
        delta: float = 0.10,
        lambda_warp: float = 0.30,
        lambda_pi:   float = 0.05,
        dbz_max: float = 60.0,
        intensity_power: float = 0.0,
    ) -> None:
        super().__init__()
        self.task_loss  = _TaskLoss(alpha, beta, gamma, delta, dbz_max, intensity_power)
        self.warp_loss  = WarpSupervisionLoss(dbz_max=dbz_max)
        self.div_loss   = FlowDivergenceLoss()
        self.lambda_warp = lambda_warp
        self.lambda_pi   = lambda_pi

    def forward(
        self,
        output:   torch.Tensor,
        flow:     torch.Tensor,
        warp_seq: torch.Tensor,
        target:   torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            output:   [B, T_out, H, W] final model prediction.
            flow:     [B, T_out, 2, H, W] flow field from FlowEstimator.
            warp_seq: [B, T_out, H, W] warp output (before detach).
            target:   [B, T_out, H, W] ground truth.

        Returns:
            (total_loss, component_dict)
        """
        l_task, comps = self.task_loss(output, target)
        l_warp = self.warp_loss(warp_seq, target)
        l_pi   = self.div_loss(flow)
        total  = l_task + self.lambda_warp * l_warp + self.lambda_pi * l_pi
        comps.update({"warp": l_warp.item(), "pi": l_pi.item(), "total": total.item()})
        return total, comps
