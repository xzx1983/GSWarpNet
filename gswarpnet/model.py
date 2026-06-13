"""GSWarpNet: Grid-Sampling Warp Network for radar echo nowcasting.

Architecture:
    [B, T_in, H, W]
        ├──────────────────────────────────────────┐
        ▼                                          │
    FlowEstimator (CBAM-UNet, 4-level)             │
        ▼                                          │
    flow [B, T_out, 2, H, W]                       │
        ▼                                          │
    WarpModule (grid_sample, zero-param)           │
        ▼                                          │
    warp_seq [B, T_out, H, W]                      │
        ▼                                          │
    warp_seq.detach()  ← GRADIENT STOP             │
        ▼                                          │
    concat(warp_for_refine, x) ←───────────────────┘
        ▼
    RefinementNet (3-level DSC-UNet)
        ▼
    output = (warp_for_refine + 0.5·tanh(residual)).clamp(0, 1)

Key design decisions:
- Zero-initialised flow head → persistence at training start.
- Gradient stop (detach) decouples FlowEstimator and RefinementNet supervision.
- FlowEstimator supervised via L_warp + L_pi; RefinementNet via task loss only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, kernels_per_layer: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch * kernels_per_layer, kernel_size,
                            padding=padding, groups=in_ch)
        self.pw = nn.Conv2d(in_ch * kernels_per_layer, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSCBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels_per_layer: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch, kernels_per_layer=kernels_per_layer),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_ch, out_ch, kernels_per_layer=kernels_per_layer),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# CBAM attention
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = torch.sigmoid(
            self.mlp(self.avg_pool(x).view(b, c)) +
            self.mlp(self.max_pool(x).view(b, c))
        ).view(b, c, 1, 1)
        return x * scale


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=3 if kernel_size == 7 else 1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction_ratio)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# FlowEstimator: 4-level CBAM-UNet → per-lead-time flow fields
# ---------------------------------------------------------------------------

class FlowEstimator(nn.Module):
    """Input: [B, T_in, H, W] → Output: [B, T_out, 2, H, W]."""

    def __init__(self, t_in: int, t_out: int, base_ch: int = 64,
                 kernels_per_layer: int = 2, use_cbam: bool = True) -> None:
        super().__init__()
        self.t_out = t_out
        C = base_ch
        _cbam = CBAM if use_cbam else lambda ch, **_: nn.Identity()

        self.enc1 = DSCBlock(t_in, C,     kernels_per_layer)
        self.enc2 = DSCBlock(C,    C * 2, kernels_per_layer)
        self.enc3 = DSCBlock(C*2,  C * 4, kernels_per_layer)
        self.enc4 = DSCBlock(C*4,  C * 4, kernels_per_layer)
        self.cbam1 = _cbam(C);     self.cbam2 = _cbam(C * 2)
        self.cbam3 = _cbam(C * 4); self.cbam4 = _cbam(C * 4)
        self.pool  = nn.MaxPool2d(2)

        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = DSCBlock(C*4 + C*4, C*2, kernels_per_layer)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = DSCBlock(C*2 + C*2, C,   kernels_per_layer)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DSCBlock(C   + C,   C,   kernels_per_layer)

        self.flow_head = nn.Conv2d(C, t_out * 2, kernel_size=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.cbam1(self.enc1(x))
        e2 = self.cbam2(self.enc2(self.pool(e1)))
        e3 = self.cbam3(self.enc3(self.pool(e2)))
        e4 = self.cbam4(self.enc4(self.pool(e3)))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        flow = self.flow_head(d1)
        B, _, H, W = flow.shape
        return flow.view(B, self.t_out, 2, H, W)


# ---------------------------------------------------------------------------
# WarpModule: parameter-free differentiable advection via grid_sample
# ---------------------------------------------------------------------------

class WarpModule(nn.Module):
    """Warps the last input frame by per-lead-time flow fields."""

    def __init__(self) -> None:
        super().__init__()
        self._cached_shape: tuple = (0, 0)
        self._cached_grid: torch.Tensor = torch.empty(0)

    def _base_grid(self, H: int, W: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        if self._cached_shape != (H, W) or self._cached_grid.device != device:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._cached_grid = torch.stack([gx, gy], dim=-1)
            self._cached_shape = (H, W)
        return self._cached_grid

    def forward(self, last_frame: torch.Tensor,
                flow_px: torch.Tensor) -> torch.Tensor:
        B, T_out, _, H, W = flow_px.shape
        base = self._base_grid(H, W, last_frame.device, last_frame.dtype)
        flow_norm = flow_px.permute(0, 1, 3, 4, 2).clone()
        flow_norm[..., 0] /= W / 2.0
        flow_norm[..., 1] /= H / 2.0
        grid = base.unsqueeze(0).unsqueeze(0) + flow_norm
        lf = last_frame.unsqueeze(1).expand(B, T_out, 1, H, W)
        warped = F.grid_sample(
            lf.reshape(B * T_out, 1, H, W),
            grid.reshape(B * T_out, H, W, 2),
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        return warped.view(B, T_out, H, W)


# ---------------------------------------------------------------------------
# RefinementNet: 3-level DSC-UNet residual predictor
# ---------------------------------------------------------------------------

class RefinementNet(nn.Module):
    """Input: [B, T_out+T_in, H, W] → Output: [B, T_out, H, W]."""

    def __init__(self, t_in: int, t_out: int, base_ch: int = 64,
                 kernels_per_layer: int = 2) -> None:
        super().__init__()
        C    = base_ch
        in_ch = t_out + t_in
        self.enc1 = DSCBlock(in_ch, C,    kernels_per_layer)
        self.enc2 = DSCBlock(C,    C*2,   kernels_per_layer)
        self.enc3 = DSCBlock(C*2,  C*4,   kernels_per_layer)
        self.btn  = DSCBlock(C*4,  C*4,   kernels_per_layer)
        self.pool = nn.MaxPool2d(2)
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = DSCBlock(C*4 + C*4, C*2, kernels_per_layer)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = DSCBlock(C*2 + C*2, C,   kernels_per_layer)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DSCBlock(C   + C,   C,   kernels_per_layer)
        self.head = nn.Conv2d(C, t_out, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.btn(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# GSWarpNet (full model)
# ---------------------------------------------------------------------------

class GSWarpNet(nn.Module):
    """Decoupled warp-and-refine nowcasting network.

    Args:
        t_in:              Number of input frames.
        t_out:             Number of output frames.
        base_channels:     Base channel width for both sub-networks (default 64).
        kernels_per_layer: DSC channel multiplier (default 2).
        use_gradient_stop: If True (default), detach warp_seq before RefinementNet.
        use_cbam:          If True (default), use CBAM in FlowEstimator.
    """

    def __init__(
        self,
        t_in: int = 4,
        t_out: int = 12,
        base_channels: int = 64,
        kernels_per_layer: int = 2,
        use_gradient_stop: bool = True,
        use_cbam: bool = True,
    ) -> None:
        super().__init__()
        self.t_in  = t_in
        self.t_out = t_out
        self.use_gradient_stop = use_gradient_stop

        self.flow_estimator = FlowEstimator(
            t_in=t_in, t_out=t_out, base_ch=base_channels,
            kernels_per_layer=kernels_per_layer, use_cbam=use_cbam,
        )
        self.warp    = WarpModule()
        self.refiner = RefinementNet(
            t_in=t_in, t_out=t_out, base_ch=base_channels,
            kernels_per_layer=kernels_per_layer,
        )

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """Forward pass.

        Args:
            x: [B, T_in, H, W] normalised reflectivity in [0, 1].
            return_intermediates: If True return (output, flow, warp_seq).

        Returns:
            output [B, T_out, H, W], or (output, flow, warp_seq) if requested.
        """
        flow     = self.flow_estimator(x)
        warp_seq = self.warp(x[:, -1:], flow)

        warp_for_refine = warp_seq.detach() if self.use_gradient_stop else warp_seq
        residual = 0.5 * torch.tanh(self.refiner(torch.cat([warp_for_refine, x], dim=1)))
        output   = (warp_for_refine + residual).clamp(0.0, 1.0)

        if return_intermediates:
            return output, flow, warp_seq
        return output

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> "GSWarpNet":
        """Load a GSWarpNet from a checkpoint saved by the training scripts."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        # Support both OmegaConf config objects and plain dicts
        def _get(obj, *keys, default=None):
            for k in keys:
                try:
                    obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
                except (KeyError, AttributeError):
                    return default
            return obj

        model = GSWarpNet(
            t_in=_get(cfg, "model", "t_in", default=4),
            t_out=_get(cfg, "model", "t_out", default=12),
            base_channels=_get(cfg, "model", "base_channels", default=64),
            kernels_per_layer=_get(cfg, "model", "kernels_per_layer", default=2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model.eval().to(device)
