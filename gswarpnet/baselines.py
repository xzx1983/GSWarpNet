"""Non-learning baselines for radar echo nowcasting.

PersistenceBaseline  — repeats the last input frame for all T_out steps.
OpticalFlowBaseline  — linear extrapolation via Farneback dense optical flow.

Both accept [B, T_in, H, W] tensors and return [B, T_out, H, W].
"""

from __future__ import annotations

import numpy as np
import torch


class PersistenceBaseline:
    """Repeat the last input frame for all future lead times.

    Args:
        t_out: Number of output frames (default 12 for SRadar).
    """

    def __init__(self, t_out: int = 12) -> None:
        self.t_out = t_out

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args: inputs [B, T_in, H, W]. Returns [B, T_out, H, W]."""
        last = inputs[:, -1:, :, :]
        return last.expand(-1, self.t_out, -1, -1).clone()


class OpticalFlowBaseline:
    """Dense optical flow extrapolation using OpenCV Farneback method.

    Motion is estimated from the last two input frames and extrapolated
    linearly for each lead-time step k = 1 … T_out.

    Args:
        t_out: Number of output frames (default 12 for SRadar).

    Raises:
        ImportError: If opencv-python-headless is not installed.
    """

    def __init__(self, t_out: int = 12) -> None:
        self.t_out = t_out
        try:
            import cv2
            self.cv2 = cv2
        except ImportError as e:
            raise ImportError(
                "opencv-python-headless is required for OpticalFlowBaseline.\n"
                "Install: uv pip install opencv-python-headless"
            ) from e

    def _compute_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        p8 = (prev * 255).astype(np.uint8)
        c8 = (curr * 255).astype(np.uint8)
        return self.cv2.calcOpticalFlowFarneback(
            p8, c8, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    def _warp(self, frame: np.ndarray, flow: np.ndarray, k: int) -> np.ndarray:
        H, W = frame.shape
        xs, ys = np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        mx, my = np.meshgrid(xs, ys)
        map_x = (mx + flow[:, :, 0] * k).astype(np.float32)
        map_y = (my + flow[:, :, 1] * k).astype(np.float32)
        return self.cv2.remap(
            frame, map_x, map_y,
            interpolation=self.cv2.INTER_LINEAR,
            borderMode=self.cv2.BORDER_REPLICATE,
        ).clip(0, 1)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args: inputs [B, T_in, H, W]. Returns [B, T_out, H, W]."""
        B, T, H, W = inputs.shape
        out    = torch.zeros(B, self.t_out, H, W, device=inputs.device)
        inp_np = inputs.cpu().numpy()
        for b in range(B):
            prev = inp_np[b, -2]
            curr = inp_np[b, -1]
            flow = self._compute_flow(prev, curr)
            for k in range(1, self.t_out + 1):
                out[b, k - 1] = torch.from_numpy(self._warp(curr, flow, k)).to(inputs.device)
        return out
