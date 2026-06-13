"""Spatial and intensity augmentation for radar sequences.

All spatial transforms are applied jointly to inputs and targets so that
the spatiotemporal relationship is preserved.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch


class RadarSequenceAugment:
    """Augmentation for [T_in, H, W] / [T_out, H, W] sequence pairs.

    Args:
        p_flip_h:        Probability of horizontal flip (default 0.5).
        p_flip_v:        Probability of vertical flip (default 0.5).
        p_rot90:         Probability of 90°/180°/270° rotation (default 0.5).
        p_noise:         Probability of additive Gaussian noise on inputs (default 0.3).
        noise_std:       Noise std in normalised units (default 1/70 ≈ 0.014).
        p_intensity:     Probability of global intensity shift (default 0.3).
        intensity_range: Shift range in normalised units (±2/70 ≈ ±0.029).
    """

    def __init__(
        self,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        p_rot90: float = 0.5,
        p_noise: float = 0.3,
        noise_std: float = 1.0 / 70.0,
        p_intensity: float = 0.3,
        intensity_range: float = 2.0 / 70.0,
    ) -> None:
        self.p_flip_h       = p_flip_h
        self.p_flip_v       = p_flip_v
        self.p_rot90        = p_rot90
        self.p_noise        = p_noise
        self.noise_std      = noise_std
        self.p_intensity    = p_intensity
        self.intensity_range = intensity_range

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation.

        Args:
            inputs:  [T_in, H, W]
            targets: [T_out, H, W]

        Returns:
            Augmented (inputs, targets).
        """
        all_frames = torch.cat([inputs, targets], dim=0)

        if random.random() < self.p_flip_h:
            all_frames = torch.flip(all_frames, dims=[-1])
        if random.random() < self.p_flip_v:
            all_frames = torch.flip(all_frames, dims=[-2])
        if random.random() < self.p_rot90:
            k = random.choice([1, 2, 3])
            all_frames = torch.rot90(all_frames, k=k, dims=[-2, -1])

        t_in   = inputs.shape[0]
        inputs  = all_frames[:t_in]
        targets = all_frames[t_in:]

        if random.random() < self.p_noise:
            inputs = (inputs + torch.randn_like(inputs) * self.noise_std).clamp(0.0, 1.0)

        if random.random() < self.p_intensity:
            shift   = random.uniform(-self.intensity_range, self.intensity_range)
            inputs  = (inputs  + shift).clamp(0.0, 1.0)
            targets = (targets + shift).clamp(0.0, 1.0)

        return inputs, targets
