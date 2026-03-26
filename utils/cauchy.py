#
# Cauchy-based activation and loss for Gaussian Splatting
#
# CauchyActivation: Smooth [0,1] mapping via arctan, replacing hard clamp.
#   Preserves gradients for over/under-exposed pixels, improving textures.
#
# cauchy_loss: Robust Lorentzian loss — down-weights outliers, letting the
#   optimizer focus on texture fidelity rather than bright/dark extremes.
#

import torch
import torch.nn as nn
from torch import Tensor
import math


class CauchyActivation(nn.Module):
    """Smooth, learnable [0,1] color activation based on arctan.

    Replaces the hard ``clamp(0, 1)`` in the renderer with:

        φ(x) = 0.5 + (1/π) · arctan((x − μ) / γ)

    where μ (center) and γ (scale) are **per-channel learnable** parameters.

    Properties:
        - Maps ℝ → (0, 1), smooth and monotonic everywhere.
        - Gradients never vanish (unlike clamp which kills grad outside [0,1]).
        - Initialized to approximate the identity on [0, 1]:
            μ = 0.5, γ = 0.15 → steep S-curve centered at 0.5.
        - Heavy-tailed: extreme values get gentle gradient push back
          into range, instead of hard cut-off.

    Args:
        channels: Number of color channels (default: 3 for RGB).
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        # Center of the activation (≈ 0.5 so midpoint maps to 0.5)
        self.mu = nn.Parameter(torch.full((channels, 1, 1), 0.5))
        # Scale (small γ → steep curve ≈ hard clamp; large γ → soft)
        # 0.15 gives a steep-enough curve that (0,1) maps to roughly (0.03, 0.97)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), 0.15))

    def forward(self, x: Tensor) -> Tensor:
        """Apply smooth Cauchy activation.

        Args:
            x: Rendered image tensor, shape (C, H, W).

        Returns:
            Activated image in (0, 1), shape (C, H, W).
        """
        # Ensure γ > 0 for numerical stability
        gamma = torch.clamp(self.gamma, min=1e-4)
        return 0.5 + (1.0 / math.pi) * torch.atan((x - self.mu) / gamma)


def cauchy_loss(pred: Tensor, target: Tensor, scale: float = 0.1) -> Tensor:
    """Cauchy (Lorentzian) robust loss.

    L(Δ) = log(1 + (Δ / scale)²)

    Properties:
        - Small Δ: behaves like L2 (quadratic).
        - Large Δ: grows logarithmically (robust to outliers).
        - ``scale`` controls the transition: smaller scale means
          more aggressive outlier rejection.

    This is beneficial for 3DGS because:
        - Specular highlights and sky regions create large pixel errors.
          L1 treats these equally — Cauchy down-weights them.
        - Texture regions have small, structured errors that matter
          for LPIPS — Cauchy emphasizes these.

    Args:
        pred: Predicted image (C, H, W).
        target: Ground truth image (C, H, W).
        scale: Controls the outlier rejection threshold.

    Returns:
        Scalar mean loss.
    """
    residual = (pred - target) / scale
    return torch.log1p(residual * residual).mean()
