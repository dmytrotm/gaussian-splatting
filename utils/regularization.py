#
# Regularization losses — modular loss terms for Gaussian Splatting
# Ported from gsplat reference (complete_trainer.py L1222-1229)
#

import torch
from torch import Tensor


def opacity_reg_loss(gaussians) -> Tensor:
    """Penalize Gaussians that are alive but nearly transparent.

    Encourages Gaussians to either become fully opaque (useful) or fully
    transparent (can be pruned), reducing wasted memory.

    Reference: gsplat ``cfg.opacity_reg * |sigmoid(opacities)|.mean()``

    Args:
        gaussians: GaussianModel instance with ``_opacity`` attribute (raw logits).

    Returns:
        Scalar loss tensor.
    """
    return torch.abs(torch.sigmoid(gaussians._opacity)).mean()


def scale_reg_loss(gaussians) -> Tensor:
    """Penalize Gaussians with excessively large scale.

    Prevents individual Gaussians from growing to cover large portions
    of the scene, which wastes rasterization bandwidth.

    Reference: gsplat ``cfg.scale_reg * |exp(scales)|.mean()``

    Args:
        gaussians: GaussianModel instance with ``_scaling`` attribute (log-space).

    Returns:
        Scalar loss tensor.
    """
    return torch.abs(torch.exp(gaussians._scaling)).mean()
