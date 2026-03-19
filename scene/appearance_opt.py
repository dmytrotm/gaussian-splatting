#
# Appearance Optimization Module
# Ported from gsplat reference (complete_trainer.py L106-155)
#
# Learns per-image appearance corrections (exposure, lighting, color shift)
# via a small MLP conditioned on:
#   - Per-image learned embeddings
#   - Gaussian feature vectors
#   - Spherical harmonic bases evaluated at view directions
#
# This allows the model to handle inconsistent lighting / exposure across
# training images (e.g., outdoor captures at different times of day).
#
# Usage:
#   app_module = AppearanceOptModule(n=len(cameras), feature_dim=32).cuda()
#   optimizer = torch.optim.Adam(app_module.parameters(), lr=1e-3)
#
#   # In rendering: add color correction
#   color_correction = app_module(features, image_ids, view_dirs, sh_degree)
#   colors = torch.sigmoid(raw_colors + color_correction)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def eval_sh_bases(num_bases: int, dirs: Tensor) -> Tensor:
    """Evaluate spherical harmonic bases at given directions.

    Pure PyTorch implementation (no gsplat dependency).
    Replaces gsplat's ``_eval_sh_bases_fast``.

    Args:
        num_bases: Number of SH bases to evaluate (must be a perfect square).
        dirs: (..., 3) tensor of unit directions.

    Returns:
        (..., num_bases) tensor of SH basis values.
    """
    result = torch.zeros(*dirs.shape[:-1], num_bases, device=dirs.device, dtype=dirs.dtype)

    # Constants
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]

    # Degree 0
    result[..., 0:1] = C0

    if num_bases > 1:
        # Degree 1
        result[..., 1:2] = -C1 * y
        result[..., 2:3] = C1 * z
        result[..., 3:4] = -C1 * x

    if num_bases > 4:
        # Degree 2
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result[..., 4:5] = C2[0] * xy
        result[..., 5:6] = C2[1] * yz
        result[..., 6:7] = C2[2] * (2.0 * zz - xx - yy)
        result[..., 7:8] = C2[3] * xz
        result[..., 8:9] = C2[4] * (xx - yy)

    if num_bases > 9:
        # Degree 3
        result[..., 9:10] = C3[0] * y * (3 * xx - yy)
        result[..., 10:11] = C3[1] * xy * z
        result[..., 11:12] = C3[2] * y * (4 * zz - xx - yy)
        result[..., 12:13] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        result[..., 13:14] = C3[4] * x * (4 * zz - xx - yy)
        result[..., 14:15] = C3[5] * z * (xx - yy)
        result[..., 15:16] = C3[6] * x * (xx - 3 * yy)

    return result


class AppearanceOptModule(nn.Module):
    """Per-image appearance correction via learned embeddings + MLP.

    Compensates for inconsistent exposure, white balance, and lighting
    across training images by learning a small color correction network
    conditioned on per-image embeddings, Gaussian features, and SH bases.

    Args:
        n: Number of training images.
        feature_dim: Dimension of per-Gaussian feature vectors.
        embed_dim: Dimension of per-image appearance embeddings (default: 16).
        sh_degree: Maximum SH degree for directional encoding (default: 3).
        mlp_width: Width of hidden layers (default: 64).
        mlp_depth: Number of hidden layers (default: 2).
    """

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = nn.Embedding(n, embed_dim)

        # MLP: [embed + features + SH_bases] -> color correction (RGB)
        in_dim = embed_dim + feature_dim + (sh_degree + 1) ** 2
        layers = []
        layers.append(nn.Linear(in_dim, mlp_width))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(mlp_width, mlp_width))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(mlp_width, 3))
        self.color_head = nn.Sequential(*layers)

    def forward(
        self,
        features: Tensor,
        embed_ids: Tensor,
        dirs: Tensor,
        sh_degree: int,
    ) -> Tensor:
        """Compute per-Gaussian color corrections.

        Args:
            features: (N, feature_dim) per-Gaussian feature vectors.
            embed_ids: (C,) integer image indices.
            dirs: (C, N, 3) view directions (Gaussian mean - camera position).
            sh_degree: Current SH degree to use.

        Returns:
            (C, N, 3) color corrections to add before sigmoid.
        """
        C, N = dirs.shape[:2]

        # Per-image embeddings: (C,) -> (C, embed_dim) -> (C, N, embed_dim)
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)
        embeds = embeds[:, None, :].expand(-1, N, -1)

        # Features: (N, D) -> (C, N, D)
        features = features[None, :, :].expand(C, -1, -1)

        # SH bases from view directions
        dirs = F.normalize(dirs, dim=-1)
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)
        sh_bases[:, :, :num_bases_to_use] = eval_sh_bases(num_bases_to_use, dirs)

        # Concatenate and forward through MLP
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)
        else:
            h = torch.cat([features, sh_bases], dim=-1)

        colors = self.color_head(h)
        return colors
