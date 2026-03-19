#
# Camera Pose Optimization Module
# Ported from gsplat reference (complete_trainer.py L68-103)
#
# Learns per-camera SE(3) pose corrections (translation + rotation deltas)
# during training. Useful when SfM poses are noisy.
#
# Usage:
#   cam_opt = CameraOptModule(n=len(train_cameras)).cuda()
#   cam_opt.zero_init()
#   optimizer = torch.optim.Adam(cam_opt.parameters(), lr=1e-5)
#
#   # In training loop:
#   corrected_c2w = cam_opt(camtoworlds, embed_ids)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks" (CVPR 2019).

    Args:
        d6: (..., 6) tensor of 6D rotation parameters.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOptModule(nn.Module):
    """Learnable per-camera SE(3) pose deltas.

    Each camera gets a 9-dimensional embedding:
    - 3 for translation delta (dx, dy, dz)
    - 6 for rotation delta (6D rotation representation)

    The deltas are applied as a right-multiplication to the original
    camera-to-world transform: ``C2W' = C2W @ delta_SE3``.

    Args:
        n: Number of cameras (training views).
    """

    def __init__(self, n: int):
        super().__init__()
        # 9D per camera: 3 translation + 6 rotation (6D repr)
        self.embeds = nn.Embedding(n, 9)
        # Identity rotation in 6D: first two columns of I_3x3 flattened
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        """Initialize all pose deltas to zero (identity transform)."""
        nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float = 1e-6):
        """Initialize pose deltas with small random noise."""
        nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Apply learned pose corrections to camera-to-world matrices.

        Args:
            camtoworlds: (..., 4, 4) original camera-to-world transforms.
            embed_ids: (...,) integer camera indices.

        Returns:
            (..., 4, 4) corrected camera-to-world transforms.
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]

        pose_deltas = self.embeds(embed_ids)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]

        # Decode rotation: add identity bias, convert 6D -> 3x3
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )

        # Build SE(3) delta transform
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx

        return torch.matmul(camtoworlds, transform)
