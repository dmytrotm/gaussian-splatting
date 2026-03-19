#
# Default Densification Strategy — Original Inria 3DGS Logic
# MODIFIED: New file, logic extracted verbatim from scene/gaussian_model.py
#

import torch
from torch import nn
from utils.general_utils import build_rotation
from .base_strategy import DensificationStrategy


class DefaultStrategy(DensificationStrategy):
    """Original Inria densification strategy using gradient-threshold heuristics.

    This strategy:
      - Accumulates viewspace gradients per Gaussian.
      - Clones small Gaussians with large gradients.
      - Splits large Gaussians with large gradients.
      - Prunes low-opacity and oversized Gaussians.
      - Periodically resets all opacities to a low value.

    All logic is moved verbatim from the original GaussianModel methods.
    """

    # ------------------------------------------------------------------
    # Public interface (called from train.py)
    # ------------------------------------------------------------------

    def step(self, gaussians, scene, iteration, visibility_filter,
             viewspace_point_tensor, radii, opt, dataset):
        """Full densification step — mirrors the original train.py block."""

        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )
            self.add_densification_stats(gaussians, viewspace_point_tensor, visibility_filter)

            if (iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                self.densify_and_prune(
                    gaussians,
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                    radii,
                )

            if (iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)):
                self.reset_opacity(gaussians)

    def post_step(self, gaussians, iteration, opt):
        """No-op for the default strategy."""
        pass

    # ------------------------------------------------------------------
    # Densification helpers (moved from GaussianModel)
    # ------------------------------------------------------------------

    @staticmethod
    def add_densification_stats(gaussians, viewspace_point_tensor, update_filter):
        """Accumulate viewspace-gradient norms for densification decisions."""
        gaussians.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True,
        )
        gaussians.denom[update_filter] += 1

    @staticmethod
    def reset_opacity(gaussians):
        """Reset all opacities to a small value, clearing optimizer momentum."""
        opacities_new = gaussians.inverse_opacity_activation(
            torch.min(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * 0.01)
        )
        optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
        gaussians._opacity = optimizable_tensors["opacity"]

    def densify_and_prune(self, gaussians, max_grad, min_opacity, extent, max_screen_size, radii):
        """Combined clone + split + prune pass."""
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0

        gaussians.tmp_radii = radii
        self.densify_and_clone(gaussians, grads, max_grad, extent)
        self.densify_and_split(gaussians, grads, max_grad, extent)

        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = gaussians.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        gaussians.prune_points(prune_mask)
        gaussians.tmp_radii = None

        torch.cuda.empty_cache()

    @staticmethod
    def densify_and_clone(gaussians, grads, grad_threshold, scene_extent):
        """Clone small Gaussians whose gradients exceed the threshold."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussians.get_scaling, dim=1).values <= gaussians.percent_dense * scene_extent,
        )

        new_xyz = gaussians._xyz[selected_pts_mask]
        new_features_dc = gaussians._features_dc[selected_pts_mask]
        new_features_rest = gaussians._features_rest[selected_pts_mask]
        new_opacities = gaussians._opacity[selected_pts_mask]
        new_scaling = gaussians._scaling[selected_pts_mask]
        new_rotation = gaussians._rotation[selected_pts_mask]
        new_tmp_radii = gaussians.tmp_radii[selected_pts_mask]

        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation, new_tmp_radii,
        )

    @staticmethod
    def densify_and_split(gaussians, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians whose gradients exceed the threshold."""
        n_init_points = gaussians.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussians.get_scaling, dim=1).values > gaussians.percent_dense * scene_extent,
        )

        stds = gaussians.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = gaussians._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = gaussians._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = gaussians.tmp_radii[selected_pts_mask].repeat(N)

        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation, new_tmp_radii,
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
        ))
        gaussians.prune_points(prune_filter)
