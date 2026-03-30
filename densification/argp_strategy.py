#
# ARGP Densification Strategy — Adaptive and Recoverable Gaussian Pruning
# Reference: "Adaptive and Recoverable 3D Gaussian Splatting Pruning
#             for Efficient Real-Time Scene Reconstruction"
#             (IEEE TCSVT 2026, Sinyo-Liu et al.)
#
# Two-phase approach:
#   Phase 1 (AOP): During densification — quantile-adaptive opacity pruning
#   Phase 2 (IRP): After densification — iterative freeze/score/recover cycles
#

import torch
from torch import nn
from tqdm import tqdm
from utils.general_utils import build_rotation
from .base_strategy import DensificationStrategy


class ARGPStrategy(DensificationStrategy):
    """ARGP: Adaptive and Recoverable Gaussian Splatting Pruning.

    Phase 1 — Adaptive Opacity Pruning (AOP):
      Same clone/split logic as DefaultStrategy, but replaces the fixed
      opacity threshold (0.005) with a quantile-based threshold that adapts
      to the evolving opacity distribution.

    Phase 2 — Iterative Recovery Pruning (IRP):
      After densification ends, at each opacity_reset_interval:
        1. Freeze bottom tp_prune_level of Gaussians by opacity
        2. Compute gradient-informed importance scores across all views
        3. Recover top recover_level of frozen Gaussians
        4. Permanently delete the rest

    Config keys (read from *opt*):
      ctprune_ratio    (float) — AOP quantile for pruning during densification [0.01]
      tp_prune_level   (float) — IRP fraction to freeze by opacity             [0.7]
      recover_level    (float) — IRP fraction of frozen to recover             [0.4]
    """

    def __init__(self):
        super().__init__()
        self._irp_round = 0  # Tracks which IRP cycle we're on

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, gaussians, scene, iteration, visibility_filter,
             viewspace_point_tensor, radii, opt, dataset):
        """Full ARGP densification step — AOP during densification, IRP after."""

        ctprune_ratio = getattr(opt, "ctprune_ratio", 0.01)
        tp_prune_level = getattr(opt, "tp_prune_level", 0.7)
        recover_level = getattr(opt, "recover_level", 0.4)

        if iteration < opt.densify_until_iter:
            # === Phase 1: AOP (Adaptive Opacity Pruning) ===
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )
            self.add_densification_stats(gaussians, viewspace_point_tensor, visibility_filter)

            if (iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                # AOP: quantile-adaptive pruning instead of fixed 0.005
                self.densify_and_prune(
                    gaussians,
                    opt.densify_grad_threshold,
                    ctprune_ratio,  # quantile ratio, not fixed threshold
                    scene.cameras_extent,
                    size_threshold,
                    radii,
                )

            if (iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)):
                self.reset_opacity(gaussians)

        else:
            # === Phase 2: IRP (Iterative Recovery Pruning) ===
            if (iteration % opt.opacity_reset_interval == 0
                    and iteration >= opt.densify_until_iter
                    and iteration < opt.iterations):

                self._irp_round += 1

                if iteration == opt.densify_until_iter:
                    # First IRP cycle: just freeze, no scoring yet
                    print(f"\n[ARGP IRP #{self._irp_round}] Initial freeze "
                          f"(tp_level={tp_prune_level})")
                    gaussians.temp_mask = gaussians.temp_prune(tp_prune_level)
                    n_frozen = gaussians.temp_mask.sum().item()
                    n_total = gaussians.temp_mask.shape[0]
                    print(f"[ARGP IRP #{self._irp_round}] Frozen {n_frozen:,} / "
                          f"{n_total:,} ({100*n_frozen/n_total:.1f}%)")

                elif iteration == (opt.iterations - opt.opacity_reset_interval):
                    # Final IRP cycle: score + recover only (no new freeze)
                    print(f"\n[ARGP IRP #{self._irp_round}] Final recovery pass")
                    from gaussian_renderer import render
                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    pipe = type('Pipe', (), {'debug': False, 'antialiasing': False,
                                             'convert_SHs_python': False,
                                             'compute_cov3D_python': False})()
                    importance = self._compute_importance_scores(
                        scene, gaussians, pipe, background)
                    recover_mask = gaussians.recover_delete(recover_level, importance)
                    n_kept = recover_mask.sum().item()
                    n_total_before = recover_mask.shape[0]
                    print(f"[ARGP IRP #{self._irp_round}] Final: kept {n_kept:,} / "
                          f"{n_total_before:,}, deleted {n_total_before - n_kept:,}")

                else:
                    # Middle IRP cycles: freeze → score → recover
                    print(f"\n[ARGP IRP #{self._irp_round}] Freeze + Score + Recover")

                    # New freeze pass
                    frozen_mask = gaussians.temp_prune(tp_prune_level)

                    # Score all Gaussians
                    from gaussian_renderer import render
                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    pipe = type('Pipe', (), {'debug': False, 'antialiasing': False,
                                             'convert_SHs_python': False,
                                             'compute_cov3D_python': False})()
                    importance = self._compute_importance_scores(
                        scene, gaussians, pipe, background)

                    # Recover important frozen Gaussians
                    recover_mask = gaussians.recover_delete(recover_level, importance)

                    # Merge freeze decisions: combine recovered + newly frozen
                    rec_grad_mask = ~gaussians.temp_mask != recover_mask
                    rec_grad_mask = rec_grad_mask[recover_mask]
                    diff_mask = ~frozen_mask[recover_mask]
                    temp_grad_mask = rec_grad_mask | diff_mask
                    gaussians.temp_mask = ~temp_grad_mask

                    n_active = (~gaussians.temp_mask).sum().item()
                    n_total = gaussians.temp_mask.shape[0]
                    print(f"[ARGP IRP #{self._irp_round}] Active: {n_active:,} / "
                          f"{n_total:,}")

                print(f"[ARGP IRP #{self._irp_round}] Current Gaussians: "
                      f"{gaussians.get_xyz.shape[0]:,}")

    def post_step(self, gaussians, iteration, opt):
        """No-op for ARGP — all work happens in step()."""
        pass

    # ------------------------------------------------------------------
    # Importance scoring (equivalent to paper's score_func + get_scores)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_func_view(view, gaussians, pipe, background):
        """Compute per-Gaussian importance for one view via opacity gradients.

        Instead of the paper's modified rasterizer with explicit scores tensor,
        we compute equivalent importance by:
        1. Enabling grad on opacity
        2. Rendering the view
        3. Using image.sum().backward() to get gradient of total intensity w.r.t. opacity
        4. The magnitude of opacity.grad gives per-Gaussian contribution

        This is mathematically equivalent to the paper's approach since both
        measure how much each Gaussian contributes to the rendered image.
        """
        # Save current grad state and enable
        opacity_param = gaussians._opacity
        had_grad = opacity_param.requires_grad
        opacity_param.requires_grad_(True)

        if opacity_param.grad is not None:
            opacity_param.grad.zero_()

        from gaussian_renderer import render as render_fn
        render_pkg = render_fn(view, gaussians, pipe, background,
                               separate_sh=False)
        image = render_pkg["render"]

        # Backward to get per-Gaussian contribution
        image.sum().backward()

        # Collect gradient magnitudes as importance scores
        scores = torch.zeros_like(gaussians.get_opacity)
        if opacity_param.grad is not None:
            scores = opacity_param.grad.abs().detach()

        # Restore state
        opacity_param.requires_grad_(had_grad)
        if opacity_param.grad is not None:
            opacity_param.grad.zero_()

        return scores

    @staticmethod
    def _compute_importance_scores(scene, gaussians, pipe, background):
        """Compute importance scores across all training views.

        Returns:
            (N, 1) tensor of per-Gaussian importance scores.
        """
        torch.cuda.reset_peak_memory_stats()
        scores = torch.zeros_like(gaussians.get_opacity)

        with torch.enable_grad():
            views = scene.getTrainCameras()
            pbar = tqdm(total=len(views), desc='[ARGP] Computing importance scores')
            for view in views:
                view_scores = ARGPStrategy._score_func_view(
                    view, gaussians, pipe, background)
                scores += view_scores
                pbar.update(1)
            pbar.close()

        return scores

    # ------------------------------------------------------------------
    # AOP: Densification helpers (adapted from DefaultStrategy)
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
        """Reset all opacities to a small value."""
        opacities_new = gaussians.inverse_opacity_activation(
            torch.min(gaussians.get_opacity,
                       torch.ones_like(gaussians.get_opacity) * 0.01)
        )
        optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
        gaussians._opacity = optimizable_tensors["opacity"]

    def densify_and_prune(self, gaussians, max_grad, ctprune_ratio, extent,
                          max_screen_size, radii):
        """AOP: Combined clone + split + quantile-adaptive pruning."""
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        grads[grads.isnan()] = 0.0

        gaussians.tmp_radii = radii
        self.densify_and_clone(gaussians, grads, max_grad, extent)
        self.densify_and_split(gaussians, grads, max_grad, extent)

        # AOP: Quantile-based pruning (paper's key difference from default)
        all_opa = gaussians.get_opacity
        opacity_level = torch.quantile(all_opa, ctprune_ratio)
        opa_mask = (all_opa < opacity_level).squeeze()
        prune_mask = opa_mask

        if max_screen_size:
            big_points_vs = gaussians.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        gaussians.prune_points(prune_mask)
        gaussians.tmp_radii = None

        torch.cuda.empty_cache()

    @staticmethod
    def densify_and_clone(gaussians, grads, grad_threshold, scene_extent):
        """Clone small Gaussians whose gradients exceed the threshold."""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False)
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
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                   + gaussians.get_xyz[selected_pts_mask].repeat(N, 1))
        new_scaling = gaussians.scaling_inverse_activation(
            gaussians.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
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
