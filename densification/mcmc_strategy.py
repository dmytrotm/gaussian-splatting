#
# MCMC Densification Strategy — Ported from gsplat reference
# MODIFIED: New file, pure-PyTorch port of gsplat MCMCStrategy
#
# Reference: "3D Gaussian Splatting as Markov Chain Monte Carlo"
# (arXiv:2404.09591)
#
# Instead of gradient-threshold heuristics (clone/split), MCMC treats
# Gaussian positions as samples in a Markov chain.  At each refinement step it:
#   1. Teleports "dead" (low-opacity) Gaussians to high-opacity locations.
#   2. Adds new Gaussians sampled proportionally to the opacity distribution.
#   3. Injects small covariance-scaled noise into all positions for exploration.
#

import math
import numpy as np
import torch
from torch import nn, Tensor
from utils.general_utils import build_scaling_rotation
from .base_strategy import DensificationStrategy


# ====================================================================
# Pure-PyTorch replacements for gsplat C++ helpers
# ====================================================================

def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample indices from a categorical distribution defined by *weights*."""
    num_elements = weights.size(0)
    if num_elements <= 2 ** 24:
        return torch.multinomial(weights, n, replacement=replacement)
    # Fallback for very large tensors (torch.multinomial limit)
    weights = weights / weights.sum()
    weights_np = weights.detach().cpu().numpy()
    idxs = np.random.choice(num_elements, size=n, p=weights_np, replace=replacement)
    return torch.from_numpy(idxs).to(weights.device)


def _compute_relocation(opacities: Tensor, scales: Tensor,
                         ratios: Tensor, binoms: Tensor):
    """Pure-PyTorch version of gsplat.relocation.compute_relocation.

    Given a set of Gaussians that will be split into *ratio* copies, compute
    the new opacity and scale so that the integral is preserved.

    Args:
        opacities: (N,) or (N,1) post-sigmoid opacities in [0, 1].
        scales: (N, 3) **activated** (positive) scales.
        ratios: (N,) integer tensor — how many copies each Gaussian spawns.
        binoms: (n_max, n_max) precomputed binomial-coefficient lookup.

    Returns:
        (new_opacities, new_scales) with the same shapes as inputs.
    """
    opacities = opacities.flatten()  # (N,)
    n_max = binoms.shape[0]
    ratios = ratios.long().clamp(min=1, max=n_max - 1)

    # For each Gaussian, the new opacity satisfies:
    #   1 - (1 - new_opa)^ratio = old_opa
    #   => new_opa = 1 - (1 - old_opa)^(1/ratio)
    new_opacities = 1.0 - (1.0 - opacities) ** (1.0 / ratios.float())

    # Scale shrinks by ratio^(1/3) to preserve volume
    new_scales = scales / (ratios.float().unsqueeze(-1) ** (1.0 / 3.0))

    return new_opacities, new_scales


def _build_covariance_from_scaling_rotation_mcmc(scaling: Tensor, rotation: Tensor):
    """Build full 3 × 3 covariance matrices from scale and quaternion tensors.

    This mirrors `quat_scale_to_covar_preci` from gsplat but uses the existing
    `build_scaling_rotation` utility already present in the Inria codebase.

    Args:
        scaling: (N, 3) **activated** (positive) scales.
        rotation: (N, 4) raw quaternions.

    Returns:
        (N, 3, 3) covariance matrices.
    """
    L = build_scaling_rotation(scaling, rotation)  # (N, 3, 3)
    return L @ L.transpose(1, 2)


# ====================================================================
# MCMCStrategy
# ====================================================================

class MCMCStrategy(DensificationStrategy):
    """MCMC densification strategy ported from gsplat.

    Instead of the gradient-based clone/split heuristics used by the original
    Inria implementation, MCMC treats Gaussian positions as samples from a
    Markov chain and controls the population through:

      1. **Relocation** — dead (low-opacity) Gaussians are teleported to
         positions sampled proportionally from the opacity of live Gaussians.
      2. **Addition** — new Gaussians are sampled from the existing opacity
         distribution up to a configurable cap.
      3. **Position noise** — small covariance-scaled perturbations are added
         to all positions every iteration for stochastic exploration.

    Config keys (read from *opt*):
      mcmc_cap_max         (int)   — max number of Gaussians        [1_000_000]
      mcmc_noise_lr        (float) — noise learning-rate multiplier  [5e5]
      mcmc_min_opacity     (float) — opacity threshold for "dead"    [0.005]
    """

    def __init__(self):
        super().__init__()
        # Precompute binomial coefficients (used by _compute_relocation)
        n_max = 51
        self._binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                self._binoms[n, k] = math.comb(n, k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, gaussians, scene, iteration, visibility_filter,
             viewspace_point_tensor, radii, opt, dataset):
        """MCMC densification step: relocate dead → add new."""

        refine_start = getattr(opt, "densify_from_iter", 500)
        refine_stop = getattr(opt, "densify_until_iter", 25_000)
        refine_every = getattr(opt, "densification_interval", 100)

        if iteration < refine_stop and iteration > refine_start and iteration % refine_every == 0:
            cap_max = getattr(opt, "mcmc_cap_max", 1_000_000)
            min_opacity = getattr(opt, "mcmc_min_opacity", 0.005)

            binoms = self._binoms.to(gaussians.get_xyz.device)

            # 1. Teleport dead Gaussians
            self._relocate_gs(gaussians, binoms, min_opacity)

            # 2. Add new Gaussians
            self._add_new_gs(gaussians, binoms, min_opacity, cap_max)

            torch.cuda.empty_cache()

    def post_step(self, gaussians, iteration, opt):
        """Inject covariance-scaled noise into positions (every iteration)."""
        noise_lr = getattr(opt, "mcmc_noise_lr", 5e5)
        # Use the current xyz learning rate as the base scaler
        lr = None
        for pg in gaussians.optimizer.param_groups:
            if pg["name"] == "xyz":
                lr = pg["lr"]
                break
        if lr is None or lr == 0.0:
            return
        self._inject_noise(gaussians, scaler=lr * noise_lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _relocate_gs(self, gaussians, binoms: Tensor, min_opacity: float):
        """Teleport dead Gaussians to positions of alive ones."""
        opacities = gaussians.get_opacity.flatten()  # post-sigmoid
        dead_mask = opacities <= min_opacity
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return

        alive_mask = ~dead_mask
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        probs = opacities[alive_indices]
        sampled_idxs = _multinomial_sample(probs, n_dead, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]

        # Compute new opacity / scale that preserves the integral
        ratios = torch.bincount(sampled_idxs, minlength=gaussians.get_xyz.shape[0])[sampled_idxs] + 1
        new_opacities, new_scales = _compute_relocation(
            opacities[sampled_idxs],
            gaussians.get_scaling[sampled_idxs],
            ratios, binoms,
        )

        eps = torch.finfo(torch.float32).eps
        new_opacities = new_opacities.clamp(min=min_opacity, max=1.0 - eps)

        # Update source Gaussians
        with torch.no_grad():
            gaussians._opacity.data[sampled_idxs] = gaussians.inverse_opacity_activation(
                new_opacities.unsqueeze(-1) if gaussians._opacity.dim() == 2 else new_opacities
            )
            gaussians._scaling.data[sampled_idxs] = gaussians.scaling_inverse_activation(new_scales)

            # Copy source → dead slots
            for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
                getattr(gaussians, attr).data[dead_indices] = getattr(gaussians, attr).data[sampled_idxs]

        # Zero optimizer state for relocated Gaussians
        for group in gaussians.optimizer.param_groups:
            stored = gaussians.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                for key in stored:
                    if key != "step" and isinstance(stored[key], torch.Tensor):
                        stored[key][sampled_idxs] = 0

    @torch.no_grad()
    def _add_new_gs(self, gaussians, binoms: Tensor, min_opacity: float, cap_max: int):
        """Add new Gaussians sampled from the opacity distribution."""
        current_n = gaussians.get_xyz.shape[0]
        n_target = min(cap_max, int(1.05 * current_n))
        n_new = max(0, n_target - current_n)
        if n_new == 0:
            return

        opacities = gaussians.get_opacity.flatten()
        probs = opacities
        sampled_idxs = _multinomial_sample(probs, n_new, replacement=True)

        ratios = torch.bincount(sampled_idxs, minlength=current_n)[sampled_idxs] + 1
        new_opacities, new_scales = _compute_relocation(
            opacities[sampled_idxs],
            gaussians.get_scaling[sampled_idxs],
            ratios, binoms,
        )
        eps = torch.finfo(torch.float32).eps
        new_opacities = new_opacities.clamp(min=min_opacity, max=1.0 - eps)

        # Update source Gaussians in-place
        gaussians._opacity.data[sampled_idxs] = gaussians.inverse_opacity_activation(
            new_opacities.unsqueeze(-1) if gaussians._opacity.dim() == 2 else new_opacities
        )
        gaussians._scaling.data[sampled_idxs] = gaussians.scaling_inverse_activation(new_scales)

        # Build new parameter tensors for the added Gaussians
        new_xyz = gaussians._xyz.data[sampled_idxs]
        new_features_dc = gaussians._features_dc.data[sampled_idxs]
        new_features_rest = gaussians._features_rest.data[sampled_idxs]
        new_opa = gaussians._opacity.data[sampled_idxs]
        new_sc = gaussians._scaling.data[sampled_idxs]
        new_rot = gaussians._rotation.data[sampled_idxs]

        # Use existing GaussianModel bookkeeping to extend tensors + optimizer
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opa,
            "scaling": new_sc,
            "rotation": new_rot,
        }
        optimizable_tensors = gaussians.cat_tensors_to_optimizer(d)
        gaussians._xyz = optimizable_tensors["xyz"]
        gaussians._features_dc = optimizable_tensors["f_dc"]
        gaussians._features_rest = optimizable_tensors["f_rest"]
        gaussians._opacity = optimizable_tensors["opacity"]
        gaussians._scaling = optimizable_tensors["scaling"]
        gaussians._rotation = optimizable_tensors["rotation"]

        # Extend auxiliary buffers
        gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    @torch.no_grad()
    def _inject_noise(self, gaussians, scaler: float):
        """Add covariance-scaled noise to Gaussian positions."""
        opacities = gaussians.get_opacity.flatten()  # post-sigmoid
        scales = gaussians.get_scaling  # post-activation (positive)

        # Build covariance matrices using existing utility
        covars = _build_covariance_from_scaling_rotation_mcmc(scales, gaussians._rotation)

        # Sigmoid gate: strongly suppress noise for high-opacity Gaussians
        def op_sigmoid(x, k=100, x0=0.995):
            return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(gaussians._xyz)
            * op_sigmoid(1.0 - opacities).unsqueeze(-1)
            * scaler
        )
        # Transform noise through covariance
        noise = torch.einsum("bij,bj->bi", covars, noise)
        gaussians._xyz.data.add_(noise)
