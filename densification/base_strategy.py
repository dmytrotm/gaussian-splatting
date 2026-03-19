#
# Densification Strategy — Abstract Base Class
# MODIFIED: New file, extracted from scene/gaussian_model.py
#

from abc import ABC, abstractmethod


class DensificationStrategy(ABC):
    """Abstract base class for Gaussian densification strategies.

    Subclasses implement different approaches to adaptively controlling the
    number and placement of 3D Gaussians during training:
      - DefaultStrategy: Original Inria gradient-threshold heuristics
      - MCMCStrategy: Markov Chain Monte Carlo resampling (ported from gsplat)
    """

    @abstractmethod
    def step(self, gaussians, scene, iteration, visibility_filter,
             viewspace_point_tensor, radii, opt, dataset):
        """Execute one densification step (accumulate stats, densify/prune, reset opacity).

        Args:
            gaussians: GaussianModel instance.
            scene: Scene instance (provides cameras_extent).
            iteration: Current training iteration.
            visibility_filter: Boolean mask of visible Gaussians.
            viewspace_point_tensor: Screen-space point positions with gradients.
            radii: Per-Gaussian screen-space radii.
            opt: OptimizationParams (densify_from_iter, densify_until_iter, etc.).
            dataset: ModelParams (white_background flag, etc.).
        """
        ...

    @abstractmethod
    def post_step(self, gaussians, iteration, opt):
        """Optional hook called after the optimizer step each iteration.

        Used by MCMCStrategy to inject noise into positions; no-op for default.

        Args:
            gaussians: GaussianModel instance.
            iteration: Current training iteration.
            opt: OptimizationParams.
        """
        ...
