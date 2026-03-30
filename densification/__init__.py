#
# Densification Strategy — Factory and public API
# MODIFIED: New file
#

from .base_strategy import DensificationStrategy
from .default_strategy import DefaultStrategy
from .mcmc_strategy import MCMCStrategy
from .argp_strategy import ARGPStrategy


def get_strategy(name: str) -> DensificationStrategy:
    """Instantiate a densification strategy by name.

    Args:
        name: One of ``'default'``, ``'mcmc'``, or ``'argp'``.

    Returns:
        A ready-to-use :class:`DensificationStrategy` instance.

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name == "default":
        return DefaultStrategy()
    elif name == "mcmc":
        return MCMCStrategy()
    elif name == "argp":
        return ARGPStrategy()
    raise ValueError(
        f"Unknown densification strategy: '{name}'. "
        f"Choose from: 'default', 'mcmc', 'argp'."
    )

