#
# Metrics Tracker — structured PSNR/SSIM/LPIPS recording + plotting
# MODIFIED: New file, ported from gsplat reference metrics + plotting logic
#

import json
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MetricsTracker:
    """Track PSNR, SSIM, LPIPS over training iterations and generate reports.

    Usage::

        tracker = MetricsTracker(output_dir="output/experiment")
        tracker.update(iteration=1000, psnr=24.5, ssim=0.81, lpips=0.18)
        # ... at end of training:
        tracker.save_json()
        tracker.plot_all()
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._history = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def update(self, iteration: int, **kwargs):
        """Append metric values for a given iteration.

        Args:
            iteration: Training step number.
            **kwargs: Metric name/value pairs, e.g. ``psnr=24.5, ssim=0.81``.
        """
        self._history["iteration"].append(iteration)
        for key, value in kwargs.items():
            self._history[key].append(float(value))

    def log_gpu_memory(self, iteration: int):
        """Record peak GPU memory usage (in GB) for this iteration.

        Calls ``torch.cuda.max_memory_allocated()`` and resets the counter.
        """
        import torch
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        self._history.setdefault("gpu_mem_gb", [])
        self._history.setdefault("gpu_mem_iter", [])
        self._history["gpu_mem_gb"].append(round(mem_gb, 3))
        self._history["gpu_mem_iter"].append(iteration)

    def log_num_gaussians(self, iteration: int, count: int):
        """Record the number of active Gaussians."""
        self._history.setdefault("num_gaussians", [])
        self._history.setdefault("num_gaussians_iter", [])
        self._history["num_gaussians"].append(count)
        self._history["num_gaussians_iter"].append(iteration)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(self, filename: str = "metrics.json"):
        """Write all recorded metrics to a JSON file in *output_dir*."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(dict(self._history), f, indent=2)
        print(f"[MetricsTracker] Saved metrics to {path}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self):
        """Generate individual and combined metric plots as PNG files."""
        if not MATPLOTLIB_AVAILABLE:
            print("[MetricsTracker] matplotlib not available — skipping plots.")
            return
        if not self._history["iteration"]:
            print("[MetricsTracker] No metrics recorded — skipping plots.")
            return

        iters = self._history["iteration"]

        # Individual plots
        for metric in ("psnr", "ssim", "lpips"):
            if metric not in self._history:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(iters, self._history[metric], marker="o", markersize=3, linewidth=1.5)
            ax.set_title(metric.upper(), fontsize=14)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.output_dir, f"metrics_{metric}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[MetricsTracker] Saved {path}")

        # Combined plot
        metrics_present = [m for m in ("psnr", "ssim", "lpips") if m in self._history]
        if metrics_present:
            fig, axes = plt.subplots(1, len(metrics_present), figsize=(6 * len(metrics_present), 5))
            if len(metrics_present) == 1:
                axes = [axes]
            for ax, metric in zip(axes, metrics_present):
                ax.plot(iters, self._history[metric], marker="o", markersize=3, linewidth=1.5)
                ax.set_title(metric.upper(), fontsize=14)
                ax.set_xlabel("Iteration")
                ax.set_ylabel(metric.upper())
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.output_dir, "metrics_combined.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[MetricsTracker] Saved {path}")
