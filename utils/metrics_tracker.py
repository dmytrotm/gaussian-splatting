#
# Metrics Tracker — structured PSNR/SSIM/LPIPS recording + plotting
# MODIFIED: New file, ported from gsplat reference metrics + plotting logic
# MODIFIED: Added per-iteration VRAM, timing, live summary, and dashboard plot
#

import json
import os
import time
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MetricsTracker:
    """Track PSNR, SSIM, LPIPS, VRAM, and timing over training iterations.

    Usage::

        tracker = MetricsTracker(output_dir="output/experiment")
        tracker.log_iter_time(iteration=100, elapsed_ms=42.5)
        tracker.log_gpu_memory(iteration=100)
        tracker.update(iteration=1000, psnr=24.5, ssim=0.81, lpips=0.18)
        # ... at end of training:
        tracker.save_json()
        tracker.plot_all()
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._history = defaultdict(list)
        self._start_time = time.time()

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

    def log_iter_time(self, iteration: int, elapsed_ms: float):
        """Record per-iteration wall-clock time in milliseconds."""
        self._history.setdefault("iter_time_ms", [])
        self._history.setdefault("iter_time_iter", [])
        self._history["iter_time_ms"].append(round(elapsed_ms, 2))
        self._history["iter_time_iter"].append(iteration)

    def log_gpu_memory(self, iteration: int):
        """Record current and peak GPU memory usage (in GB) for this iteration."""
        import torch
        mem_current_gb = torch.cuda.memory_allocated() / 1024**3
        mem_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        self._history.setdefault("gpu_mem_gb", [])
        self._history.setdefault("gpu_mem_peak_gb", [])
        self._history.setdefault("gpu_mem_iter", [])
        self._history["gpu_mem_gb"].append(round(mem_current_gb, 3))
        self._history["gpu_mem_peak_gb"].append(round(mem_peak_gb, 3))
        self._history["gpu_mem_iter"].append(iteration)

    def log_num_gaussians(self, iteration: int, count: int):
        """Record the number of active Gaussians."""
        self._history.setdefault("num_gaussians", [])
        self._history.setdefault("num_gaussians_iter", [])
        self._history["num_gaussians"].append(count)
        self._history["num_gaussians_iter"].append(iteration)

    def get_wall_time(self) -> float:
        """Return elapsed wall-clock time in seconds since tracker creation."""
        return time.time() - self._start_time

    def get_latest_vram_gb(self) -> float:
        """Return the latest recorded VRAM usage in GB, or 0 if none."""
        if self._history["gpu_mem_gb"]:
            return self._history["gpu_mem_gb"][-1]
        return 0.0

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
        """Generate individual metric plots, VRAM/timing plots, and a combined dashboard."""
        if not MATPLOTLIB_AVAILABLE:
            print("[MetricsTracker] matplotlib not available — skipping plots.")
            return

        iters = self._history.get("iteration", [])

        # Individual metric plots (PSNR, SSIM, LPIPS)
        if iters:
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

        # VRAM plot
        if self._history.get("gpu_mem_gb"):
            fig, ax = plt.subplots(figsize=(8, 5))
            mem_iters = self._history["gpu_mem_iter"]
            ax.plot(mem_iters, self._history["gpu_mem_gb"],
                    linewidth=1.2, label="Current VRAM", color="#2196F3")
            if self._history.get("gpu_mem_peak_gb"):
                ax.plot(mem_iters, self._history["gpu_mem_peak_gb"],
                        linewidth=1.2, linestyle="--", label="Peak VRAM", color="#F44336", alpha=0.7)
            ax.set_title("GPU VRAM Usage", fontsize=14)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("VRAM (GB)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.output_dir, "metrics_vram.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[MetricsTracker] Saved {path}")

        # Iteration time plot
        if self._history.get("iter_time_ms"):
            fig, ax = plt.subplots(figsize=(8, 5))
            time_iters = self._history["iter_time_iter"]
            ax.plot(time_iters, self._history["iter_time_ms"],
                    linewidth=0.8, alpha=0.4, color="#4CAF50")
            # Add a rolling average for readability
            times = self._history["iter_time_ms"]
            window = min(100, len(times))
            if window > 1:
                import numpy as np
                rolling = np.convolve(times, np.ones(window) / window, mode="valid")
                offset = window // 2
                ax.plot(time_iters[offset:offset + len(rolling)], rolling,
                        linewidth=2, color="#2E7D32", label=f"Rolling avg ({window})")
                ax.legend()
            ax.set_title("Iteration Time", fontsize=14)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Time (ms)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(self.output_dir, "metrics_iter_time.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[MetricsTracker] Saved {path}")

        # Combined dashboard (2×3 grid)
        self._plot_dashboard()

    def _plot_dashboard(self):
        """Generate a 2×3 combined dashboard with all tracked metrics."""
        panels = []
        # Row 1: PSNR, SSIM, LPIPS
        for metric in ("psnr", "ssim", "lpips"):
            if metric in self._history and self._history.get("iteration"):
                panels.append((metric.upper(), self._history["iteration"], self._history[metric]))
        # Row 2: VRAM, Iter Time, Gaussian Count
        if self._history.get("gpu_mem_gb"):
            panels.append(("VRAM (GB)", self._history["gpu_mem_iter"], self._history["gpu_mem_gb"]))
        if self._history.get("iter_time_ms"):
            panels.append(("Iter Time (ms)", self._history["iter_time_iter"], self._history["iter_time_ms"]))
        if self._history.get("num_gaussians"):
            panels.append(("Gaussians", self._history["num_gaussians_iter"], self._history["num_gaussians"]))

        if not panels:
            return

        n = len(panels)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, (title, x, y) in enumerate(panels):
            ax = axes[i]
            ax.plot(x, y, linewidth=1.2, markersize=2, marker="o" if len(x) < 50 else None)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "metrics_dashboard.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[MetricsTracker] Saved {path}")
