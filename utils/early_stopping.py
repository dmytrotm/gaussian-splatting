#
# Early Stopping — PSNR-based early termination for Gaussian Splatting training
#

import os
import json


class EarlyStopping:
    """Monitor test PSNR and stop training when it plateaus.

    Args:
        patience: Number of evaluation rounds without improvement before stopping.
        min_delta: Minimum PSNR improvement (in dB) to count as progress.
        output_dir: Directory to save the best checkpoint info.

    Usage::

        es = EarlyStopping(patience=3, min_delta=0.01, output_dir="output/exp")
        # In the evaluation loop:
        if es.check(iteration, test_psnr):
            print("Early stopping triggered!")
            break
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.01, output_dir: str = "."):
        self.patience = patience
        self.min_delta = min_delta
        self.output_dir = output_dir
        self.best_psnr = float("-inf")
        self.best_iteration = 0
        self.rounds_without_improvement = 0
        self.enabled = patience > 0

    def check(self, iteration: int, psnr: float) -> bool:
        """Check whether training should stop.

        Args:
            iteration: Current training iteration.
            psnr: Test PSNR value at this evaluation point.

        Returns:
            True if training should stop, False otherwise.
        """
        if not self.enabled:
            return False

        if psnr > self.best_psnr + self.min_delta:
            # Improvement found
            prev_best = self.best_psnr
            self.best_psnr = psnr
            self.best_iteration = iteration
            self.rounds_without_improvement = 0
            print(f"\n[EarlyStopping] PSNR improved: {prev_best:.4f} → {psnr:.4f} "
                  f"(Δ={psnr - prev_best:+.4f} dB) at iter {iteration}")
            return False
        else:
            self.rounds_without_improvement += 1
            remaining = self.patience - self.rounds_without_improvement
            print(f"\n[EarlyStopping] No improvement at iter {iteration} "
                  f"(PSNR={psnr:.4f}, best={self.best_psnr:.4f}). "
                  f"Patience: {remaining}/{self.patience} remaining")

            if self.rounds_without_improvement >= self.patience:
                print(f"\n[EarlyStopping] ⛔ Stopping training! "
                      f"No improvement for {self.patience} evaluation rounds. "
                      f"Best PSNR: {self.best_psnr:.4f} at iteration {self.best_iteration}")
                self._save_summary()
                return True
            return False

    def _save_summary(self):
        """Save early stopping summary to a JSON file."""
        summary = {
            "stopped_early": True,
            "best_psnr": round(self.best_psnr, 6),
            "best_iteration": self.best_iteration,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "rounds_without_improvement": self.rounds_without_improvement,
        }
        path = os.path.join(self.output_dir, "early_stopping.json")
        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[EarlyStopping] Saved summary to {path}")
        except Exception as e:
            print(f"[EarlyStopping] Warning: could not save summary: {e}")
