import json
import os
import matplotlib.pyplot as plt
import glob

def load_metrics(run_dir):
    events = glob.glob(os.path.join(run_dir, 'events.out.*'))
    if not events:
        return [], []
    
    # Try reading from events for high-res data or metrics.json for final data
    # For simplicity, we'll use a manual extraction from the log or event files
    # But wait, we can just use the metrics.json if it has PSNR vs Iter
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            # We don't have iter-to-metric mapping in metrics.json easily
            # Let's use a simpler approach: extract from TensorBoard events
            pass

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(events[-1])
    ea.Reload()
    if 'train/loss_viewpoint - psnr' in ea.Tags().get('scalars', []):
        sc = ea.Scalars('train/loss_viewpoint - psnr')
        iters = [s.step for s in sc]
        psnrs = [s.value for s in sc]
        return iters, psnrs
    return [], []

def main():
    runs = {
        "Baseline (No Correction)": "output/pose_free/pose_noisy_baseline",
        "Refinement (Our Method)": "output/pose_free/pose_noisy_full_pack"
    }
    
    plt.figure(figsize=(10, 6))
    plt.title("Impact of Camera Pose Refinement on Noisy Initializations", fontsize=14, fontweight='bold')
    
    for label, path in runs.items():
        iters, psnrs = load_metrics(path)
        if iters:
            plt.plot(iters, psnrs, label=label, linewidth=2)
            print(f"Loaded {len(iters)} points for {label}. Final PSNR: {psnrs[-1]:.4f}")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Train PSNR (dB)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = "plots/pose_free/noisy_pose_study.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    main()
