import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_data():
    runs_dir = "output/batch"
    metrics_files = glob.glob(os.path.join(runs_dir, "*", "metrics.json"))
    data = {}
    
    for f in metrics_files:
        run_name = os.path.basename(os.path.dirname(f))
        try:
            with open(f, "r") as json_file:
                data[run_name] = json.load(json_file)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return dict(sorted(data.items()))

# Expanded color palette for up to 20 runs
COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", 
    "#00BCD4", "#795548", "#E91E63", "#607D8B", "#FFEB3B",
    "#009688", "#FF5722", "#3F51B5", "#8BC34A", "#673AB7",
    "#CDDC39", "#03A9F4", "#FFC107", "#E91E63", "#455A64"
]

def plot_combined_metric(data_dict, y_key, x_key, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    plotted_any = False
    for i, (run_name, metrics) in enumerate(data_dict.items()):
        if y_key in metrics and x_key in metrics and len(metrics[y_key]) > 0:
            x = metrics[x_key]
            y = metrics[y_key]
            
            # Rolling average for noisy per-iteration metrics
            if y_key == "iter_time_ms":
                window = min(100, len(y))
                if window > 1:
                    rolling = np.convolve(y, np.ones(window) / window, mode="valid")
                    offset = window // 2
                    x_roll = x[offset:offset + len(rolling)]
                    ax.plot(x_roll, rolling, linewidth=2.0, color=COLORS[i % len(COLORS)],
                            label=run_name.replace("_", " ").title())
                else:
                    ax.plot(x, y, linewidth=1.5, color=COLORS[i % len(COLORS)],
                            label=run_name.replace("_", " ").title())
            else:
                use_markers = len(x) < 30
                ax.plot(x, y, linewidth=2.0, color=COLORS[i % len(COLORS)],
                        label=run_name.replace("_", " ").title(),
                        marker="o" if use_markers else None,
                        markersize=4 if use_markers else None)
            plotted_any = True
            
    if not plotted_any:
        plt.close(fig)
        return
        
    ax.legend(fontsize=9, loc="best", ncol=2 if len(data_dict) > 5 else 1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate comparison plots for multiple runs.")
    parser.add_argument("--plot_dir", type=str, default="plots_batch", help="Directory to save plots (default: plots)")
    args = parser.parse_args()

    data = load_data()
    if not data:
        print("No valid metric datasets found.")
        return

    # Create plot directory 
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    print(f"Found {len(data)} runs: {', '.join(data.keys())}")
    print(f"Saving plots to: {args.plot_dir}/")

    # PSNR (test set)
    plot_combined_metric(data, "psnr", "iteration",
                         "Test PSNR over Training", "PSNR (dB)", 
                         os.path.join(args.plot_dir, "compare_psnr.png"))
    
    # LPIPS (test set)
    plot_combined_metric(data, "lpips", "iteration",
                         "Test LPIPS over Training", "LPIPS ↓", 
                         os.path.join(args.plot_dir, "compare_lpips.png"))
    
    # Gaussians Count
    plot_combined_metric(data, "num_gaussians", "num_gaussians_iter",
                         "Gaussian Count", "Number of Gaussians", 
                         os.path.join(args.plot_dir, "compare_gaussians.png"))
    
    # VRAM
    plot_combined_metric(data, "gpu_mem_gb", "gpu_mem_iter",
                         "GPU VRAM Usage", "VRAM (GB)", 
                         os.path.join(args.plot_dir, "compare_vram.png"))
    
    # Iteration Time
    plot_combined_metric(data, "iter_time_ms", "iter_time_iter",
                         "Forward+Backward Iteration Time", "Time (ms)", 
                         os.path.join(args.plot_dir, "compare_iter_time.png"))

if __name__ == "__main__":
    main()
