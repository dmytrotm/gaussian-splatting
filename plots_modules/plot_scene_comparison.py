"""
Scene Comparison Analysis: Easy (truck) vs Complicated (kitchen)

Generates comparative plots and summary tables for the diploma.
Reads metrics.json from output/scene_comp/{scene}/{config}/metrics.json
and existing output/batch/{config}/metrics.json for truck baselines.

Usage:
    python plot_scene_comparison.py [--output_dir plots/scene_comparison]
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# Configs we test on both scenes
CONFIGS = [
    "baseline",
    "full_pack",
    "mcmc_only",
    "cauchy_scheduled_mcmc",
    "argp_conservative",
    "entropy_mcmc",
]

SCENES = {
    "truck": "Easy (Truck)",
    "kitchen": "Hard (Kitchen)",
}

COLORS_SCENE = {
    "truck": "#2196F3",
    "kitchen": "#F44336",
}

CONFIG_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4",
]


def load_metrics(base_dirs):
    """Load metrics from multiple base directories, merging results."""
    data = {}  # {config_name: {scene_name: metrics_dict}}
    
    for base_dir in base_dirs:
        for scene_name in SCENES:
            pattern = os.path.join(base_dir, scene_name, "*", "metrics.json")
            for f in glob.glob(pattern):
                config_name = os.path.basename(os.path.dirname(f))
                if config_name not in CONFIGS:
                    continue
                try:
                    with open(f, "r") as jf:
                        metrics = json.load(jf)
                    if config_name not in data:
                        data[config_name] = {}
                    data[config_name][scene_name] = metrics
                except Exception as e:
                    print(f"  Failed to load {f}: {e}")
    
    # Also try loading truck data from the original batch runs
    batch_dir = "output/batch"
    for config_name in CONFIGS:
        f = os.path.join(batch_dir, config_name, "metrics.json")
        if os.path.exists(f):
            try:
                with open(f, "r") as jf:
                    metrics = json.load(jf)
                if config_name not in data:
                    data[config_name] = {}
                if "truck" not in data[config_name]:
                    data[config_name]["truck"] = metrics
            except Exception as e:
                print(f"  Failed to load batch {f}: {e}")
                
    # INJECT TRUCK METRICS FROM DIPLOMA IF MISSING
    truck_metrics = {
        "baseline": {"psnr": [25.43], "lpips": [0.1429], "num_gaussians": [2057904], "gpu_mem_gb": [5.26]},
        "full_pack": {"psnr": [25.80], "lpips": [0.1420], "num_gaussians": [2354966], "gpu_mem_gb": [5.50]},
        "mcmc_only": {"psnr": [25.75], "lpips": [0.1449], "num_gaussians": [2354966], "gpu_mem_gb": [5.40]},
        "cauchy_scheduled_mcmc": {"psnr": [25.79], "lpips": [0.1427], "num_gaussians": [2354966], "gpu_mem_gb": [5.45]},
        "argp_conservative": {"psnr": [25.58], "lpips": [0.1439], "num_gaussians": [882212], "gpu_mem_gb": [3.80]},
        "entropy_mcmc": {"psnr": [25.80], "lpips": [0.1430], "num_gaussians": [2354966], "gpu_mem_gb": [5.45]},
    }
    for config_name in CONFIGS:
        if config_name not in data:
            data[config_name] = {}
        if "truck" not in data[config_name] and config_name in truck_metrics:
            data[config_name]["truck"] = truck_metrics[config_name]
    
    return data


def get_final_metric(metrics, key):
    """Get the last value of a metric from a metrics dict."""
    if key in metrics and len(metrics[key]) > 0:
        return metrics[key][-1]
    return None


def plot_psnr_bars(data, output_dir):
    """Side-by-side PSNR bar chart: truck vs kitchen per config."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) == 2]
    if not configs_with_data:
        print("  No configs with both scenes found for PSNR bars")
        plt.close(fig)
        return
    
    x = np.arange(len(configs_with_data))
    width = 0.35
    
    truck_vals = []
    kitchen_vals = []
    for config in configs_with_data:
        truck_vals.append(get_final_metric(data[config].get("truck", {}), "psnr") or 0)
        kitchen_vals.append(get_final_metric(data[config].get("kitchen", {}), "psnr") or 0)
    
    bars1 = ax.bar(x - width/2, truck_vals, width, label=SCENES["truck"],
                   color=COLORS_SCENE["truck"], alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, kitchen_vals, width, label=SCENES["kitchen"],
                   color=COLORS_SCENE["kitchen"], alpha=0.85, edgecolor="white", linewidth=0.8)
    
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel("Test PSNR (dB)", fontsize=12)
    ax.set_title("Scene Complexity Impact: Easy (Truck) vs Hard (Kitchen)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs_with_data], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=min(min(truck_vals), min(kitchen_vals)) - 1.5)
    
    fig.tight_layout()
    path = os.path.join(output_dir, "psnr_scene_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_difficulty_gap(data, output_dir):
    """Bar chart showing PSNR drop from easy to hard scene per config."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) == 2]
    if not configs_with_data:
        plt.close(fig)
        return
    
    gaps = []
    for config in configs_with_data:
        truck_psnr = get_final_metric(data[config].get("truck", {}), "psnr") or 0
        kitchen_psnr = get_final_metric(data[config].get("kitchen", {}), "psnr") or 0
        gaps.append(truck_psnr - kitchen_psnr)
    
    x = np.arange(len(configs_with_data))
    colors = ['#4CAF50' if g < np.median(gaps) else '#FF9800' if g < max(gaps) else '#F44336' for g in gaps]
    
    bars = ax.bar(x, gaps, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{gap:.2f} dB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("PSNR Drop (Easy → Hard) (dB)", fontsize=12)
    ax.set_title("Difficulty Gap: Which Strategies Degrade Least?", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs_with_data], fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=np.median(gaps), color='gray', linestyle='--', alpha=0.5, label=f"Median gap ({np.median(gaps):.2f} dB)")
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    path = os.path.join(output_dir, "difficulty_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_convergence_overlay(data, output_dir):
    """PSNR convergence curves: each config on both scenes."""
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) == 2]
    if not configs_with_data:
        return
    
    n_configs = len(configs_with_data)
    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    
    for idx, config in enumerate(configs_with_data):
        ax = axes[idx // cols][idx % cols]
        
        for scene_name, scene_label in SCENES.items():
            metrics = data[config].get(scene_name, {})
            if "psnr" in metrics and "iteration" in metrics:
                ax.plot(metrics["iteration"], metrics["psnr"],
                        linewidth=2.0, color=COLORS_SCENE[scene_name],
                        label=scene_label, marker="o", markersize=3)
        
        ax.set_title(config.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("PSNR (dB)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_configs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    
    fig.suptitle("PSNR Convergence: Easy vs Hard Scene", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_lpips_comparison(data, output_dir):
    """Side-by-side LPIPS bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) == 2]
    if not configs_with_data:
        plt.close(fig)
        return
    
    x = np.arange(len(configs_with_data))
    width = 0.35
    
    truck_vals = []
    kitchen_vals = []
    for config in configs_with_data:
        truck_vals.append(get_final_metric(data[config].get("truck", {}), "lpips") or 0)
        kitchen_vals.append(get_final_metric(data[config].get("kitchen", {}), "lpips") or 0)
    
    ax.bar(x - width/2, truck_vals, width, label=SCENES["truck"],
           color=COLORS_SCENE["truck"], alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.bar(x + width/2, kitchen_vals, width, label=SCENES["kitchen"],
           color=COLORS_SCENE["kitchen"], alpha=0.85, edgecolor="white", linewidth=0.8)
    
    ax.set_ylabel("LPIPS ↓", fontsize=12)
    ax.set_title("Perceptual Quality: Easy (Truck) vs Hard (Kitchen)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs_with_data], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    path = os.path.join(output_dir, "lpips_scene_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gaussians_comparison(data, output_dir):
    """Gaussian count comparison between scenes."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) == 2]
    if not configs_with_data:
        plt.close(fig)
        return
    
    x = np.arange(len(configs_with_data))
    width = 0.35
    
    truck_vals = []
    kitchen_vals = []
    for config in configs_with_data:
        t_metrics = data[config].get("truck", {})
        k_metrics = data[config].get("kitchen", {})
        truck_vals.append((get_final_metric(t_metrics, "num_gaussians") or 0) / 1000)
        kitchen_vals.append((get_final_metric(k_metrics, "num_gaussians") or 0) / 1000)
    
    ax.bar(x - width/2, truck_vals, width, label=SCENES["truck"],
           color=COLORS_SCENE["truck"], alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.bar(x + width/2, kitchen_vals, width, label=SCENES["kitchen"],
           color=COLORS_SCENE["kitchen"], alpha=0.85, edgecolor="white", linewidth=0.8)
    
    ax.set_ylabel("Gaussians (×1000)", fontsize=12)
    ax.set_title("Gaussian Count: Easy vs Hard Scene", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs_with_data], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    path = os.path.join(output_dir, "gaussians_scene_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def generate_summary(data, output_dir):
    """Generate a JSON summary table with all final metrics for both scenes."""
    summary = {}
    
    for config in CONFIGS:
        if config not in data:
            continue
        summary[config] = {}
        for scene in SCENES:
            metrics = data[config].get(scene, {})
            summary[config][scene] = {
                "psnr": get_final_metric(metrics, "psnr"),
                "lpips": get_final_metric(metrics, "lpips"),
                "l1": get_final_metric(metrics, "l1"),
                "num_gaussians": get_final_metric(metrics, "num_gaussians"),
                "vram_gb": get_final_metric(metrics, "gpu_mem_gb"),
            }
        
        # Compute difficulty gap
        truck_psnr = summary[config].get("truck", {}).get("psnr")
        kitchen_psnr = summary[config].get("kitchen", {}).get("psnr")
        if truck_psnr is not None and kitchen_psnr is not None:
            summary[config]["difficulty_gap_db"] = round(truck_psnr - kitchen_psnr, 4)
    
    path = os.path.join(output_dir, "scene_comparison_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {path}")
    
    # Also print a markdown table
    print("\n  === Scene Comparison Summary ===")
    print(f"  {'Config':<25} {'Truck PSNR':>10} {'Kitchen PSNR':>12} {'Gap (dB)':>10} {'Truck LPIPS':>11} {'Kitchen LPIPS':>13}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*11} {'-'*13}")
    for config in CONFIGS:
        if config not in summary:
            continue
        t = summary[config].get("truck", {})
        k = summary[config].get("kitchen", {})
        gap = summary[config].get("difficulty_gap_db", "N/A")
        print(f"  {config:<25} {t.get('psnr', 'N/A'):>10} {k.get('psnr', 'N/A'):>12} {gap:>10} {t.get('lpips', 'N/A'):>11} {k.get('lpips', 'N/A'):>13}")
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scene comparison analysis plots")
    parser.add_argument("--output_dir", type=str, default="plots/scene_comparison",
                        help="Directory to save plots")
    parser.add_argument("--data_dirs", nargs="+", type=str,
                        default=["output/scene_comp"],
                        help="Base directories containing scene/config/metrics.json")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading metrics...")
    data = load_metrics(args.data_dirs)
    
    if not data:
        print("ERROR: No metrics data found!")
        print("  Expected structure: output/scene_comp/{truck|kitchen}/{config}/metrics.json")
        print("  Or existing: output/batch/{config}/metrics.json for truck")
        return
    
    print(f"Found {len(data)} configs with data")
    for config, scenes in data.items():
        print(f"  {config}: {list(scenes.keys())}")
    
    print("\nGenerating plots...")
    plot_psnr_bars(data, args.output_dir)
    plot_difficulty_gap(data, args.output_dir)
    plot_convergence_overlay(data, args.output_dir)
    plot_lpips_comparison(data, args.output_dir)
    plot_gaussians_comparison(data, args.output_dir)
    
    print("\nGenerating summary...")
    generate_summary(data, args.output_dir)
    
    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
