#!/usr/bin/env python3
"""
Pose-Free 3DGS — Analysis & Plotting Script

Generates comparison plots and JSON summary for the Pose-Free experiments:
- PSNR bar chart across all configs
- LPIPS bar chart  
- Training timeline (PSNR over iterations)
- JSON summary for diploma integration
"""

import os
import json
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not available — skipping plots")


# Config definitions
CONFIGS = {
    # Config 1-2: existing baselines (different output paths)
    "baseline": {
        "path": "output/scene_comp/baseline",
        "label": "Baseline\n(SfM + COLMAP)",
        "color": "#64748b",
        "category": "reference",
    },
    "random_100k": {
        "path": "output/random_init/baseline_random_100k",
        "label": "Random 100k\n(COLMAP poses)",
        "color": "#94a3b8",
        "category": "reference",
    },
    # Config 3-6: core pose-free
    "pose_free_full_pack": {
        "path": "output/pose_free/pose_free_full_pack",
        "label": "Pose-Free\nfull_pack",
        "color": "#3b82f6",
        "category": "pose_free",
    },
    "pose_free_cauchy_sched_mcmc": {
        "path": "output/pose_free/pose_free_cauchy_sched_mcmc",
        "label": "Pose-Free\ncauchy_sched",
        "color": "#6366f1",
        "category": "pose_free",
    },
    "pose_free_mcmc_only": {
        "path": "output/pose_free/pose_free_mcmc_only",
        "label": "Pose-Free\nmcmc_only",
        "color": "#8b5cf6",
        "category": "pose_free",
    },
    "pose_free_baseline_adc": {
        "path": "output/pose_free/pose_free_baseline_adc",
        "label": "Pose-Free\nADC baseline",
        "color": "#a78bfa",
        "category": "pose_free",
    },
    # Config 7-8: noisy pose recovery
    "pose_noisy_full_pack": {
        "path": "output/pose_free/pose_noisy_full_pack",
        "label": "Noisy Pose\nfull_pack",
        "color": "#f59e0b",
        "category": "noisy",
    },
    "pose_noisy_cauchy_sched": {
        "path": "output/pose_free/pose_noisy_cauchy_sched",
        "label": "Noisy Pose\ncauchy_sched",
        "color": "#f97316",
        "category": "noisy",
    },
}


def load_metrics(config_path):
    """Load metrics.json from an experiment directory."""
    metrics_path = os.path.join(config_path, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def extract_final_metrics(metrics):
    """Extract PSNR, SSIM, LPIPS from metrics JSON."""
    if metrics is None:
        return None
    
    result = {}
    
    # Check for test_psnr entries
    if 'test_psnr' in metrics:
        test_data = metrics['test_psnr']
        if isinstance(test_data, dict):
            # Find the last iteration
            last_iter = max(test_data.keys(), key=int)
            result['psnr'] = test_data[last_iter]
        elif isinstance(test_data, list) and len(test_data) > 0:
            result['psnr'] = test_data[-1].get('value', test_data[-1]) if isinstance(test_data[-1], dict) else test_data[-1]
    
    if 'test_ssim' in metrics:
        test_data = metrics['test_ssim']
        if isinstance(test_data, dict):
            last_iter = max(test_data.keys(), key=int)
            result['ssim'] = test_data[last_iter]
    
    if 'test_lpips' in metrics:
        test_data = metrics['test_lpips']
        if isinstance(test_data, dict):
            last_iter = max(test_data.keys(), key=int)
            result['lpips'] = test_data[last_iter]
    
    if 'num_gaussians' in metrics:
        ng = metrics['num_gaussians']
        if isinstance(ng, dict):
            last_iter = max(ng.keys(), key=int)
            result['num_gaussians'] = ng[last_iter]
    
    return result if result else None


def plot_psnr_comparison(results, output_dir):
    """Create PSNR bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    names = []
    values = []
    colors = []
    
    for config_name, config in CONFIGS.items():
        if config_name in results and results[config_name] and 'psnr' in results[config_name]:
            names.append(config['label'])
            values.append(results[config_name]['psnr'])
            colors.append(config['color'])
    
    if not values:
        print("[WARNING] No PSNR data available for plotting")
        return
    
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, ha='center')
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Pose-Free 3DGS: PSNR Comparison\n(Truck Scene, 30k Iterations)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'pose_free_psnr_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_lpips_comparison(results, output_dir):
    """Create LPIPS bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    names = []
    values = []
    colors = []
    
    for config_name, config in CONFIGS.items():
        if config_name in results and results[config_name] and 'lpips' in results[config_name]:
            names.append(config['label'])
            values.append(results[config_name]['lpips'])
            colors.append(config['color'])
    
    if not values:
        print("[WARNING] No LPIPS data available for plotting")
        return
    
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='white', linewidth=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, ha='center')
    ax.set_ylabel('LPIPS ↓', fontsize=12)
    ax.set_title('Pose-Free 3DGS: LPIPS Comparison\n(Truck Scene, 30k Iterations)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'pose_free_lpips_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_convergence(output_dir):
    """Plot PSNR convergence over iterations for all configs."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    has_data = False
    for config_name, config in CONFIGS.items():
        metrics = load_metrics(config['path'])
        if metrics is None or 'test_psnr' not in metrics:
            continue
        
        test_data = metrics['test_psnr']
        if isinstance(test_data, dict):
            iters = sorted([int(k) for k in test_data.keys()])
            psnrs = [test_data[str(i)] for i in iters]
            ax.plot(iters, psnrs, 'o-', label=config_name, color=config['color'],
                    linewidth=2, markersize=5)
            has_data = True
    
    if not has_data:
        print("[WARNING] No convergence data available")
        plt.close()
        return
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Pose-Free 3DGS: PSNR Convergence\n(Truck Scene)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'pose_free_convergence.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def main():
    output_dir = "plots/pose_free"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Pose-Free 3DGS — Analysis")
    print("=" * 60)
    
    # Load all results
    results = {}
    for config_name, config in CONFIGS.items():
        metrics = load_metrics(config['path'])
        final = extract_final_metrics(metrics)
        results[config_name] = final
        
        if final:
            psnr_str = f"PSNR={final.get('psnr', 'N/A'):.2f}" if 'psnr' in final else "PSNR=N/A"
            lpips_str = f"LPIPS={final.get('lpips', 'N/A'):.4f}" if 'lpips' in final else "LPIPS=N/A"
            print(f"  {config_name:35s}: {psnr_str}, {lpips_str}")
        else:
            print(f"  {config_name:35s}: [not found]")
    
    # Generate plots
    print()
    plot_psnr_comparison(results, output_dir)
    plot_lpips_comparison(results, output_dir)
    plot_convergence(output_dir)
    
    # Save JSON summary
    summary = {
        "experiment": "pose_free_3dgs",
        "scene": "truck",
        "dataset": "tanks_and_temples",
        "iterations": 30000,
        "configs": {}
    }
    for config_name, config in CONFIGS.items():
        entry = {
            "category": config['category'],
            "label": config['label'].replace('\n', ' '),
        }
        if results.get(config_name):
            entry.update(results[config_name])
        summary["configs"][config_name] = entry
    
    json_path = os.path.join(output_dir, "pose_free_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary] Saved: {json_path}")


if __name__ == "__main__":
    main()
