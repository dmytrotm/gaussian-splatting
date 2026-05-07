"""
Random Initialization Analysis: SfM vs Random Init at different point budgets

Compares the best optimization strategies with SfM initialization (existing results)
versus random point cloud initialization at 50k, 100k, 200k points.

Reads metrics from:
  - output/batch/{config}/metrics.json            (SfM baseline)
  - output/random_init/{config}_random_{N}k/metrics.json  (random init variants)

Usage:
    python plot_random_init.py [--output_dir plots/random_init]
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configs to test with random init
CONFIGS = [
    "baseline",
    "mcmc_only",
    "full_pack",
    "argp_conservative",
]

POINT_BUDGETS = [50, 100, 200]  # in thousands

COLORS_INIT = {
    "sfm": "#2196F3",
    "50k": "#FF9800",
    "100k": "#F44336",
    "200k": "#9C27B0",
}

INIT_LABELS = {
    "sfm": "SfM (COLMAP)",
    "50k": "Random 50K",
    "100k": "Random 100K",
    "200k": "Random 200K",
}

CONFIG_COLORS = {
    "baseline": "#2196F3",
    "mcmc_only": "#4CAF50",
    "full_pack": "#FF9800",
    "argp_conservative": "#9C27B0",
}


def load_metrics():
    """Load all metrics: SfM baselines + random init variants."""
    data = {}  # {config: {init_type: metrics}}
    
    # Load SfM baselines from batch runs
    for config in CONFIGS:
        f = os.path.join("output", "batch", config, "metrics.json")
        if os.path.exists(f):
            try:
                with open(f, "r") as jf:
                    metrics = json.load(jf)
                if config not in data:
                    data[config] = {}
                data[config]["sfm"] = metrics
            except Exception as e:
                print(f"  Failed to load SfM {f}: {e}")
    
    # Load random init variants
    random_dir = "output/random_init"
    for config in CONFIGS:
        for budget in POINT_BUDGETS:
            run_name = f"{config}_random_{budget}k"
            f = os.path.join(random_dir, run_name, "metrics.json")
            if os.path.exists(f):
                try:
                    with open(f, "r") as jf:
                        metrics = json.load(jf)
                    if config not in data:
                        data[config] = {}
                    data[config][f"{budget}k"] = metrics
                except Exception as e:
                    print(f"  Failed to load {f}: {e}")
    
    return data


def get_final_metric(metrics, key):
    if key in metrics and len(metrics[key]) > 0:
        return metrics[key][-1]
    return None


def get_convergence_iter(metrics, target_fraction=0.9):
    """Find iteration at which PSNR reaches target_fraction of final value."""
    if "psnr" not in metrics or "iteration" not in metrics:
        return None
    psnr_vals = metrics["psnr"]
    iters = metrics["iteration"]
    if not psnr_vals:
        return None
    
    final_psnr = psnr_vals[-1]
    target = final_psnr * target_fraction
    
    for i, p in enumerate(psnr_vals):
        if p >= target:
            return iters[i]
    return iters[-1]  # never reached


def plot_convergence_subplots(data, output_dir):
    """PSNR convergence: SfM vs random init variants, one subplot per config."""
    configs_with_data = [c for c in CONFIGS if c in data and len(data[c]) > 1]
    if not configs_with_data:
        print("  No configs with both SfM and random data found")
        return
    
    n = len(configs_with_data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    
    for idx, config in enumerate(configs_with_data):
        ax = axes[idx // cols][idx % cols]
        
        for init_type in ["sfm"] + [f"{b}k" for b in POINT_BUDGETS]:
            if init_type not in data[config]:
                continue
            metrics = data[config][init_type]
            if "psnr" in metrics and "iteration" in metrics:
                ax.plot(metrics["iteration"], metrics["psnr"],
                        linewidth=2.5 if init_type == "sfm" else 1.8,
                        color=COLORS_INIT[init_type],
                        label=INIT_LABELS[init_type],
                        linestyle="-" if init_type == "sfm" else "--",
                        marker="o" if len(metrics["iteration"]) < 20 else None,
                        markersize=4)
        
        ax.set_title(config.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    
    fig.suptitle("SfM vs Random Initialization: PSNR Convergence", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_sfm_vs_random.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_point_budget_sensitivity(data, output_dir):
    """Grouped bar chart: final PSNR for each config × point budget."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    configs_with_data = [c for c in CONFIGS if c in data]
    if not configs_with_data:
        plt.close(fig)
        return
    
    init_types = ["sfm"] + [f"{b}k" for b in POINT_BUDGETS]
    n_inits = len(init_types)
    x = np.arange(len(configs_with_data))
    width = 0.8 / n_inits
    
    for i, init_type in enumerate(init_types):
        vals = []
        for config in configs_with_data:
            metrics = data.get(config, {}).get(init_type, {})
            vals.append(get_final_metric(metrics, "psnr") or 0)
        
        offset = (i - n_inits / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=INIT_LABELS[init_type],
                      color=COLORS_INIT[init_type], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{bar.get_height():.1f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
    
    ax.set_ylabel("Final Test PSNR (dB)", fontsize=12)
    ax.set_title("Point Budget Sensitivity: SfM vs Random Initialization", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs_with_data], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    min_val = min(
        get_final_metric(data.get(c, {}).get(it, {}), "psnr") or 100
        for c in configs_with_data for it in init_types
        if get_final_metric(data.get(c, {}).get(it, {}), "psnr") is not None
    )
    ax.set_ylim(bottom=max(0, min_val - 2))
    
    fig.tight_layout()
    path = os.path.join(output_dir, "point_budget_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sfm_gap(data, output_dir):
    """Show how close each random init combo gets to SfM quality."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    entries = []
    for config in CONFIGS:
        if config not in data or "sfm" not in data[config]:
            continue
        sfm_psnr = get_final_metric(data[config]["sfm"], "psnr")
        if sfm_psnr is None:
            continue
        
        for budget in POINT_BUDGETS:
            key = f"{budget}k"
            if key not in data[config]:
                continue
            rand_psnr = get_final_metric(data[config][key], "psnr")
            if rand_psnr is not None:
                gap = sfm_psnr - rand_psnr
                entries.append({
                    "label": f"{config}\n({budget}k)",
                    "gap": gap,
                    "config": config,
                    "budget": budget,
                })
    
    if not entries:
        plt.close(fig)
        return
    
    # Sort by gap (smallest first = best)
    entries.sort(key=lambda e: e["gap"])
    
    labels = [e["label"] for e in entries]
    gaps = [e["gap"] for e in entries]
    colors = [CONFIG_COLORS.get(e["config"], "#607D8B") for e in entries]
    
    bars = ax.barh(range(len(entries)), gaps, color=colors, alpha=0.85, edgecolor="white")
    
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{gap:.2f} dB', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("PSNR Gap vs SfM (dB) — lower is better", fontsize=11)
    ax.set_title("Best Random Init Combos (Ranked by Proximity to SfM)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    fig.tight_layout()
    path = os.path.join(output_dir, "sfm_gap_ranking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_convergence_speed(data, output_dir):
    """Bar chart: iterations to reach 90% of final PSNR."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    entries = []
    for config in CONFIGS:
        if config not in data:
            continue
        for init_type in ["sfm"] + [f"{b}k" for b in POINT_BUDGETS]:
            if init_type not in data[config]:
                continue
            conv_iter = get_convergence_iter(data[config][init_type], 0.9)
            if conv_iter is not None:
                entries.append({
                    "label": f"{config}\n({INIT_LABELS[init_type]})",
                    "iter": conv_iter,
                    "init": init_type,
                    "config": config,
                })
    
    if not entries:
        plt.close(fig)
        return
    
    labels = [e["label"] for e in entries]
    iters = [e["iter"] for e in entries]
    colors = [COLORS_INIT.get(e["init"], "#607D8B") for e in entries]
    
    ax.barh(range(len(entries)), iters, color=colors, alpha=0.85, edgecolor="white")
    
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Iterations to 90% of Final PSNR", fontsize=11)
    ax.set_title("Convergence Speed: SfM vs Random Initialization", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_speed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def generate_summary(data, output_dir):
    """Generate JSON summary and print markdown table."""
    summary = {}
    
    for config in CONFIGS:
        if config not in data:
            continue
        summary[config] = {}
        
        for init_type in ["sfm"] + [f"{b}k" for b in POINT_BUDGETS]:
            if init_type not in data[config]:
                continue
            metrics = data[config][init_type]
            summary[config][init_type] = {
                "psnr": get_final_metric(metrics, "psnr"),
                "lpips": get_final_metric(metrics, "lpips"),
                "l1": get_final_metric(metrics, "l1"),
                "num_gaussians": get_final_metric(metrics, "num_gaussians"),
                "convergence_iter_90pct": get_convergence_iter(metrics, 0.9),
            }
        
        # Compute gaps vs SfM
        sfm_psnr = summary[config].get("sfm", {}).get("psnr")
        if sfm_psnr is not None:
            for budget in POINT_BUDGETS:
                key = f"{budget}k"
                if key in summary[config] and summary[config][key]["psnr"] is not None:
                    summary[config][key]["sfm_gap_db"] = round(sfm_psnr - summary[config][key]["psnr"], 4)
    
    path = os.path.join(output_dir, "random_init_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {path}")
    
    # Find best combo
    best_combo = None
    best_gap = float('inf')
    for config in CONFIGS:
        if config not in summary:
            continue
        for budget in POINT_BUDGETS:
            key = f"{budget}k"
            entry = summary[config].get(key, {})
            gap = entry.get("sfm_gap_db")
            if gap is not None and gap < best_gap:
                best_gap = gap
                best_combo = f"{config} + {key} points"
    
    if best_combo:
        print(f"\n  🏆 BEST RANDOM INIT COMBO: {best_combo} (only {best_gap:.2f} dB below SfM)")
    
    # Print table
    print(f"\n  === Random Initialization Summary ===")
    print(f"  {'Config':<22} {'Init':<12} {'PSNR':>8} {'LPIPS':>8} {'Gap vs SfM':>12} {'Conv@90%':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    for config in CONFIGS:
        if config not in summary:
            continue
        for init_type in ["sfm"] + [f"{b}k" for b in POINT_BUDGETS]:
            if init_type not in summary[config]:
                continue
            e = summary[config][init_type]
            gap = e.get("sfm_gap_db", "-")
            conv = e.get("convergence_iter_90pct", "-")
            psnr = f"{e['psnr']:.2f}" if e.get("psnr") else "N/A"
            lpips = f"{e['lpips']:.4f}" if e.get("lpips") else "N/A"
            print(f"  {config:<22} {init_type:<12} {psnr:>8} {lpips:>8} {str(gap):>12} {str(conv):>10}")
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Random initialization analysis plots")
    parser.add_argument("--output_dir", type=str, default="plots/random_init",
                        help="Directory to save plots and summary JSON")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading metrics...")
    data = load_metrics()
    
    if not data:
        print("ERROR: No metrics data found!")
        print("  Expected: output/batch/{config}/metrics.json for SfM baselines")
        print("  Expected: output/random_init/{config}_random_{N}k/metrics.json for random init")
        return
    
    print(f"Found {len(data)} configs:")
    for config, inits in data.items():
        print(f"  {config}: {list(inits.keys())}")
    
    print("\nGenerating plots...")
    plot_convergence_subplots(data, args.output_dir)
    plot_point_budget_sensitivity(data, args.output_dir)
    plot_sfm_gap(data, args.output_dir)
    plot_convergence_speed(data, args.output_dir)
    
    print("\nGenerating summary...")
    generate_summary(data, args.output_dir)
    
    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
