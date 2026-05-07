#!/usr/bin/env python3
"""
Unified Plot Generator for all diploma experiments.

Usage:
    python plot_all.py batch           — 20-config benchmark plots (§3.3-3.8)
    python plot_all.py scene           — Scene comparison: truck vs kitchen (§3.9)
    python plot_all.py random          — Random initialization analysis (§3.10)
    python plot_all.py pose_free       — Pose-free experiment plots
    python plot_all.py noisy           — Noisy pose refinement comparison (§3.11)
    python plot_all.py all             — Run all of the above

Each subcommand wraps the corresponding plot module's main() function.
"""

import sys
import os

def run_batch():
    """20-config benchmark (plot_compare_all)."""
    from plots_modules.plot_compare_all import main
    sys.argv = ['plot_all.py', '--plot_dir', 'plots_batch']
    main()

def run_scene():
    """Scene comparison: truck vs kitchen."""
    from plots_modules.plot_scene_comparison import main
    sys.argv = ['plot_all.py', '--output_dir', 'plots/scene_comparison',
                '--data_dirs', 'output/scene_comp']
    main()

def run_random():
    """Random initialization analysis."""
    from plots_modules.plot_random_init import main
    sys.argv = ['plot_all.py', '--output_dir', 'plots/random_init']
    main()

def run_pose_free():
    """Pose-free experiment plots."""
    from plots_modules.plot_pose_free import main
    main()

def run_noisy():
    """Noisy pose refinement comparison."""
    from plots_modules.plot_noisy_study import main
    main()

COMMANDS = {
    'batch': run_batch,
    'scene': run_scene,
    'random': run_random,
    'pose_free': run_pose_free,
    'noisy': run_noisy,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'all':
        for name, func in COMMANDS.items():
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")
            try:
                func()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}, all")
        sys.exit(1)

if __name__ == '__main__':
    main()
