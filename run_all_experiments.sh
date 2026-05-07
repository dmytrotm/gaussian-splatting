#!/bin/bash
# =============================================================================
# UNIFIED EXPERIMENT RUNNER
# =============================================================================
#
# Reproduces ALL experiments from the diploma thesis in sequence:
#   Part A: Scene Comparison   (Easy: truck vs Hard: kitchen)
#   Part B: Random Init        (SfM vs Random @ 50k/100k/200k)
#   Part C: Pose-Free / Noisy Pose Refinement
#
# Prerequisites:
#   1. Truck dataset at data/tandt/truck
#   2. Kitchen dataset at data/360_v2/kitchen
#      (download: python download_dataset.py --dataset mipnerf360 --save_dir data)
#   3. Existing SfM baseline results in output/batch/ (for Part A reuse)
#
# Estimated GPU time: ~12 hours total (RTX 3090)
#
# Output:
#   output/scene_comp/       — scene comparison training outputs + renders
#   output/random_init/      — random init training outputs + renders
#   output/pose_free/        — pose-free and noisy pose experiments
#   plots/scene_comparison/  — plots + JSON summary
#   plots/random_init/       — plots + JSON summary
#   plots/pose_free/         — plots + JSON summary
# =============================================================================
set -e

PYTHON="/workspace/gaussian-splatting/.venv/bin/python"

# Fallback: find python if .venv doesn't exist
if [ ! -f "$PYTHON" ]; then
    PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
fi

echo "Using Python: $PYTHON"
echo ""

TRUCK_SRC="data/tandt/truck"
KITCHEN_SRC="data/360_v2/kitchen"
TEST="--test_iterations 1000 2000 3000 5000 7000 10000 15000 20000 25000 30000"
SAVE="--save_iterations 7000 30000"
ITERS="--iterations 30000"


# #############################################################################
#  PART A: SCENE COMPARISON (§3.9)
#  6 configs × kitchen + reuse truck from output/batch/
# #############################################################################

run_part_a() {
    local BASE_OUT="output/scene_comp"

    if [ ! -d "$KITCHEN_SRC" ]; then
        echo "WARNING: Kitchen dataset not found at $KITCHEN_SRC — skipping Part A"
        echo "Download it: $PYTHON download_dataset.py --dataset mipnerf360 --save_dir data"
        return
    fi

    echo "============================================================"
    echo "  PART A: Scene Comparison (Kitchen)"
    echo "============================================================"

    mkdir -p "$BASE_OUT/kitchen"

    # Link existing truck results
    mkdir -p "$BASE_OUT/truck"
    for config in baseline full_pack mcmc_only cauchy_scheduled_mcmc argp_conservative entropy_mcmc; do
        if [ -d "output/batch/$config" ]; then
            ln -sf "$(realpath output/batch/$config)" "$BASE_OUT/truck/$config" 2>/dev/null || true
        fi
    done

    echo "[A1/6] Kitchen: Baseline..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/baseline" --run_name baseline
    echo ""

    echo "[A2/6] Kitchen: Full Pack..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/full_pack" \
        --cauchy_loss --cauchy_scale_schedule --entropy_reg \
        --densification_strategy mcmc --run_name full_pack
    echo ""

    echo "[A3/6] Kitchen: MCMC Only..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/mcmc_only" \
        --densification_strategy mcmc --run_name mcmc_only
    echo ""

    echo "[A4/6] Kitchen: Cauchy Scheduled MCMC..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/cauchy_scheduled_mcmc" \
        --cauchy_loss --cauchy_scale_schedule \
        --densification_strategy mcmc --run_name cauchy_scheduled_mcmc
    echo ""

    echo "[A5/6] Kitchen: ARGP Conservative..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/argp_conservative" \
        --densification_strategy argp \
        --densify_until_iter 15000 --ctprune_ratio 0.005 \
        --tp_prune_level 0.5 --recover_level 0.6 \
        --optimizer_type default --run_name argp_conservative
    echo ""

    echo "[A6/6] Kitchen: Entropy MCMC..."
    $PYTHON train.py -s "$KITCHEN_SRC" -i images_2 --eval $ITERS $TEST $SAVE \
        -m "$BASE_OUT/kitchen/entropy_mcmc" \
        --entropy_reg --densification_strategy mcmc --run_name entropy_mcmc
    echo ""

    # Render test views
    for d in "$BASE_OUT"/kitchen/*/; do
        $PYTHON render.py -m "$d" --skip_train --quiet 2>/dev/null || true
    done

    # Generate plots
    $PYTHON plot_all.py scene

    echo "  PART A COMPLETE"
    echo ""
}


# #############################################################################
#  PART B: RANDOM INITIALIZATION (§3.10)
#  4 configs × 3 point budgets = 12 runs
# #############################################################################

run_part_b() {
    local BASE_OUT="output/random_init"
    local RUN=0
    local TOTAL=12

    if [ ! -d "$TRUCK_SRC" ]; then
        echo "WARNING: Truck dataset not found — skipping Part B"
        return
    fi

    echo "============================================================"
    echo "  PART B: Random Initialization (12 runs)"
    echo "============================================================"

    mkdir -p "$BASE_OUT"

    run_training() {
        local CONFIG_NAME=$1
        local NUM_POINTS=$2
        local BUDGET_K=$3
        local EXTRA_ARGS=$4

        RUN=$((RUN + 1))
        local RUN_NAME="${CONFIG_NAME}_random_${BUDGET_K}k"

        echo "[$RUN/$TOTAL] $RUN_NAME (${NUM_POINTS} random points)..."
        $PYTHON train.py -s "$TRUCK_SRC" --eval $ITERS $TEST $SAVE \
            --random_init --random_init_num_points "$NUM_POINTS" \
            -m "$BASE_OUT/$RUN_NAME" --run_name "$RUN_NAME" $EXTRA_ARGS
        echo ""
    }

    # Baseline
    run_training "baseline" 50000  50  ""
    run_training "baseline" 100000 100 ""
    run_training "baseline" 200000 200 ""

    # MCMC Only
    run_training "mcmc_only" 50000  50  "--densification_strategy mcmc"
    run_training "mcmc_only" 100000 100 "--densification_strategy mcmc"
    run_training "mcmc_only" 200000 200 "--densification_strategy mcmc"

    # Full Pack
    run_training "full_pack" 50000  50  "--cauchy_loss --cauchy_scale_schedule --entropy_reg --densification_strategy mcmc"
    run_training "full_pack" 100000 100 "--cauchy_loss --cauchy_scale_schedule --entropy_reg --densification_strategy mcmc"
    run_training "full_pack" 200000 200 "--cauchy_loss --cauchy_scale_schedule --entropy_reg --densification_strategy mcmc"

    # ARGP Conservative
    ARGP_ARGS="--densification_strategy argp --densify_until_iter 15000 --ctprune_ratio 0.005 --tp_prune_level 0.5 --recover_level 0.6 --optimizer_type default"
    run_training "argp_conservative" 50000  50  "$ARGP_ARGS"
    run_training "argp_conservative" 100000 100 "$ARGP_ARGS"
    run_training "argp_conservative" 200000 200 "$ARGP_ARGS"

    # Render + plots
    for d in "$BASE_OUT"/*/; do
        $PYTHON render.py -m "$d" --skip_train --quiet 2>/dev/null || true
    done
    $PYTHON plot_all.py random

    echo "  PART B COMPLETE"
    echo ""
}


# #############################################################################
#  PART C: POSE-FREE & NOISY POSE REFINEMENT (§3.11)
# #############################################################################

run_part_c() {
    local BASE_OUT="output/pose_free"
    local POSE_COMMON="-s ${TRUCK_SRC} --eval --iterations 30000 \
        --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 \
        --save_iterations 7000 15000 30000 \
        --pose_lr 0.001 --pose_lr_final 0.0001 \
        --densify_until_iter 25000 --densification_interval 200"

    if [ ! -d "$TRUCK_SRC" ]; then
        echo "WARNING: Truck dataset not found — skipping Part C"
        return
    fi

    echo "============================================================"
    echo "  PART C: Pose-Free & Noisy Pose Refinement"
    echo "============================================================"

    mkdir -p "$BASE_OUT"

    # --- C1: Fully pose-free (Fibonacci sphere init) ---
    echo "[C1/4] Pose-Free: Full Pack..."
    $PYTHON train.py $POSE_COMMON -m "$BASE_OUT/pose_free_full_pack" \
        --pose_free --densification_strategy mcmc \
        --entropy_reg --cauchy_loss --cauchy_scale_schedule \
        --run_name pose_free_full_pack --plot_dir plots/pose_free
    echo ""

    echo "[C2/4] Pose-Free: Cauchy Sched MCMC..."
    $PYTHON train.py $POSE_COMMON -m "$BASE_OUT/pose_free_cauchy_sched_mcmc" \
        --pose_free --densification_strategy mcmc \
        --cauchy_loss --cauchy_scale_schedule \
        --run_name pose_free_cauchy_sched_mcmc --plot_dir plots/pose_free
    echo ""

    # --- C3: Noisy pose with refinement ---
    echo "[C3/4] Noisy Pose: Refinement (σ=0.1)..."
    $PYTHON train.py $POSE_COMMON --densify_from_iter 2000 \
        -m "$BASE_OUT/pose_noisy_full_pack" \
        --pose_noise 0.1 --pose_refine \
        --densification_strategy mcmc \
        --entropy_reg --cauchy_loss --cauchy_scale_schedule \
        --run_name pose_noisy_full_pack --plot_dir plots/pose_free
    echo ""

    # --- C4: Noisy pose baseline (no refinement) ---
    echo "[C4/4] Noisy Pose: Baseline (σ=0.1, no correction)..."
    $PYTHON train.py $POSE_COMMON --densify_from_iter 2000 \
        -m "$BASE_OUT/pose_noisy_baseline" \
        --pose_noise 0.1 \
        --densification_strategy mcmc \
        --entropy_reg --cauchy_loss --cauchy_scale_schedule \
        --run_name pose_noisy_baseline --plot_dir plots/pose_free
    echo ""

    # Generate comparison plot
    $PYTHON plot_all.py noisy

    echo "  PART C COMPLETE"
    echo ""
}


# #############################################################################
#  MAIN: Run all parts
# #############################################################################

echo "================================================================"
echo "  3DGS Adaptive Pipeline — Full Experiment Suite"
echo "  Parts: A (Scene Comparison), B (Random Init), C (Pose Refinement)"
echo "================================================================"
echo ""

# Parse optional arguments: --part a, --part b, --part c, or run all
if [ "$1" == "--part" ]; then
    case "$2" in
        a|A) run_part_a ;;
        b|B) run_part_b ;;
        c|C) run_part_c ;;
        *) echo "Unknown part: $2. Use a, b, or c."; exit 1 ;;
    esac
else
    run_part_a
    run_part_b
    run_part_c
fi

echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Plots:  plots/scene_comparison/"
echo "          plots/random_init/"
echo "          plots/pose_free/"
echo "================================================================"
