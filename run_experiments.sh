#!/bin/bash
set -e

PYTHON="/workspace/gaussian-splatting/.venv/bin/python"
BASE="-s data/tandt/truck --eval --iterations 30000"
TEST="--test_iterations 1000 2000 3000 5000 7000 10000 15000 20000 25000 30000"
SAVE="--save_iterations 7000 30000"

echo "=== Starting 18-Variant 30K 3DGS Experiment Sweep ==="
echo "Estimated time: ~8-9 hours"
echo ""

# Clear previous batch
rm -rf output/batch
mkdir -p output/batch

echo "[1/14] Baseline (Standard L1 + SSIM)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/baseline --run_name baseline
echo "[1/14] DONE"

echo "[2/14] MCMC Only..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/mcmc_only --densification_strategy mcmc --run_name mcmc_only
echo "[2/14] DONE"

echo "[3/14] Cauchy Loss Only..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/cauchy_only --cauchy_loss --densify_grad_threshold 0.0006 --run_name cauchy_only
echo "[3/14] DONE"

echo "[4/14] Entropy Only..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/entropy_only --entropy_reg --run_name entropy_only
echo "[4/14] DONE"

echo "[5/14] Cauchy + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/cauchy_mcmc --cauchy_loss --densification_strategy mcmc --run_name cauchy_mcmc
echo "[5/14] DONE"

echo "[6/14] Cauchy Scheduled + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/cauchy_scheduled_mcmc --cauchy_loss --cauchy_scale_schedule --densification_strategy mcmc --run_name cauchy_scheduled_mcmc
echo "[6/14] DONE"

echo "[7/14] Entropy + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/entropy_mcmc --entropy_reg --densification_strategy mcmc --run_name entropy_mcmc
echo "[7/14] DONE"

echo "[8/14] Full Pack (Cauchy Sched + Entropy + MCMC)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/full_pack --cauchy_loss --cauchy_scale_schedule --entropy_reg --densification_strategy mcmc --run_name full_pack
echo "[8/14] DONE"

echo "[9/18] Cauchy Activation + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/cauchy_activation_mcmc --cauchy_activation --densification_strategy mcmc --run_name cauchy_activation_mcmc
echo "[9/18] DONE"

echo "[10/18] ARGP Vanilla (paper defaults)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/argp_vanilla \
    --densification_strategy argp \
    --densify_until_iter 12000 \
    --ctprune_ratio 0.01 \
    --tp_prune_level 0.7 \
    --recover_level 0.4 \
    --optimizer_type default \
    --run_name argp_vanilla
echo "[10/18] DONE"

echo "[11/18] ARGP + Cauchy Loss..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/argp_cauchy \
    --densification_strategy argp \
    --densify_until_iter 12000 \
    --ctprune_ratio 0.01 \
    --tp_prune_level 0.7 \
    --recover_level 0.4 \
    --cauchy_loss --cauchy_scale_schedule \
    --densify_grad_threshold 0.0006 \
    --optimizer_type default \
    --run_name argp_cauchy
echo "[11/18] DONE"

echo "[12/18] ARGP + Entropy Reg..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/argp_entropy \
    --densification_strategy argp \
    --densify_until_iter 12000 \
    --ctprune_ratio 0.01 \
    --tp_prune_level 0.7 \
    --recover_level 0.4 \
    --entropy_reg \
    --optimizer_type default \
    --run_name argp_entropy
echo "[12/18] DONE"

echo "[13/18] ARGP Aggressive (stronger pruning + more recovery)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/argp_aggressive \
    --densification_strategy argp \
    --densify_until_iter 12000 \
    --ctprune_ratio 0.05 \
    --tp_prune_level 0.85 \
    --recover_level 0.3 \
    --optimizer_type default \
    --run_name argp_aggressive
echo "[13/18] DONE"

echo "[14/18] ARGP Conservative (gentler pruning + more recovery)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/argp_conservative \
    --densification_strategy argp \
    --densify_until_iter 15000 \
    --ctprune_ratio 0.005 \
    --tp_prune_level 0.5 \
    --recover_level 0.6 \
    --optimizer_type default \
    --run_name argp_conservative
echo "[14/18] DONE"

echo "[15/18] Lion Optimizer (baseline densification)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/lion_baseline \
    --optimizer_type lion \
    --run_name lion_baseline
echo "[15/18] DONE"

echo "[16/18] Lion + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/lion_mcmc \
    --optimizer_type lion \
    --densification_strategy mcmc \
    --run_name lion_mcmc
echo "[16/18] DONE"

echo "[17/18] ApproxSLN Optimizer (baseline densification)..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/approx_sln_baseline \
    --optimizer_type approx_sln \
    --run_name approx_sln_baseline
echo "[17/18] DONE"

echo "[18/18] ApproxSLN + MCMC..."
$PYTHON train.py $BASE $TEST $SAVE -m output/batch/approx_sln_mcmc \
    --optimizer_type approx_sln \
    --densification_strategy mcmc \
    --run_name approx_sln_mcmc
echo "[18/18] DONE"

echo ""
echo "=== All 18 training runs complete! ==="
echo ""

echo "=== Rendering test views ==="
for d in output/batch/*/; do
    echo "Rendering $(basename $d)..."
    $PYTHON render.py -m "$d" --skip_train --quiet
done

echo ""
echo "=== Generating comparison plots ==="
$PYTHON plot_compare_all.py

echo ""
echo "=== ALL DONE ==="

