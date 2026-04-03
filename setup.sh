#!/bin/bash
set -e

echo "=== Gaussian Splatting + UI Server Setup ==="

# 1. Update submodules
echo "[1/4] Initializing git submodules..."
git submodule update --init --recursive

# 2. Install PyTorch FIRST (since submodules require it during compilation)
echo "[2/4] Installing PyTorch..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Install other Python dependencies
echo "[3/4] Installing Python dependencies..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Install CUDA submodules
echo "[4/4] Compiling CUDA submodules (this may take a few minutes)..."

# Apply gcc13 patch to diff-gaussian-rasterization if needed
if ! grep -q "<cstdint>" submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h; then
    sed -i '10i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
fi

pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

echo "=== Setup Complete! ==="
echo "You can now run: python server.py"
