#!/bin/bash

# Build script for mnk project
# Handles the complex CUDA container build environment
# Builds CUDA-enabled binaries; runtime `--cpu` forces CPU inference in mnk_game.

set -e

LD_LIB="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib"

# Build CUDA binary (burn-cuda via CubeCL for GPU training + inference)
echo "Building CUDA binaries (training + GPU tournament)..."
CUDA_BUILD_CMD="cd /workspace/mnk && $LD_LIB /root/.cargo/bin/cargo build --release --features cuda"
podman exec cuda-dev bash -c "$CUDA_BUILD_CMD"

echo ""
echo "Build completed successfully!"
echo ""
echo "Available binaries:"
echo "  ./target/release/train_alphazero  - Training binary (CUDA)"
echo "  ./target/release/mnk_game         - Tournament/Eval binary (GPU by default; add --cpu for CPU inference)"
echo ""
echo "NOTE: Ensure container has matching NVIDIA libs (libcuda, libnvidia-ptxjitcompiler,"
echo "libnvidia-gpucomp) for the host driver version. Copy from /usr/lib64/ on host if needed."
