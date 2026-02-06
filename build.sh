#!/bin/bash

# Build script for mnk project
# Handles the complex CUDA container build environment
# Builds both CUDA (training+GPU tournament) and CPU-only (CPU tournament) binaries

set -e

LD_LIB="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib"

# Step 1: Build CPU-only binary (NdArray backend, no CUDA dependency)
echo "Building CPU-only binary (for tournaments)..."
CPU_BUILD_CMD="cd /workspace/mnk && $LD_LIB /root/.cargo/bin/cargo build --release --bin mnk_game 2>&1"
podman exec cuda-dev bash -c "$CPU_BUILD_CMD"
# Save CPU binary before CUDA build overwrites it
podman exec cuda-dev bash -c "cp /workspace/mnk/target/release/mnk_game /workspace/mnk/target/release/mnk_game_cpu"
echo "  -> target/release/mnk_game_cpu"

# Step 2: Build CUDA binary (burn-cuda via CubeCL for GPU training + inference)
echo "Building CUDA binaries (training + GPU tournament)..."
CUDA_BUILD_CMD="cd /workspace/mnk && $LD_LIB /root/.cargo/bin/cargo build --release --features cuda"
podman exec cuda-dev bash -c "$CUDA_BUILD_CMD"

echo ""
echo "Build completed successfully!"
echo ""
echo "Available binaries:"
echo "  ./target/release/train_alphazero  - Training binary (CUDA)"
echo "  ./target/release/mnk_game         - Tournament binary (CUDA)"
echo "  ./target/release/mnk_game_cpu     - Tournament binary (CPU-only)"
echo ""
echo "NOTE: Ensure container has matching NVIDIA libs (libcuda, libnvidia-ptxjitcompiler,"
echo "libnvidia-gpucomp) for the host driver version. Copy from /usr/lib64/ on host if needed."
