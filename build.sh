#!/bin/bash

# Build script for mnk project
# Handles the complex CUDA container build environment

set -e

echo "🔨 Building mnk project in CUDA container..."

# Build command that works with CUDA environment
BUILD_CMD="cd /workspace/mnk && CUDA_COMPUTE_CAP=86 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib:/usr/local/cuda-12.8/compat /root/.cargo/bin/cargo build --release --features cuda"

# Execute in container
podman exec cuda-dev bash -c "$BUILD_CMD"

echo "✅ Build completed successfully!"
echo ""
echo "Available binaries:"
echo "  ./target/release/train_alphazero - Training binary"
echo "  ./target/release/mnk_game - Tournament binary"
echo ""
echo "Usage examples:"
echo "  podman exec cuda-dev bash -c \"cd /workspace/mnk && LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib:/usr/local/cuda-12.8/compat ./target/release/train_alphazero\""
echo "  ./target/release/mnk_game --model-path model.bin"