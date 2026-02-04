# GPU Acceleration Setup for AlphaZero with Rust and Burn Framework

This document details the complete process of enabling GPU acceleration for an AlphaZero implementation using Rust, the Burn framework, and NVIDIA CUDA on a modern Linux system.

## System Overview

- **Host**: Fedora 43 with GCC 15.2.1 and glibc 2.42
- **GPU**: NVIDIA GeForce RTX 3090 (Compute Capability 8.6)
- **Driver**: NVIDIA 580.105.08 (CUDA 13.0 compatible)
- **Target Framework**: Burn 0.17.1 with Candle backend for CUDA acceleration

## The Core Challenge

The primary obstacle was **host system incompatibility**: CUDA 12.4/13.x requires GCC ≤13, but Fedora 43 ships with GCC 15.2.1 and glibc 2.42, creating an insurmountable compilation barrier for CUDA code.

## Solution Strategy: Containerization

We solved this by using **Podman containers** to isolate the CUDA environment from the incompatible host system, while maintaining GPU access through device passthrough.

## Step-by-Step Implementation

### 1. Container Environment Setup

Created a CUDA development container with proper GPU passthrough:

```bash
podman run -d --privileged --name cuda-dev \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools \
  -v /code/mnk:/workspace/mnk:Z \
  docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04 sleep infinity
```

**Key Elements:**
- `--privileged`: Required for proper GPU device access
- Device mounts: All necessary NVIDIA device files
- CUDA 13.1.1: Compatible with host driver 580.105.08
- Ubuntu 22.04: Provides GCC 11.x (CUDA compatible)

### 2. Development Environment Installation

Inside the container:

```bash
# Install build tools and Rust
apt-get update && apt-get install -y curl build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

### 3. Cargo.toml Configuration

Critical dependency management for CUDA compatibility:

```toml
[dependencies]
# Burn framework with CUDA support
burn = { version = "0.17.1", features = ["ndarray", "autodiff"] }
burn-ndarray = { version = "0.17.1" }
burn-candle = { version = "0.17.1", optional = true }

# Candle backend with consistent versions
candle-core = { version = "0.8.4", features = ["cuda"] }
candle-nn = { version = "0.8.4", features = ["cuda"] }

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "burn-candle/cuda"]

[[bin]]
name = "train_alphazero_gpu"
path = "src/train_alphazero_gpu.rs"
```

**Critical Points:**
- Version consistency between burn-candle (0.17.1) and candle-core (0.8.4)
- Explicit CUDA features on all candle dependencies
- Optional burn-candle for conditional compilation

### 4. GPU-Enabled Source Code

Created `src/train_alphazero_gpu.rs` with conditional compilation:

```rust
use burn::prelude::*;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};

// GPU backend configuration
#[cfg(feature = "cuda")]
use burn_candle::{Candle, CandleDevice};

#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Candle>;

// CPU fallback
#[cfg(not(feature = "cuda"))]
use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;

fn main() {
    #[cfg(feature = "cuda")]
    println!("🚀 GPU ACCELERATION ENABLED (Candle/CUDA)");

    #[cfg(not(feature = "cuda"))]
    println!("💻 Running on CPU");

    // Device initialization
    #[cfg(feature = "cuda")]
    let device = CandleDevice::cuda(0);

    #[cfg(not(feature = "cuda"))]
    let device = NdArrayDevice::default();

    // ... rest of AlphaZero implementation
}
```

### 5. Build Process

Compilation with proper CUDA configuration:

```bash
cd /workspace/mnk
CUDA_COMPUTE_CAP=86 cargo build --release --bin train_alphazero_gpu --features cuda --no-default-features
```

**Build Parameters:**
- `CUDA_COMPUTE_CAP=86`: RTX 3090 compute capability
- `--features cuda`: Enable CUDA compilation path
- `--no-default-features`: Clean feature selection

### 6. Runtime Dependencies

The critical missing piece was the **NVIDIA PTX JIT compiler**:

```bash
# Install NVIDIA runtime libraries including PTX compiler
apt-get install -y libnvidia-compute-580 nvidia-cuda-toolkit
```

**What this provides:**
- `libnvidia-compute-580`: GPU compute runtime matching host driver
- PTX JIT compiler: Required for runtime CUDA kernel compilation
- CUDA runtime libraries: Full execution environment

## Technical Challenges Resolved

### 1. **Host System Incompatibility**
- **Problem**: GCC 15.2.1 vs CUDA requirement of GCC ≤13
- **Solution**: Container isolation with Ubuntu 22.04 (GCC 11.x)

### 2. **GPU Device Access**
- **Problem**: Container couldn't access NVIDIA devices
- **Solution**: Privileged container with explicit device passthrough

### 3. **CUDA Driver Mismatch**
- **Problem**: Container CUDA vs host driver version conflicts
- **Solution**: Matching CUDA 13.1.1 container with host driver 580.105.08

### 4. **Library Version Conflicts**
- **Problem**: Burn-candle 0.17.1 vs candle-core version mismatches
- **Solution**: Explicit version pinning to compatible combinations

### 5. **Runtime PTX Compiler Missing**
- **Problem**: `CUDA_ERROR_JIT_COMPILER_NOT_FOUND` during execution
- **Solution**: Install `libnvidia-compute-580` for PTX JIT compiler

## Performance Results

### Benchmark Comparison

| Platform | Time per Iteration | Speedup |
|----------|-------------------|---------|
| CPU (Host) | ~12.24s | 1.0x |
| GPU (RTX 3090) | ~5.31s | **2.3x** |

### Training Quality Metrics

GPU training showed excellent convergence:
- **Iteration 5**: 77% win rate against random player
- **Loss reduction**: From 3.03 to 0.67 over 5 iterations
- **Stable performance**: Consistent ~5.3s iteration times

## Key Learnings

### 1. **Containerization is Essential**
Modern Linux distributions often have CUDA compatibility issues. Containers provide perfect isolation while maintaining GPU access.

### 2. **Version Compatibility Matrix**
The Rust ML ecosystem requires careful version alignment:
- Burn framework version
- Candle backend version
- CUDA toolkit version
- Host NVIDIA driver version

### 3. **Runtime vs Compile-time Dependencies**
Successfully compiling with CUDA doesn't guarantee runtime execution. The PTX JIT compiler is a critical runtime dependency often overlooked.

### 4. **Device Passthrough Complexity**
GPU access in containers requires:
- All NVIDIA device files (`/dev/nvidia*`)
- Privileged access for device management
- Proper driver library mounting

## Reproduction Instructions

To reproduce this setup on a similar system:

1. **Verify GPU and driver compatibility**:
   ```bash
   nvidia-smi  # Check driver version
   ```

2. **Create GPU-enabled container**:
   ```bash
   podman run -d --privileged --name cuda-dev \
     --device=/dev/nvidia0 --device=/dev/nvidiactl \
     --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools \
     -v /path/to/project:/workspace/project:Z \
     docker.io/nvidia/cuda:13.1.1-devel-ubuntu22.04 sleep infinity
   ```

3. **Install development environment**:
   ```bash
   podman exec cuda-dev bash -c "
     apt-get update && apt-get install -y curl build-essential &&
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   "
   ```

4. **Install NVIDIA runtime**:
   ```bash
   podman exec cuda-dev apt-get install -y libnvidia-compute-580 nvidia-cuda-toolkit
   ```

5. **Build and run**:
   ```bash
   podman exec cuda-dev bash -c "
     cd /workspace/project &&
     source ~/.cargo/env &&
     CUDA_COMPUTE_CAP=86 cargo build --release --features cuda &&
     ./target/release/your_gpu_binary
   "
   ```

## Current Status

✅ **GPU acceleration fully working!**

- **Performance**: 2.3x speedup over CPU (5.31s vs 12.24s per iteration)
- **Training quality**: 77% win rate by iteration 5
- **Stability**: Consistent performance with proper convergence

## Quick Start Commands

For immediate use with the working setup:

```bash
# Start the container (if not running)
podman start cuda-dev

# Run GPU training
podman exec cuda-dev bash -c "
  cd /workspace/mnk &&
  timeout 60 ./target/release/train_alphazero_gpu
"
```

## Conclusion

Enabling GPU acceleration for Rust ML workloads requires navigating complex compatibility matrices between:
- Host system toolchain
- CUDA toolkit versions
- Rust framework versions
- NVIDIA driver versions

**Containerization proved to be the key solution**, providing a stable, reproducible environment while maintaining high-performance GPU access. The resulting 2.3x speedup demonstrates the value of this approach for computationally intensive ML training.

The investment in proper GPU setup pays immediate dividends in development velocity and enables exploration of larger, more complex neural network architectures that would be prohibitively slow on CPU alone.