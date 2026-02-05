# M,N,K Game with AlphaZero Training (Burn Framework)

This project implements Tic-Tac-Toe (3,3,3 M,N,K game) with AlphaZero-style training using the Burn deep learning framework in Rust. Features GPU acceleration with CUDA support and high-performance batched MCTS.

## Current Status - Working Implementation

- ✅ **AlphaZero Training**: Complete self-play training pipeline with Burn neural networks
- ✅ **GPU Acceleration**: CUDA support with batched inference (1500+ games/sec)
- ✅ **Unified MCTS**: Standalone Monte Carlo Tree Search module with batching
- ✅ **Tournament System**: Automated model evaluation against different strategies
- ✅ **Parameter Sweeps**: Parallel hyperparameter optimization with performance tracking
- ✅ **Performance Tracking**: Real-time games/second measurement for training and tournaments
- ✅ **Batch Optimization**: Automatic batch size optimization for optimal GPU utilization

## Architecture

### Core Components

1. **AlphaZero Neural Network** (`src/alphazero.rs`)
   - Burn framework with GPU support (CUDA backend)
   - Two-headed architecture: Policy (move probabilities) + Value (position evaluation)
   - Batched inference for high-performance evaluation
   - 8x symmetry augmentation during training

2. **Unified MCTS** (`src/unified_mcts.rs`)
   - Standalone Monte Carlo Tree Search implementation
   - Batched neural network evaluation for performance
   - PUCT formula for exploration/exploitation balance
   - InterleavedGamesManager for multi-game parallelization

3. **Self-Play Training** (`src/train_alphazero_unified.rs`)
   - High-performance training with batched position evaluation
   - Experience replay with symmetry augmentation
   - Configurable hyperparameters (learning rate, batch size, etc.)
   - Automatic model saving and compatibility checks

4. **Tournament System** (`play.rs`)
   - **Interleaved tournaments**: Batches AlphaZero neural network calls across concurrent games
   - Model evaluation against Random, Deep, and Medium strategies
   - Configurable number of tournament games (default 100)
   - Performance measurement and win rate tracking
   - 2x speed improvement over sequential tournaments

## Project Structure

```
mnk/
├── Cargo.toml                      # Dependencies: Burn, CUDA, etc.
├── README.md                       # Project documentation
├── play.rs                         # Tournament system and CLI
├── parallel_sweep.py              # Hyperparameter sweep automation
├── src/
│   ├── lib.rs                     # Library exports
│   ├── alphazero.rs               # Neural network (Burn framework)
│   ├── unified_mcts.rs            # Standalone MCTS implementation
│   ├── mcts_bridge.rs             # Bridge for network integration
│   ├── self_play.rs               # Self-play functionality
│   ├── train_alphazero_unified.rs # Training executable
│   └── inference_backend.rs       # Model loading and inference
└── sweep_results/                 # Training results and logs
```

## Building and Running

### Requirements
- Rust 1.70+
- CUDA 12.8+ (optional, for GPU acceleration)
- Python 3.8+ (for parameter sweeps)

### Build
```bash
# CPU-only build
cargo build --release

# GPU build with CUDA support
cargo build --release --features cuda
```

### Training
```bash
# Quick training test
./target/release/train_alphazero --iterations 2 --games 5 --epochs 1

# Full training session
./target/release/train_alphazero --iterations 100 --games 50 --epochs 5 --batch-size 512 --mcts-simulations 25
```

### Tournament Evaluation
```bash
# Evaluate trained model
./target/release/mnk_game --model-path alphazero_model.bin --tournament-games 100
```

### Parameter Sweeps
```bash
# Automated hyperparameter optimization
python3 parallel_sweep.py --training-jobs 4 --tournament-jobs 2 --concurrent
```

## Key Features

### GPU Acceleration
- CUDA backend support through Burn framework
- Batched neural network inference (1500+ games/sec)
- Automatic fallback to CPU if CUDA unavailable

### Performance Optimizations
- InterleavedGamesManager for parallel game simulation
- Batched position evaluation across multiple games
- Optimized MCTS with batch size tuning
- 8x symmetry augmentation for data efficiency

### Training Pipeline
- Self-play game generation with MCTS
- Experience replay with augmented training data
- Automatic model checkpointing and validation
- Configurable hyperparameters (learning rate, batch size, MCTS simulations)

### Evaluation System
- Tournament play against multiple strategies (Random, Deep, Medium)
- Win rate tracking and performance measurement
- Games/second performance monitoring
- Automated sweep result analysis

## Example Training Session

```bash
$ ./target/release/train_alphazero --iterations 10 --games 25 --epochs 2 --batch-size 512

Testing AlphaZero with Burn Framework
=====================================
💻 Running on GPU (CUDA available)

Training Configuration:
  Iterations: 10
  Games per iteration: 25
  Epochs: 2
  Batch size: 512
  Learning rate: 0.001
  Value weight: 1
  MCTS simulations: 25

  OPTIMIZED position batching: 0.15s for 25 games (166.7 games/sec)
  Batch size: 128, ALL game states batched!
Iteration 1: 95 → 760 examples (8x symmetry)
  Epoch 1: Total Loss = 2.8456
    Value Loss (weighted): 1.6234, Policy Loss: 1.2222
    Value:Policy ratio 1:1, 25 MCTS simulations
  Self-play: 0.15s, Training: 1.25s, Total: 1.40s

✅ Training completed! Model saved and ready for tournament use.
```

## Performance Notes

### Current Performance (Optimized GPU + Interleaved Tournaments)
- **Training**: 1130+ games/sec with GPU batch optimization (5x faster than CPU)
- **Tournament**: 2.5-2.8 games/sec with interleaved batching (2x improvement)
- **Optimal batch size**: 512-1024 for GPU training
- **Memory**: ~2GB GPU memory usage during training
- **Build Time**: <1 second incremental builds

### GPU vs CPU Comparison
- **GPU Training**: 1130+ games/sec (batch size 512+), 0.7s total time
- **CPU Training**: 400-611 games/sec (batch size 64), 6.5s total time
- **GPU Advantage**: 5-9x faster training, especially for longer runs

### Build Requirements
GPU acceleration requires container build:
```bash
./build.sh  # Builds with CUDA support in container
```

For CPU-only development:
```bash
cargo build --release  # CPU-only build
```

## Dependencies

Key dependencies from `Cargo.toml`:
- **burn**: Deep learning framework with CUDA support
- **burn-ndarray**: CPU backend
- **burn-cuda**: GPU acceleration
- **rand**: Random number generation
- **clap**: Command-line argument parsing

## References

- [Burn Deep Learning Framework](https://burn.dev/)
- [AlphaZero Paper](https://www.nature.com/articles/nature24270)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)