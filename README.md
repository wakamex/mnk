# M,N,K Game with AlphaZero-style Self-Play Training

This project implements an M,N,K game (generalized Tic-Tac-Toe) with an AlphaZero-inspired self-play training system using NNUE (Efficiently Updatable Neural Network) architecture.

## Overview

The M,N,K game is played on an M×N board where players take turns placing pieces, and the first to get K pieces in a row wins. This implementation combines:

- **NNUE Neural Network**: A lightweight, CPU-efficient neural network architecture originally from Shogi/Chess engines
- **Monte Carlo Tree Search (MCTS)**: Tree search with PUCT (Predictor + Upper Confidence bounds applied to Trees) for move selection
- **Self-Play Training**: The AI learns by playing against itself, continuously improving through reinforcement learning

## Architecture

### Core Components

1. **NNUE Network** (`src/nnue.rs`)
   - Input: Sparse board representation (768 features for 3x3, scales with board size)
   - Architecture: `(input -> 256) x2 -> 32 -> 32 -> 1`
   - Two-headed output: Policy (move probabilities) and Value (position evaluation)
   - Efficiently updatable through incremental computation

2. **MCTS with PUCT** (`src/mcts.rs`)
   - Tree search guided by neural network predictions
   - PUCT formula balances exploration vs exploitation
   - Configurable simulations per move (default: 800)

3. **Self-Play System** (`src/selfplay.rs`)
   - Generates training games using MCTS
   - Temperature-based move selection for exploration
   - Stores game trajectories with final outcomes

4. **Training Pipeline** (`src/training.rs`)
   - Processes self-play games into training data
   - Updates NNUE weights using gradient descent
   - Implements experience replay buffer

## Project Structure

```
mnk/
├── Cargo.toml          # Project dependencies
├── README.md           # This file
├── play.rs             # Existing M,N,K game implementation
├── src/
│   ├── main.rs         # Main training loop
│   ├── nnue.rs         # NNUE neural network implementation
│   ├── mcts.rs         # Monte Carlo Tree Search
│   ├── selfplay.rs     # Self-play game generation
│   ├── training.rs     # Neural network training
│   └── integration.rs  # Integration with existing game code
└── models/             # Saved neural network weights
```

## TODO List

### Phase 1: Core Implementation
- [ ] Implement NNUE network structure with sparse input encoding
- [ ] Create MCTS with PUCT selection and neural network integration
- [ ] Build self-play game generation system
- [ ] Develop training pipeline with batch processing

### Phase 2: Integration
- [ ] Integrate with existing M,N,K game structures
- [ ] Add serialization for model weights
- [ ] Create evaluation and testing framework
- [ ] Implement progressive training with checkpoint system

### Phase 3: Optimization
- [ ] Add SIMD optimizations for NNUE forward pass
- [ ] Implement parallel self-play workers
- [ ] Add tensorboard logging for training metrics
- [ ] Create tournament system for model comparison

### Phase 4: Extensions
- [ ] Support variable board sizes (different M,N,K configurations)
- [ ] Add opening book generation from self-play
- [ ] Implement AlphaZero-style policy iteration
- [ ] Create interactive play interface

## Building and Running

```bash
# Build the project
cargo build --release

# Run training
cargo run --release -- train --games 10000 --iterations 100

# Play against the AI
cargo run --release -- play --model models/best.nnue

# Run tournament between models
cargo run --release -- tournament --model1 models/v1.nnue --model2 models/v2.nnue
```

## Training Process

1. **Initialization**: Start with random neural network weights
2. **Self-Play**: Generate games using MCTS guided by current network
3. **Training**: Update network weights to predict game outcomes better
4. **Evaluation**: Test new network against previous version
5. **Iteration**: Keep best performing network, repeat from step 2

## Key Differences from Original AlphaZero

- **NNUE Architecture**: More efficient for CPU evaluation than deep CNNs
- **Incremental Updates**: Exploits sparse position changes in M,N,K games
- **Smaller Scale**: Designed for learning on consumer hardware
- **M,N,K Specific**: Optimized for games with simple win conditions

## Dependencies

```toml
[dependencies]
rand = "0.8"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
rayon = "1.7"  # For parallel self-play
indicatif = "0.17"  # Progress bars
```

## References

- [AlphaZero Paper](https://www.nature.com/articles/nature24270)
- [NNUE Introduction (Stockfish)](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
- [Leela Chess Zero](https://lczero.org/)