# M,N,K Game with AlphaZero Training

Tic-Tac-Toe (3,3,3 game) with AlphaZero-style training using the [Burn](https://burn.dev/) deep learning framework in Rust. GPU-accelerated with CUDA.

## Architecture

- **CNN Network** (`src/alphazero.rs`) — 3-layer conv backbone, policy head (raw logits), value head (tanh-bounded)
- **Transformer Network** (`src/minibt4.rs`) — Variable board size support
- **Network Enum** (`src/network.rs`) — Swappable CNN/Transformer behind a common interface
- **MCTS** (`src/unified_mcts.rs`) — Proper AlphaZero tree search with PUCT, Dirichlet noise, batched multi-game self-play
- **Training** (`src/train_alphazero_unified.rs`) — SGD with momentum, logit clamping, 8x symmetry augmentation
- **Tournament** (`play.rs`) — Evaluation against Random/Deep/Medium strategies

## Building

Requires the `cuda-dev` podman container (host GCC is too new for CUDA):

```bash
bash build.sh
```

## Training

```bash
podman exec cuda-dev bash -c "cd /workspace/mnk && \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib \
  ./target/release/train_alphazero \
    --iterations 20 --games-per-iter 500 --mcts-simulations 50"
```

Key flags:
- `--iterations N` — self-play/train cycles (default: 30)
- `--games-per-iter N` — games per self-play phase (default: 1000)
- `--mcts-simulations N` — MCTS sims per move (default: 50)
- `--epochs N` — training epochs per iteration (default: 8)
- `--learning-rate F` — SGD learning rate (default: 0.01)
- `--net-type cnn|transformer` — network architecture (default: cnn)

## Performance

With batched MCTS (interleaved multi-game self-play with batch NN inference):

| Metric | Value |
|--------|-------|
| Self-play throughput | 350-400 games/sec |
| Training loss (20 iter) | 2.7 → 0.9 |
| Win rate vs random (20 iter) | 87% |

## Training Details

- **Optimizer**: SGD with momentum 0.9, weight decay 1e-4, gradient clipping norm 1.0
- **Logit clamping**: Policy logits clamped to [-20, 20] before cross-entropy to prevent unbounded growth
- **Data augmentation**: 8x via dihedral symmetries of the square board
- **Self-play batching**: N games run simultaneously; leaf nodes collected into batches for single GPU inference call
