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
- `--learning-rate F` — SGD learning rate (default: 0.02)
- `--net-type cnn|transformer` — network architecture (default: cnn)

## Sweep Status

Latest sweep progress and decisions are tracked in `EXPERIMENTS.md`.

Current CNN sweep snapshot (`sweep_results/cnn_lr_mcts_20260206_125932.csv`):

| Config | Train Time | vs Random | vs Deep | vs Medium |
|--------|------------|-----------|---------|-----------|
| `lr0.005_mcts100` | `1091.2s` | `91.5%` | `24.0%` | `24.5%` |
| `lr0.005_mcts25` | `919.3s` | `82.0%` | `38.5%` | `24.5%` |
| `lr0.05_mcts50` | `945.0s` | `80.0%` | `25.0%` | `44.0%` |
| `lr0.02_mcts50` | `857.5s` | `86.0%` | `27.0%` | `26.5%` |

## Training Details

- **Optimizer**: SGD with momentum 0.9, weight decay 1e-4, gradient clipping norm 1.0
- **Logit clamping**: Policy logits clamped to [-20, 20] before cross-entropy to prevent unbounded growth
- **Data augmentation**: 8x via dihedral symmetries of the square board
- **Self-play batching**: N games run simultaneously; leaf nodes collected into batches for single GPU inference call
