# M,N,K Game with AlphaZero Training

Tic-Tac-Toe (3,3,3 game) with AlphaZero-style training using the [Burn](https://burn.dev/) deep learning framework in Rust. GPU-accelerated with CUDA.

## Architecture

- **CNN Network** (`src/alphazero.rs`) — 3-layer conv backbone with board-agnostic heads: policy uses conv-only per-cell logits, value uses global pooling + MLP (transfer-ready across board sizes)
- **Transformer Network** (`src/minibt4.rs`) — Variable board size support with CNN-comparable default budget (~101K params)
- **Network Enum** (`src/network.rs`) — Swappable CNN/Transformer behind a common interface
- **MCTS** (`src/unified_mcts.rs`) — Proper AlphaZero tree search with PUCT, Dirichlet noise, batched multi-game self-play
- **Training** (`src/train_alphazero_unified.rs`) — SGD with momentum, logit clamping, 8x symmetry augmentation
- **Tournament** (`play.rs`) — Evaluation against Random/Deep/Medium strategies

## Building

Requires the `cuda-dev` podman container (host GCC is too new for CUDA):

```bash
bash build.sh
```

### Host vs Container (quick note)

- **Build**: use `bash build.sh` (runs inside `cuda-dev` via `podman exec`).
- **Run eval/tournament**: run on host with `./target/release/mnk_game ...` (no container required). Add `--cpu` to force CPU inference.
- **Run training**: host or `cuda-dev` both work; host requires a working NVIDIA driver/CUDA runtime setup.

## Training

```bash
./target/release/train_alphazero
```

Key flags:
- `--iterations N` — self-play/train cycles (default: 100)
- `--preset NAME|PATH` — load a named JSON preset from `configs/train/` (or explicit path). CLI flags override preset values.
- `--games-per-iter N` — games per self-play phase (default: 1000)
- `--mcts-simulations N` — MCTS sims per move (default: 50)
- `--epochs N` — training epochs per iteration (default: 8)
- `--learning-rate F` — SGD learning rate (default: 0.0015)
- `--optimizer sgd|adamw` — optimizer (default: sgd)
- `--lr-schedule constant|step|cosine` — LR schedule (default: step)
- `--lr-decay-step N` — step schedule interval (default: 25)
- `--lr-decay-gamma F` — step decay factor (default: 0.45)
- `--value-weight F` — value-loss weight (default: 2.0)
- `--temperature F` — opening move temperature (default: 1.25)
- `--temperature-cutoff-moves N` — opening moves with non-zero temperature (default: 1)
- `--dirichlet-alpha F` — self-play root-noise alpha (default: 0.1)
- `--cpuct F` — PUCT exploration constant (default: 0.75)
- `--net-type cnn|minibt4|transformer` — network architecture (default: cnn)
- `--board-width N` — square board width (default: 3)
- `--win-k N` — K in a row to win (default: 3)
- `--init-model-path PATH` — warm-start from an existing checkpoint (useful for transfer learning)

Preset examples:

```bash
# 3x3 production baseline (old strong 3x3 settings)
./target/release/train_alphazero --preset cnn_3x3_prod

# 5x5 k=4 transfer baseline (current transfer settings)
./target/release/train_alphazero --preset cnn_5x5k4_transfer \
  --init-model-path research_runs/transfer_ab/seed_20260209/b5_k4/cnn_3x3k3_seed20260209.bin
```

MiniBT4 baseline run (recommended starting point):

```bash
./target/release/train_alphazero \
  --net-type minibt4 \
  --optimizer sgd \
  --learning-rate 0.001 \
  --lr-schedule step \
  --lr-decay-step 25 \
  --lr-decay-gamma 0.65 \
  --iterations 100 \
  --model-path minibt4_i100.bin
```

## Transfer Learning (Kickoff)

The CNN architecture has board-agnostic heads, so you can initialize a larger-board CNN from a 3x3 checkpoint.

Notes:
- Fixed-suite evaluation now follows `--board-width` and `--win-k`. For larger boards, consider reducing `--fixed-suite-openings` / `--fixed-suite-sims` or evaluating every few iterations (`--fixed-suite-every 5`) to control wall-clock.

Scratch vs transfer example (5x5, still k=3 to keep the win condition comparable):

```bash
# Scratch
./target/release/train_alphazero \
  --net-type cnn \
  --board-width 5 --win-k 3 \
  --fixed-suite-every 0 \
  --model-path cnn_5x5k3_scratch.bin

# Transfer-init from a 3x3 CNN checkpoint
./target/release/train_alphazero \
  --net-type cnn \
  --board-width 5 --win-k 3 \
  --fixed-suite-every 0 \
  --init-model-path cnn_3x3_best.bin \
  --model-path cnn_5x5k3_from3x3.bin
```

Reproducible scratch vs transfer (writes artifacts under `research_runs/`, not committed):

```bash
SEED=20260209 BOARD=5 K=3 ./scripts/transfer_ab.sh
```

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
- **Data augmentation**: 8x via dihedral symmetries of square boards (generic size support in trainer)
- **Self-play batching**: N games run simultaneously; leaf nodes collected into batches for single GPU inference call
- **CNN transfer readiness**: no board-size-dependent FC head parameters, so a 3x3 checkpoint can initialize larger-board CNN models directly
