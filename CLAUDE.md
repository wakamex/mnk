# Claude Instructions

## Objective Assessment
- Be factually accurate about success/failure rates
- Don't be sycophantic or overly positive about mediocre results
- Focus on identifying and fixing root causes rather than celebrating partial success
- Poor results are poor results - call them what they are

## Technical Standards
- 100% success rate should be the goal for automated systems
- Investigate failures systematically
- Don't declare victory until the system works reliably

## Code Analysis
- Don't hallucinate problems that don't exist
- Actually read and analyze code before claiming there are issues

## Module Organization
- `src/alphazero.rs` — CNN neural network (AlphaZeroNet), board evaluation, batch inference
- `src/minibt4.rs` — Transformer neural network (MiniBT4Net) for variable board sizes
- `src/network.rs` — `Network` enum wrapping CNN/Transformer, implements `PolicyValueNetwork` trait
- `src/unified_mcts.rs` — MCTS tree search: `mcts_search()` for single-game, `generate_training_data_batched()` for multi-game batched self-play
- `src/mcts_bridge.rs` — Implements `NetworkInference` trait for `AlphaZeroNet`
- `src/inference_backend.rs` — Backend type aliases and model loading for play.rs
- `src/train_alphazero_unified.rs` — Training binary (SGD + momentum, logit clamping, symmetry augmentation)
- `play.rs` — Tournament system and CLI

## Build
- All compilation must happen in the `cuda-dev` podman container
- Use `bash build.sh` to build
- Run binaries via `podman exec cuda-dev bash -c "cd /workspace/mnk && LD_LIBRARY_PATH=... ./target/release/train_alphazero"`
