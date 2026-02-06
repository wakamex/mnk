# Experiment Log

This file tracks what has actually been run and which defaults we picked from sweep data.
Old roadmap/planning notes were removed to keep this document current.

## Source of truth

- Sweep artifacts: `sweep_results/`
- Latest completed CNN summary: `sweep_results/cnn_lr_mcts_20260206_125932.csv`
- Curve analysis tool: `analyze_sweep.py`

## Current defaults (as of 2026-02-06)

- `--learning-rate`: `0.02`
- `--mcts-simulations`: `50`

Rationale: this pair is a better balance than the old `0.01/50` on the current sweep, with stronger `vs_Deep` and `vs_Medium` while staying fast.

## Latest completed sweep (CNN lr x mcts)

Sweep grid:

- Iterations: `20`
- Games/iter: `1000`
- Epochs: `8`
- Batch size: `1024`
- LR: `0.005, 0.01, 0.02, 0.05`
- MCTS: `25, 50, 100, 200`

Selected outcomes from `sweep_results/cnn_lr_mcts_20260206_125932.csv`:

| Config | Train Time | vs Random | vs Deep | vs Medium | Notes |
|--------|------------|-----------|---------|-----------|-------|
| `lr0.005_mcts100` | `1091.2s` | `91.5%` | `24.0%` | `24.5%` | Best `vs Random` |
| `lr0.005_mcts25` | `919.3s` | `82.0%` | `38.5%` | `24.5%` | Best `vs Deep` |
| `lr0.05_mcts50` | `945.0s` | `80.0%` | `25.0%` | `44.0%` | Best `vs Medium` and best weighted composite in `analyze_sweep.py` |
| `lr0.02_mcts50` | `857.5s` | `86.0%` | `27.0%` | `26.5%` | Chosen default compromise |

## Reliability notes

- `mcts=200` had only 1 successful run; 3/4 runs failed.
- Failed runs show `CUDA_ERROR_OUT_OF_MEMORY` in:
  - `sweep_results/i20_g1000_lr0.005_mcts200_netcnn/training.log`
  - `sweep_results/i20_g1000_lr0.01_mcts200_netcnn/training.log`
  - `sweep_results/i20_g1000_lr0.02_mcts200_netcnn/training.log`
- Partial `training_log.csv` data for failed runs is still useful for convergence trend analysis, but not for final tournament ranking.

## VRAM status

- Main training VRAM-growth issue was traced to inference through `Autodiff<Cuda>` during self-play/eval.
- Fix now in trainer: self-play/eval/sample inference use `net.valid()` (non-autodiff inner backend).
- With no competing GPU jobs, monitored runs showed stable VRAM after warm-up instead of linear growth.

## Next queued sweeps

- Transformer baseline sweep on same lr/mcts grid (3x3).
- Value-weight sweep (`--value-weight`) to improve policy behavior vs medium-depth opponents.
- Re-test `mcts=200` in isolated runs only if needed, after confirming stable memory behavior under current code.

## Useful commands

Analyze latest sweep:

```bash
python analyze_sweep.py sweep_results/
```

Analyze a specific summary:

```bash
python analyze_sweep.py sweep_results/cnn_lr_mcts_20260206_125932.csv
```

Run the CNN lr/mcts sweep:

```bash
python parallel_sweep.py \
  --net-type cnn \
  -i 20 \
  -g 1000 \
  --learning-rate 0.005,0.01,0.02,0.05 \
  --mcts 25,50,100,200 \
  --sweep-name cnn_lr_mcts
```
