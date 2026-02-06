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
- `--value-weight`: `4.0`
- `--temperature`: `1.75`

Rationale: lr/mcts from Phase 1 CNN sweep. Value weight and temperature from Phase 2 policy investigation — `vw=4.0, temp=1.75` had the best composite score (vsR=83%, vsD=49.5%, vsM=46%), first config to approach 50% against both Deep and Medium minimax.

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

## Phase 2: Policy Head Investigation (completed)

Replaced the binary temp_threshold (sample N moves then argmax) with continuous temperature `tau` applied via `N_i^(1/tau)` to MCTS visit counts for all moves.

### Value weight sweep (`sweep_results/policy_value_weight_20260206_140629.csv`)

- `vw=4.0`: best vsM (40.5%), decent vsD (13.0%)
- `vw=0.25`: most stable, balanced across opponents
- High variance across opponents at n=1 per config

### Temperature sweep (`sweep_results/policy_temperature_v2_20260206_142644.csv`)

- `tau=1.5`: best composite (vsD=38%, vsM=35%)
- Low tau (0.1, 0.5): good vsR but passive policy (0% vsD)
- `tau=2.0`: too noisy, drops off

### Combined grid (`sweep_results/policy_temp_vw_grid_20260206_164451.csv`)

4x4 grid: `temp={1.0,1.25,1.5,1.75}` x `vw={1.0,2.0,4.0,6.0}`

| Config | vsR | vsD | vsM | Avg |
|--------|-----|-----|-----|-----|
| **vw4.0_temp1.75** | **83.0%** | **49.5%** | **46.0%** | **59.5%** |
| vw2.0_temp1.0 | 69.5% | 50.0% | 50.0% | 56.5% |
| vw1.0_temp1.0 | 88.5% | 25.0% | 47.5% | 53.7% |
| vw2.0_temp1.5 | 82.5% | 37.5% | 41.5% | 53.8% |

**Note**: `vw=2.0, temp=1.0` scored vsD=50%, vsM=50% but vsR=69.5% — needs further investigation. This region of the parameter space may contain a better balanced config. Consider a focused sweep around `vw=1.5-3.0, temp=0.75-1.25`.

### Focused sweep around vw=2.0, temp=1.0

The `vw=2.0, temp=1.0` config hit 50%/50% vs Deep/Medium but only 69.5% vs Random. A focused grid around this area may find a config that keeps the strong policy without sacrificing vs Random:

```bash
python parallel_sweep.py \
  --net-type cnn -i 20 -g 1000 \
  --temperature 0.75,1.0,1.25 \
  --value-weight 1.5,2.0,3.0 \
  --sweep-name policy_focused_vw2_temp1
```

### MCTS sims scaling

More search produces sharper policy targets. Try 200-400 sims in isolated runs now that VRAM is stable.

### Phase 3: Transformer on Larger Boards

The Transformer (800K params) is ~20x overparameterized for 3x3. Running a full sweep on 3x3 would just confirm "right-sized CNN beats overparameterized Transformer on trivial problem."

Instead:
- **Sanity check**: One quick Transformer run on 3x3 (5 iterations) to confirm it trains without crashing
- **Real test**: Transformer on 5x5 or 7x7 boards where CNN can't go and the capacity is justified
- Compare Transformer learning curves on larger boards against random/minimax baselines

### Housekeeping

- Re-test `mcts=200` in isolated runs if needed after policy head improvements
- Consider training intensity sweep (`--epochs 4,8,16`, `--games-per-iter 500,1000,2000`) once policy head is unblocked

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
