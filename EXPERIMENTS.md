# Experiment Log

This file tracks what has actually been run and which defaults we picked from sweep data.
Old roadmap/planning notes were removed to keep this document current.

## Source of truth

- Sweep artifacts: `sweep_results/`
- Latest completed CNN summary: `sweep_results/cnn_lr_mcts_20260206_125932.csv`
- Curve analysis tool: `analyze_sweep.py`

## Overnight harness run (2026-02-07)

Run outputs live under local `research_runs/` and are not committed.

Key additions in code:
- Fixed deterministic evaluation suite mode in `mnk_game` (25 openings x 2 sides, eval sims=100, root noise off, CSV output).
- Strict model-load behavior in fixed-suite mode (fail fast instead of silently evaluating with an untrained fallback model).
- Exposed MCTS eval-time hyperparameter path via `mcts_search_with_hyperparams`.

Critical finding:
- Many legacy checkpoints are not loadable with the current recorder path (`Utf8` record error). Earlier sweeps that allowed fallback-to-untrained were partly invalid for model ranking.

## Current defaults (as of 2026-02-11)

- `--iterations`: `100`
- `--learning-rate`: `0.0015`
- `--mcts-simulations`: `50`
- `--value-weight`: `2.0`
- `--temperature`: `1.25`
- `--temperature-cutoff-moves`: `1`
- `--dirichlet-alpha`: `0.1`
- `--cpuct`: `0.75`
- `--optimizer`: `sgd`
- `--lr-schedule`: `step`
- `--lr-decay-step`: `25`
- `--lr-decay-gamma`: `0.45`

Notes:
- These defaults now prioritize the current transfer-learning track (`5x5 k=4`, init from `3x3` CNN), not legacy `3x3` fixed-suite-only optimization.
- For pure `3x3` work, the older `lr=0.02, gamma=0.65` setting remains a valid baseline.
- To avoid ambiguity, use explicit presets:
  - `--preset cnn_3x3_prod` for 3x3
  - `--preset cnn_5x5k4_transfer` for 5x5 k=4 transfer

## Optimizer + LR-schedule sweep (2026-02-07)

Run:
- `python parallel_sweep.py --optimizer sgd,adamw --lr-schedule constant,cosine --learning-rate 0.01,0.02 --sweep-name optimizer_lr_schedule_v1`

Summary from `sweep_results/optimizer_lr_schedule_v1_20260207_184552.csv`:
- `lr0.02_optsgd_lrsconstant`: `vs_Deep=43.0%` (best)
- `lr0.02_optsgd_lrscosine`: `39.0%`
- `lr0.01_optsgd_lrsconstant`: `38.0%`
- `lr0.01_optsgd_lrscosine`: `33.0%`
- AdamW variants: `21.0%` to `27.0%`

Decision (superseded by step-schedule sweeps below):
- Under the 30-iteration regime, `optimizer=sgd`, `lr_schedule=constant`, `learning_rate=0.02` looked best.
- Under the newer 100-iteration regime, step schedules dominate constant for peak `vs_Deep` and are now default.

## LR drop @ iter 25 (100 iters) (2026-02-07)

Goal: reduce late-run drift/collapse by lowering LR after the mid-run peak (~iter 25).

Run:
- `python parallel_sweep.py --iterations 100 --optimizer sgd --learning-rate 0.02 --lr-schedule step --lr-decay-step 25 --lr-decay-gamma 0.5,0.7,0.85 --sweep-name lr_drop_iter25`

Important: as of 2026-02-07, sweeps are ranked by `vs_Deep_Max` (peak over iterations), not `vs_Deep` at the final iteration, since the trainer exports the best checkpoint by `vs_Deep`.

3-run snapshot (max `vs_Deep` over 100 iters):
- `gamma=0.5`: max `49-50%`
- `gamma=0.7`: max `48-52%` (highest peaks, higher variance)
- `gamma=0.85`: max `48-49%` (worst)

Conclusion (tentative):
- Step LR drop at iter 25 helps produce >50% peaks against Deep more often than constant LR.
- Prefer `gamma=0.5` for stability or `gamma=0.7` when optimizing for peak strength.

## Step LR gamma sweep (step=25) (2026-02-08)

Goal: pick a default `--lr-decay-gamma` for `--lr-schedule step` (keeping `--lr-decay-step 25` fixed).

Runs (3 repeats):
- `python parallel_sweep.py --iterations 100 --optimizer sgd --learning-rate 0.02 --lr-schedule step --lr-decay-step 25 --lr-decay-gamma 0.45,0.5,0.55,0.6,0.65,0.7,0.75 --sweep-name lr_gamma_step25_v{1,2,3}`

Aggregate metric: `vs_Deep_Max` (peak `vs_Deep` over iterations; trainer exports the best checkpoint by `vs_Deep`).

Results (mean `vs_Deep_Max` over 3 runs):
- `gamma=0.65`: `50.33%` (min `49%`, max `52%`)
- `gamma=0.60`: `49.67%` (min `48%`, max `52%`)
- `gamma=0.50`: `49.67%` (min `47%`, max `52%`)
- `gamma=0.70`: `49.33%` (min `49%`, max `50%`, lowest variance)
- `gamma=0.55`: `47.67%` (min `46%`, max `50%`)
- `gamma=0.45`: `47.67%` (min `47%`, max `49%`)
- `gamma=0.75`: `47.33%` (min `45%`, max `49%`)

Decision:
- Set `--lr-schedule step` as the default schedule (best peaks and less late-run collapse).
- Set `--lr-decay-gamma 0.65` as the default (when using `--lr-schedule step`).
- Set `--lr-decay-step 25` as the default for `--lr-schedule step` (matches our best-performing schedule).

### Correlated step-size check (2026-02-08)

We tried smaller step sizes with correspondingly higher gammas (to keep total decay roughly comparable) and then ran 5 repeats for the most promising pair.

Result:
- `step=20, gamma=0.70` (n=5): mean `vs_Deep_Max=48.8%` (min `46%`, max `50%`)
- `step=25, gamma=0.65` (n=3, from `lr_gamma_step25_v1..v3`): mean `vs_Deep_Max=50.3%` (min `49%`, max `52%`)

Conclusion:
- Do not switch to smaller steps; keep `--lr-decay-step 25` and `--lr-decay-gamma ~0.65` as the step-schedule default.

### Base LR sweep (step=25, gamma=0.65) (2026-02-08)

Run:
- `python parallel_sweep.py --iterations 100 --optimizer sgd --lr-schedule step --lr-decay-step 25 --lr-decay-gamma 0.65 --learning-rate 0.01,0.015,0.02,0.025,0.03 --sweep-name lr_base_step25_g0.65_v1`

Summary from `sweep_results/lr_base_step25_g0.65_v1_20260208_155001.csv` (ranked by `vs_Deep_Max`):
- `lr=0.02`: `vs_Deep_max=50%` @iter 42 (final `36%`)
- `lr=0.025`: `49%` @iter 29 (final `32%`)
- `lr=0.01`: `48%` @iter 29 (final `43%` best final, but lower peak)
- `lr=0.015`: `48%` @iter 49 (final `30%`)
- `lr=0.03`: `43%` @iter 2 (final `27%`)

Conclusion:
- Keep `lr=0.02` as the base LR default (best peak in this run).
- `lr=0.01` looks more stable (better final score), but did not beat the peak; consider repeating `lr=0.01` vs `lr=0.02` if we want a stability/throughput tradeoff.

## MiniBT4 hyperparams (3x3) (2026-02-08)

Goal: stabilize MiniBT4 training (avoid very-early peaks and late collapse) and find a good LR/optimizer/schedule combo under the same fixed-suite vs Deep metric (`vs_Deep_Max`).

Wide sweep:
- `python parallel_sweep.py --net-type minibt4 --optimizer sgd,adamw --learning-rate 0.0003,0.001,0.003 --lr-schedule cosine,step --sweep-name minibt4_opt_lr_sched_wide_v1`

Summary from `sweep_results/minibt4_opt_lr_sched_wide_v1_20260208_205951.csv` (ranked by `vs_Deep_Max`):
- Best: `optimizer=sgd`, `lr-schedule=step`, `lr=0.001` => `vs_Deep_max=48%` @iter 48 (final `41%`)
- AdamW variants were generally worse; `lr=0.003` often peaked in the first few iterations then degraded.

Follow-up LR local sweep (SGD + step):
- `python parallel_sweep.py --net-type minibt4 --optimizer sgd --lr-schedule step --learning-rate 0.0006,0.0008,0.001,0.0012,0.0015 --sweep-name minibt4_lr_local_step_v1`

Result: `lr=0.0008` and `lr=0.001` are the best region; higher LRs degrade more.

Reproducibility check (`lr=0.0008` vs `lr=0.001`, 3 repeats):
- `sweep_results/minibt4_lr_0p0008_vs_0p001_r1_20260208_224849.csv`
- `sweep_results/minibt4_lr_0p0008_vs_0p001_r2_20260208_232239.csv`
- `sweep_results/minibt4_lr_0p0008_vs_0p001_r3_20260208_235519.csv`

Aggregate (metric `vs_Deep_Max`):
- `lr=0.001`: mean `45.33%` (min `45%`, max `46%`)
- `lr=0.0008`: mean `45.00%` (min `45%`, max `45%`)

Decision (for now):
- Default MiniBT4 recommendation: `--optimizer sgd --lr-schedule step --learning-rate 0.001`.

## MCTS-only scaling sweep (2026-02-07)

Run:
- `python parallel_sweep.py --mcts 100,200,400,800,1200 --sweep-name mcts_only_high`
- Training-sweep mode (no tournament), ranking by in-training `fixed_suite_vs_deep`.

Summary from `sweep_results/mcts_only_high_20260207_160627.csv`:
- `mcts100`: final `vs_Deep=47.0%`, max `48.0%`, training time `431.5s`
- `mcts200`: final `36.0%`, max `43.0%`, training time `593.4s`
- `mcts400`: final `41.0%`, max `49.0%`, training time `841.5s`
- `mcts800`: final `40.0%`, max `45.0%`, training time `1277.8s`
- `mcts1200`: final `46.0%`, max `49.0%`, training time `1568.6s`

Conclusion:
- Keep `mcts=100` as the practical default for now.
- Higher MCTS (`400/1200`) can hit similar or slightly higher peak `vs_Deep`, but did not produce a clearly better final/stable score in this n=1 run and costs 2-4x wall-clock.
- Treat 1-2 point differences as likely eval noise at this suite size; only change default after multi-seed confirmation.

Follow-up multi-seed runs:
- `sweep_results/mcts_only_high_2_20260207_171707.csv`
- `sweep_results/mcts_only_high_3_20260207_175130.csv`

Aggregate over 3 runs (`mcts_only_high`, `_2`, `_3`):
- `mcts100`: mean `vs_Deep=44.33%`, std `2.49`, mean time `429.0s`
- `mcts200`: mean `42.67%`, std `4.71`, mean time `581.1s`
- `mcts400`: mean `42.33%`, std `4.19`, mean time `843.4s`
- `mcts800`: mean `41.67%`, std `1.25`, mean time `1257.4s`
- `mcts1200`: mean `45.33%`, std `2.49`, mean time `1560.5s`

Decision:
- Keep `mcts=100` as interim default relative to `200/400/800/1200`.
- `mcts1200` is ~1.0pp higher on mean `vs_Deep`, but at ~3.6x wall-clock cost; not worth defaulting to for current iteration speed goals.

Focused default check (`mcts50` vs `mcts100`):
- `sweep_results/mcts_50_vs_100_20260207_181239.csv`
- `sweep_results/mcts_50_vs_100_2_20260207_181743.csv`
- `sweep_results/mcts_50_vs_100_3_20260207_182258.csv`

Aggregate over 3 runs:
- `mcts50`: mean `vs_Deep=46.67%`, std `0.47`, mean time `225.8s`, `vs_Deep/hour=744.0`
- `mcts100`: mean `vs_Deep=46.67%`, std `2.36`, mean time `283.4s`, `vs_Deep/hour=592.7`

Final default decision:
- Revert to `mcts=50` as global default.
- `mcts100` does not show better mean `vs_Deep` in repeated runs and is slower (~26%).

Rationale:
- Temp-cutoff sweep (`sweep_results/temp_cutoff_v1_20260207_000025.csv`) selected `tcut=1` as the only setting that kept `vsDeep=25%` while reaching `vsMedium=50%`.
- Dirichlet sweep at `tcut=1` (`sweep_results/dirichlet_tcut1_v1_20260207_001159.csv`) selected `dalpha=0.1` as top balanced config (`vsR=89.5%, vsD=25.0%, vsM=50.0%`).
- CPUCT sweep (`sweep_results/cpuct_tcut1_dalpha0.1_v1_20260207_002811.csv`) selected `cpuct=0.75` as best combined strength/diversity tradeoff.

Position-diversity note:
- Duplicate analysis showed very high repetition with `tcut=1` (`8978` samples, `145` exact uniques, `34` canonical uniques), versus `tcut=4` (`6635` samples, `934` exact uniques, `241` canonical uniques).
- We keep `tcut=1` for strength and track diversity as a secondary metric during sweeps.

## Architecture update (2026-02-07)

- CNN heads were refactored to be board-size-agnostic:
  - Policy: conv-only per-cell logits (`1x1` conv to one channel, then flatten to `H*W`)
  - Value: spatial global average pooling first, then small FC stack to scalar
- Result: no `H*W`-dependent parameter shapes in CNN heads, so a checkpoint trained on 3x3 can be loaded into larger-board CNN models for transfer learning.
- Trainer internals were also updated to use dynamic board-size tensors/symmetry transforms; current self-play/MCTS logic is still 3x3-specific and remains the next blocker for full larger-board CNN training.

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
This was later superseded in current defaults by a short opening schedule (`--temperature-cutoff-moves 1`), based on the 2026-02-07 sweeps above.

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

### Focused sweep (`sweep_results/policy_focused_vw2_temp1_20260206_171949.csv`)

3x3 grid: `temp={0.75,1.0,1.25}` x `vw={1.5,2.0,3.0}`

| Config | vsR | vsD | vsM | Avg |
|--------|-----|-----|-----|-----|
| **vw2.0_temp1.25** | **87.0%** | **38.0%** | **45.5%** | **56.8%** |
| vw1.5_temp1.0 | 74.0% | 25.0% | 43.0% | 47.3% |
| vw1.5_temp1.25 | 86.5% | 23.5% | 27.5% | 45.8% |
| vw3.0_temp0.75 | 90.5% | 1.5% | 25.0% | 39.0% |

The `vw=2.0, temp=1.0` result from the 4x4 grid (50%/50% vsD/vsM) did not reproduce (0.5%/0.0%) — confirmed as noise at n=1. `temp=1.25` is consistently the best temperature in this region.

## Reference implementation comparison

Both `/code/alpha-zero` (Connect4) and `/code/AlphaZero_Gomoku` (Gomoku) use ~103K param CNNs on 6x6 boards — same architecture as ours. Key hyperparameter gaps:

| | AlphaZero_Gomoku | alpha-zero | Our setup |
|--|--|--|--|
| **Board** | 6x6 k=4 | 6x6 Connect4 | 5x5 k=4 |
| **Params** | ~104K | ~103K | ~103K |
| **MCTS sims** | **400** | **500** | 50 |
| **Replay buffer** | 10K | 20K | 20K |
| **Grad updates/game** | **5 epochs x batch** | **50 per game** | 8 epochs x ~few batches |
| **Optimizer** | **Adam lr=0.002** | **Adam lr=0.001** | SGD lr=0.02 |
| **Input channels** | **4** (pieces, last move, color) | 1 | 1 |
| **Total games** | 1,500 | 3,000 | 1,000/iter x 100 iter |
| **c_puct** | **5.0** | **5.0** | 0.75 |
| **Dirichlet alpha** | 0.3 | 0.3 | 0.1 |
| **Training time** | ~2 hrs (6x6) | 7-8 hrs | ~2.4 hrs (5x5, 100 iter) |

Top gaps to close (in priority order):
1. **c_puct**: both references use 5.0, we use 0.75 — low c_puct means MCTS doesn't explore the tree, so more sims just deepen the same narrow line
2. **MCTS sims**: 400-500 vs 50 — now that we have a replay buffer, higher sims should produce sharper targets the buffer can train on repeatedly
3. **Optimizer**: both use Adam, we use SGD — Adam is more forgiving, especially for larger/harder problems

## 5x5 k=4 transfer learning sweeps (2026-02-13 to 2026-02-15)

All runs use `--preset cnn_5x5k4_transfer` (transfer from 3x3 CNN checkpoint, 1000 games/iter, replay buffer 20K).

### c_puct x MCTS sims (`b5k4_cpuct_high_v1`, vs_Deep)

Goal: close the c_puct gap with reference implementations (they use 5.0, we use 0.75).

Grid: `cpuct={2.0,5.0}` x `mcts={50,100,200}`, 100 iterations.

| Config | Time | vs_Deep max | vs_Deep final |
|--------|------|-------------|---------------|
| mcts50, cpuct2.0 | 4083s | 3.0% @i90 | 2.0% |
| mcts50, cpuct5.0 | 3635s | 3.0% @i30 | 0.0% |
| mcts100, cpuct2.0 | 5994s | 3.0% @i5 | 0.0% |
| mcts100, cpuct5.0 | 5662s | 2.0% @i80 | 0.0% |
| mcts200, cpuct2.0 | 9024s | 2.0% @i55 | 0.0% |
| mcts200, cpuct5.0 | 8947s | 2.0% @i25 | 0.0% |

Conclusion: c_puct doesn't matter at these levels. More MCTS sims don't help either — 2.5x wall time for no improvement. All configs 0-3% vs_Deep.

### Adam optimizer (`b5k4_adam_v1`, vs_Deep)

Goal: test Adam (gap #3 from reference comparison).

| Optimizer | vs_Deep max | Final |
|-----------|-------------|-------|
| AdamW (lr=0.0015) | **5.0%** @i45 | 3.0% |
| SGD (lr=0.0015) | 0.0% | 0.0% |

Adam is clearly better. But lr=0.0015 was tuned for SGD.

### Adam LR sweep (`b5k4_adam_lr_v1`, vs_Deep)

| LR | vs_Deep max | Peak iter | Final | Pattern |
|----|-------------|-----------|-------|---------|
| **0.0001** | **6.0%** | **100** | **6.0%** | **Still climbing** |
| 0.0003 | 4.0% | 25 | 0.0% | Collapsed |
| 0.001 | 2.0% | 80 | 0.0% | Collapsed |
| 0.003 | 4.0% | 60 | 1.0% | Collapsed |

lr=0.0001 is the only config that didn't collapse — peaked at the last iteration. But extending to 300 iters (`b5k4_adam_long_v1`) didn't help: lr=0.0001 only hit 2% max over 300 iters, lr=5e-05 hit 4% max. The 6% result was noise.

### Value target blending (new feature)

Inspired by [wazir-drop](https://github.com/tczajka/wazir-drop): instead of training the value head on pure game outcomes (+1/-1), blend with the MCTS root Q-value:

```
value = blend * game_outcome + (1 - blend) * mcts_root_q
```

CLI: `--value-target-blend` (default 1.0 = pure game outcome, backward compatible).

Rationale: with 50 MCTS sims on a 25-cell board (~2 visits/move), self-play games are nearly random. Game outcomes have almost no correlation with position quality. The MCTS root Q-value, even from a weak net, provides a lower-variance bootstrap signal.

### vtb sweep vs_Deep (`b5k4_vtb_v1`)

| vtb | vs_Deep max | Final |
|-----|-------------|-------|
| 0.1 | 2.0% @i10 | 1.0% |
| 0.5 | 1.0% @i40 | 1.0% |
| 1.0 | 0.0% | 0.0% |

Tiny signal in right direction, but Deep is too strong to measure at these levels.

### vtb sweep vs_Medium (`b5k4_vs_medium_v1`)

Switched fixed-suite to `--fixed-suite-mode medium` (new feature) to get more signal.

| vtb | vs_Medium max | Peak iter | Final |
|-----|---------------|-----------|-------|
| **0.1** | **13.0%** | 35 | 6.0% |
| 1.0 | 6.0% | 40 | 5.0% |

The model IS learning — we were just measuring against too strong an opponent. vtb=0.1 doubles the peak score.

### vtb aggressive sweep vs_Medium (`b5k4_vtb_medium_v2`, 200 iters)

| vtb | vs_Medium max | Peak iter | Final | Collapse |
|-----|---------------|-----------|-------|----------|
| **0.0** (pure MCTS) | **22.0%** | 25 | **18.0%** | 4% drop |
| 0.05 | 18.0% | 40 | 11.0% | 7% drop |
| 0.1 | 17.0% | 10 | 6.0% | 11% drop |

Key findings:
1. **Less game outcome = better**: vtb=0.0 > 0.05 > 0.1 monotonically. Game outcomes are pure noise on 5x5 k=4 with 50-sim MCTS.
2. **Less game outcome = more stable**: vtb=0.0 only dropped 4% from peak, while vtb=0.1 dropped 11%. Game outcome noise causes training collapse.
3. **Pure MCTS value (vtb=0.0) is best**: 22% vs_Medium max, 18% final — best 5x5 result so far.

### Current best 5x5 k=4 config

```
--preset cnn_5x5k4_transfer
--optimizer adamw
--learning-rate 0.0001
--cpuct 2.0
--value-target-blend 0.0
--fixed-suite-mode medium
```

### Summary of what helped vs what didn't for 5x5 k=4

| Change | Impact |
|--------|--------|
| Adam optimizer | Major (0% → 6% vs_Deep) |
| Value target blend (vtb=0.0) | Major (6% → 22% vs_Medium) |
| Lower LR (0.0001) | Moderate (prevents collapse) |
| Higher c_puct (2.0, 5.0) | None |
| More MCTS sims (100-1000) | None |
| More iterations (300) | None (model collapses) |

## Next experiments

### Push vtb=0.0 further

- Longer training (300+ iters) — vtb=0.0 showed only 4% collapse, might keep climbing
- Test vtb=0.0 with higher MCTS sims (100, 200) — now that value targets are stable, more sims might actually help
- Test vtb=0.0 with vs_Deep to see if we've crossed the threshold

### Batch size exploration

Reference implementations use batch_size=512-8192, we use 256. Larger batches stabilize Adam training. Worth testing 512, 1024, 2048.

### Phase 3: Transformer on Larger Boards

The current MiniBT4 default is right-sized to ~101K params (close to CNN ~103K), so capacity mismatch is no longer the main concern for 3x3. Instead of broad sweeps on 3x3:
- **Sanity check**: One quick Transformer run on 3x3 (5 iterations) to confirm it trains
- **Real test**: Transformer on 5x5 or 7x7 boards where capacity is more justified than 3x3
- Compare Transformer learning curves on larger boards against random/minimax baselines

## Useful commands

Analyze latest sweep:

```bash
python analyze_sweep.py sweep_results/
```

Run a 5x5 k=4 training sweep:

```bash
python parallel_sweep.py \
  --preset cnn_5x5k4_transfer \
  -i 200 --epochs 8 \
  --optimizer adamw --learning-rate 0.0001 \
  --cpuct 2.0 --value-target-blend 0.0 \
  --fixed-suite-mode medium --fixed-suite-every 5 \
  --sweep-name my_sweep
```
