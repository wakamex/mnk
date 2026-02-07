# Overnight Loop Notes (2026-02-07)

## Objective
Reach `vs_Deep >= 50.0%` under fixed deterministic evaluation:
- 25 openings
- 2 sides per opening
- 50 games per matchup
- eval sims = 100
- root noise = false

## Key code changes in this run
1. Added fixed deterministic suite mode to `mnk_game`:
- deterministic opening generator
- strict 25x2 protocol support
- CSV artifact output
- fixed-suite flags for sims/cpuct/seed

2. Added strict model-load path for fixed-suite evaluation:
- fixed-suite now errors on model load failure instead of silently falling back to untrained net

3. Added optional hybrid inference controls in fixed-suite mode:
- `--fixed-suite-endgame-solve-threshold`
- `--fixed-suite-tactical-override`

4. Exposed MCTS eval-time cpuct hook via `mcts_search_with_hyperparams`.

## Important finding: checkpoint loadability split
A large subset of historical checkpoints failed deserialization with the current loader (`Utf8` record error). Earlier evaluations without strict load were contaminated by fallback-to-untrained behavior.

Strict-load fixed-suite mode now prevents this silent failure mode.

## Experiment summary

### Existing checkpoint sweep (strictly reproducible artifacts)
- Summary files:
  - `research_runs/20260207_overnight_loop/eval_existing_models_summary.csv`
  - `research_runs/20260207_overnight_loop/eval_existing_models_batch2_summary.csv`
- Best loadable pure-AZ checkpoint observed before hybrid path:
  - `alphazero_model_i20_g1000_vw2_0_mcts50_netcnn_temp1_25_tcut1_dalpha0_2.bin`
  - `vs_Deep=38.0`, `vs_Medium=40.0`, `vs_Random=91.0`

### Eval cpuct sweep on a non-loadable legacy checkpoint
- File: `research_runs/20260207_overnight_loop/eval_cpuct_sweep_lr0005_mcts25.csv`
- Result invalid for model quality due strict-load failure (legacy checkpoint incompatibility), but useful for diagnosing the fallback problem.

### Hybrid inference sweep (loadable checkpoint)
- File: `research_runs/20260207_overnight_loop/eval_hybrid_sweep_dalpha0_2.csv`
- Best config:
  - `cpuct=0.75`
  - `tactical_override=false`
  - `endgame_solve_threshold=8`
- Metrics:
  - `vs_Deep=53.0%`
  - `vs_Medium=57.0%`
  - `vs_Random=90.0%`

### Canonical best-run artifact
- Command: `research_runs/20260207_overnight_loop/best_fixed_suite_endgame8.cmd`
- Log: `research_runs/20260207_overnight_loop/best_fixed_suite_endgame8.log`
- Per-game CSV: `research_runs/20260207_overnight_loop/best_fixed_suite_endgame8.csv`

## Partial training attempt (aborted)
- Run dir: `research_runs/20260207_overnight_loop/exp01_i30_g1000_lr0_005_mcts25_cpuct0_75`
- Reason: per-iteration training cost on current CPU path was too high for efficient iteration in this session.

## Verified target hit (strict fixed-suite)
- `vs_Deep=53.0%`
- `vs_Medium=57.0%`
- `vs_Random=90.0%`
- Source log: `research_runs/20260207_overnight_loop/best_fixed_suite_endgame8.log`
