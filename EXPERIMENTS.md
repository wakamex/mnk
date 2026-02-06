# Training Efficiency Roadmap

Goal: maximize model quality per wall-clock second. Find optimal hyperparameters for both CNN and Transformer, determine which architecture learns faster, and identify the best training configuration for each.

## Current Baseline (CNN, 30 iter x 1000 games, defaults)

| Metric | Value |
|--------|-------|
| Wall clock | 268s |
| Self-play throughput | 370-415 games/sec |
| Final loss | 0.83 (value: 0.01, policy: 0.81) |
| vs Random | 88.5% |
| vs Medium (depth 2) | 37.0% |
| vs Deep (depth 3) | 25.0% |

The model draws against perfect minimax but never wins. Policy head has plateaued.

## Known Issues to Fix First

### 1. Sweep framework defaults are stale
`parallel_sweep.py` defaults were set for Adam optimizer:
- `learning_rate = [0.0005]` -- should be `[0.01]` for SGD
- `epochs = [2]` -- current training default is 8
- `mcts_simulations = [25]` -- current default is 50

### 2. Sweep framework lacks `--net-type` support
Cannot sweep CNN vs Transformer. Need to add `--net-type` as a sweep parameter in `SweepConfig`, `generate_experiments()`, and the argparse CLI.

### 3. Transformer batch inference is not actually batched
In `network.rs`, the Transformer's `forward_batch_inference()` loops over positions sequentially. This will make Transformer self-play much slower than CNN. Needs a real batched forward pass (same pattern as CNN).

## Experiment Phases

### Phase 0: Framework Fixes
Fix issues #1-3 above before running any sweeps. Verify with `--dry-run`.

### Phase 1: CNN Hyperparameter Sensitivity (quick, 1-2 hours)
Goal: find which hyperparameters matter most for CNN quality-per-second.

Sweep learning rate and MCTS sims (the two most impactful parameters), holding everything else at defaults:

```bash
python parallel_sweep.py \
  --net-type cnn \
  -i 20 \
  -g 1000 \
  --learning-rate 0.005,0.01,0.02,0.05 \
  --mcts 25,50,100,200 \
  --sweep-name cnn_lr_mcts \
  --dry-run
```
16 experiments. Key questions:
- Does more MCTS sims improve final quality enough to justify slower self-play?
- Is LR=0.01 actually optimal for SGD, or can we push higher?

Then sweep training intensity (epochs and games-per-iter):
```bash
python parallel_sweep.py \
  --net-type cnn \
  -i 20 \
  -g 500,1000,2000 \
  -e 4,8,16 \
  --sweep-name cnn_training_intensity \
  --dry-run
```
9 experiments. Key questions:
- Is 8 epochs overfitting within each iteration?
- Does 2000 games/iter help or just waste wall clock?

### Phase 2: Transformer Baseline (after fixing batch inference)
Goal: establish Transformer baseline on 3x3 with same hyperparameter grid.

```bash
python parallel_sweep.py \
  --net-type transformer \
  -i 20 \
  -g 1000 \
  --learning-rate 0.005,0.01,0.02,0.05 \
  --mcts 25,50,100 \
  --sweep-name transformer_lr_mcts \
  --dry-run
```
12 experiments. The transformer has ~800K params vs CNN's ~42K -- it may need different LR or more data.

### Phase 3: Head-to-Head CNN vs Transformer (long, overnight)
Goal: compare architectures at their respective best hyperparameters (from phases 1-2).

Run both architectures at longer training horizons:
```bash
# Use best LR/MCTS from phases 1-2
python parallel_sweep.py \
  --net-type cnn,transformer \
  -i 30,50,100 \
  -g 1000 \
  --learning-rate <best_cnn_lr>,<best_transformer_lr> \
  --mcts <best_mcts> \
  --sweep-name architecture_comparison \
  --dry-run
```
Key questions:
- Does the transformer catch up or surpass CNN with more iterations?
- Which architecture has better quality-per-second at convergence?
- Does CNN plateau earlier due to smaller capacity?

### Phase 4: Policy Head Investigation
The biggest open problem is the passive policy (draws but never wins). Targeted experiments:

- **Value weight sweep**: `--value-weight 0.25,0.5,1.0,2.0,4.0` -- maybe we're over-weighting value loss, starving the policy gradient
- **MCTS sims scaling**: Try 200-800 sims -- more search may produce sharper policy targets
- **Temperature threshold**: Currently 2 (first 2 moves sampled, rest argmax). Try 0,4,6 to see if more exploration during self-play produces better policy training data

## Metrics to Track

For each experiment, the sweep framework already captures:
- Training time (wall clock)
- Games/sec (self-play throughput)
- vs Random / vs Medium / vs Deep (tournament scores)

Derived metrics to compute from sweep CSVs:
- **Quality/second**: tournament score / wall_clock_time
- **Convergence speed**: iterations to reach 80% vs Random
- **Policy quality**: vs_Medium score (sensitive to policy -- must exploit mistakes to beat depth-2)

## Architecture Notes

**CNN** (~42K params): 3 conv layers (32/64/128 channels), 1x1 conv heads. Only supports 3x3. Fast inference, fast training. Good baseline.

**Transformer** (~800K params): 4 layers, 8 heads, d_model=128, 2D positional encoding. Supports 3x3 to 15x15. ~20x more parameters than CNN. Potentially better for larger boards but may be overkill for 3x3.

**Transformer gap**: `forward_batch_inference()` in `network.rs` is sequential (loops over positions one at a time). Must fix before meaningful comparison -- otherwise Transformer self-play will be artificially slow.
