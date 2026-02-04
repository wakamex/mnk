# MNK NNUE Training – Progress Log

Date: 2025-08-07

## What I did
- Added a minimal sanity binary `nnue_sanity` (`src/nnue_sanity.rs`) to run tiny self-play + one training epoch.
- Exposed `Trainer::evaluate` as public for quick pre/post loss checks.
- Fixed `ndarray` shapes in `forward_with_features` (1D concat) and enabled `ndarray` serde support; added `bincode` and `ndarray` deps.
- Added a new Cargo bin target for `nnue_sanity`.

## How to run
```
cargo run --bin nnue_sanity -q
```

## Sanity run result (3x3, 6 games, 1 epoch)
- Pre-train loss: 1.4793
- Final train loss: 1.5010, Val loss: 1.3826
- Post-train loss: 1.4792 (slight decrease on full set; validation improved more noticeably)

Interpretation: Head-only SGD updates execute and loss does not explode; validation decreased. With more data/epochs, we expect clearer downward trend.

## Larger sanity run (3x3, 50 games, 3 epochs)
Command:
```
cargo run --bin nnue_sanity -q -- --games 50 --epochs 3 --lr 0.003
```
Results:
- Generated 317 examples
- Pre-train loss: 1.4998
- Epoch 1/3: Train 1.4978, Val 1.5052
- Epoch 2/3: Train 1.4976, Val 1.5061
- Epoch 3/3: Train 1.4960, Val 1.5070
- Post-train loss: 1.4973

Interpretation: Training remains stable; train loss ticks down slightly. Validation slightly increases on this small sample; likely noise or overfitting on limited data. Next, increase data or regularization (smaller LR or weight decay tweak), and ensure policy masking is correct.

## Next steps
- Increase self-play games (e.g., 50–200) and epochs (e.g., 3–10) to confirm steady loss decrease.
- Consider smaller LR or gradient clipping if instability appears.
- Verify policy loss masking strictly matches legal moves in training batches.
- Add brief comments where head-only gradients are computed for clarity.

---

## 2025-08-07 — MCTS backprop fix + eval-only guard

What I changed
- Fixed MCTS backpropagation to update the entire node path (was only updating the first child). This substantially improves search quality.
- Allowed `nnue_sanity` to run in eval-only mode by skipping training when no examples or epochs=0.

How to reproduce
```
cargo run --bin nnue_sanity -q -- --games 0 --epochs 0 --eval-random 200 --eval-sims 10000
```

Result
- Eval vs random (3x3, 10k sims, 200 games): W-L-D = 188-0-12 (zero losses).

Next
- Verify/correct policy loss masking in `training.rs` (current goal).
- Optional: use network policy priors at expansion in MCTS for faster convergence (store per-child priors).
- Optional: gradient clipping on heads if needed.

## 2025-08-07 — Policy loss masking correction

What I changed
- Masked policy loss and gradients strictly to legal moves and re-normalized predicted policy over legal actions in `training.rs` (both train and eval paths). Numerical clamp added for stability.

How to reproduce
```
cargo run --bin nnue_sanity -q -- \
  --train-vs-random 200 --sp-sims 200 --epochs 2 --lr 0.002 --save checkpoints/e2.bin \
  --eval-mode greedy --eval-random 200 --eval-sims-low 32
```

Result (post-fix)
- Generated 649 examples
- Pre-train loss: 1.6670
- Epoch 1/2: Train 1.4077, Val 1.6173
- Epoch 2/2: Train 1.3537, Val 1.5698
- Post-train loss: 1.5644
- Eval[greedy] vs random (200): W-L-D = 103-53-44
- Eval[mcts-low:32] vs random (200): W-L-D = 167-3-30

Notes
- Loss scale changed due to masking with legal-move renormalization; values are not directly comparable to pre-fix runs but trend and eval strength improved.

Next
- Optional: add head-only gradient clipping for extra stability on noisy batches.

## 2025-08-07 — Head gradient clipping (L2)

What I changed
- Added L2 gradient clipping on output heads in `Trainer::update_weights` with `HEAD_GRAD_CLIP_NORM = 1.0`.

How to reproduce
```
cargo run --bin nnue_sanity -q -- \
  --train-vs-random 200 --sp-sims 200 --epochs 2 --lr 0.002 --save checkpoints/e2.bin \
  --eval-mode greedy --eval-random 200 --eval-sims-low 32
```

Result (with clipping)
- Generated 667 examples
- Pre-train loss: 1.8986
- Epoch 1/2: Train 1.6490, Val 1.9772
- Epoch 2/2: Train 1.6298, Val 1.9732
- Post-train loss: 1.8903
- Eval[greedy] vs random (200): W-L-D = 34-83-83
- Eval[mcts-low:32] vs random (200): W-L-D = 181-2-17

Notes
- Low-sims MCTS strength improved; greedy dipped on this seed. Likely sampling variance + different datasets between runs. Clip norm is conservative and can be tuned.

Next
- Optionally tune `HEAD_GRAD_CLIP_NORM` (e.g., 0.5–2.0) and/or increase epochs to 3–5 for more stable comparison.

## 2025-08-07 — Efficiency: augmentation + parallel vs-random + LR decay

Changes
- Enabled data augmentation in `nnue_sanity.rs` before training.
- Reduced self-play defaults: `SIMULATIONS=32`, `TEMP_MOVES=2`, `MAX_MOVES=BOARD_W*BOARD_H`.
- Parallelized `generate_vs_random_games` with rayon.
- Added simple LR decay in `Trainer::train`: halve LR after epoch 3 (logged per-epoch).

Runs
1) 400 vs-random, 3 epochs, lr=0.003 (baseline with aug)
   - 8226 examples (after aug)
   - Final train/val: 3.3058 / 3.6428
   - Eval[mcts]: 173-1-26; Eval[mcts-low:32]: 165-5-30

2) 800 vs-random, 3 epochs, lr=0.003 (parallel)
   - 16200 examples (after aug)
   - Final train/val: 3.2985 / 3.6605
   - Eval[mcts]: 183-0-17; Eval[mcts-low:32]: 174-5-21

3) 1400 vs-random, 5 epochs, lr=0.003 with decay->0.0015 @ epoch4
   - 29046 examples (after aug)
   - Final train/val: 3.1445 / 3.5796
   - Eval[mcts]: 171-1-28; Eval[mcts-low:32]: 171-13-16

4) 2000 vs-random, 3 epochs, lr=0.003 (no decay triggered with 3 epochs)
   - 41580 examples (after aug)
   - Final train/val: 3.1519 / 3.5439
   - Eval[mcts]: 135-4-61; Eval[mcts-low:32]: 132-23-45

Notes
- Parallelization increases data throughput; mcts-low improved notably in run (2).
- LR decay at epoch 4 didn’t help on this seed; mcts-low got worse in (3), likely data variance. More runs or fixed seeds recommended for fair compare.
- Run (4) underperformed; dataset composition variance suspected. Next: try 1200–1600 games, 4–5 epochs with decay, or switch to low-sims self-play data.
