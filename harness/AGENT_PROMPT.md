# Overnight Autonomous Research Prompt

You are running an autonomous research loop in `/code/mnk`.

## Mission

Reach `vs_Deep >= 50%` under this evaluation protocol:

- Deterministic fixed suite
- Openings: 25
- Sides per opening: 2
- Total games: 50
- Root noise: disabled
- Eval sims: 100

## Hard Constraints

- Primary metric must use pure AlphaZero move selection only:
  - neural inference + MCTS
- Do not use non-MCTS move logic for primary metrics:
  - minimax endgame solve
  - tactical override rules
  - handcrafted fallback shortcuts
- Track regressions vs `vs_Medium` and `vs_Random`.

## Operating Loop

1. Start from the current best checkpoint and notes.
2. Propose a small experiment batch with explicit hypotheses.
3. Run experiments and keep reproducible command/log/CSV outputs.
4. Update notes with what changed and what evidence was observed.
5. Commit cohesive improvements only.
6. Repeat until target is reached or budget is exhausted.

## End-of-Run Deliverable

- Best `vs_Deep`, `vs_Medium`, `vs_Random`
- Best config and why
- Code and experiment deltas from baseline
- Remaining blockers to `vs_Deep >= 50%`
