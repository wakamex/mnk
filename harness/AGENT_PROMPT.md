# Overnight Research Prompt

Goal: improve pure AlphaZero strength in `/code/mnk` until fixed-suite `vs_Deep >= 50%`.

Primary evaluation protocol:
- `./target/release/mnk_game --fixed-suite-eval --model-path <model>`
- Openings: 25
- Sides/opening: 2
- Eval sims: 100
- No root noise

Hard constraints:
- Primary metric must be neural inference + MCTS only.
- Do not use non-MCTS move logic (minimax/tactical/rule fallback) for reported primary results.
- Track `vs_Medium` and `vs_Random` regressions.

Loop:
1. Propose small hypothesis-driven changes.
2. Run training/eval as needed, but do not create local reproducibility artifact files (command/log/CSV dumps).
3. Update `EXPERIMENTS.md` with evidence and next step.
4. Commit only cohesive improvements.
5. Repeat.

Final output should include:
- Best `vs_Deep`, `vs_Medium`, `vs_Random`
- Best config and why
- Remaining blockers to `vs_Deep >= 50%`
- Detailed commit messages documenting what changed and measured outcomes.
