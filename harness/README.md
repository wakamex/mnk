# Overnight Research Harness

This harness prepares an autonomous research run package for Codex.

## What it creates

Given a branch and config, it writes:
- `research_runs/<timestamp>_<run-name>/codex_prompt.md`
- `research_runs/<timestamp>_<run-name>/runner_settings.md`
- `research_runs/<timestamp>_<run-name>/run_state.json`

## Why this exists

The project needs broad research loops (including code changes) to push
`vs_Deep` to 50% with controlled budgets and reproducible notes.

## Research goal guardrail

Primary objective is pure AlphaZero net strength:
- Only neural inference + MCTS move selection counts toward the target metric.
- Do not count runs that rely on minimax endgame solves, tactical override rules, or other non-MCTS move logic.

## Usage

```bash
python scripts/overnight_research_harness.py \
  --repo /code/mnk \
  --branch research/overnight-harness-20260207 \
  --run-name deep50 \
  --prepare --show-settings --print-prompt
```

## Permission model recommendation

- Use `workspace-write`, not full permissions, by default.
- Grant write access only to `/code/mnk`.
- Grant read-only access to `/code/alpha-zero` and `/code/AlphaZero_Gomoku`.
- Enable network access so web research is available.
