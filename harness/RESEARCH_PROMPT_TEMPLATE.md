# Overnight Autonomous Research Prompt

You are running an autonomous research loop in `{{REPO_PATH}}` on branch `{{BRANCH_NAME}}`.

## Mission
Reach `{{PRIMARY_METRIC}} >= {{TARGET_PERCENT}}%` under this evaluation protocol:

{{EVAL_PROTOCOL}}

## Constraints
{{SECONDARY_CONSTRAINTS}}

For the primary metric, treat pure AlphaZero as mandatory: move choice must come from neural policy/value + MCTS only.

## What You Are Allowed To Do
- Run broad experiments across hyperparameters, training pipeline changes, and architecture choices.
- Propose and implement new code paths when they are the most promising route.
- Search the web for relevant papers, repos, and practical ideas, then test adapted variants locally.
- If exploring non-MCTS helper logic, report it separately as non-primary and do not use it to claim target hit.
- Compare implementation and training choices against:
{{REFERENCE_REPOS}}

## Research Space
{{RESEARCH_SPACE}}

## Operating Loop
1. Start with the best-known checkpoint/config and current notes.
2. Propose a small batch of experiments or code changes with explicit hypotheses.
3. Run experiments, collect metrics, and save reproducible artifacts (CSV/logs/commands).
4. Update notes with what changed, why, and what evidence supports the next step.
5. Commit only cohesive improvements on this branch.
6. Repeat until budget is reached or target is hit.

## Budget
{{BUDGETS}}

## Commit Policy (Required)
Every commit message must include these sections:
{{COMMIT_REQUIREMENTS}}

When a plan does not improve results, commit the failed but informative result if it narrows search space.

## Deliverable At End Of Run
- Best achieved metrics (vs_Random, vs_Deep, vs_Medium)
- Most promising configuration and why
- Code and experiment deltas from baseline
- Remaining blockers to hit `{{TARGET_PERCENT}}%` on `{{PRIMARY_METRIC}}`
