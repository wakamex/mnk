# Overnight Harness (Minimal)

Use one prompt plus one loop script.

Policy: prioritize code changes + detailed commit messages; avoid generating local experiment artifact files (command/log/CSV dumps).

## Files

- `harness/AGENT_PROMPT.md` - static research prompt
- `scripts/overnight_loop.sh` - restart loop that runs the agent and logs output

## Usage

```bash
bash scripts/overnight_loop.sh
```

Optional environment variables:

- `PROMPT_FILE` (default: `harness/AGENT_PROMPT.md`)
- `LOG_DIR` (default: empty = no loop logs; set e.g. `agent_logs` to enable)
- `CODEX_MODEL` (default: empty, uses codex default model/profile)
- `CODEX_WORKDIR` (default: `.`)
- `SLEEP_SECONDS` (default: `1`)
- `MAX_LOOPS` (default: `0`, meaning infinite)
