# Overnight Harness (Minimal)

Use one prompt plus one loop script.

## Files

- `harness/AGENT_PROMPT.md` - static research prompt
- `scripts/overnight_loop.sh` - restart loop that runs the agent and logs output

## Usage

```bash
bash scripts/overnight_loop.sh
```

Optional environment variables:

- `PROMPT_FILE` (default: `harness/AGENT_PROMPT.md`)
- `LOG_DIR` (default: `agent_logs`)
- `AGENT_BIN` (default: `claude`)
- `AGENT_MODEL` (default: `claude-opus-X-Y`)
- `SLEEP_SECONDS` (default: `1`)
- `MAX_LOOPS` (default: `0`, meaning infinite)
