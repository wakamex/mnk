#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-harness/AGENT_PROMPT.md}"
LOG_DIR="${LOG_DIR:-agent_logs}"
AGENT_BIN="${AGENT_BIN:-claude}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-X-Y}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
MAX_LOOPS="${MAX_LOOPS:-0}" # 0 = infinite

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

iter=0
while true; do
  commit="$(git rev-parse --short=6 HEAD)"
  ts="$(date -u +%Y%m%d_%H%M%S)"
  logfile="$LOG_DIR/agent_${ts}_${commit}.log"

  "$AGENT_BIN" --dangerously-skip-permissions \
    -p "$(cat "$PROMPT_FILE")" \
    --model "$AGENT_MODEL" &> "$logfile"

  iter=$((iter + 1))
  if [[ "$MAX_LOOPS" -gt 0 && "$iter" -ge "$MAX_LOOPS" ]]; then
    break
  fi

  sleep "$SLEEP_SECONDS"
done
