#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-harness/AGENT_PROMPT.md}"
LOG_DIR="${LOG_DIR:-}"
CODEX_MODEL="${CODEX_MODEL:-}"
CODEX_WORKDIR="${CODEX_WORKDIR:-.}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
MAX_LOOPS="${MAX_LOOPS:-0}" # 0 = infinite

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ -n "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
fi

iter=0
while true; do
  commit="$(git rev-parse --short=6 HEAD)"
  ts="$(date -u +%Y%m%d_%H%M%S)"
  if [[ -n "$LOG_DIR" ]]; then
    logfile="$LOG_DIR/agent_${ts}_${commit}.log"
    if [[ -n "$CODEX_MODEL" ]]; then
      codex exec --full-auto --cd "$CODEX_WORKDIR" --model "$CODEX_MODEL" - \
        < "$PROMPT_FILE" &> "$logfile"
    else
      codex exec --full-auto --cd "$CODEX_WORKDIR" - \
        < "$PROMPT_FILE" &> "$logfile"
    fi
  else
    if [[ -n "$CODEX_MODEL" ]]; then
      codex exec --full-auto --cd "$CODEX_WORKDIR" --model "$CODEX_MODEL" - \
        < "$PROMPT_FILE"
    else
      codex exec --full-auto --cd "$CODEX_WORKDIR" - \
        < "$PROMPT_FILE"
    fi
  fi

  iter=$((iter + 1))
  if [[ "$MAX_LOOPS" -gt 0 && "$iter" -ge "$MAX_LOOPS" ]]; then
    break
  fi

  sleep "$SLEEP_SECONDS"
done
