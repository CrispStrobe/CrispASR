#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${CRISPASR_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${CRISPASR_SERVER_PORT:-8080}"
MODEL_PATH="${CRISPASR_MODEL:-/models/model.gguf}"
LANGUAGE="${CRISPASR_LANGUAGE:-en}"
BACKEND="${CRISPASR_BACKEND:-whisper}"
AUTO_DOWNLOAD="${CRISPASR_AUTO_DOWNLOAD:-1}"
EXTRA_ARGS="${CRISPASR_EXTRA_ARGS:-}"

declare -a cmd
cmd=(crispasr --server --host "$SERVER_HOST" --port "$SERVER_PORT" -l "$LANGUAGE")

if [[ "$AUTO_DOWNLOAD" == "1" ]]; then
    cmd+=(-m auto)
else
    cmd+=(-m "$MODEL_PATH")
fi

if [[ -n "$BACKEND" ]]; then
    cmd+=(--backend "$BACKEND")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    eval "cmd+=($EXTRA_ARGS)"
fi

"${cmd[@]}" &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

python3 /space/app.py
