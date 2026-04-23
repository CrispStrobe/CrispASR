#!/usr/bin/env bash
set -euo pipefail

ts() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(ts)] hf-space: $*" >&2
}

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
    cmd+=(-m auto --auto-download)
else
    cmd+=(-m "$MODEL_PATH")
fi

if [[ -n "$BACKEND" ]]; then
    cmd+=(--backend "$BACKEND")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    eval "cmd+=($EXTRA_ARGS)"
fi

log "startup"
log "server_host=$SERVER_HOST server_port=$SERVER_PORT backend=$BACKEND language=$LANGUAGE auto_download=$AUTO_DOWNLOAD"
if [[ "$AUTO_DOWNLOAD" == "1" ]]; then
    log "model=auto"
else
    log "model=$MODEL_PATH"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    log "extra_args=$EXTRA_ARGS"
fi
log "launching crispasr server: ${cmd[*]}"

"${cmd[@]}" &
server_pid=$!
log "crispasr server pid=$server_pid"

cleanup() {
    log "cleanup"
    if kill -0 "$server_pid" 2>/dev/null; then
        log "stopping crispasr server pid=$server_pid"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

log "launching gradio app"
python3 /space/app.py
