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
CACHE_DIR="${CRISPASR_CACHE_DIR:-/cache}"
EXTRA_ARGS="${CRISPASR_EXTRA_ARGS:-}"
API_KEYS="${CRISPASR_API_KEYS:-}"

ensure_writable_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        if ! mkdir -p "$dir" 2>/dev/null; then
            log "ERROR: could not create cache directory '$dir'"
            exit 70
        fi
    fi
    if [[ ! -w "$dir" ]]; then
        log "ERROR: cache directory '$dir' is not writable by uid=$(id -u) gid=$(id -g)"
        ls -ld "$dir" >&2 || true
        log "If this is a bind mount, chown the host directory or set CRISPASR_CACHE_DIR to a writable path."
        exit 70
    fi
}

ensure_writable_dir "$CACHE_DIR"

declare -a cmd
cmd=(crispasr --server --host "$SERVER_HOST" --port "$SERVER_PORT" -l "$LANGUAGE" --cache-dir "$CACHE_DIR")

if [[ "$AUTO_DOWNLOAD" == "1" ]]; then
    cmd+=(-m auto --auto-download)
else
    cmd+=(-m "$MODEL_PATH")
fi

if [[ -n "$BACKEND" ]]; then
    cmd+=(--backend "$BACKEND")
fi

if [[ -n "$API_KEYS" ]]; then
    cmd+=(--api-keys "$API_KEYS")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    eval "cmd+=($EXTRA_ARGS)"
fi

display_cmd=("${cmd[@]}")
if [[ -n "$API_KEYS" ]]; then
    for i in "${!display_cmd[@]}"; do
        if [[ "${display_cmd[$i]}" == "--api-keys" && $((i + 1)) -lt ${#display_cmd[@]} ]]; then
            display_cmd[$((i + 1))]="(redacted)"
        fi
    done
fi

log "startup"
log "server_host=$SERVER_HOST server_port=$SERVER_PORT backend=$BACKEND language=$LANGUAGE auto_download=$AUTO_DOWNLOAD cache_dir=$CACHE_DIR"
if [[ "$AUTO_DOWNLOAD" == "1" ]]; then
    log "model=auto"
else
    log "model=$MODEL_PATH"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    log "extra_args=$EXTRA_ARGS"
fi
if [[ -n "$API_KEYS" ]]; then
    log "api_keys=enabled"
fi
log "launching crispasr server: ${display_cmd[*]}"

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
