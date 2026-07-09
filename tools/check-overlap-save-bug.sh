#!/usr/bin/env bash
# Detect backends that have the Cohere-style "external overlap-save context
# breaks model-internal chunking" bug (PR #124 / issue #114 follow-up).
#
# A backend is AT RISK on paper if it doesn't declare CAP_UNBOUNDED_INPUT or
# CAP_INTERNAL_CHUNKING — meaning it goes through crispasr_run.cpp's fallback
# chunking + external overlap-save path. Whether it ACTUALLY exhibits the bug
# depends on whether its own transcribe() does additional chunking when handed
# a fallback-chunk-sized buffer.
#
# Empirical check: run the same long audio with default overlap (3.0s) vs
# --chunk-overlap 0. If the no-overlap pass produces materially more output
# (later last-timestamp or larger char count), the backend has the bug.

set -u

BIN="${BIN:-/Users/christianstrobele/code/CrispASR/build/bin/crispasr}"
AUDIO="${AUDIO:-/Volumes/backups/ai/crispasr-regression/issue-89/first300.wav}"
MODELS_DIR="${MODELS_DIR:-/Volumes/backups/ai/crispasr}"
OUT_DIR="${OUT_DIR:-/Volumes/backups/ai/bench-results/overlap-bug-check}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-900}"  # 15 min wallclock per run

mkdir -p "$OUT_DIR"

# (backend_arg, model_file_in_$MODELS_DIR, optional_extra_args)
# Only ASR backends in the AT-RISK bucket (no CAP_UNBOUNDED_INPUT, no
# CAP_INTERNAL_CHUNKING). Source of truth for "is this backend in the
# overlap-save opt-out" is `examples/cli/crispasr_chunk_context_gate.h`'s
# `backend_allows_chunk_context()` — currently blocks canary-qwen, cohere,
# gemma4-e2b, glm-asr, granite, kyutai-stt, moss-transcribe, qwen3, and voxtral.
# Those opt-out backends are KEPT in the sweep as regression-positive checks:
# their default and no-overlap runs
# should match because the gate skips the overlap-save wrap. If any
# of them ever flips back to a SUSPECTED-BUG verdict, the opt-out
# regressed.
#
# voxtral4b is the canonical control that the bug is per-architecture
# not per-name-family — same name prefix as voxtral, different model
# under the hood, passed the original A/B cleanly.
CASES=(
  # ── opt-out backends (regression-positive — should print OK) ──
  "canary-qwen|canary-qwen-2.5b-q8_0.gguf|"
  "cohere|cohere-transcribe-q4_k.gguf|"
  "gemma4-e2b|gemma4-e2b-it-q4_k.gguf|"
  "glm-asr|glm-asr-nano-q4_k.gguf|"
  "granite|granite-speech-4.1-2b-plus-q4_k.gguf|"
  "kyutai-stt|kyutai-stt-1b-q4_k.gguf|"
  "moss-transcribe|moss-transcribe-preview-2b-q8_0.gguf|"
  "qwen3|qwen3-asr-0.6b-q4_k.gguf|"
  "voxtral|voxtral-mini-3b-2507-q4_k.gguf|"
  # ── never-opted-out backends (sweep guards future regressions) ──
  "funasr|funasr-nano-2512-q4_k.gguf|"
  "granite-4.1|granite-speech-4.1-2b-q4_k.gguf|"
  "mimo-asr|mimo-asr-q4_k.gguf|"
  "moonshine|moonshine-base-q4_k.gguf|"
  "moonshine-streaming|moonshine-streaming-tiny-f32.gguf|"
  "omniasr|omniasr-ctc-1b-v2-q4_k.gguf|"
  "omniasr-llm|omniasr-llm-300m-v2-q4_k.gguf|"
  "sensevoice|sensevoice-small-q4_k.gguf|"
  "vibevoice|vibevoice-asr-7b-q4_k-fixed.gguf|"
  "voxtral4b|voxtral-mini-4b-realtime-q4_k.gguf|"
)

# Optional filter: pass backend names as args to run only those.
FILTER=("$@")
in_filter() {
  [ ${#FILTER[@]} -eq 0 ] && return 0
  for f in "${FILTER[@]}"; do [ "$f" = "$1" ] && return 0; done
  return 1
}

# srt → seconds-of-last-timestamp + line count
parse_srt() {
  local srt="$1"
  if [ ! -s "$srt" ]; then echo "0|0|0"; return; fi
  python3 - "$srt" <<'PY'
import re, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("0|0|0"); raise SystemExit
text = open(p, errors='replace').read()
times = re.findall(r'\d{2}:\d{2}:\d{2}[,.]\d{3} --> (\d{2}:\d{2}:\d{2})[,.]\d{3}', text)
if not times:
    print(f"0|0|{len(text)}"); raise SystemExit
def to_s(t):
    h,m,s = t.split(':'); return int(h)*3600+int(m)*60+int(s)
last = max(to_s(t) for t in times)
segs = text.count(' --> ')
chars = sum(len(l) for l in text.splitlines() if l and not l[0].isdigit() and ' --> ' not in l)
print(f"{last}|{segs}|{chars}")
PY
}

run_one() {
  local backend="$1" model="$2" extra="$3" label="$4" extra_flag="$5"
  local out="$OUT_DIR/${backend}.${label}"
  local srt="${out}.srt"
  local log="${out}.log"
  if [ -s "$srt" ]; then
    echo "    [$label] cached: $srt"
    return 0
  fi
  local cmd=(
    "$BIN" -m "$MODELS_DIR/$model" --backend "$backend"
    -f "$AUDIO" -of "$out" -osrt -np
  )
  [ -n "$extra_flag" ] && cmd+=($extra_flag)
  [ -n "$extra" ] && cmd+=($extra)
  echo "    [$label] $(date +%H:%M:%S) start"
  /usr/bin/time -lp gtimeout "$PER_RUN_TIMEOUT" "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    [$label] EXIT=$rc (timeout=$PER_RUN_TIMEOUT) — see $log"
  fi
  return $rc
}

printf "%-22s %-9s %-9s %-9s %-9s %-9s %-9s  %s\n" \
  "BACKEND" "DEF_LAST" "DEF_SEGS" "DEF_CHRS" "NO_LAST" "NO_SEGS" "NO_CHRS" "VERDICT"
printf -- "%.s-" {1..120}; echo

for entry in "${CASES[@]}"; do
  IFS='|' read -r backend model extra <<<"$entry"
  in_filter "$backend" || continue

  if [ ! -f "$MODELS_DIR/$model" ]; then
    printf "%-22s  MODEL_NOT_FOUND  %s\n" "$backend" "$model"
    continue
  fi

  echo "[$(date +%H:%M:%S)] $backend ($model)"
  run_one "$backend" "$model" "$extra" "default" ""
  run_one "$backend" "$model" "$extra" "nooverlap" "--chunk-overlap 0"

  read def_last def_segs def_chrs <<<"$(parse_srt "$OUT_DIR/${backend}.default.srt" | tr '|' ' ')"
  read no_last no_segs no_chrs <<<"$(parse_srt "$OUT_DIR/${backend}.nooverlap.srt" | tr '|' ' ')"

  verdict="OK"
  if [ "$def_chrs" = "0" ] && [ "$no_chrs" = "0" ]; then
    verdict="BOTH_EMPTY (check log)"
  else
    # Bug signal: no-overlap pass extends materially further OR has materially more text
    last_delta=$(( no_last - def_last ))
    chr_delta=$(( no_chrs - def_chrs ))
    abs_chr_delta=${chr_delta#-}
    if [ "$last_delta" -gt 30 ] || [ "$abs_chr_delta" -gt $(( no_chrs / 5 + 50 )) ] && [ "$chr_delta" -gt 0 ]; then
      verdict="*** SUSPECTED BUG (Δlast=${last_delta}s Δchars=+${chr_delta}) ***"
    fi
  fi
  printf "%-22s %-9s %-9s %-9s %-9s %-9s %-9s  %s\n" \
    "$backend" "$def_last" "$def_segs" "$def_chrs" "$no_last" "$no_segs" "$no_chrs" "$verdict"
done
