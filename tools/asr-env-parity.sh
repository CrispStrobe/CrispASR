#!/usr/bin/env bash
# asr-env-parity.sh
#
# Generic byte-parity harness for CrispASR environment-gated code paths.
#
# For every matching audio file, run:
#   baseline -> each test env individually -> all test envs together (when 2+ test envs)
# and compare stdout byte-for-byte against that file's baseline.
#
# The harness is intentionally backend-agnostic. It knows nothing about
# Nemotron, CUDA/Vulkan/Metal, or individual optimization activation logs.
# Everything after "--" is passed unchanged to the CrispASR command; the
# harness only appends "-f <audio>" for each input file.

set -u
set -o pipefail

SAMPLES="samples"
PATTERN="*.wav"
OUT=""
FAIL_FAST=0
RUN_LABEL="default"
TEST_ENVS=()
BASE_ENVS=()
UNSET_ENVS=()
CMD=()

usage() {
  cat <<'USAGE'
Usage:
  tools/asr-env-parity.sh [harness options] -- CRISPASR [crispasr args ...]

For each matching audio file, runs:
  baseline
  each -e/--env setting individually
  all -e/--env settings together (only when two or more -e settings are given)

stdout is compared byte-for-byte against the per-file baseline. stderr and
stdout from every run are retained for diagnosis.

Harness options (before --):
  -s, --samples DIR      Input directory (default: samples)
  -p, --pattern GLOB     File-name glob (default: *.wav)
  -o, --out DIR          Results directory (default: /tmp/asr-env-parity-$$)
  -n, --name NAME        Run/profile label shown in output (default: default)

  -e, --env SPEC         Test env. Repeatable.
                         SPEC may be NAME or NAME=VALUE.
                         NAME means NAME=1.

  -E, --base-env SPEC    Env applied to baseline and every variant. Repeatable.
                         SPEC may be NAME or NAME=VALUE.
                         NAME means NAME=1.

  -u, --unset NAME       Unset env in every run before applying -E/-e.
                         Repeatable. Test env names from -e are always unset
                         automatically before each run, preventing shell leakage.

      --fail-fast        Stop on first command error or parity mismatch
  -h, --help             Show this help

Everything after -- is passed unchanged to CrispASR. Do not pass -f/--file;
the harness appends -f <audio> itself.

Examples:
  # Generic single-gate test
  tools/asr-env-parity.sh \
    -e CRISPASR_SOME_OPT \
    -- ./build/bin/crispasr -m /path/model.gguf --lid-backend off

  # Several gates + a common execution mode
  tools/asr-env-parity.sh \
    -n streaming \
    -E CRISPASR_NEMOTRON_STREAMING=1 \
    -u CRISPASR_NEMOTRON_GPU_FASTPATH \
    -e CRISPASR_NEMOTRON_GPU_JOINT \
    -e CRISPASR_NEMOTRON_GPU_DIRECT_CONV \
    -e CRISPASR_NEMOTRON_GPU_STREAM_CACHE \
    -e CRISPASR_NEMOTRON_GPU_PROMPT \
    -- ./build-vulkan/bin/crispasr \
       --gpu-backend vulkan -dev 0 --lid-backend off -m /path/model.gguf

  # Same harness with another ASR backend (Parakeet example):
  # compare its default streamed path against forced single-pass encoding.
  tools/asr-env-parity.sh \
    -e CRISPASR_PARAKEET_STREAM_THRESHOLD=999 \
    -- ./build-vulkan/bin/crispasr \
       --backend parakeet -m /path/parakeet.gguf -l en
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

valid_name() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

# Normalize NAME -> NAME=1; preserve NAME=VALUE exactly.
normalize_spec() {
  local spec=$1 name
  if [[ "$spec" == *=* ]]; then
    name=${spec%%=*}
  else
    name=$spec
    spec="$spec=1"
  fi
  valid_name "$name" || die "invalid environment variable name in spec: $1"
  NORMALIZED_SPEC=$spec
}

spec_name() {
  printf '%s' "${1%%=*}"
}

safe_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

variant_label() {
  local spec=$1 name value
  name=$(spec_name "$spec")
  value=${spec#*=}
  if [[ "$value" == "1" ]]; then
    printf '%s' "$name"
  else
    printf '%s=%s' "$name" "$value"
  fi
}

transcribe_seconds() {
  local log=$1
  sed -n 's/.*transcribed .* audio in \([0-9][0-9.]*\)s.*/\1/p' "$log" | tail -n 1
}

shell_join() {
  local out=() x
  for x in "$@"; do
    printf -v x '%q' "$x"
    out+=("$x")
  done
  local IFS=' '
  printf '%s' "${out[*]}"
}

while (($#)); do
  case "$1" in
    -s|--samples)
      (($# >= 2)) || die "$1 needs a value"
      SAMPLES=$2; shift 2 ;;
    -p|--pattern)
      (($# >= 2)) || die "$1 needs a value"
      PATTERN=$2; shift 2 ;;
    -o|--out)
      (($# >= 2)) || die "$1 needs a value"
      OUT=$2; shift 2 ;;
    -n|--name)
      (($# >= 2)) || die "$1 needs a value"
      RUN_LABEL=$2; shift 2 ;;
    -e|--env)
      (($# >= 2)) || die "$1 needs a value"
      normalize_spec "$2"
      TEST_ENVS+=("$NORMALIZED_SPEC"); shift 2 ;;
    -E|--base-env)
      (($# >= 2)) || die "$1 needs a value"
      normalize_spec "$2"
      BASE_ENVS+=("$NORMALIZED_SPEC"); shift 2 ;;
    -u|--unset)
      (($# >= 2)) || die "$1 needs a value"
      valid_name "$2" || die "invalid environment variable name: $2"
      UNSET_ENVS+=("$2"); shift 2 ;;
    --fail-fast)
      FAIL_FAST=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      CMD=("$@")
      break ;;
    -*)
      die "unknown harness option: $1 (CrispASR args belong after --)" ;;
    *)
      die "unexpected argument before --: $1" ;;
  esac
done

[[ -d "$SAMPLES" ]] || die "samples directory not found: $SAMPLES"
((${#TEST_ENVS[@]} > 0)) || die "at least one -e/--env is required"
((${#CMD[@]} > 0)) || die "missing CrispASR command after --"
[[ -x "${CMD[0]}" ]] || die "CrispASR command is not executable: ${CMD[0]}"

# A test variable must represent one baseline-vs-variant dimension. Repeating
# the same name (possibly with different values) makes the aggregate "all"
# variant order-dependent, and using the same name as a base env means it is
# already enabled in the baseline. Reject both cases.
declare -A TEST_NAMES=()
for spec in "${TEST_ENVS[@]}"; do
  name=$(spec_name "$spec")
  [[ -z ${TEST_NAMES[$name]+x} ]] || die "duplicate test env name: $name"
  TEST_NAMES[$name]=1
done
for spec in "${BASE_ENVS[@]}"; do
  name=$(spec_name "$spec")
  [[ -z ${TEST_NAMES[$name]+x} ]] || die "env cannot be both --base-env and --env: $name"
done

# Avoid ambiguous double input selection. Exact parsing of every CrispASR alias
# is deliberately out of scope; catch the common forms.
for arg in "${CMD[@]:1}"; do
  case "$arg" in
    -f|--file|--file=*) die "do not pass $arg after --; the harness supplies -f <audio>" ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="/tmp/asr-env-parity-$$"
fi
mkdir -p "$OUT"

FILES=()
while IFS= read -r -d '' f; do
  FILES+=("$f")
done < <(find "$SAMPLES" -maxdepth 1 -type f -name "$PATTERN" -print0 | sort -z)
((${#FILES[@]} > 0)) || die "no files matching '$PATTERN' in $SAMPLES"

# Width for human-readable per-file output. Compute it from the actual variant
# labels so long ENV=VALUE names never run into the status column.
LABEL_WIDTH=${#RUN_LABEL}
(( LABEL_WIDTH < 8 )) && LABEL_WIDTH=8
for spec in "${TEST_ENVS[@]}"; do
  label=$(variant_label "$spec")
  ((${#label} > LABEL_WIDTH)) && LABEL_WIDTH=${#label}
done
if ((${#TEST_ENVS[@]} > 1)) && (( LABEL_WIDTH < 3 )); then
  LABEL_WIDTH=3
fi

# Build a deduplicated unset list. Every tested gate is scrubbed from the
# caller environment before baseline and each variant. Explicit -u entries are
# for aggregate/related controls that the generic harness cannot know about.
ALL_UNSETS=()
declare -A SEEN_UNSET=()
for spec in "${TEST_ENVS[@]}"; do
  name=$(spec_name "$spec")
  if [[ -z ${SEEN_UNSET[$name]+x} ]]; then
    ALL_UNSETS+=("$name")
    SEEN_UNSET[$name]=1
  fi
done
for name in "${UNSET_ENVS[@]}"; do
  if [[ -z ${SEEN_UNSET[$name]+x} ]]; then
    ALL_UNSETS+=("$name")
    SEEN_UNSET[$name]=1
  fi
done

{
  printf 'profile=%s\n' "$RUN_LABEL"
  printf 'samples=%s\n' "$SAMPLES"
  printf 'pattern=%s\n' "$PATTERN"
  printf 'file_count=%d\n' "${#FILES[@]}"
  printf 'command='; shell_join "${CMD[@]}"; printf '\n'
  printf 'test_envs='; shell_join "${TEST_ENVS[@]}"; printf '\n'
  printf 'base_envs='; shell_join "${BASE_ENVS[@]}"; printf '\n'
  printf 'unset_envs='; shell_join "${ALL_UNSETS[@]}"; printf '\n'
  printf 'cli_version='; "${CMD[0]}" --version 2>&1 | head -n 1 || true
} > "$OUT/meta.txt"

pass_count=0
fail_count=0
error_count=0

RUN_RC=0
RUN_STDOUT=""
RUN_STDERR=""
RUN_TSEC="-"

run_one() {
  local audio=$1 variant=$2
  shift 2
  local variant_envs=("$@")
  local sample_name variant_safe dir stdout stderr rc tsec

  sample_name=$(safe_name "$(basename "$audio")")
  variant_safe=$(safe_name "$variant")
  dir="$OUT/$sample_name"
  mkdir -p "$dir"
  stdout="$dir/$variant_safe.stdout"
  stderr="$dir/$variant_safe.stderr"

  local env_cmd=(env)
  local name
  for name in "${ALL_UNSETS[@]}"; do
    env_cmd+=(-u "$name")
  done
  env_cmd+=("${BASE_ENVS[@]}")
  env_cmd+=("${variant_envs[@]}")

  if "${env_cmd[@]}" "${CMD[@]}" -f "$audio" >"$stdout" 2>"$stderr"; then
    rc=0
  else
    rc=$?
  fi

  tsec=$(transcribe_seconds "$stderr")

  RUN_RC=$rc
  RUN_STDOUT=$stdout
  RUN_STDERR=$stderr
  RUN_TSEC=${tsec:--}
}

for audio in "${FILES[@]}"; do
  sample=$(basename "$audio")
  printf '\n[%s] %s\n' "$RUN_LABEL" "$sample"

  run_one "$audio" baseline
  if ((RUN_RC != 0)); then
    printf '  %-*s  %-5s  rc=%d  log=%s\n' "$LABEL_WIDTH" baseline ERROR "$RUN_RC" "$RUN_STDERR"
    ((error_count++))
    if ((FAIL_FAST)); then exit 1; fi
    continue
  fi

  baseline_stdout=$RUN_STDOUT
  printf '  %-*s  %-5s  transcribe=%ss\n' "$LABEL_WIDTH" baseline REF "$RUN_TSEC"

  for spec in "${TEST_ENVS[@]}"; do
    label=$(variant_label "$spec")
    run_one "$audio" "$label" "$spec"

    if ((RUN_RC != 0)); then
      status=ERROR
      diff_path=-
      ((error_count++))
      printf '  %-*s  %-5s  rc=%d  log=%s\n' "$LABEL_WIDTH" "$label" ERROR "$RUN_RC" "$RUN_STDERR"
    elif cmp -s "$baseline_stdout" "$RUN_STDOUT"; then
      status=PASS
      diff_path=-
      ((pass_count++))
      printf '  %-*s  %-5s  transcribe=%ss\n' "$LABEL_WIDTH" "$label" PASS "$RUN_TSEC"
    else
      status=FAIL
      diff_path="${RUN_STDOUT%.stdout}.diff"
      ((fail_count++))
      diff -u "$baseline_stdout" "$RUN_STDOUT" > "$diff_path" || true
      printf '  %-*s  %-5s  transcribe=%ss  diff=%s\n' "$LABEL_WIDTH" "$label" FAIL "$RUN_TSEC" "$diff_path"
    fi

    if ((FAIL_FAST)) && [[ "$status" != PASS ]]; then exit 1; fi
  done

  # The aggregate "all" case is meaningful only when there are multiple
  # independent test dimensions. With a single -e it would duplicate that
  # variant exactly and waste time.
  if ((${#TEST_ENVS[@]} > 1)); then
    run_one "$audio" all "${TEST_ENVS[@]}"
    if ((RUN_RC != 0)); then
      status=ERROR
      diff_path=-
      ((error_count++))
      printf '  %-*s  %-5s  rc=%d  log=%s\n' "$LABEL_WIDTH" all ERROR "$RUN_RC" "$RUN_STDERR"
    elif cmp -s "$baseline_stdout" "$RUN_STDOUT"; then
      status=PASS
      diff_path=-
      ((pass_count++))
      printf '  %-*s  %-5s  transcribe=%ss\n' "$LABEL_WIDTH" all PASS "$RUN_TSEC"
    else
      status=FAIL
      diff_path="${RUN_STDOUT%.stdout}.diff"
      ((fail_count++))
      diff -u "$baseline_stdout" "$RUN_STDOUT" > "$diff_path" || true
      printf '  %-*s  %-5s  transcribe=%ss  diff=%s\n' "$LABEL_WIDTH" all FAIL "$RUN_TSEC" "$diff_path"
    fi
    if ((FAIL_FAST)) && [[ "$status" != PASS ]]; then exit 1; fi
  fi
done

printf '\nParity variants: PASS=%d FAIL=%d ERROR=%d\n' "$pass_count" "$fail_count" "$error_count"
printf 'Results: %s\n' "$OUT"

if ((fail_count > 0 || error_count > 0)); then
  exit 1
fi
exit 0
