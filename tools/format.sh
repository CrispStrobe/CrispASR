#!/usr/bin/env bash
# tools/format.sh — clang-format wrapper that ALWAYS uses v18.
#
# Why this script exists: CI pins clang-format-18 (.github/workflows/lint.yml)
# because version bumps between 14 / 18 / 20 / 22 silently flip wrapping
# decisions and cause "passed on my machine, failed in CI" drift. Running
# the system `clang-format` (which is v22 on a current Xcode / Homebrew
# install) WILL produce output that fails the CI lint step.
#
# Usage:
#   ./tools/format.sh                 # check-only, prints any violations
#   ./tools/format.sh --fix           # rewrite files in place
#   ./tools/format.sh src/foo.cpp     # operate on specific files only
set -euo pipefail

# Locate clang-format-18. Refuse to fall back to anything else — the whole
# point is to fail loudly when v18 isn't available, not silently use v22.
find_clang_format_18() {
    # macOS Homebrew (the recipe this project documents)
    local mac_path="/opt/homebrew/opt/llvm@18/bin/clang-format"
    [[ -x "$mac_path" ]] && { echo "$mac_path"; return 0; }
    # Linux / apt-pinned name
    if command -v clang-format-18 >/dev/null 2>&1; then
        command -v clang-format-18; return 0
    fi
    # Versioned binary on PATH
    if command -v clang-format >/dev/null 2>&1; then
        local v
        v=$(clang-format --version 2>/dev/null | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
        if [[ "$v" == "18" ]]; then
            command -v clang-format; return 0
        fi
    fi
    return 1
}

if ! CLANG_FMT=$(find_clang_format_18); then
    cat <<'EOF' >&2
error: clang-format-18 not found.

CI pins this project to clang-format-18; v17/v19/v20/v22 produce
different wrapping and cause "passed locally, failed in CI" drift.

Install:
  macOS:  brew install llvm@18
          (binary lands at /opt/homebrew/opt/llvm@18/bin/clang-format)
  Ubuntu: sudo apt install clang-format-18
EOF
    exit 2
fi

mode="check"
files=()
for arg in "$@"; do
    case "$arg" in
        --fix) mode="fix" ;;
        --check) mode="check" ;;
        --help|-h)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*) echo "error: unknown flag: $arg" >&2; exit 2 ;;
        *) files+=("$arg") ;;
    esac
done

if [[ ${#files[@]} -eq 0 ]]; then
    # Default scope mirrors lint.yml: project-owned C++ only.
    while IFS= read -r f; do files+=("$f"); done < <(
        find src examples/cli crisp_audio \
             \( -name '*.cpp' -o -name '*.h' -o -name '*.c' \) \
             ! -name 'httplib.h' \
             ! -name 'miniaudio.h' \
             ! -name 'json.hpp' \
             ! -name 'stb_vorbis.c' \
             ! -path '*/coreml/*'
    )
fi

if [[ "$mode" == "fix" ]]; then
    "$CLANG_FMT" -i "${files[@]}"
    echo "format.sh: rewrote ${#files[@]} file(s) with $("$CLANG_FMT" --version | head -1)"
else
    out=$("$CLANG_FMT" --dry-run --Werror "${files[@]}" 2>&1 || true)
    n=$(echo "$out" | grep -c 'error:.*clang-format' || true)
    if [[ "$n" -ne 0 ]]; then
        echo "$out" | grep 'error:.*clang-format' | head -50
        echo ""
        echo "format.sh: $n violation(s). Fix with: ./tools/format.sh --fix"
        exit 1
    fi
    echo "format.sh: OK ($("$CLANG_FMT" --version | head -1), ${#files[@]} files)"
fi
