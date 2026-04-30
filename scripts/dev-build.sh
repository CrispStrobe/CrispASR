#!/usr/bin/env bash
# dev-build.sh — fast iterative build of crispasr with platform-aware
# optimizations. Picks Ninja, ccache, OpenMP, and (where available)
# the mold linker. Cross-platform via uname.
#
# First run from a fresh checkout:    scripts/dev-build.sh
# Re-configure from scratch:          scripts/dev-build.sh --reconfigure
# Build a different target:           scripts/dev-build.sh --target crispasr-quantize
# Pass extra cmake args:              scripts/dev-build.sh -DGGML_VULKAN=ON
#
# All extra args are passed to cmake configure (when reconfiguring) or
# straight through if --target is recognized.

set -e

cd "$(dirname "$0")/.."

OS="$(uname -s)"
RECONFIGURE=0
TARGET=crispasr
CMAKE_EXTRA=()

# Argument parsing — capture flags we care about, pass the rest to cmake.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reconfigure) RECONFIGURE=1; shift ;;
    --target)      TARGET="$2"; shift 2 ;;
    *)             CMAKE_EXTRA+=("$1"); shift ;;
  esac
done

# Per-platform tuning. Only flags that the platform's toolchain
# actually supports are added — others stay at cmake's autodetected
# defaults so the script never makes things worse than a vanilla cmake.
ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCRISPASR_BUILD_TESTS=OFF
)

case "$OS" in
  Darwin)
    # Apple clang needs OpenMP_ROOT to find Homebrew's libomp.
    if [ -d /opt/homebrew/opt/libomp ]; then
      ARGS+=(-DOpenMP_ROOT=/opt/homebrew/opt/libomp)
    elif [ -d /usr/local/opt/libomp ]; then
      ARGS+=(-DOpenMP_ROOT=/usr/local/opt/libomp)
    fi
    # mold doesn't ship a Mach-O linker — leave LINKER_TYPE at default (Apple ld).
    ;;
  Linux)
    # mold is much faster than GNU ld for big projects. Use when present.
    if command -v mold >/dev/null; then
      ARGS+=(-DCMAKE_LINKER_TYPE=MOLD)
    fi
    # OpenMP/BLAS find themselves on Linux without further hints.
    ;;
esac

# (Re)configure if no build dir, --reconfigure was passed, or extra args were given.
if [ ! -d build ] || [ "$RECONFIGURE" = 1 ] || [ ${#CMAKE_EXTRA[@]} -gt 0 ]; then
  rm -rf build
  echo "Configuring with: ${ARGS[*]} ${CMAKE_EXTRA[*]}"
  cmake -S . -B build "${ARGS[@]}" "${CMAKE_EXTRA[@]}"
fi

# Build. Ninja parallelises automatically; -j with no number uses all cores.
exec cmake --build build --target "$TARGET" -j
