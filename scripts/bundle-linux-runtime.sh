#!/usr/bin/env bash
# bundle-linux-runtime.sh — make a staged Linux release directory self-contained.
#
# Two jobs, and THE ORDER IS PART OF THE CONTRACT:
#
#   1. Copy in every non-system shared library the binaries need.
#   2. Rewrite RUNPATH to `$ORIGIN` on every ELF in the directory.
#
# Both are required. Step 2 was the one originally missing: the published
# v0.8.25 artifact carried
#
#     crispasr           RUNPATH=$ORIGIN:/home/runner/work/CrispASR/.../c2pa/lib
#     crispasr-quantize  RUNPATH=/home/runner/work/CrispASR/.../c2pa/lib
#
# — a build-machine path that exists on no user's disk, and for
# `crispasr-quantize` no `$ORIGIN` at all, so it could not find the
# `libc2pa_c.so` sitting right beside it. Bundling a library the loader has no
# way to look for buys nothing, which is why this script does both.
#
# WHY RESOLVE BEFORE REWRITING. The first version did these in the opposite
# order, and that silently dropped exactly the libraries most in need of
# bundling. `ldd` resolves through the binary's own RUNPATH; erasing it first
# turns every such dependency into `=> not found`, which the copy loop then
# filtered out along with the blank lines. The HIP leg of v0.8.27 is what this
# cost: ROCm's clang links OpenMP against `libomp.so` in
# /opt/rocm-6.3.0/lib/llvm/lib, a directory reachable only via the RUNPATH this
# script had just erased, so it was never copied. (gcc's `libgomp.so.1` sits in
# the default loader path, which is why every non-HIP leg was unaffected and the
# bug stayed hidden.) Resolve first, rewrite second, and a dependency's
# discoverability no longer depends on this script's own side effects.
#
# Copied libraries are scanned at their ORIGINAL path, not at the staged copy,
# for the same reason: a `$ORIGIN`-relative RUNPATH means something different
# once the file has moved.
#
# WHY AN UNRESOLVED DEPENDENCY IS FATAL HERE. It used to be silent, and the
# script still reported how many libraries it had bundled — a green line over a
# missing one. Now anything the loader cannot find, and that is not on the
# excluded list below, stops the release. check-bundled-deps.py remains the
# authority on the finished directory (it reads DT_NEEDED rather than trusting
# ldd), but a failure there names a symptom; a failure here names the library
# the bundler could not reach.
#
# GPU runtimes are deliberately NOT bundled — libcuda/libcudart/libcublas,
# libamdhip64/librocblas and libvulkan belong to the host's driver or toolkit
# install. They are also legitimately absent from CI runners (no driver), so
# they are excluded from the fatal check as well. Pass them to
# check-bundled-deps.py with --allow so the contract is recorded rather than
# assumed.
#
# Usage: scripts/bundle-linux-runtime.sh <staged-dir>
set -euo pipefail

DEST="${1:-}"
[ -n "$DEST" ] && [ -d "$DEST" ] || { echo "bundle-linux-runtime: no dest dir '$DEST'" >&2; exit 1; }

command -v patchelf >/dev/null 2>&1 || {
    echo "bundle-linux-runtime: patchelf not installed — RUNPATH cannot be fixed" >&2; exit 1; }

is_elf() { head -c 4 "$1" 2>/dev/null | grep -q $'\x7fELF'; }

# One policy, consulted by both the copy loop and the unresolved-dependency
# check. Splitting them is how a library ends up excluded from the copy but
# still fatal, or bundled but not required.
skip_lib() {
    case "$1" in
        libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libutil.so.*) return 0 ;;
        libgcc_s.so.*|libstdc++.so.*|libresolv.so.*|ld-linux*) return 0 ;;
        libcuda.so.*|libcudart.so.*|libcublas*.so.*|libnv*.so.*) return 0 ;;
        libamdhip64.so.*|librocblas.so.*|libhsa*.so.*|libvulkan.so.*) return 0 ;;
    esac
    return 1
}

# ── 1. copy the non-system dependency closure ───────────────────────────────
# A work queue rather than a single pass: `ldd` gives the transitive closure of
# each file it is pointed at, but a library copied in during the sweep may pull
# in something the original binaries never named directly.
queue=()
while IFS= read -r f; do
    [ -L "$f" ] && continue
    is_elf "$f" || continue
    queue+=("$f")
done < <(find "$DEST" -maxdepth 1 -type f | sort)

copied=0
missing=""
i=0
while [ "$i" -lt "${#queue[@]}" ]; do
    f="${queue[$i]}"
    i=$((i + 1))
    ldd_out=$(ldd "$f" 2>/dev/null || true)

    while read -r base; do
        [ -n "$base" ] || continue
        skip_lib "$base" && continue
        case "$missing" in *" $base "*) continue ;; esac
        missing="$missing $base "
        echo "  MISSING $base  (needed by $(basename "$f"), unresolvable on this machine)" >&2
    done < <(printf '%s\n' "$ldd_out" | awk '/not found/ {print $1}' | sort -u)

    while read -r lib; do
        [ -n "$lib" ] || continue
        base=$(basename "$lib")
        skip_lib "$base" && continue
        [ -e "$DEST/$base" ] && continue
        cp -Lf "$lib" "$DEST/$base"
        echo "  bundle $base  (from $lib)"
        copied=$((copied + 1))
        # Scan the source, whose RUNPATH still means what it meant when the
        # library was built.
        queue+=("$lib")
    done < <(printf '%s\n' "$ldd_out" | awk '/=>/ {print $3}' | grep '^/' | sort -u)
done

if [ -n "$missing" ]; then
    echo "bundle-linux-runtime: cannot bundle the dependencies listed above." >&2
    echo "They are neither resolvable on this machine nor host-provided by contract." >&2
    exit 1
fi

# ── 2. RUNPATH -> $ORIGIN ────────────────────────────────────────────────────
# Unconditional, and last: an inherited build-tree RUNPATH is at best useless
# and at worst points somewhere that exists on the build machine only.
for f in "$DEST"/*; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    is_elf "$f" || continue
    before=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
    patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || {
        echo "bundle-linux-runtime: patchelf failed on $(basename "$f")" >&2; exit 1; }
    echo "  rpath  $(basename "$f"): '${before}' -> '\$ORIGIN'"
done

echo "bundle-linux-runtime: $DEST — $copied librar(ies) bundled, rpaths normalised"
