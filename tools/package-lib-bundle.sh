#!/usr/bin/env bash
# package-lib-bundle.sh — make a libcrispasr bundle loadable and ergonomic.
#
# Usage: tools/package-lib-bundle.sh <bundle-dir>
#
# Run after the job has copied libs into <bundle>/src and <bundle>/ggml/src.
# Does three things, in order:
#
#   1. FLATTEN   every library into <bundle>/lib, leaving <bundle>/src and
#                <bundle>/ggml/src as symlinks so existing consumers still work.
#   2. RPATH     rewrite to loader-relative so the bundle resolves wherever it
#                is extracted (@loader_path on macOS, $ORIGIN on Linux).
#   3. VERIFY    delegate to verify-lib-bundle.sh, which relocates and dlopens.
#
# WHY FLATTEN. Downstream app bundles (Flutter/Xcode Frameworks, Android
# jniLibs, a Rust target dir) want one directory of libraries. Splitting them
# across src/ and ggml/src makes every consumer write a copy loop first.
# crispasr-sys/build.rs already probes BOTH <dir>/src/lib* and <dir>/lib*, so a
# flat dir is something it can consume today.
#
# WHY SYMLINKS RATHER THAN MOVING. release.yml has long told consumers to point
# CRISPASR_SYS_LIB_DIR at the extract dir, which resolves via <dir>/src/. Moving
# the files would break that silently. Real files live in lib/; src/ and
# ggml/src/ become symlinks to it, so both spellings work and nothing is
# duplicated.
#
# WHY RPATH IS NOT OPTIONAL. v0.8.18 shipped bundles that could not be loaded at
# all: macOS baked the CI runner's build tree into LC_RPATH, and Linux used
# $ORIGIN/../../ggml/src — one level too high, pointing outside the bundle. Both
# passed the old packaging gate because it only checked that dependencies were
# PRESENT. Presence is not resolvability.

set -euo pipefail

OUT="${1:?usage: package-lib-bundle.sh <bundle-dir>}"
[ -d "$OUT" ] || { echo "ERROR: no such bundle dir: $OUT" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "packaging lib bundle: $OUT"

# ── 1. flatten ──────────────────────────────────────────────────────────────
mkdir -p "$OUT/lib"
for d in "$OUT/src" "$OUT/ggml/src"; do
    [ -d "$d" ] || continue
    # -P: move symlinks as symlinks. The SOVERSION aliases (libcrispasr.so ->
    # libcrispasr.so.1) are relative, so they keep resolving in the flat dir.
    find "$d" -maxdepth 1 \( -type f -o -type l \) -name 'lib*' -exec mv {} "$OUT/lib/" \;
done
# Replace the old dirs with symlinks to lib/ (relative, so the bundle stays
# relocatable). Guard each rmdir: a non-empty dir means something unexpected is
# in there and silently discarding it would be worse than failing.
if [ -d "$OUT/src" ] && [ ! -L "$OUT/src" ]; then
    rmdir "$OUT/src" && ln -s lib "$OUT/src"
fi
if [ -d "$OUT/ggml/src" ] && [ ! -L "$OUT/ggml/src" ]; then
    rmdir "$OUT/ggml/src" && ln -s ../lib "$OUT/ggml/src"
fi
echo "  flattened into lib/ ($(find "$OUT/lib" -maxdepth 1 -type f | wc -l | tr -d ' ') files); src/ and ggml/src/ are symlinks"

# ── 2. rpath ────────────────────────────────────────────────────────────────
case "$(uname -s)" in
Darwin)
    for lib in "$OUT"/lib/*.dylib; do
        [ -f "$lib" ] || continue          # skip symlinks
        # Drop absolute entries. A stale absolute rpath that happens to exist on
        # a consumer's machine would silently load THAT copy instead of ours.
        otool -l "$lib" | awk '/LC_RPATH/{f=1} f&&/ path /{print $2; f=0}' |
            while read -r old; do
                case "$old" in
                    @*) : ;;
                    *) install_name_tool -delete_rpath "$old" "$lib" 2>/dev/null || true ;;
                esac
            done
        install_name_tool -add_rpath "@loader_path" "$lib" 2>/dev/null || true
        # Kept for bundles consumed through the src/ symlink, where @loader_path
        # is reported as the symlinked dir.
        install_name_tool -add_rpath "@loader_path/../lib" "$lib" 2>/dev/null || true
        # install_name_tool invalidates the signature and arm64 macOS refuses an
        # incorrectly-signed dylib, so re-sign ad-hoc.
        codesign --force --sign - "$lib" 2>/dev/null || true
    done
    echo "  rpaths → @loader_path (macOS), re-signed"
    ;;
Linux)
    # Copy the external dependency closure FIRST, while the libraries still
    # carry the RUNPATH that can find it (CrispASR #341, and #339 for why the
    # order is not a detail). The published HIP bundle needed libomp.so — ROCm
    # clang's OpenMP runtime, which lives in /opt/rocm/lib/llvm/lib and is on no
    # loader search path — and shipped without it, because nothing here ever
    # asked about a dependency the bundle did not already provide.
    #
    # Safe to point at the flattened lib/: anything the bundle already provides
    # is skipped by name, so the ggml sonames the libraries resolve through the
    # build tree are not re-copied over the SOVERSION symlinks.
    bash "$HERE/../scripts/bundle-linux-runtime.sh" "$OUT/lib"
    for lib in "$OUT"/lib/*.so*; do
        [ -f "$lib" ] || continue
        [ -L "$lib" ] && continue
        # Overwrites the plain $ORIGIN the bundler just set: a consumer reaching
        # the bundle through the src/ symlink sees $ORIGIN as the symlinked dir.
        patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib' "$lib" 2>/dev/null || true
    done
    echo "  rpaths → \$ORIGIN (Linux)"
    ;;
*)
    echo "  (no rpath step for $(uname -s))"
    ;;
esac

# ── 3. verify ───────────────────────────────────────────────────────────────
bash "$HERE/verify-lib-bundle.sh" "$OUT"
