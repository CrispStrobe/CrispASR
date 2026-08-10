#!/usr/bin/env bash
# test-package-lib-bundle.sh — regression test for tools/package-lib-bundle.sh.
#
# CrispASR #341. The published libcrispasr-linux-x86_64-hip bundle had
# `libomp.so` in DT_NEEDED and did not carry it: ROCm clang's OpenMP runtime,
# which lives in /opt/rocm/lib/llvm/lib and is on no loader search path. Nothing
# caught it, because `verify-lib-bundle.sh` gates INTRA-bundle resolvability —
# for every library the bundle ships, every dependency whose soname is ALSO
# shipped must be reachable via rpath — and a dependency absent from the bundle
# entirely is outside the question it asks.
#
# So this checks the other half: an external dependency reachable only through
# the build-time RUNPATH must end up in the bundle, and the bundle must still
# load once that directory is gone. Needs no GPU and no release.
#
# Skips (exit 77) anywhere the tools are missing, so it is inert on macOS —
# the packaging script's macOS branch is deliberately not changed by #341.

set -euo pipefail

SKIP=77
[ "$(uname -s)" = "Linux" ] || { echo "SKIP: Linux-only (ELF/patchelf/ldd)"; exit $SKIP; }
for tool in cc patchelf ldd python3; do
    command -v "$tool" >/dev/null 2>&1 || { echo "SKIP: $tool not installed"; exit $SKIP; }
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$HERE/../tools/package-lib-bundle.sh"
CHECK="$HERE/../scripts/check-bundled-deps.py"
[ -f "$PKG" ] || { echo "FAIL: no $PKG"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

cat > "$WORK/ext.c" <<'EOF'
int crispasr_test_ext(void) { return 7; }
EOF
# verify-lib-bundle.sh insists the concrete libcrispasr dlopens AND exports the
# session ABI, so the stand-in has to carry those two symbols. That check is the
# point of the script and is not worth weakening for a test.
cat > "$WORK/main.c" <<'EOF'
int crispasr_test_ext(void);
int crispasr_test_main(void) { return crispasr_test_ext(); }
void *crispasr_session_open_explicit(void) { return 0; }
void crispasr_session_close(void *s) { (void)s; }
EOF
cat > "$WORK/ggml.c" <<'EOF'
int crispasr_test_ggml(void) { return 1; }
EOF

# A bundle shaped the way release.yml stages one: real libraries in src/ and
# ggml/src/, SOVERSION symlinks beside them, and one dependency living outside
# in a directory only the RUNPATH knows about.
OUT="$WORK/bundle"
priv="$WORK/toolchain-lib"
mkdir -p "$OUT/src" "$OUT/ggml/src" "$priv"

cc -shared -fPIC -o "$priv/libcrispasrtestext.so" "$WORK/ext.c"
cc -shared -fPIC -Wl,-soname,libcrispasr.so.1 \
   -o "$OUT/src/libcrispasr.so.1.0.0" "$WORK/main.c" \
   -L"$priv" -lcrispasrtestext -Wl,-rpath,"$priv"
ln -s libcrispasr.so.1.0.0 "$OUT/src/libcrispasr.so.1"
ln -s libcrispasr.so.1     "$OUT/src/libcrispasr.so"
cc -shared -fPIC -o "$OUT/ggml/src/libcrispasrtestggml.so" "$WORK/ggml.c"

bash "$PKG" "$OUT" > "$WORK/pkg.log" 2>&1 || { cat "$WORK/pkg.log"; fail "packaging failed"; }

# 1. the external dependency is in the bundle
[ -f "$OUT/lib/libcrispasrtestext.so" ] || {
    cat "$WORK/pkg.log"
    fail "libcrispasrtestext.so was not bundled — the RUNPATH-only dependency was dropped"; }

# 2. the existing contract is intact: flattened, with src/ and ggml/src/ as
#    symlinks, and the SOVERSION aliases still resolving
[ -L "$OUT/src" ] || fail "src/ is not a symlink to lib/"
[ -L "$OUT/ggml/src" ] || fail "ggml/src/ is not a symlink to lib/"
[ -f "$OUT/lib/libcrispasr.so.1.0.0" ] || fail "the real library did not reach lib/"
[ -e "$OUT/lib/libcrispasr.so.1" ] || fail "SOVERSION alias did not survive the flatten"
[ -f "$OUT/lib/libcrispasrtestggml.so" ] || fail "the ggml library did not reach lib/"

# 3. nothing unbundled is left over. This is the gate release.yml now runs.
rm -rf "$priv"
python3 "$CHECK" "$OUT/lib" > "$WORK/check.log" 2>&1 || {
    cat "$WORK/check.log"; fail "check-bundled-deps rejects the packaged bundle"; }

# 4. and it loads, relocated, with the build-time toolchain dir deleted —
#    presence is not resolvability (v0.8.18 shipped bundles that had every
#    dependency beside them and one unreachable).
moved="$WORK/relocated"
cp -R "$OUT" "$moved"
python3 - "$moved" <<'PY' || exit 1
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1] + "/lib/libcrispasr.so.1")
lib.crispasr_test_main.restype = ctypes.c_int
v = lib.crispasr_test_main()
if v != 7:
    print(f"FAIL: relocated bundle returned {v}, expected 7"); sys.exit(1)
print("  ok: relocated bundle dlopens and calls through to the bundled dependency")
PY

echo "PASS: package-lib-bundle"
