#!/usr/bin/env bash
# test-bundle-linux-runtime.sh — regression test for scripts/bundle-linux-runtime.sh.
#
# The bug this pins down shipped twice before anyone could see it, because the
# packaging scripts only ever ran inside a release job: `bundle-linux-runtime.sh`
# rewrote RUNPATH to $ORIGIN *before* asking `ldd` what the binaries needed, so
# any dependency reachable only through the binary's own RUNPATH became
# `=> not found` and was quietly filtered out with the blank lines. On the HIP
# leg that was `libomp.so` (ROCm's clang links OpenMP against LLVM's, which
# lives in /opt/rocm/lib/llvm/lib); the tarball failed to package at all, and
# only because check-bundled-deps.py happened to be downstream.
#
# Nothing here needs ROCm, or a GPU, or a release. A private directory plus
# -Wl,-rpath reproduces the exact condition: a library the loader can find only
# via the RUNPATH the bundler is about to erase.
#
# Skips (exit 77) anywhere the tools are missing, so it is inert on macOS.

set -euo pipefail

SKIP=77
[ "$(uname -s)" = "Linux" ] || { echo "SKIP: Linux-only (ELF/patchelf/ldd)"; exit $SKIP; }
for tool in cc patchelf ldd; do
    command -v "$tool" >/dev/null 2>&1 || { echo "SKIP: $tool not installed"; exit $SKIP; }
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$HERE/../scripts/bundle-linux-runtime.sh"
[ -f "$BUNDLE" ] || { echo "FAIL: no $BUNDLE"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

cat > "$WORK/dep.c" <<'EOF'
int crispasr_test_dep(void) { return 42; }
EOF
cat > "$WORK/app.c" <<'EOF'
#include <stdio.h>
int crispasr_test_dep(void);
int main(void) { printf("dep=%d\n", crispasr_test_dep()); return 0; }
EOF

# ── 1. a dependency reachable only via RUNPATH must be bundled ───────────────
# This is the regression. Pre-fix the bundler reported success and copied
# nothing, and the staged binary could not start once the private dir was gone.
priv="$WORK/private-lib"
stage="$WORK/stage"
mkdir -p "$priv" "$stage"
cc -shared -fPIC -o "$priv/libcrispasrtestdep.so" "$WORK/dep.c"
cc -o "$WORK/app" "$WORK/app.c" -L"$priv" -lcrispasrtestdep -Wl,-rpath,"$priv"
cp "$WORK/app" "$stage/app"

bash "$BUNDLE" "$stage" > "$WORK/bundle.log" 2>&1 || {
    cat "$WORK/bundle.log"; fail "bundler exited nonzero on a resolvable dependency"; }

[ -f "$stage/libcrispasrtestdep.so" ] || {
    cat "$WORK/bundle.log"
    fail "libcrispasrtestdep.so was not bundled — the RUNPATH-only dependency was dropped"; }

rp="$(patchelf --print-rpath "$stage/app")"
[ "$rp" = '$ORIGIN' ] || fail "staged app RUNPATH is '$rp', expected \$ORIGIN"

# The acceptance test is the binary starting with the original directory gone,
# not the file being present. v0.8.18 shipped bundles that had every dependency
# beside them and still could not load one.
mv "$priv" "$priv.gone"
out="$("$stage/app" 2>&1)" || { echo "$out"; fail "staged app does not run once its build-time libdir is gone"; }
[ "$out" = "dep=42" ] || fail "staged app printed '$out', expected 'dep=42'"
echo "  ok: RUNPATH-only dependency bundled, and the relocated binary runs"

# ── 2. a genuinely unresolvable dependency must fail the release ─────────────
# Without this arm arm 1 would also pass against a bundler that simply reports
# everything as fine.
priv2="$WORK/private-lib2"
stage2="$WORK/stage2"
mkdir -p "$priv2" "$stage2"
cc -shared -fPIC -o "$priv2/libcrispasrtestgone.so" "$WORK/dep.c"
cc -o "$WORK/app2" "$WORK/app.c" -L"$priv2" -lcrispasrtestgone -Wl,-rpath,"$priv2"
cp "$WORK/app2" "$stage2/app2"
rm -rf "$priv2"   # the library no longer exists anywhere

if bash "$BUNDLE" "$stage2" > "$WORK/bundle2.log" 2>&1; then
    cat "$WORK/bundle2.log"
    fail "bundler reported success with an unresolvable dependency"
fi
grep -q "libcrispasrtestgone.so" "$WORK/bundle2.log" || {
    cat "$WORK/bundle2.log"; fail "failure did not name the missing library"; }
echo "  ok: an unresolvable dependency fails the bundler, by name"

# ── 3. host-provided GPU runtimes must NOT fail it ───────────────────────────
# CI runners have no GPU driver, so `libcuda.so.1 => not found` is the normal
# state of the CUDA legs. If arm 2's check did not consult the same exclusion
# list the copy loop uses, this is where it would take the whole release down.
priv3="$WORK/private-lib3"
stage3="$WORK/stage3"
mkdir -p "$priv3" "$stage3"
cc -shared -fPIC -Wl,-soname,libcuda.so.1 -o "$priv3/libcuda.so.1" "$WORK/dep.c"
cc -o "$WORK/app3" "$WORK/app.c" "$priv3/libcuda.so.1" -Wl,-rpath,"$priv3"
cp "$WORK/app3" "$stage3/app3"
rm -rf "$priv3"

bash "$BUNDLE" "$stage3" > "$WORK/bundle3.log" 2>&1 || {
    cat "$WORK/bundle3.log"; fail "an absent host-provided runtime must not fail the bundler"; }
# `[ … ] && fail` would take the whole script down under `set -e` on the
# success path, since the AND-list itself then returns 1.
if [ -f "$stage3/libcuda.so.1" ]; then fail "libcuda.so.1 must never be bundled"; fi
echo "  ok: an absent host-provided runtime is tolerated and not bundled"

echo "PASS: bundle-linux-runtime"
