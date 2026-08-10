#!/usr/bin/env bash
# verify-lib-bundle.sh — prove a packaged libcrispasr bundle actually LOADS.
#
# Usage: tools/verify-lib-bundle.sh <bundle-dir>
#
# WHY THIS EXISTS
#
# release.yml already gated on "every @rpath/DT_NEEDED dependency is present in
# the bundle". That check passed for v0.8.18 and the bundles were still broken
# for every consumer, because presence is not resolvability:
#
#   macOS: LC_RPATH was the CI runner's build tree
#          (/Users/runner/work/CrispASR/CrispASR/build-libs/ggml/src), which
#          exists nowhere else, so @rpath/libggml.0.dylib resolved to nothing.
#   Linux: DT_RUNPATH was '$ORIGIN:$ORIGIN/../../ggml/src' — one level too high,
#          pointing outside the bundle. Relative, so it leaked no CI path and
#          looked correct.
#
# Both shipped. The dependency really was in the tarball; the loader simply
# could not find it.
#
# TWO CLASSES OF DEPENDENCY. A GPU bundle (CUDA/HIP) legitimately depends on the
# host DRIVER — libcuda.so.1, libamdhip64.so — which is NEVER bundled and is
# provided at runtime on the user's machine. A plain dlopen in CI, where no GPU
# driver is installed, fails on libcuda and tells you nothing about whether the
# bundle's OWN libs resolve. So the authoritative gate here is a static
# rpath-CLOSURE check: for every library the bundle ships, every dependency
# whose soname is ALSO shipped in the bundle must be reachable via that library's
# rpath. That is exactly the v0.8.18 bug and it needs no driver. dlopen is then a
# best-effort confirmation that tolerates a missing external driver.
#
# Exits non-zero on a real failure, so it fails the RELEASE instead of the
# consumer.

set -euo pipefail

BUNDLE="${1:?usage: verify-lib-bundle.sh <bundle-dir>}"
[ -d "$BUNDLE" ] || { echo "ERROR: no such bundle dir: $BUNDLE" >&2; exit 1; }

# Relocate. Loading in place can succeed by accident — a build-tree rpath still
# resolves on the machine that built it, which is exactly how this shipped.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -R "$BUNDLE" "$TMP/relocated"
ROOT="$TMP/relocated"

echo "verifying bundle rpath closure + load after relocation: $ROOT"

python3 - "$ROOT" <<'PY'
import ctypes, glob, os, re, subprocess, sys

root = sys.argv[1]
IS_MAC = sys.platform == "darwin"

# Driver / loader-provided sonames that may legitimately be absent here.
#
# ⚠ This list is a TOLERANCE, and the dlopen below `sys.exit(0)`s on the FIRST
# match — so anything named here stops the check rather than passing it. That is
# how #341 shipped: the HIP bundle's dlopen failed on `libomp.so`, matched this
# list, printed "dlopen deferred: external driver absent in CI" and exited 0,
# and the bundle went out needing a library that exists only under
# /opt/rocm/lib/llvm/lib. The claim that libomp/libgomp are "provided by the
# user's toolchain" holds for gcc's versioned libgomp.so.1 on the default loader
# path; it does not hold for ROCm clang's unversioned libomp.so.
#
# On Linux those two are now bundled (package-lib-bundle.sh copies the
# dependency closure), so they land in `provided` and the tolerance no longer
# applies to them — the `not in provided` guard below is what makes that work.
# They stay listed for macOS, where OpenMP really does come from the toolchain.
# Add nothing here without asking what a bundle would look like if the entry
# were wrong.
EXTERNAL = re.compile(
    r"^(libcuda|libamdhip64|libhsa-runtime|librocm|libnvidia|libcudart|libcublas|"
    r"libomp|libgomp|librt|libdl|libpthread|libm|libc|libstdc\+\+|libgcc_s|libSystem)\b"
)


def libdir():
    for sub in ("lib", "src"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            return d
    return None


LIBDIR = libdir()
if not LIBDIR:
    print("  ERROR: bundle has neither lib/ nor src/"); sys.exit(1)

# Every soname the bundle actually provides (real files, and symlink names).
provided = set()
for f in os.listdir(LIBDIR):
    provided.add(f)


def needed_and_rpaths(lib):
    """Return (list of dependency strings, list of rpath entries)."""
    if IS_MAC:
        out = subprocess.run(["otool", "-L", lib], capture_output=True, text=True).stdout
        deps = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        rp, cur = [], subprocess.run(["otool", "-l", lib], capture_output=True, text=True).stdout.splitlines()
        for i, ln in enumerate(cur):
            if "LC_RPATH" in ln:
                for j in range(i, min(i + 4, len(cur))):
                    m = re.search(r"\bpath (\S+)", cur[j])
                    if m:
                        rp.append(m.group(1)); break
        return deps, rp
    # Linux
    needed = subprocess.run(["patchelf", "--print-needed", lib], capture_output=True, text=True).stdout.split()
    rpath = subprocess.run(["patchelf", "--print-rpath", lib], capture_output=True, text=True).stdout.strip()
    return needed, [p for p in rpath.split(":") if p]


def expand(rp, lib):
    d = os.path.dirname(lib)
    return rp.replace("@loader_path", d).replace("$ORIGIN", d)


# ── Authoritative gate: rpath closure over bundle-internal deps ───────────────
concrete = [os.path.join(LIBDIR, f) for f in os.listdir(LIBDIR)
            if not os.path.islink(os.path.join(LIBDIR, f))
            and (f.endswith(".dylib") or ".so" in f)]
failures = []
for lib in concrete:
    deps, rpaths = needed_and_rpaths(lib)
    for dep in deps:
        base = os.path.basename(dep)
        # Only bundle-internal deps must resolve via rpath. A dep whose soname
        # the bundle does NOT ship is external (driver/system) — skip it.
        if base not in provided or EXTERNAL.match(base):
            continue
        if dep.startswith("/") or dep == os.path.basename(lib):
            continue  # absolute self-id or the library's own install name
        found = any(os.path.exists(os.path.join(expand(rp, lib), base)) for rp in rpaths)
        if not found:
            failures.append((os.path.basename(lib), base, rpaths))

if failures:
    print("  ERROR: bundle-internal dependency not reachable via rpath:")
    for who, dep, rps in failures:
        print(f"    {who} needs {dep}, but no rpath entry resolves it: {rps or '(no rpath)'}")
    print("  This is the v0.8.18 bug: the .so/.dylib is shipped but the loader")
    print("  cannot find it. Fix the rpath, do not ship.")
    sys.exit(1)
print(f"  rpath closure OK: every bundled dependency of {len(concrete)} libs resolves internally")

# ── Best-effort confirmation: actually dlopen ────────────────────────────────
pats = ("libcrispasr.*.dylib", "libcrispasr.so.*")
cands = [f for p in pats for f in glob.glob(os.path.join(LIBDIR, p)) if not os.path.islink(f)]
if not cands:
    print("  ERROR: no concrete libcrispasr in the bundle"); sys.exit(1)
lib = sorted(cands)[0]

try:
    h = ctypes.CDLL(lib)
except OSError as e:
    msg = str(e)
    m = re.search(r"(lib[\w.+-]+\.(?:so|dylib)[\w.]*)", msg)
    missing = m.group(1) if m else ""
    # Tolerate ONLY a missing external driver — the closure check above already
    # proved the internal libs resolve. Anything else is a real failure.
    if missing and EXTERNAL.match(os.path.basename(missing)) and os.path.basename(missing) not in provided:
        print(f"  dlopen deferred: external driver {missing} absent in CI "
              f"(expected for GPU bundles); internal rpath closure verified above.")
        sys.exit(0)
    print("  ERROR: dlopen FAILED on the packaged bundle:")
    for line in msg.splitlines()[:4]:
        print("   ", line[:300])
    sys.exit(1)

required = ["crispasr_session_open_explicit", "crispasr_session_close"]
absent = [s for s in required if not hasattr(h, s)]
if absent:
    print("  ERROR: loaded, but these symbols do not resolve:", ", ".join(absent))
    sys.exit(1)
print(f"  OK: {os.path.basename(lib)} dlopens after relocation; session ABI resolves")
PY
