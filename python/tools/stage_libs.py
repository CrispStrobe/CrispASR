#!/usr/bin/env python3
"""Stage a prebuilt libcrispasr bundle into the `crispasr` package for wheel
building.

The release workflow (`release.yml`) already builds relocatable
`libcrispasr-<platform>[-cuda|-vulkan].tar.gz` bundles whose libraries have
their rpaths rewritten to `$ORIGIN` / `@loader_path` (see
`tools/package-lib-bundle.sh`). This script copies those libraries next to
`crispasr/_binding.py` — where `_find_lib()` probes first — and best-effort
compiles the tiny `_helpers.c` shim, so that `python -m build` produces a
self-contained platform wheel.

Usage:
    python tools/stage_libs.py --bundle <extracted-bundle-dir> [--pkg crispasr]

Symlinks are materialised as real files: the wheel (a zip) cannot be relied on
to preserve symlinks across pip versions, and soname-based inter-library
resolution needs the versioned file to exist as real bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LIB_GLOBS = ("lib*.so", "lib*.so.*", "*.so", "*.dylib", "*.dll")
OPTIONAL_HELPERS = {"libcrispasr_helpers.so", "libcrispasr_helpers.dylib", "crispasr_helpers.dll"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# Shared objects that must NEVER be vendored into a wheel: the C runtime and its
# friends. Copying these in is how you get two libcs in one process.
LINUX_CORE = (
    "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
    "libgcc_s.so", "libstdc++.so", "ld-linux", "libresolv.so", "libutil.so",
    "libnsl.so", "libanl.so", "libcrypt.so",
)

# GPU driver + runtime libraries. These come from the HOST by definition — a
# wheel that shipped the user's NVIDIA driver would be both wrong and unusable —
# so they are neither vendored nor treated as a missing dependency. The GPU
# wheels carry a `+cuda` / `+vulkan` local version precisely because the user
# supplies this half.
#
# Getting this wrong is not hypothetical: failing on any unresolved dep is right
# for libopenblas and broke the previously-working linux-CUDA wheel, which
# legitimately has libcuda.so.1 / libcudart.so.12 / libcublas.so.12 unresolved on
# a runner with no CUDA installed.
LINUX_HOST_PROVIDED = (
    "libcuda.so", "libcudart.so", "libcublas", "libcufft", "libcurand",
    "libcusparse", "libcusolver", "libnvrtc", "libnvidia-", "libcudnn",
    "libamdhip", "libhip", "libhsa-runtime", "librocblas", "libroc",
    "libOpenCL.so", "libvulkan.so",
)


def find_lib_dir(bundle: Path) -> Path:
    # `bin` FIRST because that is where the Windows bundle puts its DLLs; `src`
    # holds the static import libs (src/Release/*.lib) which are not loadable, so
    # probing src before bin found a directory with no shared object in it and
    # bailed out — v0.8.24's three Windows wheels all died here with
    # "no shared libraries found".
    for sub in ("lib", "bin", "src", "."):
        d = bundle / sub if sub != "." else bundle
        if d.is_dir() and any(_is_lib(p) for p in d.iterdir()):
            return d
    raise SystemExit(f"stage_libs: no shared libraries found under {bundle}")


def _is_lib(p: Path) -> bool:
    n = p.name
    return (
        n.endswith(".dylib")
        or n.endswith(".dll")
        or n.endswith(".so")
        or ".so." in n
    )


def copy_libs(lib_dir: Path, pkg: Path, provenance=None) -> list[str]:
    copied = []
    for entry in sorted(lib_dir.iterdir()):
        if not _is_lib(entry):
            continue
        # Resolve symlinks to real bytes but keep the entry's own name, so both
        # `libcrispasr.so` (what _find_lib opens) and `libcrispasr.so.1` (what
        # the ggml backends' SONAME refers to) exist as loadable real files.
        src = entry.resolve()
        if not src.is_file():
            continue
        dst = pkg / entry.name
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)
        copied.append(entry.name)
        if provenance is not None:
            provenance[entry.name] = dict(kind="release-bundle", bundle_name=entry.name,
                                         source_sha256=sha(src), staged_sha256=sha(dst))
    if not copied:
        raise SystemExit(f"stage_libs: nothing copied from {lib_dir}")
    return copied


def vendor_linux_deps(pkg: Path, provenance=None,
                      extra_host: tuple[str, ...] = ()) -> tuple[list[str], list[str], list[str]]:
    """Copy the staged libraries' EXTERNAL dependencies into the package.

    The release bundle is relocatable but NOT self-contained: `libcrispasr.so`
    and `libwhisper.so` carry a DT_NEEDED on `libopenblas.so.0` (and on
    libespeak-ng / libfdk-aac / libasound) which the bundle does not ship. On a
    CI runner that happens to have them installed nothing looks wrong; on a
    user's machine `pip install crispasr` then fails at import with
    "libopenblas.so.0: cannot open shared object file" — exactly what v0.8.24's
    smoke test caught on linux-x86_64 and linux-arm64.

    This is auditwheel's job, done explicitly: walk `ldd` transitively, copy in
    everything that is not core-libc, and repoint each copy's RUNPATH at
    `$ORIGIN` so its own dependencies resolve from inside the wheel too (the
    loader uses each library's OWN RUNPATH, not the one that pulled it in — the
    easy thing to get wrong here).

    Returns (vendored, unresolved, host_provided). A non-empty `unresolved`
    means the wheel would be broken on any machine lacking those libraries; the
    caller fails. `host_provided` is reported but never fatal.
    """
    if not shutil.which("ldd"):
        print("stage_libs: no ldd, skipping dependency vendoring", flush=True)
        return [], [], []
    have_patchelf = shutil.which("patchelf") is not None
    if not have_patchelf:
        print("stage_libs: WARNING patchelf not found — vendored libraries will "
              "keep their original RUNPATH and may not find each other",
              flush=True)

    staged = {p.name for p in pkg.iterdir() if _is_lib(p)}
    queue = [p for p in pkg.iterdir() if _is_lib(p)]
    seen: set[str] = set()
    vendored: list[str] = []
    unresolved: list[str] = []
    host_provided: list[str] = []

    while queue:
        lib = queue.pop()
        try:
            out = subprocess.run(["ldd", str(lib)], capture_output=True,
                                 text=True, check=False).stdout
        except OSError:
            continue
        for line in out.splitlines():
            line = line.strip()
            if "=>" not in line:
                continue
            soname, _, rest = line.partition("=>")
            soname, rest = soname.strip(), rest.strip()
            if not soname or soname in seen:
                continue
            seen.add(soname)
            if any(soname.startswith(c) for c in LINUX_CORE) or soname in staged:
                continue
            if any(soname.startswith(c) for c in LINUX_HOST_PROVIDED + extra_host):
                host_provided.append(soname)
                continue
            if rest.startswith("not found"):
                unresolved.append(soname)
                continue
            path = rest.split(" (")[0].strip()
            if not path or not os.path.isfile(path):
                continue
            dst = pkg / soname
            shutil.copyfile(os.path.realpath(path), dst)
            shutil.copymode(os.path.realpath(path), dst)
            os.chmod(dst, os.stat(dst).st_mode | 0o200)  # copied libs are 0444
            if have_patchelf:
                subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(dst)],
                               check=False)
            if provenance is not None:
                provenance[soname] = dict(kind="installed-runtime", source_path=os.path.realpath(path),
                                         source_sha256=sha(Path(path)), staged_sha256=sha(dst))
            staged.add(soname)
            vendored.append(soname)
            queue.append(dst)
    return vendored, unresolved, host_provided


def compile_helpers(pkg: Path, include_dir: Path, extra_includes: list[Path]) -> None:
    """Compile the optional compatibility shim into the package directory.

    Current libraries export native by-reference wrappers, which the legacy
    `CrispASR` class prefers. Older libraries need this shim for those calls.
    The `Session` API does not use it. Compilation remains best-effort; the
    installed-wheel smoke test checks the current native exports separately.

    `extra_includes` matters more than it looks: `crispasr.h` line 4 includes
    `ggml.h`, which lives in the bundle's `ggml/include`, NOT next to
    `crispasr.h`. Passing only the latter made every wheel — on every platform —
    print "helpers compile failed: 'ggml.h' file not found" and ship without the
    legacy class. Because the failure is deliberately non-fatal it was a warning
    nobody read."""
    src = pkg / "_helpers.c"
    if not src.exists():
        print("stage_libs: no _helpers.c, skipping helpers", flush=True)
        return
    if not (include_dir / "crispasr.h").exists():
        print(f"stage_libs: no crispasr.h under {include_dir}, skipping helpers",
              flush=True)
        return
    inc_flags = [f"-I{include_dir}"] + [f"-I{p}" for p in extra_includes]
    system = platform.system()
    try:
        if system == "Windows":
            # cl needs the import lib; the bundle ships crispasr.lib alongside.
            implib = next(iter(pkg.glob("crispasr.lib")), None) or next(
                iter(include_dir.parent.rglob("crispasr.lib")), None)
            if implib is None:
                print("stage_libs: no crispasr.lib, skipping Windows helpers",
                      flush=True)
                return
            out = pkg / "crispasr_helpers.dll"
            subprocess.run(
                ["cl", "/nologo", "/LD", str(src)]
                + [f"/I{p}" for p in [include_dir] + extra_includes]
                + [str(implib), f"/Fe:{out}"],
                check=True,
            )
        else:
            ext = "dylib" if system == "Darwin" else "so"
            out = pkg / f"libcrispasr_helpers.{ext}"
            rpath = "@loader_path" if system == "Darwin" else "$ORIGIN"
            cc = os.environ.get("CC", "cc")
            subprocess.run(
                [cc, "-shared", "-fPIC", str(src)] + inc_flags
                + [f"-L{pkg}", "-lcrispasr", f"-Wl,-rpath,{rpath}", "-o", str(out)],
                check=True,
            )
            if system == "Darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(out)],
                               check=False)
        print(f"stage_libs: compiled helpers -> {out.name}", flush=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"stage_libs: WARNING helpers compile failed ({exc}); Session "
              "does not use this shim. Legacy Whisper requires native "
              "by-reference exports or this helper library", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path,
                    help="extracted libcrispasr-<platform> bundle directory")
    ap.add_argument("--pkg", type=Path, default=Path(__file__).parent.parent / "crispasr",
                    help="the crispasr package directory to stage into")
    ap.add_argument("--provenance", type=Path, help="write original/staged library identities for notice QA")
    ap.add_argument("--host-provided", action="append", default=[], metavar="SONAME_PREFIX",
                    help="extra variant-specific host runtime (e.g. ROCm's libomp.so for +hip); "
                         "never vendored, never treated as missing")
    args = ap.parse_args()

    bundle = args.bundle
    if not bundle.is_dir():
        raise SystemExit(f"stage_libs: bundle not found: {bundle}")
    pkg = args.pkg
    if not (pkg / "_binding.py").exists():
        raise SystemExit(f"stage_libs: {pkg} is not the crispasr package")

    lib_dir = find_lib_dir(bundle)
    provenance = {}
    copied = copy_libs(lib_dir, pkg, provenance)
    print(f"stage_libs: staged {len(copied)} libs from {lib_dir}:", flush=True)
    for n in copied:
        print(f"  {n}", flush=True)

    # crispasr.h includes ggml.h, which the bundle keeps in a separate tree.
    include_dir = bundle / "include"
    extra_includes = [d for d in (bundle / "ggml" / "include", bundle / "ggml")
                      if d.is_dir()]

    if platform.system() == "Linux":
        vendored, unresolved, host_provided = vendor_linux_deps(pkg, provenance, tuple(args.host_provided))
        if vendored:
            print(f"stage_libs: vendored {len(vendored)} external deps:", flush=True)
            for n in sorted(vendored):
                print(f"  {n}", flush=True)
        if host_provided:
            print("stage_libs: expected from the host (not vendored): "
                  + ", ".join(sorted(set(host_provided))), flush=True)
        if unresolved:
            # Failing here is the point. These are libraries the staged objects
            # NEED and neither the bundle nor this machine has; shipping the
            # wheel anyway just moves the ImportError to the user.
            print("stage_libs: ERROR unresolved dependencies — the wheel would "
                  "fail to import on any machine without them:", flush=True)
            for n in sorted(unresolved):
                print(f"  {n}", flush=True)
            raise SystemExit(1)

    compile_helpers(pkg, include_dir, extra_includes)
    for path in pkg.iterdir():
        if not _is_lib(path) or not path.is_file():
            continue
        identity = sha(path)
        if path.name not in provenance or provenance[path.name]["staged_sha256"] != identity:
            if path.name not in OPTIONAL_HELPERS:
                raise SystemExit(f"stage_libs: unexpected native payload change: {path.name}")
            provenance[path.name] = dict(kind="compatibility-helper", staged_sha256=identity,
                                         source_sha256=sha(pkg / "_helpers.c"))
    if args.provenance:
        args.provenance.write_text(json.dumps({"libraries": provenance}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
