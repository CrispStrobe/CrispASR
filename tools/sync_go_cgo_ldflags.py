#!/usr/bin/env python
"""Sync the #cgo LDFLAGS library list in bindings/go/whisper.go with
the CMake dependency graph.

Usage:
    python tools/sync_go_cgo_ldflags.py [--check] [--dot /path/to/crispasr.dot]

Without --dot, runs cmake --graphviz to generate a fresh dot file.
With --check, exits non-zero if whisper.go would change (CI drift check).
Without --check, rewrites whisper.go in place.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WHISPER_GO = os.path.join(REPO_ROOT, "bindings", "go", "whisper.go")

# System libs appended after the CMake-derived list
SYSTEM_LIBS_LINUX = ["-lm", "-lstdc++", "-fopenmp"]
SYSTEM_LIBS_DARWIN = ["-lm", "-lstdc++"]

# Darwin-only extra libs (Metal/BLAS backends, not in the base dot on Linux)
DARWIN_EXTRA_LIBS = ["-lggml-metal", "-lggml-blas"]

# Targets that appear in the CMake dependency graph but should not be part of
# the generated cross-platform base list. The Go binding adds platform backend
# libraries separately so this check is stable across Linux and Darwin graphs.
EXCLUDE_LIBS = {"common", "ggml-metal", "ggml-blas"}

# CMake target names that differ from their OUTPUT_NAME. The linker
# sees the output filename (libcrispasr.a), not the cmake target name
# (crispasr-lib). Map target → output name here.
LIB_NAME_MAP = {"crispasr-lib": "crispasr"}


def generate_dot(dot_path, build_dir=None):
    """Run cmake --graphviz to produce the dependency dot file.

    The scratch build dir must be UNIQUE per invocation. It used to be a fixed
    `<tmp>/crispasr_graphviz_build`, which collides whenever two checkouts of
    this repo run the tool on one machine — several sessions plus git worktrees
    is the normal case here. Observed 2026-09-03: a stale cache left by a
    worktree made `--check` die with "The source ... does not match the source
    ... used to generate cache", which reads like an LDFLAGS drift failure and
    is not one. The worse version of the collision is silent: a cache from a
    tree cmake considers compatible gets REUSED, and the dot then describes a
    different backend set than the one being checked — a false verdict rather
    than an error.
    """
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="crispasr_graphviz_")
    cmd = [
        "cmake",
        "-S", REPO_ROOT,
        "-B", build_dir,
        "--graphviz", dot_path,
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCRISPASR_BUILD_TESTS=OFF",
        "-DCRISPASR_BUILD_EXAMPLES=OFF",
        "-DCRISPASR_BUILD_SERVER=OFF",
        "-DGGML_CUDA=OFF",
        "-DCRISPASR_OPUS_FETCH=ON",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def get_libs(dot_path):
    """Get the static lib list from the dot file."""
    sys.path.insert(0, SCRIPT_DIR)
    from cmake_graphviz_targets import get_static_libs

    libs = get_static_libs(dot_path, ["crispasr-lib"])
    return [LIB_NAME_MAP.get(lib, lib) for lib in libs if lib not in EXCLUDE_LIBS]


def build_linux_line(libs):
    """Build the #cgo linux LDFLAGS line."""
    lib_flags = " ".join(f"-l{lib}" for lib in libs)
    system = " ".join(SYSTEM_LIBS_LINUX)
    return (
        f"#cgo linux LDFLAGS: -Wl,--start-group {lib_flags} "
        f"-Wl,--end-group {system}"
    )


def build_darwin_line(libs):
    """Build the #cgo darwin LDFLAGS line (main lib list only)."""
    lib_flags = " ".join(f"-l{lib}" for lib in libs)
    system = " ".join(SYSTEM_LIBS_DARWIN)
    return f"#cgo darwin LDFLAGS: {lib_flags} {system}"


# Regex patterns matching the lines we replace
RE_LINUX = re.compile(r"^#cgo linux LDFLAGS:.*--start-group.*--end-group.*$")
RE_DARWIN_LIBS = re.compile(
    r"^#cgo darwin LDFLAGS: -l\w.*-lstdc\+\+$"
)


def sync(dot_path=None, check=False):
    if dot_path is None:
        tmp_build = tempfile.mkdtemp(prefix="crispasr_graphviz_")
        dot_path = os.path.join(tmp_build, "crispasr_sync.dot")
        print(f"Generating dot file at {dot_path} ...", file=sys.stderr)
        generate_dot(dot_path, build_dir=tmp_build)

    libs = get_libs(dot_path)
    if not libs:
        print("ERROR: no libraries found in dot file", file=sys.stderr)
        return 1

    new_linux = build_linux_line(libs)
    new_darwin = build_darwin_line(libs)

    with open(WHISPER_GO) as f:
        original = f.read()

    lines = original.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if RE_LINUX.match(line):
            if lines[i] != new_linux:
                lines[i] = new_linux
                changed = True
        elif RE_DARWIN_LIBS.match(line):
            if lines[i] != new_darwin:
                lines[i] = new_darwin
                changed = True

    if not changed:
        print("whisper.go LDFLAGS are up to date.", file=sys.stderr)
        return 0

    if check:
        print(
            "ERROR: whisper.go LDFLAGS are out of sync with CMake targets.\n"
            "Run: python tools/sync_go_cgo_ldflags.py\n"
            "to update the #cgo LDFLAGS lines automatically.",
            file=sys.stderr,
        )
        return 1

    result = "\n".join(lines)
    with open(WHISPER_GO, "w") as f:
        f.write(result)
    print(f"Updated {WHISPER_GO}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if whisper.go would change")
    parser.add_argument("--dot", help="Path to existing dot file (skip cmake)")
    args = parser.parse_args()
    sys.exit(sync(dot_path=args.dot, check=args.check))


if __name__ == "__main__":
    main()
