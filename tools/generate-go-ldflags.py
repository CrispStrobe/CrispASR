#!/usr/bin/env python3
"""Generate CGO LDFLAGS for bindings/go/whisper.go from src/CMakeLists.txt.

Parses target_link_libraries(crispasr PUBLIC <lib>) lines and emits
the Go CGO LDFLAGS strings for linux and darwin.

Usage:
    python tools/generate-go-ldflags.py           # print to stdout
    python tools/generate-go-ldflags.py --check   # verify whisper.go is in sync
    python tools/generate-go-ldflags.py --update   # update whisper.go in place
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CMAKE = ROOT / "src" / "CMakeLists.txt"
GO_FILE = ROOT / "bindings" / "go" / "whisper.go"

# Libraries that are system/cmake targets, not our static libs
SKIP = {
    "ggml", "Threads::Threads", "winhttp", "crispasr.coreml",
    "crispasr.openvino", "MKL::MKL", "crispasr-llama-core",
    "CURL::libcurl",
}

# ggml libraries handled separately at the end of the link line
GGML_LIBS = ["ggml", "ggml-base", "ggml-cpu"]

# Transitive dependencies: libraries linked by backends (or their deps)
# that aren't direct crispasr deps. Go links all static libs flat — no
# CMake transitive resolution. Auto-detected from CMakeLists.txt with
# manual additions for second-level transitive deps.
_TRANSITIVE_MANUAL = ["crispasr-core", "crisp_audio", "moonshine_tokenizer"]

# GPU-specific libs that only exist when CUDA/Metal is enabled
_GPU_ONLY = {"ggml-cuda", "ggml-metal", "ggml-blas", "ggml-vulkan"}


def extract_libs() -> list[str]:
    """Extract PUBLIC link libraries from CMakeLists.txt."""
    text = CMAKE.read_text()
    libs = []
    for m in re.finditer(
        r"target_link_libraries\(crispasr\s+PUBLIC\s+([^)]+)\)", text
    ):
        for token in m.group(1).split():
            token = token.strip()
            if not token:
                continue
            # Skip cmake generator expressions, quoted strings, system targets
            if token.startswith("$") or token.startswith('"') or token.startswith("<"):
                continue
            if token in SKIP:
                continue
            # Must look like a library name (alphanumeric + hyphens/underscores)
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', token):
                continue
            libs.append(token)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for lib in libs:
        if lib not in seen:
            seen.add(lib)
            unique.append(lib)
    return unique


def extract_transitive() -> list[str]:
    """Find libraries linked by backends but not by crispasr directly."""
    text = CMAKE.read_text()
    direct = set(extract_libs())
    all_linked = set()
    lib_re = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')
    for m in re.finditer(
        r"target_link_libraries\(\s*\w+\s+PUBLIC\s+([^)]+)\)", text
    ):
        for token in m.group(1).split():
            token = token.strip()
            if token and lib_re.match(token) and token not in SKIP:
                all_linked.add(token)
    # Transitive = linked somewhere but not a direct crispasr dep
    transitive = sorted(all_linked - direct - set(GGML_LIBS) - _GPU_ONLY - {"crispasr"})
    # Add manual entries that auto-detection misses (2nd-level deps)
    for m in _TRANSITIVE_MANUAL:
        if m not in transitive:
            transitive.append(m)
    return transitive


def format_ldflags(libs: list[str], platform: str) -> str:
    """Format as a CGO LDFLAGS line."""
    trans = extract_transitive()
    flags = [f"-l{lib}" for lib in libs]
    flags.extend(f"-l{lib}" for lib in trans)
    flags.extend(f"-l{lib}" for lib in GGML_LIBS)
    if platform == "linux":
        return (
            "#cgo linux LDFLAGS: -Wl,--start-group -lcrispasr "
            + " ".join(flags)
            + " -Wl,--end-group -lm -lstdc++ -fopenmp"
        )
    else:  # darwin
        return (
            "#cgo darwin LDFLAGS: -lcrispasr "
            + " ".join(flags)
            + " -lm -lstdc++"
        )


def main():
    libs = extract_libs()
    linux_line = format_ldflags(libs, "linux")
    darwin_line = format_ldflags(libs, "darwin")

    if "--check" in sys.argv:
        go_text = GO_FILE.read_text()
        ok = True
        for lib in libs:
            flag = f"-l{lib}"
            if flag not in go_text:
                print(f"MISSING in whisper.go: {flag}")
                ok = False
        if ok:
            print("OK: all libraries present in whisper.go")
        else:
            print("\nExpected linux line:")
            print(linux_line)
            print("\nExpected darwin line:")
            print(darwin_line)
            sys.exit(1)

    elif "--update" in sys.argv:
        go_text = GO_FILE.read_text()
        lines = go_text.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("#cgo linux LDFLAGS: -Wl,--start-group"):
                new_lines.append(linux_line)
            elif line.startswith("#cgo darwin LDFLAGS: -lcrispasr"):
                new_lines.append(darwin_line)
            else:
                new_lines.append(line)
        GO_FILE.write_text("\n".join(new_lines))
        print(f"Updated {GO_FILE}")
        print(f"  {len(libs)} libraries from CMakeLists.txt")

    else:
        print("Libraries from CMakeLists.txt:")
        for lib in libs:
            print(f"  -l{lib}")
        print(f"\n{len(libs)} libraries total\n")
        print("Linux:")
        print(linux_line)
        print("\nDarwin:")
        print(darwin_line)


if __name__ == "__main__":
    main()
