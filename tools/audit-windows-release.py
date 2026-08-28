#!/usr/bin/env python
"""Audit a packaged Windows CLI archive before it is published.

This checks the artifact rather than trusting the CMake command that produced
it. Issue #374/#397 is the reason: v0.8.29's Windows CUDA workflow looked like a
CUDA build, but its delivered ggml-cpu.dll inherited AVX-512 from the GitHub
runner and SIGILL'd on an AVX2-only laptop.

Checks:
  * crispasr.exe embeds the expected release commit prefix;
  * a packaged ggml-cpu.dll contains no instructions wider than its baseline;
  * on Windows, a dynamic ggml-cpu.dll's exported build-feature predicates
    agree with the promised baseline.

The CPU-only CLI archives link ggml statically. Their executables may contain
runtime-dispatched implementations wider than the minimum baseline, so a raw
register scan would reject valid packages. The workflow's hermetic CMake audit
covers those builds; this artifact-level ISA check intentionally targets the
dynamic ggml-cpu.dll that caused #374/#397.

Usage:
  python tools/audit-windows-release.py archive.zip --expected-sha "$GITHUB_SHA" --baseline avx2
  python tools/audit-windows-release.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def _forbidden_registers(line: str, baseline: str) -> list[str]:
    prefixes = ("zmm",) if baseline == "avx2" else ("ymm", "zmm")
    found: list[str] = []
    lowered = line.lower()
    for prefix in prefixes:
        if re.search(rf"\b{prefix}(?:[0-9]|[12][0-9]|3[01])\b", lowered):
            found.append(prefix)
    return found


def _find_dumpbin() -> str | None:
    direct = shutil.which("dumpbin") or shutil.which("dumpbin.exe")
    if direct:
        return direct
    if os.name != "nt":
        return None

    roots = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")]
    for root in filter(None, roots):
        vswhere = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.is_file():
            continue
        proc = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-find",
             r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe"],
            capture_output=True, text=True, check=False,
        )
        candidates = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if candidates:
            return candidates[-1]
    return None


def _disassembler(path_override: str | None) -> tuple[str, list[str]]:
    if path_override:
        tool = path_override
    else:
        tool = _find_dumpbin() or shutil.which("llvm-objdump") or shutil.which("objdump")
    if not tool:
        raise RuntimeError("no disassembler found (need dumpbin, llvm-objdump, or objdump)")

    name = Path(tool).name.lower()
    if name.startswith("dumpbin"):
        return tool, ["/NOLOGO", "/DISASM"]
    return tool, ["-d", "-M", "intel"]


def _scan_isa(binary: Path, baseline: str, path_override: str | None) -> None:
    tool, args = _disassembler(path_override)
    proc = subprocess.Popen(
        [tool, *args, str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None
    hits: list[str] = []
    for line in proc.stdout:
        if _forbidden_registers(line, baseline) and len(hits) < 8:
            hits.append(line.strip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"disassembler failed for {binary} (rc={rc})")
    if hits:
        detail = "\n    ".join(hits)
        forbidden = "ZMM" if baseline == "avx2" else "YMM/ZMM"
        raise RuntimeError(
            f"{binary.name} violates the {baseline} artifact contract: found {forbidden} instructions\n"
            f"    {detail}"
        )
    print(f"PASS isa: {binary.name} contains no registers wider than {baseline}")


def _check_dynamic_features(root: Path, cpu_dll: Path, baseline: str) -> None:
    if os.name != "nt":
        return
    cookie = os.add_dll_directory(str(root)) if hasattr(os, "add_dll_directory") else None
    try:
        lib = ctypes.WinDLL(str(cpu_dll))
        if baseline == "avx2":
            expected = {"avx512": 0, "avx2": 1, "fma": 1, "f16c": 1}
        else:
            expected = {"avx512": 0, "avx2": 0, "fma": 0, "f16c": 0,
                        "bmi2": 0, "avx": 0}
        for feature, want in expected.items():
            fn = getattr(lib, f"ggml_cpu_has_{feature}")
            fn.restype = ctypes.c_int
            got = int(fn())
            if got != want:
                raise RuntimeError(
                    f"ggml-cpu.dll feature contract mismatch: {feature}={got}, expected {want}"
                )
        print(f"PASS features: ggml-cpu.dll reports the {baseline} baseline")
    finally:
        if cookie is not None:
            cookie.close()


def audit(archive: Path, expected_sha: str, baseline: str,
          disassembler: str | None = None) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{8,40}", expected_sha):
        raise RuntimeError("--expected-sha must be an 8-40 character hexadecimal Git SHA")
    with tempfile.TemporaryDirectory(prefix="crispasr-release-audit-") as tmp:
        out = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)

        executables = list(out.rglob("crispasr.exe"))
        if len(executables) != 1:
            raise RuntimeError(f"expected exactly one crispasr.exe, found {len(executables)}")
        exe = executables[0]
        root = exe.parent
        sha_prefix = expected_sha[:8].lower().encode("ascii")
        if sha_prefix not in exe.read_bytes().lower():
            raise RuntimeError(
                f"{exe.name} does not embed expected Git prefix {expected_sha[:8]}"
            )
        print(f"PASS provenance: crispasr.exe embeds {expected_sha[:8]}")

        cpu_dll = root / "ggml-cpu.dll"
        if cpu_dll.is_file():
            _scan_isa(cpu_dll, baseline, disassembler)
            _check_dynamic_features(root, cpu_dll, baseline)
            contract = f"dynamic CPU ISA is {baseline}"
        else:
            contract = "no dynamic ggml-cpu.dll (static ISA is checked at build configuration)"
            print(f"PASS layout: {contract}")
        print(
            f"RESULT: PASS — {archive.name} embeds {expected_sha[:8]}; {contract}"
        )


def self_test() -> None:
    assert _forbidden_registers("vmovups zmm0, [rax]", "avx2") == ["zmm"]
    assert _forbidden_registers("vmovups ymm15, [rax]", "avx2") == []
    assert _forbidden_registers("vmovups ymm15, [rax]", "legacy") == ["ymm"]
    assert _forbidden_registers("symbol named zmm_cache", "avx2") == []
    assert _forbidden_registers("vmovups xmm0, [rax]", "legacy") == []
    print("RESULT: PASS — release-audit parser self-test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--expected-sha")
    parser.add_argument("--baseline", choices=("avx2", "legacy"), default="avx2")
    parser.add_argument("--disassembler")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.archive is None or args.expected_sha is None:
                parser.error("archive and --expected-sha are required unless --self-test is used")
            audit(args.archive, args.expected_sha, args.baseline, args.disassembler)
        return 0
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"RESULT: FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
