#!/usr/bin/env python3
"""Exercise the actual CLI backend factory with lightweight constructor stubs."""
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("GITHUB_ACTIONS") != "true":
    raise RuntimeError("Hosted-only compiler fixture")
mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
if int(mem["MemAvailable"].split()[0]) < 3 * 1024**2 or os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
    raise RuntimeError("Insufficient hosted compiler headroom")
source = (ROOT / "examples/cli/crispasr_backend.cpp").read_text()
start = source.index("std::unique_ptr<CrispasrBackend> crispasr_create_backend(")
opening = source.index("{", start)
end, depth = opening + 1, 1
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
factory = source[start:end]
declarations = re.findall(r"std::unique_ptr<CrispasrBackend>\s+(crispasr_(?:make|create)_\w+)\s*\(([^;]*)\)\s*;", source[:start])
called = set(re.findall(r"\b(crispasr_(?:make|create)_\w+)\s*\(", factory)) - {"crispasr_create_backend"}
if called - {name for name, _ in declarations}:
    raise RuntimeError("Factory contains an unstubbed constructor")
parts = ["#include <cstdio>\n#include <memory>\n#include <string>\nclass CrispasrBackend {};\nint observed = -1;\n"]
for name, args in declarations:
    if name == "crispasr_make_qwen3_backend":
        parts.append(f"std::unique_ptr<CrispasrBackend> {name}({args}) {{ observed = context_enabled ? 1 : 0; return nullptr; }}\n")
    else:
        parts.append(f"std::unique_ptr<CrispasrBackend> {name}({args}) {{ return nullptr; }}\n")
parts.append(factory)
parts.append(r"""
int main() {
    const char* enabled[] = {"qwen3", "qwen3-1.7b", "qwen3_1.7b", "qwen3_17b"};
    const char* excluded[] = {"mega-asr", "mega_asr", "megaasr", "raon-speech", "raon-speech-9b", "raon_speech"};
    for (const char* name : enabled) { observed = -1; crispasr_create_backend(name); if (observed != 1) return 1; }
    for (const char* name : excluded) { observed = -1; crispasr_create_backend(name); if (observed != 0) return 2; }
    observed = -1; crispasr_create_backend("whisper"); if (observed != -1) return 3;
    return 0;
}
""")
work = Path(os.environ["HEAVY_SCRATCH"]) / "qwen3-factory-probe"
work.mkdir(parents=True, exist_ok=True)
fixture, binary = work / "factory.cpp", work / "factory"
fixture.write_text("\n".join(parts))
subprocess.run(["c++", "-std=c++17", "-O0", str(fixture), "-o", str(binary)], check=True, timeout=120)
subprocess.run([str(binary)], check=True, timeout=30)
print("Actual CLI factory: Qwen aliases enabled; Mega-ASR and Raon aliases excluded.")
