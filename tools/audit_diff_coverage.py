#!/usr/bin/env python3
"""
Audit `crispasr-diff` coverage across all registered reference backends.

Reports, per backend:
  - whether a Python reference backend module exists (always YES if registered)
  - whether a frozen reference archive (.gguf) exists in the conventional
    locations under /Volumes/backups/ai/ or the project's tools/ref-dumps/
  - whether the C++ diff harness in examples/cli/crispasr_diff_main.cpp has
    a `backend_name == "<backend>"` branch wired up
  - the heavy python dependencies the backend imports (transformers, torch,
    upstream model packages) so a reproducible env can be set up

Output is a markdown table written to docs/diff-harness-coverage.md, plus a
summary printed to stdout. Run periodically to spot stale archives, missing
harness branches, and uncovered backends.

Usage:
    python tools/audit_diff_coverage.py                       # write doc + print
    python tools/audit_diff_coverage.py --print-only          # don't write doc
    python tools/audit_diff_coverage.py --emit-env-scaffolds  # also write
                                                              # tools/reference_envs/<name>/requirements.txt
                                                              # for any backend that doesn't have one
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

# Make tools/ a package import path so we can pull REGISTERED_BACKENDS
# without executing the full dump driver.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

# We don't import dump_reference (it has heavy import-time work); re-parse the
# REGISTERED_BACKENDS dict directly.
DUMP_DRIVER = HERE / "dump_reference.py"


def parse_registered_backends() -> dict[str, str]:
    text = DUMP_DRIVER.read_text()
    m = re.search(r"REGISTERED_BACKENDS:\s*Dict\[str,\s*str\]\s*=\s*\{(.*?)^\}", text,
                  re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError("could not locate REGISTERED_BACKENDS in dump_reference.py")
    body = m.group(1)
    out: dict[str, str] = {}
    # Match  "name":  "reference_backends.module",
    for key, mod in re.findall(r'"([\w.\-]+)"\s*:\s*"([\w.]+)"', body):
        out[key] = mod
    return out


# Conventional locations for frozen reference archives.
ARCHIVE_ROOTS = [
    Path("/Volumes/backups/ai"),
    Path("/Volumes/backups/ai/crispasr"),
    Path("/Volumes/backups/ai/crispasr/ref-dumps"),
    REPO / "tools" / "ref-dumps",
]

# Map backend name → list of expected archive filenames (basenames). For
# backends whose canonical archive name isn't <name>-ref.gguf, fall through
# to a permissive glob.
ARCHIVE_HINTS = {
    "chatterbox": ["chatterbox-ref.gguf", "chatterbox-base-ref.gguf"],
    "chatterbox_turbo": ["chatterbox-turbo-ref.gguf", "chatterbox_turbo-ref.gguf"],
    "qwen3-tts": ["qwen3-tts-ref.gguf"],
    "qwen3-tts-codec": ["qwen3-tts-codec-ref.gguf"],
    "qwen3-tts-spk": ["qwen3-tts-spk-ref.gguf"],
    "qwen3-tts-cenc": ["qwen3-tts-cenc-ref.gguf"],
    "vibevoice": ["vibevoice-ref-jfk.gguf", "vibevoice-ref.gguf"],
    "mimo-asr": ["mimo-asr-ref.gguf"],
    "mimo-tokenizer": ["mimo-tokenizer-ref.gguf"],
    "orpheus": ["orpheus-snac-ref.gguf"],
    "canary": ["canary-ref.gguf"],
    "cohere": ["cohere-ref.gguf"],
    "parakeet": ["parakeet-ref.gguf"],
    "kokoro": ["kokoro-ref.gguf"],
    "voxtral": ["voxtral-ref.gguf"],
    "voxtral4b": ["voxtral4b-ref.gguf"],
    "qwen3": ["qwen3-ref.gguf"],
    "granite": ["granite-ref.gguf"],
    "granite-4.1": ["granite-4.1-ref.gguf", "granite-ref.gguf"],
    "granite-nle": ["granite-nle-ref.gguf"],
    "gemma4": ["gemma4-ref.gguf"],
    "moonshine": ["moonshine-ref.gguf"],
    "moonshine-base": ["moonshine-base-ref.gguf"],
    "moonshine-streaming": ["moonshine-streaming-ref.gguf"],
    "voxcpm2-tts": ["voxcpm2-tts-ref.gguf"],
    "indextts": ["indextts-ref.gguf"],
    "lid-cld3": ["lid-cld3-ref.gguf"],
    "lid-glotlid": ["lid-glotlid-ref.gguf"],
    "lid-fasttext176": ["lid-fasttext176-ref.gguf"],
    "titanet": ["titanet-ref.gguf"],
    "glm-asr": ["glm-asr-ref.gguf"],
    "firered-asr": ["firered-asr-ref.gguf"],
    "funasr": ["funasr-ref.gguf"],
    "paraformer": ["paraformer-ref.gguf"],
    "sensevoice": ["sensevoice-ref.gguf"],
}


def find_archive(backend: str) -> Path | None:
    hints = ARCHIVE_HINTS.get(backend, [f"{backend}-ref.gguf"])
    for root in ARCHIVE_ROOTS:
        if not root.exists():
            continue
        for hint in hints:
            cand = root / hint
            if cand.exists():
                return cand
    return None


# Map detected import name → pip-installable package name. Most match
# directly; a few don't (kaldi_native_fbank → kaldi-native-fbank,
# openai_whisper → openai-whisper, etc.). Packages whose source-of-truth
# is git (chatterbox / vibevoice / voxcpm / etc.) are marked with a
# `git:` prefix the scaffold emitter expands to a TODO line.
DEP_TO_PIP = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "gguf": "gguf",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "scipy": "scipy",
    "onnxruntime": "onnxruntime",
    "kaldi_native_fbank": "kaldi-native-fbank",
    "snac": "snac",
    "fasttext": "fasttext",
    "pycld3": "pycld3",
    "openai_whisper": "openai-whisper",
    "moonshine_onnx": "moonshine-onnx",
    # Source-of-truth is git; leave as TODO so the maintainer adds the
    # correct git+https URL at the version the GGUF weights were converted from.
    "chatterbox": "git:resemble-ai/chatterbox",
    "funasr": "funasr",
    "nemo": "nemo_toolkit[asr]",
    "kokoro": "git:hexgrad/kokoro",
    "voxcpm": "git:openbmb/VoxCPM",
    "vibevoice": "git:microsoft/VibeVoice",
    "indextts": "git:index-tts/index-tts",
    "mimo_audio": "git:XiaomiMiMo/mimo-audio",
    "fireredasr": "fireredasr",
}


# Heavy-dep markers worth surfacing as install requirements.
# Each pattern needs to catch four forms of import that show up in the
# reference backends:
#   1. `import X`              (top-level or lazy inside dump())
#   2. `from X import ...`     (any submodule)
#   3. `import X.sub`          (submodule form, NeMo uses this)
#   4. `import a, X, b`        (comma-separated, parakeet has this)
def _imp(name: str) -> str:
    n = re.escape(name)
    # The bare-`import X` form needs to match at line start, after a
    # comma (`import a, X`), or after the literal word `import` itself
    # (so `import nemo.collections.asr` is caught). The follow context
    # allows `.`, whitespace, `,`, or end-of-line. The single (?m) flag
    # at the start applies to the whole pattern (alternation included).
    return (rf"(?m)(?:(?:^|[,;]|\bimport)\s*{n}(?=[.\s,]|$)"
            rf"|\bfrom\s+{n}(?:\.\w+)*\s+import\b)")


HEAVY_DEP_PATTERNS = [
    ("torch", _imp("torch")),
    ("torchaudio", _imp("torchaudio")),
    ("transformers", _imp("transformers")),
    ("accelerate", _imp("accelerate")),
    ("safetensors", _imp("safetensors")),
    ("gguf", _imp("gguf")),
    ("librosa", _imp("librosa")),
    ("soundfile", _imp("soundfile")),
    ("scipy", _imp("scipy")),
    ("onnxruntime", _imp("onnxruntime")),
    ("chatterbox", _imp("chatterbox")),
    ("funasr", _imp("funasr")),
    ("kaldi_native_fbank", _imp("kaldi_native_fbank")),
    ("snac", _imp("snac")),
    ("nemo", _imp("nemo")),
    ("kokoro", _imp("kokoro")),
    ("voxcpm", _imp("voxcpm")),
    ("vibevoice", _imp("vibevoice")),
    ("indextts", _imp("indextts")),
    ("mimo_audio", _imp("mimo_audio")),
    ("fasttext", _imp("fasttext")),
    ("pycld3", _imp("pycld3")),
    ("fireredasr", _imp("fireredasr")),
    ("moonshine_onnx", _imp("moonshine_onnx")),
    ("openai_whisper", _imp("whisper")),
]


def detect_heavy_deps(module_path: str) -> list[str]:
    """Read the backend's python file (and any tools/-resolved subimports) and
    return the heavy-dep package names that show up in import statements,
    including ones inside function bodies (lazy imports)."""
    rel = module_path.replace(".", "/") + ".py"
    abspath = REPO / "tools" / rel
    if not abspath.exists():
        return []
    text = abspath.read_text()
    found: list[str] = []
    for name, pat in HEAVY_DEP_PATTERNS:
        if re.search(pat, text):
            found.append(name)
    return found


def parse_harness_branches(harness_path: Path) -> set[str]:
    text = harness_path.read_text()
    branches: set[str] = set()
    for m in re.finditer(r'backend_name\s*==\s*"([^"]+)"', text):
        branches.add(m.group(1))
    return branches


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO / "docs" / "diff-harness-coverage.md",
                   help="markdown file to write")
    p.add_argument("--print-only", action="store_true",
                   help="don't write the markdown file, just print to stdout")
    p.add_argument("--emit-env-scaffolds", action="store_true",
                   help="also write tools/reference_envs/<name>/requirements.txt scaffolds "
                        "for any backend that doesn't already have one")
    args = p.parse_args()

    backends = parse_registered_backends()
    harness_branches = parse_harness_branches(REPO / "examples" / "cli" / "crispasr_diff_main.cpp")
    today = dt.date.today().isoformat()

    rows: list[dict] = []
    for name in sorted(backends):
        module_path = backends[name]
        module_file_rel = "tools/" + module_path.replace(".", "/") + ".py"
        module_exists = (REPO / module_file_rel).exists()
        archive = find_archive(name)
        archive_mtime = (
            dt.datetime.fromtimestamp(archive.stat().st_mtime).date().isoformat()
            if archive else None
        )
        has_harness = name in harness_branches
        deps = detect_heavy_deps(module_path)
        rows.append({
            "name": name,
            "module": module_file_rel if module_exists else "(missing)",
            "archive": str(archive) if archive else None,
            "archive_mtime": archive_mtime,
            "has_harness": has_harness,
            "deps": deps,
        })

    # Print summary to stdout
    n_total = len(rows)
    n_with_archive = sum(1 for r in rows if r["archive"])
    n_with_harness = sum(1 for r in rows if r["has_harness"])
    n_fully_wired = sum(1 for r in rows if r["archive"] and r["has_harness"])
    print(f"diff harness coverage audit ({today}):")
    print(f"  {n_total} registered ref backends")
    print(f"  {n_with_archive} have a frozen archive")
    print(f"  {n_with_harness} have a C++ harness branch")
    print(f"  {n_fully_wired} are fully wired (archive + harness branch)")
    print(f"  {n_total - n_fully_wired} have at least one gap")

    if args.print_only:
        return 0

    # Write markdown doc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# crispasr-diff coverage")
    lines.append("")
    lines.append(f"Auto-generated by `tools/audit_diff_coverage.py` on {today}. Do")
    lines.append("not edit by hand — re-run the tool to refresh.")
    lines.append("")
    lines.append(f"- {n_total} registered reference backends")
    lines.append(f"- {n_with_archive} have a frozen reference archive on disk")
    lines.append(f"- {n_with_harness} have a `crispasr-diff` C++ harness branch")
    lines.append(f"- **{n_fully_wired} are fully wired** (archive + harness branch)")
    lines.append(f"- {n_total - n_fully_wired} have at least one gap")
    lines.append("")
    lines.append("Adding a new entry: register the backend in")
    lines.append("`tools/dump_reference.py::REGISTERED_BACKENDS`, drop a ref module")
    lines.append("at `tools/reference_backends/<name>.py`, add an env spec at")
    lines.append("`tools/reference_envs/<name>/requirements.txt`, and wire a")
    lines.append("`backend_name == \"<name>\"` branch in")
    lines.append("`examples/cli/crispasr_diff_main.cpp`. Then `tools/bootstrap_ref_env.sh")
    lines.append("<name>` + `python tools/dump_reference.py --backend <name> ...`.")
    lines.append("")
    lines.append("| backend | ref module | archive | mtime | harness | deps |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        archive_cell = f"`{r['archive']}`" if r["archive"] else "—"
        mtime_cell = r["archive_mtime"] or "—"
        harness_cell = "yes" if r["has_harness"] else "**no**"
        deps_cell = ", ".join(sorted(r["deps"])) if r["deps"] else "—"
        lines.append(f"| `{r['name']}` | `{r['module']}` | {archive_cell} | "
                     f"{mtime_cell} | {harness_cell} | {deps_cell} |")
    lines.append("")

    # Gap summary lists
    needs_archive = [r for r in rows if r["has_harness"] and not r["archive"]]
    needs_harness = [r for r in rows if r["archive"] and not r["has_harness"]]
    needs_both = [r for r in rows if not r["has_harness"] and not r["archive"]]

    if needs_archive:
        lines.append("## Backends with harness branch but no frozen archive")
        lines.append("")
        lines.append("These are wired into `crispasr-diff` but skip every stage at runtime")
        lines.append("because the reference archive is missing. Bootstrap the env, dump an")
        lines.append("archive, place it where the harness expects, and the existing diff")
        lines.append("code starts working.")
        lines.append("")
        for r in needs_archive:
            lines.append(f"- `{r['name']}` — `tools/bootstrap_ref_env.sh {r['name']}` "
                         f"then `python tools/dump_reference.py --backend {r['name']} "
                         f"--model-dir <dir> --audio samples/jfk.wav "
                         f"--output /Volumes/backups/ai/{ARCHIVE_HINTS.get(r['name'], [r['name']+'-ref.gguf'])[0]}`")
        lines.append("")

    if needs_harness:
        lines.append("## Backends with frozen archive but no harness branch")
        lines.append("")
        lines.append("Archive is on disk but `crispasr-diff` has no `backend_name == \"<name>\"`")
        lines.append("branch yet — wire one into `examples/cli/crispasr_diff_main.cpp` to start")
        lines.append("using it.")
        lines.append("")
        for r in needs_harness:
            lines.append(f"- `{r['name']}` — archive at `{r['archive']}`")
        lines.append("")

    if needs_both:
        lines.append("## Backends with neither archive nor harness branch")
        lines.append("")
        for r in needs_both:
            lines.append(f"- `{r['name']}`")
        lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")

    if args.emit_env_scaffolds:
        env_root = REPO / "tools" / "reference_envs"
        env_root.mkdir(parents=True, exist_ok=True)
        n_written = 0
        n_skipped = 0
        for r in rows:
            backend_dir = env_root / r["name"]
            req = backend_dir / "requirements.txt"
            if req.exists():
                n_skipped += 1
                continue
            backend_dir.mkdir(parents=True, exist_ok=True)
            scaf: list[str] = []
            scaf.append(f"# Reference-dump env for `--backend {r['name']}`")
            scaf.append("# Scaffold generated by tools/audit_diff_coverage.py.")
            scaf.append("# Edit to add git-only deps and pin versions, then commit.")
            scaf.append("# Bootstrap: tools/bootstrap_ref_env.sh " + r["name"])
            scaf.append("")
            scaf.append("numpy>=2.0")
            scaf.append("# Always required by the dump driver itself")
            scaf.append("gguf>=0.10")
            scaf.append("")
            todos: list[str] = []
            installs: list[str] = []
            for dep in sorted(r["deps"]):
                pip = DEP_TO_PIP.get(dep, dep)
                if pip.startswith("git:"):
                    repo = pip[len("git:"):]
                    todos.append(f"# TODO: install upstream {dep} from git "
                                 f"(pin sha matching the GGUF weights you'll diff against)")
                    todos.append(f"# e.g.  {dep} @ git+https://github.com/{repo}.git@<sha>")
                else:
                    installs.append(pip)
            for line in installs:
                scaf.append(line)
            if todos:
                scaf.append("")
                scaf.extend(todos)
            scaf.append("")
            req.write_text("\n".join(scaf))
            n_written += 1
        print(f"env scaffolds: wrote {n_written}, skipped {n_skipped} that already exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
