#!/usr/bin/env python3
"""
Automated completeness test for CrispASR test/benchmark scripts.

Checks that every backend declared in the CLI's list_backends() is covered
by the major test and benchmark scripts. Run with:
    python3 tests/test_script_completeness.py

Exit 0 = all checks pass, exit 1 = gaps found.
"""
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known aliases / variants that don't need their own entry in every script.
# Key = alias, value = the primary name that covers it.
KNOWN_ALIASES = {
    "bark-tts": "bark",
    "csm-tts": "csm",
    "dia-tts": "dia",
    "zonos-tts": "zonos",
    "sesame": "csm",
}

# Backends that are intentionally excluded from certain scripts with a reason.
INTENTIONAL_EXCLUSIONS = {
    # test-all-backends.py: backends without HF repos or paused
    "test-all-backends.py": {
        "zonos": "HF repo not yet uploaded",
        "zonos-tts": "alias for zonos",
        "fastpitch": "paused per user request",
        "qwen3-1.7b": "ASR variant — same runtime as qwen3, tested via mega-asr (1.7B LoRA variant)",
        "omniasr-llm-1b": "size variant — tested via omniasr-llm (300m)",
    },
    # manifest.json: backend_ids for non-CLI utility backends
    "manifest.json": {
        "lid-cld3": "text-LID utility, not a CLI ASR/TTS backend",
        "titanet": "speaker verification utility, not a CLI ASR/TTS backend",
        "chatterbox_turbo": "underscore variant accepted by CLI factory (line 112)",
        "mimo-tokenizer": "audio tokenizer utility, not a CLI ASR/TTS backend",
    },
    # test-backends.sh: only tests backends with -m auto support on CPU
    "test-backends.sh": {
        # TTS backends are in TTS_BACKENDS, not FAST/MEDIUM/SLOW
        # Large/slow backends may be intentionally excluded
    },
    # benchmark_all.sh: ASR benchmarks only, needs local GGUF paths
    "benchmark_all.sh": {
        # TTS backends excluded (ASR benchmark)
        # Backends needing multiple GGUFs or special setup excluded
    },
}


def read_file(relpath):
    with open(os.path.join(REPO_ROOT, relpath)) as f:
        return f.read()


def get_cli_backends():
    """Extract canonical backend list from crispasr_list_backends()."""
    src = read_file("examples/cli/crispasr_backend.cpp")
    m = re.search(
        r"crispasr_list_backends\(\)\s*\{[^}]*return\s*\{([^}]+)\}", src, re.DOTALL
    )
    if not m:
        return []
    raw = m.group(1)
    return [s.strip().strip('"') for s in raw.split(",") if s.strip().strip('"')]


def get_test_all_backends():
    """Extract Backend() names from test-all-backends.py."""
    src = read_file("tools/test-all-backends.py")
    return set(re.findall(r'Backend\("([^"]+)"', src))


def get_test_backends_sh():
    """Extract backends from test-backends.sh FAST/MEDIUM/SLOW/TTS vars."""
    src = read_file("tests/test-backends.sh")
    backends = set()
    for m in re.findall(r'(?:FAST|MEDIUM|SLOW|TTS)_BACKENDS="([^"]+)"', src):
        backends.update(m.split())
    return backends


def get_benchmark_all():
    """Extract model keys from benchmark_all.sh MODELS array."""
    src = read_file("tools/benchmark_all.sh")
    return set(re.findall(r'\["([^"]+)"\]="', src))


def get_kaggle_benchmark():
    """Extract backend names from kaggle-benchmark-all-backends.py."""
    src = read_file("tools/kaggle-benchmark-all-backends.py")
    return set(re.findall(r'\("(\w[\w-]*)"', src))


def get_manifest():
    """Extract backend names from regression manifest.json."""
    m = json.loads(read_file("tests/regression/manifest.json"))
    backends = {b["name"] for b in m["backends"]}
    tts = {b["name"] for b in m.get("tts_backends", [])}
    return backends, tts


def get_model_registry():
    """Extract names from model registry."""
    src = read_file("src/crispasr_model_registry.cpp")
    return set(re.findall(r'\{"(\w[\w-]*)"', src))


def check_hf_repo_refs(script_name, script_src):
    """Check that all cstr/ HF repo references look valid (basic format check)."""
    errors = []
    repos = re.findall(r'"(cstr/[^"]+)"', script_src)
    for repo in repos:
        # Must match pattern: cstr/<name>-GGUF or cstr/<name>-<variant>-GGUF
        if not re.match(r"cstr/[\w.-]+-GGUF$", repo) and not re.match(
            r"cstr/[\w.-]+$", repo
        ):
            errors.append(f"{script_name}: suspicious HF repo format: {repo}")
    return errors


def check_gguf_filenames(script_name, script_src):
    """Check that GGUF filenames match expected patterns."""
    errors = []
    files = re.findall(r'"([\w.-]+\.gguf)"', script_src)
    for f in files:
        # Basic sanity: should contain a quant suffix or f16/f32
        if not re.search(r"(q[48]_[0k]|f16|f32|q8_0)", f):
            # Some files like tokenizer.bin, dac-44khz.gguf are OK
            if not any(
                x in f for x in ["tokenizer", "dac-", "voices", "campplus", "s3tok",
                                   "hift", "voice-", "bigvgan", "wavtokenizer",
                                   "speaker", "snac"]
            ):
                errors.append(
                    f"{script_name}: GGUF file without quant suffix: {f}"
                )
    return errors


def main():
    cli_backends = get_cli_backends()
    tab = get_test_all_backends()
    tbs = get_test_backends_sh()
    ba = get_benchmark_all()
    kag = get_kaggle_benchmark()
    manifest_asr, manifest_tts = get_manifest()
    registry = get_model_registry()

    errors = []
    warnings = []

    # Resolve aliases
    canonical = set()
    for b in cli_backends:
        if b not in KNOWN_ALIASES:
            canonical.add(b)

    print(f"Canonical backends (excluding aliases): {len(canonical)}")
    print(f"test-all-backends.py:    {len(tab)}")
    print(f"test-backends.sh:        {len(tbs)}")
    print(f"benchmark_all.sh:        {len(ba)}")
    print(f"kaggle-benchmark.py:     {len(kag)}")
    print(f"manifest.json backends:  {len(manifest_asr)}")
    print(f"manifest.json tts:       {len(manifest_tts)}")
    print(f"model registry:          {len(registry)}")
    print()

    # Check 1: Every canonical backend should be in test-all-backends.py
    tab_excl = INTENTIONAL_EXCLUSIONS.get("test-all-backends.py", {})
    for b in sorted(canonical):
        if b not in tab and b not in tab_excl:
            errors.append(f"test-all-backends.py: missing backend '{b}'")

    # Check 2: Every backend in test-all-backends.py should be canonical
    for b in sorted(tab):
        if b not in set(cli_backends):
            # Could be a model variant (moonshine-de, etc.)
            warnings.append(
                f"test-all-backends.py: '{b}' not in CLI list_backends "
                f"(OK if it's a model variant)"
            )

    # Check 3: Every backend in model registry should be in test-all-backends.py
    # (registry has auto-download entries; if it's downloadable, it should be testable)
    for b in sorted(registry):
        if b not in tab and b not in KNOWN_ALIASES and b not in tab_excl:
            # Filter out non-backend registry entries (companion files, etc.)
            if not any(x in b for x in ["voice", "snac", "tokenizer", "campplus",
                                         "s3tok", "hift", "bigvgan", "dac", "flow"]):
                warnings.append(
                    f"model registry has '{b}' but test-all-backends.py doesn't"
                )

    # Check 4: manifest.json backend_id should match valid CLI backends
    manifest_data = json.loads(read_file("tests/regression/manifest.json"))
    manifest_excl = INTENTIONAL_EXCLUSIONS.get("manifest.json", {})
    for b in manifest_data["backends"]:
        bid = b["backend_id"]
        if bid not in set(cli_backends) and bid not in manifest_excl:
            errors.append(
                f"manifest.json: backend_id '{bid}' (entry '{b['name']}') "
                f"not in CLI list_backends"
            )

    # Check 5: manifest.json GGUF files should have valid format
    for b in manifest_data["backends"]:
        gguf = b.get("gguf", {})
        rev = gguf.get("revision", "")
        if len(rev) != 40:
            errors.append(
                f"manifest.json: '{b['name']}' revision SHA is "
                f"{len(rev)} chars (expected 40)"
            )
        if not gguf.get("file", "").endswith(".gguf") and not gguf.get("file", "").endswith(".bin"):
            errors.append(
                f"manifest.json: '{b['name']}' file doesn't end with .gguf/.bin: "
                f"{gguf.get('file', 'MISSING')}"
            )

    # Check 6: manifest tts_backends should have required fields
    for t in manifest_data.get("tts_backends", []):
        for field in ["name", "backend_id", "gguf", "tts_phrase",
                      "roundtrip_asr_backend", "wer_max"]:
            if field not in t:
                errors.append(
                    f"manifest.json tts: '{t.get('name', '?')}' missing field '{field}'"
                )
        # wer_max should be 0 < x <= 1
        wer = t.get("wer_max", 0)
        if not (0 < wer <= 1):
            errors.append(
                f"manifest.json tts: '{t.get('name', '?')}' wer_max={wer} "
                f"out of range (0, 1]"
            )

    # Check 7: regression.yml matrix entries should match manifest names
    yml = read_file(".github/workflows/regression.yml")
    yml_backends = set(re.findall(r"^\s+- ([\w][\w.-]+)\s*$", yml, re.MULTILINE))
    all_manifest_names = manifest_asr | manifest_tts
    for b in yml_backends:
        if b not in all_manifest_names:
            errors.append(
                f"regression.yml: matrix entry '{b}' not in manifest.json"
            )

    # Check 8: HF repo format in test-all-backends.py
    tab_src = read_file("tools/test-all-backends.py")
    errors.extend(check_hf_repo_refs("test-all-backends.py", tab_src))

    # Print results
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors:
        print("\n✓ All completeness checks passed.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
