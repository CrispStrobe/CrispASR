#!/usr/bin/env python3
"""Pick the regression matrix for a Regression workflow run.

schedule / workflow_dispatch -> every backend in tests/regression/nightly_matrix.json.
push / pull_request          -> only the nightly backends the change can affect:
  * a backend's own sources (src/<id>*, examples/cli/crispasr_backend_<id>*,
    models/convert-<id>*) select that backend;
  * shared code (src/core/, src/CMakeLists.txt, CMakeLists.txt, ggml, the CLI
    dispatcher/output code, the regression driver) selects the CORE set - a few
    cheap backends spanning the main code paths;
  * changed manifest entries also select their own nightly backends;
    an unavailable manifest diff or changed nightly list selects all nightly;
  * nothing relevant -> an empty matrix (the regression job is skipped).

    python tools/regression_select.py --event push --base <sha> [--head HEAD]
Prints a JSON list (GitHub Actions `fromJSON` input) on stdout.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Cheap entries that together cover mel/conformer/CTC/RNNT/AED/LLM-decoder/TTS paths.
CORE = ["moonshine-tiny", "parakeet-tdt-0.6b-en", "canary-1b-v2", "qwen3-asr-0.6b", "wav2vec2-xlsr-en",
        "kokoro-82m-en"]
# backend_id -> extra source stems when the file name does not start with the id
ALIASES = {
    "hubert": ["wav2vec2"], "data2vec": ["wav2vec2"], "fastconformer-ctc": ["canary_ctc", "fastconformer"],
    "moonshine-streaming": ["moonshine_streaming"], "qwen3": ["qwen3_asr"], "qwen3-tts": ["qwen3_tts"],
    "bark": ["bark_tts"], "csm": ["csm_tts"], "speecht5": ["speecht5_tts"], "piper": ["piper_tts"],
    "pocket-tts": ["pocket_tts"], "mini-omni2": ["mini_omni2"], "firered-asr": ["firered_asr"],
    "glm-asr": ["glm_asr"], "kyutai-stt": ["kyutai_stt"], "granite": ["granite_speech"],
}
# Model-specific importers/reference readers do not share the runtime's stem.
MODEL_SOURCES = {
    "index-echo-2b": ("models/convert-index-echo-to-gguf.py", "tools/reference_backends/index_echo.py",
                      "tools/ci-heavy/index_echo_", "tools/index_echo_acceptance.py",
                      "tests/test_index_echo_acceptance.py", "examples/talk-llama/qwen35",
                      "examples/talk-llama/models/qwen35"),
    "phonon2": ("models/phonon2_container.py", "tools/reference_backends/phonon2/",
                "tools/reference_backends/parakeet_hf.py"),
}
SHARED_PREFIXES = ("src/core/", "src/CMakeLists.txt", "CMakeLists.txt", "ggml", "cmake/",
                   "examples/cli/crispasr_run", "examples/cli/crispasr_output", "examples/cli/cli.cpp",
                   "examples/cli/CMakeLists.txt", "examples/cli/crispasr_backend.", "examples/cli/whisper_params",
                   "tests/regression/", "tools/regression_select.py", ".github/workflows/regression.yml",
                   "examples/crispasr-diff", "examples/cli/crispasr_diff")


def manifest_ids():
    m = json.loads((ROOT / "tests/regression/manifest.json").read_text())
    ids = {e["name"]: e["backend_id"] for e in m["backends"]}
    for e in m.get("tts_backends", []):
        ids[e["name"]] = e.get("backend_id") or e.get("tts_backend") or e["name"]
    return ids


def stems_for(backend_id):
    base = backend_id.replace("-", "_")
    return [base] + ALIASES.get(backend_id, [])


def changed_entries(before, after):
    """Return names whose complete ASR/TTS manifest entry changed."""
    def entries(manifest):
        return {e["name"]: e for group in ("backends", "tts_backends")
                for e in manifest.get(group, [])}
    old, new = entries(before), entries(after)
    return {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}


def manifest_changes(base, head):
    """Read both immutable Git snapshots; None means comparison unavailable."""
    try:
        snapshots = []
        for rev in (base, head):
            out = subprocess.run(["git", "show", f"{rev}:tests/regression/manifest.json"],
                                 cwd=ROOT, capture_output=True, text=True, check=True)
            snapshots.append(json.loads(out.stdout))
        return changed_entries(*snapshots)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def select(changed, nightly, changed_manifest=None):
    ids = manifest_ids()
    picked = []
    if "tests/regression/nightly_matrix.json" in changed:
        return list(nightly)
    if "tests/regression/manifest.json" in changed:
        if changed_manifest is None:
            return list(nightly)
        picked += [n for n in nightly if n in changed_manifest]
    if any(f.startswith(SHARED_PREFIXES) for f in changed):
        picked += [n for n in CORE if n in nightly]
    for name in nightly:
        if name in MODEL_SOURCES and any(f.startswith(MODEL_SOURCES[name]) for f in changed):
            picked.append(name)
        bid = ids.get(name, name)
        for stem in stems_for(bid):
            if any(f.startswith((f"src/{stem}", f"examples/cli/crispasr_backend_{stem}", f"models/convert-{stem}",
                                 f"models/convert_{stem}")) or f.startswith(f"src/{stem.replace('_', '-')}")
                   for f in changed):
                picked.append(name)
                break
    return sorted(dict.fromkeys(picked), key=nightly.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--changed", nargs="*", help="explicit changed-file list (tests)")
    a = ap.parse_args()
    nightly = json.loads((ROOT / "tests/regression/nightly_matrix.json").read_text())["nightly"]
    if a.event in ("schedule", "workflow_dispatch"):
        print(json.dumps(nightly))
        return
    changed = a.changed
    if changed is None:
        if not a.base or set(a.base) == {"0"}:  # new branch / unknown base: be safe, run the core set
            print(json.dumps([n for n in CORE if n in nightly]))
            return
        out = subprocess.run(["git", "diff", "--name-only", a.base, a.head], capture_output=True, text=True, cwd=ROOT)
        if out.returncode != 0:
            print(json.dumps([n for n in CORE if n in nightly]))
            return
        changed = [l for l in out.stdout.splitlines() if l]
    affected = None
    if "tests/regression/manifest.json" in changed and a.base and set(a.base) != {"0"}:
        affected = manifest_changes(a.base, a.head)
    print(json.dumps(select(changed, nightly, affected)))


if __name__ == "__main__":
    sys.exit(main())
