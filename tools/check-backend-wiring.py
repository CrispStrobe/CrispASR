#!/usr/bin/env python3
"""check-backend-wiring.py — audit that every backend is wired across the surface.

Generalises the manual cross-check done when adding a backend (docs/contributing.md
checklist). Uses `crispasr --list-backends-json` as the authoritative backend +
capability list, then verifies each backend is present everywhere it must be.

Two tiers:

  REQUIRED (exact string match on the CLI name; a miss is a real bug → exit 1):
    - CLI factory dispatch        examples/cli/crispasr_backend.cpp
    - c_api open/detect           src/crispasr_c_api.cpp
    - c_api available_backends    the `list += ",<name>"` line (the easy-to-miss one)
    - feature matrix              docs/feature-matrix.md (auto-generated; stale if missing)
    - cli.md beam list            only when the backend declares the beam-search cap

  ADVISORY (per *canonical* backend — one that owns a dedicated CLI adapter file;
  aliases and shared runtimes are skipped so they aren't false-flagged. A miss is a
  warning, not a failure):
    - README mention
    - a test file                 tests/test_<x>_live.cpp OR tests/test-<x>-params.cpp
    - a reference dumper          tools/reference_backends/<x>*.py (standalone OR registered)
    - an env-live-tests entry     tests/env-live-tests.sh
    - a registry entry            src/crispasr_model_registry.cpp
    - streaming.md row            only when the backend declares the streaming cap

The Go cgo LDFLAGS check is delegated to the existing authoritative tool
(`tools/sync_go_cgo_ldflags.py --check`), which CI also runs.

Usage:
    python tools/check-backend-wiring.py [--crispasr ./build/bin/crispasr] [--verbose]

Exit code: 0 if all REQUIRED checks pass, 1 otherwise (advisory gaps never fail).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    p = ROOT / rel
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def stem_variants(name):
    """Candidate file/identifier stems for a CLI backend name."""
    u = name.replace("-", "_")
    return {name, u, name.replace("-", "")}


def list_backends(crispasr):
    out = subprocess.run([crispasr, "--list-backends-json"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"error: `{crispasr} --list-backends-json` failed; build crispasr first.\n"
                 f"{out.stderr[:400]}")
    d = json.loads(out.stdout)
    items = d if isinstance(d, list) else d.get("backends", d)
    return [(b["name"], set(b.get("caps", []))) for b in items]


def main():
    ap = argparse.ArgumentParser(description="Audit backend wiring completeness.")
    ap.add_argument("--crispasr", default=str(ROOT / "build/bin/crispasr"),
                    help="path to the crispasr binary (default: build/bin/crispasr)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every backend, not just problems")
    args = ap.parse_args()

    if not Path(args.crispasr).exists():
        sys.exit(f"error: {args.crispasr} not found — build it first "
                 f"(cmake --build build --target crispasr).")

    backends = list_backends(args.crispasr)

    factory = read("examples/cli/crispasr_backend.cpp")
    capi = read("src/crispasr_c_api.cpp")
    registry = read("src/crispasr_model_registry.cpp")
    fmatrix = read("docs/feature-matrix.md")
    cli_md = read("docs/cli.md")
    streaming = read("docs/streaming.md")
    readme = read("README.md")

    tests_dir = sorted(p.name for p in (ROOT / "tests").glob("*"))
    refs_dir = sorted(p.name for p in (ROOT / "tools/reference_backends").glob("*.py"))
    adapters = {p.name for p in (ROOT / "examples/cli").glob("crispasr_backend_*.cpp")}

    def in_available_backends(name):
        # entries look like:  list += ",moss-transcribe";  or packed:
        # list += ",granite,granite-4.1,granite-4.1-plus";
        return (f',{name}"' in capi) or (f",{name}," in capi)

    def in_beam_list(name):
        # the single --beam-size row in cli.md enumerates the supported backends
        return name in cli_md

    def any_file_has(files, name):
        return any(any(s in f for s in stem_variants(name)) for f in files)

    def has_adapter(name):
        return f"crispasr_backend_{name.replace('-', '_')}.cpp" in adapters

    required_fail = []   # (name, [missing required checks])
    advisory_gap = []    # (name, [missing advisory checks])
    n_canonical = 0
    n_alias = 0

    for name, caps in backends:
        # Only audit CANONICAL backends — those that own a dedicated CLI adapter
        # (`crispasr_backend_<x>.cpp`). Aliases / family variants (bark-tts,
        # qwen3-1.7b, chatterbox-turbo, …) resolve through a canonical backend's
        # dispatch, so the binary *listing* them already proves reachability;
        # requiring each to have its own literal wiring entry would be all
        # false-positives. (`env-var(_)`)
        if not has_adapter(name):
            n_alias += 1
            continue
        n_canonical += 1

        req_missing = []
        if f'"{name}"' not in factory:
            req_missing.append("factory")
        if f'"{name}"' not in capi:
            req_missing.append("c_api-dispatch")
        if not in_available_backends(name):
            req_missing.append("available_backends")
        if f"`{name}`" not in fmatrix:
            req_missing.append("feature-matrix(regen?)")
        if "beam-search" in caps and not in_beam_list(name):
            req_missing.append("cli.md-beam-list")
        if req_missing:
            required_fail.append((name, req_missing))

        adv_missing = []
        if name not in readme:
            adv_missing.append("README")
        if not any_file_has(tests_dir, name):
            adv_missing.append("test")
        if not any_file_has(refs_dir, name):
            adv_missing.append("ref-dumper")
        if f'"{name}"' not in registry:
            adv_missing.append("registry")
        # streaming.md documents ASR live transcription only. The `streaming` cap on
        # a TTS backend means incremental PCM synthesis (documented in tts.md), so
        # only expect a streaming.md row for ASR backends.
        if "streaming" in caps and "tts" not in caps and name not in streaming:
            adv_missing.append("streaming.md")
        if adv_missing:
            advisory_gap.append((name, adv_missing))

        if args.verbose:
            tag = "FAIL" if req_missing else ("warn" if adv_missing else "ok")
            print(f"  [{tag:4}] {name:24} caps={sorted(caps)}")

    # Go cgo LDFLAGS — advisory. The authoritative drift check is
    # tools/sync_go_cgo_ldflags.py, but a bare `--check` on macOS false-positives:
    # `cmake --graphviz` defaults Metal/BLAS ON and leaks -lggml-metal/-lggml-blas
    # into the `#cgo linux` line (see docs/contributing.md macOS gotcha). So we
    # report it but never fail on it — CI runs the real check on ubuntu.
    is_macos = sys.platform == "darwin"
    go = subprocess.run([sys.executable, str(ROOT / "tools/sync_go_cgo_ldflags.py"), "--check"],
                        capture_output=True, text=True)
    go_ok = go.returncode == 0

    print()
    print(f"Backends: {len(backends)} total — {n_canonical} canonical (audited), "
          f"{n_alias} aliases/variants (reachable, skipped).")
    if required_fail:
        print(f"\n❌ REQUIRED wiring gaps ({len(required_fail)}):")
        for name, miss in required_fail:
            print(f"   {name:24} missing: {', '.join(miss)}")
    else:
        print("✅ REQUIRED wiring: every canonical backend is in the factory, c_api "
              "dispatch, available_backends list, feature-matrix, and (if beam-capable) "
              "the cli.md beam list.")

    if advisory_gap:
        print(f"\n⚠️  Advisory coverage gaps ({len(advisory_gap)}):")
        for name, miss in advisory_gap:
            print(f"   {name:24} missing: {', '.join(miss)}")
        print("   (advisory — review, don't auto-fail. Reference dumpers may be standalone\n"
              "    (run directly, like bark/melotts); some older backends predate the test/\n"
              "    registry conventions.)")

    go_label = "✅ in sync" if go_ok else ("⚠️  reported out-of-sync (unreliable on macOS — "
                                           "re-check with --dot)" if is_macos else "❌ OUT OF SYNC")
    print(f"\nGo cgo LDFLAGS drift check: {go_label}")
    if not go_ok and not is_macos:
        print("   run: python tools/sync_go_cgo_ldflags.py   (see docs/contributing.md)")

    fail = bool(required_fail) or (not go_ok and not is_macos)
    print()
    print("RESULT:", "FAIL (required gap)" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
