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
import re
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
    # The shipped-library check needs the .so/.dylib that BELONGS TO the binary
    # being audited. Two escapes from the hardcoded <repo>/build/src guess:
    ap.add_argument("--lib", default=None,
                    help="path to the built libcrispasr shared library "
                         "(default: found next to --crispasr, then <repo>/build/src)")
    ap.add_argument("--require-lib", action="store_true",
                    help="fail instead of silently skipping when no shared "
                         "libcrispasr can be found (use in CI)")
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
    tts_md = read("docs/tts.md")
    arch_md = read("docs/architecture.md")
    src_cmake = read("src/CMakeLists.txt")
    py_binding = read("python/crispasr/_binding.py")
    env_live = read("tests/env-live-tests.sh")

    tests_dir = sorted(p.name for p in (ROOT / "tests").glob("*"))
    refs_dir = sorted(p.name for p in (ROOT / "tools/reference_backends").glob("*.py"))
    adapters = {p.name for p in (ROOT / "examples/cli").glob("crispasr_backend_*.cpp")}

    # ADAPTERS THAT CLAIM A NAME THE ROSTER DOES NOT LIST.
    #
    # Canonicality in the rest of this file is derived FROM the roster, which is
    # circular: a backend absent from crispasr_list_backends() is not canonical,
    # so it is never iterated, so nothing checks it. The reverse c_api check
    # above does not close this either, because it accepts any name the FACTORY
    # resolves -- and a forgotten roster entry is still factory-resolvable.
    #
    # Caught two real cases: supertonic (#434) shipped a working backend, a
    # factory entry, a c_api entry and a dedicated adapter while --list-backends
    # did not know it existed, so it was missing from the feature matrix and
    # from backend_caps_table.h, and the audit reported PASS. irodori-tts had
    # been in that state already, and a comment in this very file listed it as
    # an "alias" -- it is not: canary-ctc, omniasr-llm-unlimited and
    # vibevoice-tts each have a BASE entry in the roster (canary, omniasr,
    # vibevoice), while irodori-tts had no related roster entry at all.
    #
    # The authority is the adapter's own name() -- matching on FILENAME STEMS
    # gives 7 false positives (btc -> btc-chords, rvc -> rvc-svc, ...), while
    # name() gives an exact answer with none.
    adapter_claims = {}
    for ap in (ROOT / "examples/cli").glob("crispasr_backend_*.cpp"):
        try:
            src = ap.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for nm in re.findall(r'const char\*\s*name\(\)\s*const\s*override\s*\{\s*return\s*"([^"]+)"', src):
            adapter_claims.setdefault(nm, ap.name)

    # Prose names a backend the way a READER would, not the way the CLI does:
    # `crepe` appears as "CREPE", `tabcnn` as "TabCNN", `beat-this` as
    # "Beat This!". A raw case-sensitive substring test called all three
    # undocumented when the README documents every one of them, under `--pitch`,
    # `--tab` and `--beats`. Three false positives out of four advisory gaps is
    # exactly the noise ratio that teaches people to ignore the audit.
    #
    # So compare case-insensitively, and also with separators removed on BOTH
    # sides so "beat-this" finds "Beat This!". Kept as explicit variants rather
    # than stripping the whole document once, to limit the chance of a short
    # name matching inside an unrelated word.
    def mentioned(haystack, name):
        hay = haystack.lower()
        if name.lower() in hay:
            return True
        if name.replace("-", " ").replace("_", " ").lower() in hay:
            return True
        squash = lambda t: t.replace("-", "").replace("_", "").replace(" ", "")
        return squash(name.lower()) in squash(hay)

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

    # REVERSE CHECK: backends the C ABI advertises but the CLI does not know.
    #
    # Every other check in this file iterates the CLI's --list-backends-json, so
    # a backend missing from the CLI roster is not "canonical" and is never
    # audited at all -- the audit is blind to it BY CONSTRUCTION. That is not
    # hypothetical: btc-chords shipped with a runtime, a --chords dispatcher, a
    # session C ABI and wasm bindings while being absent from the CLI factory
    # and roster, so --list-backends did not know it existed and this script
    # reported PASS. Walking the c_api list and checking the other direction
    # closes the loop.
    capi_names = set(re.findall(r'list \+= ",([^"]+)"', capi))
    capi_flat = {n.strip() for entry in capi_names for n in entry.split(",") if n.strip()}
    cli_names = {name for name, _caps in backends}
    # A name is fine if the CLI ROSTER lists it *or* the CLI FACTORY resolves it
    # as an alias -- several backends are advertised by the c_api under an alias
    # (canary-ctc, vibevoice-tts, omniasr-llm-unlimited) and are genuinely
    # reachable. Only a name with NEITHER is unreachable from the CLI, which is
    # the state btc-chords was in.
    #
    # irodori-tts USED TO BE LISTED HERE and did not belong: the test for a real
    # alias is that its BASE name is in the roster (canary, vibevoice, omniasr
    # all are), and irodori-tts had no related roster entry at all. It was a
    # missing roster line wearing an alias label, which is why this check kept
    # passing over it. See the adapter_claims check above.
    # Reachability is decided by ASKING THE BINARY, not by parsing the dispatch
    # chain: some backends resolve by prefix (`name.rfind("omniasr", 0) == 0`)
    # or through multi-alias conditions that no regex will reliably cover.
    # Only the handful not already in the roster need probing.
    def cli_resolves(name):
        r = subprocess.run([args.crispasr, "--backend", name, "-m", os.devnull, "-f", os.devnull],
                           capture_output=True, text=True)
        return f"unknown backend '{name}'" not in (r.stderr + r.stdout)

    capi_only = sorted(n for n in capi_flat if n not in cli_names and not cli_resolves(n))

    # ---------------------------------------------------------------------
    # SHIPPED-LIBRARY check: is the backend's runtime actually IN the dylib?
    #
    # Every other check in this file reads SOURCE TEXT, and source text cannot
    # see this failure. mel-band-roformer was linked into crispasr-lib by
    # CMake, exactly as it looked in the CMakeLists -- but nothing in
    # crispasr_c_api.cpp referenced its symbols, so the linker dropped the
    # whole object from the shared library. It was not merely unreachable from
    # the session API: it was NOT PRESENT IN THE SHIPPED .dylib AT ALL, while
    # the CLI worked because crispasr-cli links the static lib directly.
    # Confirmed against the released v0.8.17 artifact, where
    # mel_band_roformer_separate is absent.
    #
    # Symbol presence is ground truth, so this has no alias false positives --
    # unlike name-matching, which produced 21/76 noise. Demangling matters:
    # some runtimes are C++-linkage, so `sidon_init_from_file` appears only as
    # `__Z20sidon_init_from_file...` and a raw grep misses it.
    # Scanned once and used by BOTH the shipped-library check and the
    # orphan-runtime check below; the latter must run even with no built library.
    inits_all = {}
    for h in (ROOT / "src").glob("*.h"):
        try:
            for m in re.finditer(r"\b([a-z0-9_]+)_init_from_file\s*\(", h.read_text(errors="ignore")):
                inits_all[m.group(1)] = h.name
        except OSError:
            pass

    lib_fail = []
    lib_unreadable = None
    libpath = None

    # WHERE THE LIBRARY IS LOOKED FOR, AND WHY THE ORDER MATTERS.
    #
    # This used to test ONLY <repo>/build/src/libcrispasr.{so,dylib}, with no
    # relation to the --crispasr binary it was auditing. That makes two silent
    # wrong answers possible, and both have been observed:
    #
    #   * SILENT SKIP. Every out-of-tree build (Kaggle builds into
    #     /kaggle/temp/build-cuda; ci.yml's own audit builds into ./build but
    #     with BUILD_SHARED_LIBS off, so no .so is produced at all) leaves
    #     <repo>/build/src empty, the check prints "skipped" and the run passes
    #     while proving nothing. The gate had never once executed in CI.
    #   * STALE FALSE POSITIVES. A months-old <repo>/build/src/libcrispasr.so
    #     left over from an earlier checkout, audited against a freshly built
    #     binary, reports every backend added since as "absent from the shipped
    #     library" -- a list of ~23 names that are all correctly wired. Symbol
    #     presence is only ground truth when the symbols come from the SAME
    #     build as the roster they are checked against.
    #
    # So: an explicit --lib wins, then the library that sits in the binary's own
    # build tree (build/bin/crispasr -> build/src/libcrispasr.so), and only then
    # the legacy repo-relative guess.
    cli_dir = Path(args.crispasr).resolve().parent          # <build>/bin
    cands = []
    if args.lib:
        cands.append(Path(args.lib))
    for d in (cli_dir.parent / "src", cli_dir.parent, cli_dir):
        cands += [d / n for n in ("libcrispasr.dylib", "libcrispasr.so",
                                  "libcrispasr.1.dylib")]
    cands += [ROOT / c for c in ("build/src/libcrispasr.dylib",
                                 "build/src/libcrispasr.so",
                                 "build/src/libcrispasr.1.dylib")]
    for c in cands:
        if c.exists():
            libpath = c
            break
    if args.lib and libpath != Path(args.lib):
        sys.exit(f"error: --lib {args.lib} does not exist.")

    if libpath:
        # `nm -gU` IS NOT PORTABLE, AND ITS FAILURE IS SILENT-SHAPED.
        # -U means --defined-only on macOS nm and on binutils >= 2.39, but on
        # binutils 2.38 (ubuntu-22.04, which is what bindings-rust.yml runs on)
        # -U is --unicode and `nm -gU lib.so` exits 1 with "invalid argument to
        # -U/--unicode". stdout is then empty, which reads as "no backend
        # symbols are present" -- a fabricated list of ~70 missing backends from
        # a library that contains every one of them.
        #
        # So ask nm for nothing but the global symbols, which every nm spells
        # the same way, and do the defined/undefined split here: the type column
        # is U (undefined), v/w (weak undefined) for symbols that are merely
        # REFERENCED. That distinction is the whole point of the check -- a
        # backend whose object was dropped still leaves an undefined reference
        # behind in a shared library, so counting those as present would make
        # the check pass on exactly the bug it exists to catch.
        nmr = subprocess.run(["nm", "-g", str(libpath)], capture_output=True, text=True)
        sym_re = re.compile(r"^\s*(?:[0-9a-fA-F]+)?\s*([A-Za-z?])\s+(\S+)\s*$")
        defined = [m.group(2) for m in (sym_re.match(l) for l in nmr.stdout.splitlines())
                   if m and m.group(1) not in "UuvwV"]
        dem = subprocess.run(["c++filt"], input="\n".join(defined),
                             capture_output=True, text=True).stdout

        # A READOUT THAT CANNOT REPORT ITS OWN FAILURE IS NOT A GATE.
        # Three ways to end up with an empty/garbage symbol list: nm errored,
        # the library is stripped ("no symbols"), or the output format did not
        # parse. All three make EVERY backend look absent, and 100+ bogus
        # failures are indistinguishable from a real regression.
        #
        # The last line is a POSITIVE CONTROL, not a formality: crispasr_session_open
        # is in this library in every configuration that can build it at all, so
        # if the table cannot produce it the table is wrong, whatever else it
        # seems to say about backends.
        if nmr.returncode != 0 or not dem.strip():
            lib_unreadable = (nmr.stderr.strip().splitlines() or ["nm produced no output"])[-1]
            libpath = None
        elif "crispasr_session_open" not in dem:
            lib_unreadable = (f"{len(defined)} defined symbols read from {libpath.name}, "
                              f"but not crispasr_session_open — the symbol table is not "
                              f"being parsed correctly")
            libpath = None
        else:
            inits = dict(inits_all)

            def runtime_stem(n):
                b = n.replace("-", "_")
                return [b, b.replace("_tts", ""), b + "_tts", b.replace("_asr", ""), b + "_asr"]

            for name, _caps in backends:
                hit = next((v for v in runtime_stem(name) if v in inits), None)
                if hit and (hit + "_init_from_file") not in dem:
                    lib_fail.append((name, hit))

    # ---------------------------------------------------------------------
    # ORPHAN-RUNTIME check: a runtime in NEITHER the CLI roster NOR the c_api
    # list is invisible to every check above — the state mel-band-roformer was
    # in while being the default `--separate` model.
    #
    # The signal is "src/*.h declares <x>_init_from_file but nothing is named
    # <x>". Raw, it fires on 20 of 82 stems and 17 are legitimate components, so
    # it was long left unshipped: a gate with 17 false positives trains everyone
    # to ignore the audit. tools/backend-components.txt names those 17 once, with
    # their consumer, which turns each into a recorded decision and drops the
    # false-positive count to zero — so this CAN fail the run.
    #
    # Alias-reachable backends (canary-ctc, irodori-tts, t5-translate) are NOT
    # allowlisted: cli_resolves() asks the binary about them, and allowlisting
    # would hide a real regression if an alias ever broke.
    components, comp_missing = set(), None
    comp_path = ROOT / "tools/backend-components.txt"
    if comp_path.exists():
        for line in comp_path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                components.add(line)
    else:
        comp_missing = str(comp_path)

    def name_variants(stem):
        b = stem.replace("_", "-")
        return {stem, b, b.replace("-tts", ""), b + "-tts", b.replace("-asr", ""), b + "-asr", b.replace("-", "")}

    orphans = []
    for stem in sorted(inits_all):
        if name_variants(stem) & cli_names:
            continue
        if stem in components:
            continue
        # Ask the binary before calling it an orphan — several runtimes are
        # reachable only under an alias.
        if any(cli_resolves(v) for v in (stem, stem.replace("_", "-"))):
            continue
        orphans.append((stem, inits_all[stem]))

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
        if not mentioned(readme, name):
            adv_missing.append("README")
        if not any_file_has(tests_dir, name):
            adv_missing.append("test")
        if not any_file_has(refs_dir, name):
            adv_missing.append("ref-dumper")
        # Convert-only backends ship no published GGUF — the user converts them
        # locally (documented in the README), so there is nothing to auto-download
        # and a registry entry would be a dead URL. voxcpm2-vae is converted from
        # openbmb/VoxCPM2 with `--vae-only`; exempt it from the registry advisory.
        CONVERT_ONLY = {"voxcpm2-vae"}
        if name not in CONVERT_ONLY and f'"{name}"' not in registry:
            adv_missing.append("registry")
        # env-live-tests.sh: only flag if the backend has a *_live.cpp test
        # that actually needs model env vars (params-only tests don't need them)
        has_live_test = any(any(s in f and "live" in f for s in stem_variants(name))
                           for f in tests_dir)
        if has_live_test and not any(s in env_live for s in stem_variants(name)):
            adv_missing.append("env-live-tests")
        # Python binding docstring should list TTS backends (ASR backends
        # are dispatched generically via transcribe() and don't need listing)
        if "tts" in caps and name not in py_binding:
            adv_missing.append("py-binding-doc")
        # src/CMakeLists.txt should link the backend lib into crispasr-lib.
        # Some backends share a lib (e.g. fastconformer-ctc → canary_ctc,
        # wav2vec2 → wav2vec2-ggml), so also check the CLI adapter's includes.
        in_src_cmake = any(s in src_cmake for s in stem_variants(name))
        if not in_src_cmake:
            adapter_path = ROOT / "examples/cli" / f"crispasr_backend_{name.replace('-', '_')}.cpp"
            adapter_src = adapter_path.read_text(errors="ignore") if adapter_path.exists() else ""
            # grep for #include "<lib>.h" and check that lib is in CMake
            import re as _re
            includes = _re.findall(r'#include\s+"(\w+)\.h"', adapter_src)
            in_src_cmake = any(inc in src_cmake for inc in includes)
        if not in_src_cmake:
            adv_missing.append("src-cmake")
        # TTS backends should be in docs/tts.md
        if "tts" in caps and name not in tts_md:
            adv_missing.append("tts.md")
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
    if lib_fail:
        print(f"\n❌ Declared as a backend but ABSENT from the shipped library ({len(lib_fail)}):")
        for name, stem in lib_fail:
            print(f"   {name:24} {stem}_init_from_file not in {libpath.name if libpath else '?'}")
        print("   The linker drops a static-lib object nothing references, so CMake linkage\n"
              "   is NOT evidence the code ships. Reference it from src/crispasr_c_api.cpp\n"
              "   (a session arm), then rebuild and re-check.")
    elif lib_unreadable:
        print(f"\n❌ shipped-library check could not read symbols: {lib_unreadable}")
        print("   (stripped library, or an `nm` without -U/--defined-only — the check\n"
              "    is NOT passing, it is blind. Do not read this as a green run.)")
    elif not libpath:
        print("\n(shipped-library check skipped: no built libcrispasr found — build it to enable)")
    else:
        print(f"✅ Shipped library: every backend runtime is present in {libpath.name}.")

    if capi_only:
        print(f"\n❌ Advertised by the C ABI but ABSENT from the CLI roster ({len(capi_only)}):")
        for name in capi_only:
            print(f"   {name:24} add a factory entry + roster line in examples/cli/crispasr_backend.cpp")
        print("   (A task-shaped backend still needs a redirect shim + capability bit so it\n"
              "    appears in --list-backends and the generated docs/feature-matrix.md.\n"
              "    See examples/cli/crispasr_backend_btc.cpp for the pattern.)")

    if comp_missing:
        print(f"\n⚠️  component allowlist not found at {comp_missing} — orphan-runtime check skipped")
    elif orphans:
        print(f"\n❌ Runtime declared in src/ but reachable from NOWHERE ({len(orphans)}):")
        for stem, hdr in orphans:
            print(f"   {stem:24} {hdr}: not a backend name, no alias resolves, not in tools/backend-components.txt")
        print("   Either wire it up (CLI factory + roster + c_api) so users can select it,\n"
              "   or record it as a sub-module in tools/backend-components.txt with its consumer.")
    else:
        print(f"✅ Orphan runtimes: none ({len(components)} known components allowlisted).")

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

    # Name the ACTUAL cause. This used to print "FAIL (required gap)" for all
    # four conditions, so a run whose only problem was Go LDFLAGS drift reported
    # a required *wiring* gap two lines below "✅ REQUIRED wiring: ..." — the
    # reader then hunts through the advisory list for a gap that isn't there.
    # Compare adapter claims against the roster IN SOURCE, not against the
    # binary's --list-backends-json.
    #
    # cli_names comes from the BINARY, and a binary older than the roster it is
    # being judged against manufactures false positives: a local build from
    # 04:19 audited against a roster committed at 10:44 reported supertonic,
    # irodori-tts and fireredtts3 as "claimed by an adapter but absent from the
    # roster" when the source roster listed all three. That is the same
    # stale-artifact failure as the shipped-library check above, in the check
    # written to catch roster omissions -- so it is fixed the same way: both
    # sides of THIS comparison are source-derived, and cannot skew apart.
    roster_src = set()
    try:
        _be = (ROOT / "examples/cli/crispasr_backend.cpp").read_text(errors="ignore")
        _i = _be.index("std::vector<std::string> crispasr_list_backends()")
        roster_src = set(re.findall(r'"([^"]+)"', _be[_i:_be.index("};", _i)]))
    except (OSError, ValueError):
        roster_src = set(cli_names)  # fall back rather than fabricate a gap
    unrostered = sorted(n for n in adapter_claims if n not in roster_src)
    if unrostered:
        print()
        print(f"\u274c Adapters claiming a name the CLI roster omits ({len(unrostered)}):")
        for n in unrostered:
            print(f"   {n:24s} {adapter_claims[n]}")
        print("   These are NOT aliases: an alias has a base name in the roster.")
        print("   Add them to crispasr_list_backends() in examples/cli/crispasr_backend.cpp,")
        print("   then regenerate docs/feature-matrix.* and src/core/backend_caps_table.h.")
        print("   Until then --list-backends cannot see them and this audit skips them.")

    # STALENESS GUARD. Several checks below ask the BINARY what it knows, which
    # is the right design -- some backends resolve by prefix or through
    # multi-alias conditions no regex covers. But a binary older than the
    # sources it is judged against turns every backend added since into a
    # fabricated gap, and the report then names backends instead of naming the
    # stale artifact. Observed: a build from 04:19 audited against a roster
    # committed at 10:44 produced six findings across three checks, all of them
    # the same two new backends.
    try:
        _bin_mtime = os.path.getmtime(args.crispasr)
        _newer = [
            rel for rel in ("examples/cli/crispasr_backend.cpp", "src/crispasr_c_api.cpp")
            if (ROOT / rel).is_file() and os.path.getmtime(ROOT / rel) > _bin_mtime
        ]
        if _newer:
            print()
            print("\u26a0\ufe0f  THE BINARY IS OLDER THAN THE SOURCES IT IS BEING JUDGED AGAINST:")
            for rel in _newer:
                print(f"   {rel} is newer than {args.crispasr}")
            print("   Any backend added since that build will be reported as missing from the")
            print("   roster / unreachable / orphaned. REBUILD before believing the gaps below.")
    except OSError:
        pass

    causes = []
    if required_fail:
        causes.append("required wiring gap")
    if unrostered:
        causes.append("adapter missing from the CLI roster")
    if capi_only:
        causes.append("c_api-only backend")
    if lib_fail:
        causes.append("missing symbol in shipped library")
    if lib_unreadable:
        causes.append("shipped library symbols unreadable")
    if args.require_lib and not libpath and not lib_unreadable:
        causes.append("--require-lib: no shared libcrispasr to audit")
    if not comp_missing and orphans:
        causes.append("orphan runtime")
    if not go_ok and not is_macos:
        causes.append("Go cgo LDFLAGS drift")
    print()
    print("RESULT:", f"FAIL ({'; '.join(causes)})" if causes else "PASS")
    return 1 if causes else 0


if __name__ == "__main__":
    sys.exit(main())
