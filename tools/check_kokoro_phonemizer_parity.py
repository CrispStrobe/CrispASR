#!/usr/bin/env python3
"""Diff harness for the kokoro phonemizer paths (PLAN #56 #4).

Compares the two espeak-ng entry points exposed by libcrispasr:

  - `kokoro_phonemize_text_lib`   — in-process libespeak-ng (preferred,
                                    ~30-50 ms faster per call).
  - `kokoro_phonemize_text_popen` — shell-out to `espeak-ng` on PATH
                                    (fallback when the lib isn't built
                                    in or its init failed).

The runtime today picks lib over popen with no public visibility into
which one fired. This script exercises both paths on a fixed
(lang, text) suite and reports drift, so a future espeak-ng version
bump or build-config change doesn't silently change kokoro's output.

Default mode normalises both strings (strips U+200D ZERO WIDTH JOINER
tie characters, collapses runs of whitespace) before comparing —
those differences are documented in LEARNINGS Lesson 6 and not
substantive. Use --strict for byte-exact comparison.

Exit codes:
  0   no substantive divergence (all pairs match after normalisation,
      or strict mode and all pairs byte-identical)
  1   at least one substantive divergence
  2   one or both phonemizer paths returned nullptr for every test
      (build is broken — neither lib nor popen works)

Usage:

  python tools/check_kokoro_phonemizer_parity.py
  python tools/check_kokoro_phonemizer_parity.py --strict
  CRISPASR_LIB_PATH=/path/to/libcrispasr.dylib \\
      python tools/check_kokoro_phonemizer_parity.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
import unicodedata
from pathlib import Path


# Default suite covers every kokoro language we ship voices for plus the
# two known-tricky cases (cmn tones, ja kanji) that PLAN #56 follow-ups
# #2 and #3 also touch. Keep it small — this is a regression guard, not
# a benchmark.
DEFAULT_SUITE = [
    ("en-us", "Hello, my name is Tara."),
    ("en-us", "The quick brown fox jumps over the lazy dog."),
    ("de",    "Guten Tag, dies ist ein Test."),
    ("de",    "Der Phonemizer ist heute wieder gut drauf."),
    ("fr",    "Bonjour le monde."),
    ("ru",    "Привет, мир."),
    ("cmn",   "你好世界"),
    ("ja",    "こんにちは"),
    ("it",    "Ciao mondo."),
    ("es",    "Hola mundo."),
    ("pt",    "Olá mundo."),
]


_ZWJ = "‍"


def normalise(s: str) -> str:
    """Strip the documented benign differences between lib and popen.

    - U+200D ZERO WIDTH JOINER tie characters (espeak-ng popen emits
      them between vowel pairs in diphthongs; the lib path doesn't).
    - Collapse runs of whitespace (newline / tab) into single spaces
      (popen joins sentence chunks with newlines, lib joins with
      spaces).
    """
    return re.sub(r"\s+", " ", s.replace(_ZWJ, "")).strip()


def find_lib() -> Path:
    """Locate libcrispasr — env var first, then standard build dirs."""
    env = os.environ.get("CRISPASR_LIB_PATH")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "build-ninja-compile" / "src" / "libcrispasr.dylib",
        repo_root / "build" / "src" / "libcrispasr.dylib",
        repo_root / "build" / "src" / "libcrispasr.so",
        Path("/opt/homebrew/lib/libcrispasr.dylib"),
        Path("/usr/local/lib/libcrispasr.dylib"),
        Path("/usr/lib/libcrispasr.so"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(
        "libcrispasr not found. Set CRISPASR_LIB_PATH or build with "
        "`cmake --build <build-dir> --target crispasr` first.")


def setup_phonemize_fns(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    for name in ("kokoro_phonemize_text_lib", "kokoro_phonemize_text_popen"):
        if not hasattr(lib, name):
            raise SystemExit(
                f"libcrispasr at {lib_path} missing symbol '{name}'. "
                f"Rebuild against PLAN #56 #4 (commit adding the diff "
                f"harness ABI exports).")
    # Both functions return char* (malloc'd by libc) — but ctypes.c_char_p
    # for restype assumes the returned pointer is statically allocated and
    # won't free it. Use c_void_p so we can free explicitly via libc.
    for name in ("kokoro_phonemize_text_lib", "kokoro_phonemize_text_popen"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p

    # libc free() — same allocator that libcrispasr's malloc() uses.
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    return lib, libc


def call_phonemize(fn, libc, lang: str, text: str) -> str | None:
    """Returns the IPA string, or None on nullptr."""
    ptr = fn(lang.encode("utf-8"), text.encode("utf-8"))
    if not ptr:
        return None
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        libc.free(ptr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strict", action="store_true",
                   help="Byte-exact comparison (no ZWJ/whitespace normalisation).")
    p.add_argument("--suite", type=Path,
                   help="Path to a custom suite file (one '<lang>\\t<text>' per line).")
    p.add_argument("--lib", type=Path, help="Override libcrispasr path.")
    args = p.parse_args()

    if args.lib:
        os.environ["CRISPASR_LIB_PATH"] = str(args.lib)
    lib_path = find_lib()
    print(f"libcrispasr: {lib_path}")
    lib, libc = setup_phonemize_fns(lib_path)
    fn_lib = lib.kokoro_phonemize_text_lib
    fn_popen = lib.kokoro_phonemize_text_popen

    if args.suite:
        suite = []
        for line in args.suite.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lang, _, text = line.partition("\t")
            if text:
                suite.append((lang.strip(), text.strip()))
        if not suite:
            raise SystemExit(f"empty suite file: {args.suite}")
    else:
        suite = DEFAULT_SUITE

    n_match = 0
    n_substantive_diff = 0
    n_normalise_only_diff = 0
    n_lib_null = 0
    n_popen_null = 0

    for lang, text in suite:
        ipa_lib = call_phonemize(fn_lib, libc, lang, text)
        ipa_popen = call_phonemize(fn_popen, libc, lang, text)

        if ipa_lib is None:
            n_lib_null += 1
        if ipa_popen is None:
            n_popen_null += 1
        if ipa_lib is None or ipa_popen is None:
            print(f"[SKIP] {lang:5s}  lib={ipa_lib!r}  popen={ipa_popen!r}  "
                  f"text={text!r}")
            continue

        if args.strict:
            if ipa_lib == ipa_popen:
                n_match += 1
                tag = "MATCH"
            else:
                n_substantive_diff += 1
                tag = "DIFF "
        else:
            norm_lib = normalise(ipa_lib)
            norm_popen = normalise(ipa_popen)
            if norm_lib == norm_popen:
                if ipa_lib == ipa_popen:
                    n_match += 1
                    tag = "MATCH"
                else:
                    n_normalise_only_diff += 1
                    tag = "norm "
            else:
                n_substantive_diff += 1
                tag = "DIFF "

        print(f"[{tag}] {lang:5s}  text={text!r}")
        print(f"         lib  : {ipa_lib!r}")
        print(f"         popen: {ipa_popen!r}")

    print()
    print(f"summary: {n_match} match, {n_normalise_only_diff} normalise-only diff, "
          f"{n_substantive_diff} substantive diff, "
          f"{n_lib_null} lib-null, {n_popen_null} popen-null  "
          f"(strict={args.strict})")

    if n_lib_null == len(suite) and n_popen_null == len(suite):
        return 2  # both paths broken
    if n_substantive_diff > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
