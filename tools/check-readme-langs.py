#!/usr/bin/env python3
"""
README language-list drift check.

Issue #50 was a copy-paste regression: the granite-4.1-plus and
granite-4.1-nar README rows claimed Japanese support because the
language column was carried over verbatim from granite-4.1 base
when the new variants were registered. Neither lint nor any test
caught it because language metadata doesn't drive any code path.

This script parses every model row in README.md, finds the upstream
HuggingFace link, fetches its `language:` frontmatter via
`huggingface_hub.model_info`, and compares against the language
column in our README. Mismatches are printed and the script exits
non-zero — CI fails before the drift reaches users.

Run locally:
    python tools/check-readme-langs.py
    python tools/check-readme-langs.py --strict   # mismatch == failure (default in CI)
    python tools/check-readme-langs.py --warn     # mismatch == warning, still exits 0

Skip-on-network-error semantics: HF API unreachable doesn't fail the
lint; it prints a warning and exits 0. The check is best-effort —
the goal is to catch drift, not to gate on HF uptime.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# Regex for an HF model link inside markdown — we want the repo id
# (e.g. "ibm-granite/granite-speech-4.1-2b-plus"), not the surrounding
# backtick-wrapped display text.
HF_LINK_RE = re.compile(r"https?://huggingface\.co/([\w.-]+/[\w.-]+)")

# A "language column" in our markdown table is a cell whose stripped
# content is space-separated lowercase 2-3-letter ISO codes — e.g.
# "en", "en de fr", "en fr de es pt ja". We accept dashes for region
# codes (e.g. "en-US") and digits for some HF cards.
LANG_CELL_RE = re.compile(r"^([a-z]{2,3}(?:-[A-Z]{2})?)(?:\s+[a-z]{2,3}(?:-[A-Z]{2})?)*$")


def parse_table_rows(text: str) -> list[dict]:
    """Yield (repo, langs, line_no) for every HF link in every model row.
    A row may list multiple upstream models that share the same language
    column — each is checked separately. Skips cstr/* mirrors (downstream,
    not source of truth)."""
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.startswith("| "):
            continue
        repos = [m.group(1).rstrip("/") for m in HF_LINK_RE.finditer(line)]
        repos = [r for r in repos if not r.lower().startswith("cstr/")]
        if not repos:
            continue
        # Find the rightmost cell matching the language pattern.
        cells = [c.strip() for c in line.split("|")[1:-1]]
        lang_cell = None
        for c in reversed(cells):
            if LANG_CELL_RE.match(c):
                lang_cell = c
                break
        if not lang_cell:
            continue
        # De-dup repos in case the same URL appears twice in one row.
        for repo in dict.fromkeys(repos):
            rows.append({"hf_repo": repo, "langs": lang_cell.split(),
                         "line_no": i, "raw": line})
    return rows


def fetch_upstream_langs(repo: str) -> list[str] | None:
    """Return the `language:` list from the upstream HF model card, or
    None on any error (network, missing field, etc.)."""
    try:
        from huggingface_hub import model_info
    except ImportError:
        print("warning: huggingface_hub not installed — skipping check",
              file=sys.stderr)
        return None
    try:
        info = model_info(repo)
    except Exception as e:
        print(f"warning: HF model_info({repo!r}) failed: {e}", file=sys.stderr)
        return None
    if not info.card_data:
        return None
    langs = info.card_data.get("language") if hasattr(info.card_data, "get") else getattr(info.card_data, "language", None)
    if not langs:
        return None
    if isinstance(langs, str):
        return [langs]
    return [str(l) for l in langs]


def normalize(langs: list[str]) -> set[str]:
    """Lowercase + drop region suffix so 'en-US' == 'en'. Also drops
    'multilingual' which some cards use as an umbrella tag — it isn't
    a real language code and shouldn't trigger a mismatch."""
    out = set()
    for l in langs:
        if not l:
            continue
        l = l.lower().split("-", 1)[0]
        if l == "multilingual":
            continue
        out.add(l)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--readme", default=str(README),
                    help="path to README.md")
    ap.add_argument("--warn", action="store_true",
                    help="report mismatches as warnings, exit 0 (default: fail)")
    args = ap.parse_args()

    text = Path(args.readme).read_text()
    rows = parse_table_rows(text)
    if not rows:
        print(f"no model rows found in {args.readme}", file=sys.stderr)
        return 0

    # Group by source line so multi-URL rows get aggregate semantics:
    #   single URL → README must match upstream exactly
    #   multi URL  → README must equal UNION(each upstream's langs)
    # The aggregate "granite" row is the prototypical example: each
    # granite model supports a subset, our README claims the union.
    by_line: dict[int, list[dict]] = {}
    for r in rows:
        by_line.setdefault(r["line_no"], []).append(r)

    print(f"checking {len(by_line)} model rows ({len(rows)} HF links) in {args.readme}")
    ok = mismatch = skip = 0

    for line_no, group in sorted(by_line.items()):
        readme_langs = normalize(group[0]["langs"])
        # Fetch each upstream; missing metadata → skip the row (don't FAIL
        # the lint on a flaky / removed repo).
        per_repo: dict[str, set[str] | None] = {}
        for r in group:
            up = fetch_upstream_langs(r["hf_repo"])
            per_repo[r["hf_repo"]] = normalize(up) if up else None
        if any(v is None for v in per_repo.values()):
            unreachable = [k for k, v in per_repo.items() if v is None]
            print(f"  skip  line {line_no}  (no upstream metadata: {', '.join(unreachable)})")
            skip += 1
            continue

        union = set()
        for v in per_repo.values():
            assert v is not None
            union |= v

        if len(group) == 1:
            label = group[0]["hf_repo"]
        else:
            label = f"AGG[{', '.join(r['hf_repo'].split('/')[-1] for r in group)}]"

        if readme_langs == union:
            print(f"  ok    {label:60}  {' '.join(sorted(readme_langs))}")
            ok += 1
            continue
        mismatch += 1
        only_readme = readme_langs - union
        only_upstream = union - readme_langs
        print(f"  DIFF  {label:60}  README:{sorted(readme_langs)}  upstream:{sorted(union)}")
        if only_readme:
            print(f"        README has but no upstream supports:  {sorted(only_readme)}  (line {line_no})")
        if only_upstream:
            print(f"        ≥1 upstream supports but README missing:  {sorted(only_upstream)}")
        if len(group) > 1:
            for repo, langs in per_repo.items():
                assert langs is not None
                print(f"          {repo}: {sorted(langs)}")

    print()
    print(f"summary: {ok} ok, {mismatch} mismatch, {skip} skipped")

    if mismatch and not args.warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
