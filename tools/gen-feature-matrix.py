#!/usr/bin/env python3
"""gen-feature-matrix.py — single source of truth for the feature matrix.

Calls `crispasr --list-backends-json` and emits two artifacts:

  - docs/feature-matrix.md   — Markdown table (checked-in; CI can diff)
  - docs/feature-matrix.html — vanilla-JS standalone, sortable + filterable

Why generate from the binary's JSON instead of hand-maintaining the
README's table? Because every backend addition required editing 17
rows by hand and the table drifted from reality (PLAN #61, #63, #71
all spent effort closing exactly that drift). Regenerating from
`--list-backends-json` makes drift impossible.

Both files are ALWAYS written into <repo-root>/docs/ — the location is
resolved from this script's own path, NOT from the current working
directory (#407). Running from inside tools/ therefore writes to
../docs/; the "wrote" lines print the absolute paths.

Usage:
    python tools/gen-feature-matrix.py
    python tools/gen-feature-matrix.py --crispasr ./build/bin/crispasr
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Pretty labels + one-line meanings (<=80 chars, #407) for the cap slugs
# the binary reports. Order is the display order in both outputs. Caps
# not listed here are appended in sorted order with a fallback "Title
# Case" label and no description, so adding a cap to the C++ side
# automatically shows up in both outputs.
CAP_LABELS: list[tuple[str, str, str]] = [
    ("transcribe", "Transcribe",
     "Speech-to-text transcription (the base ASR path)."),
    ("tts", "TTS",
     "Text-to-speech synthesis (--tts)."),
    ("voice-cloning", "Voice cloning",
     "TTS clones a target voice from a short reference sample (--voice ref.wav)."),
    ("translate", "Translate",
     "Translates speech into another language during transcription."),
    ("src-tgt-language", "Src/Tgt language",
     "Takes an explicit source/target language pair for translation."),
    ("language-detect", "Language detect",
     "Detects the spoken language automatically."),
    ("auto-download", "Auto-download",
     "`-m auto` downloads the model from the HF registry on first use."),
    ("timestamps-native", "Timestamps (native)",
     "Model produces segment timestamps natively."),
    ("timestamps-ctc", "Timestamps (CTC)",
     "Timestamps recovered by CTC alignment, not predicted by the model."),
    ("word-timestamps", "Word timestamps",
     "Per-word start/end timestamps."),
    ("token-confidence", "Token confidence",
     "Per-token probability/confidence scores."),
    ("temperature", "Temperature",
     "Sampling temperature control for decoding."),
    ("beam-search", "Beam search",
     "Beam-search decoding (-bs N)."),
    ("punctuation-toggle", "Punctuation toggle",
     "Punctuation can be enabled/disabled at runtime."),
    ("flash-attn", "Flash attention",
     "Flash-attention toggle for faster, lower-memory inference."),
    ("diarize", "Diarize",
     "Speaker diarization: labels who spoke when."),
    ("grammar", "GBNF grammar",
     "Output constrained to a GBNF grammar."),
    ("vad-internal", "VAD (internal)",
     "Backend runs voice-activity detection internally."),
    ("parallel-processors", "Parallel processors",
     "Whisper-style parallel processors (-p N) over audio slices."),
    ("beats", "Beats",
     "Beat / downbeat tracking (--beats)."),
    ("chords", "Chords",
     "Chord-recognition timeline (--chords)."),
    ("piano", "Piano",
     "Polyphonic piano-note transcription (--piano)."),
    ("pitch", "Pitch",
     "Monophonic F0 / pitch-track estimation (--pitch)."),
    ("punctuation-native", "Punctuation Native",
     "Model already emits punctuation; no post-processing needed."),
    ("s2s", "S2S",
     "Speech-to-speech: audio in, spoken response out."),
    ("separate", "Separate",
     "Source separation into stems, e.g. vocals vs. instruments (--separate)."),
    ("streaming", "Streaming",
     "True token-level streaming output."),
    ("tab", "Tab",
     "Guitar tablature: per-frame per-string fret scores (--tab)."),
    ("tts-speed", "TTS Speed",
     "Scales speaking rate natively in the acoustic model (--tts-speed). The "
     "server skips its post-synthesis linear resampler for these backends, so "
     "pitch is not shifted and the rate is not applied twice."),
]


def find_crispasr() -> Path | None:
    candidates = [
        REPO_ROOT / "build-ninja-compile" / "bin" / "crispasr",
        REPO_ROOT / "build" / "bin" / "crispasr",
        REPO_ROOT / "build-test" / "bin" / "crispasr",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_backend_data(crispasr: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    # The CLI emits banner + warnings on stderr; only stdout is JSON.
    r = subprocess.run([str(crispasr), "--list-backends-json"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"crispasr failed: rc={r.returncode}", file=sys.stderr)
        print(r.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    raw = json.loads(r.stdout)
    backends = raw.get("backends", [])
    # All caps that appear anywhere, in CAP_LABELS order, then fallback.
    seen: set[str] = set()
    for b in backends:
        for c in b.get("caps", []):
            seen.add(c)
    ordered: list[tuple[str, str, str]] = []
    used: set[str] = set()
    for slug, label, desc in CAP_LABELS:
        if slug in seen:
            ordered.append((slug, label, desc))
            used.add(slug)
    for c in sorted(seen):
        if c not in used:
            ordered.append((c, c.replace("-", " ").title(), ""))
    return backends, ordered


def write_markdown(backends: list[dict],
                   caps: list[tuple[str, str]],
                   out: Path) -> None:
    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by tools/gen-feature-matrix.py — do not hand-edit. -->")
    lines.append("<!-- Source of truth: `crispasr --list-backends-json`. -->")
    lines.append("")
    lines.append("# Feature matrix")
    lines.append("")
    lines.append(f"All {len(backends)} backends compiled into the `crispasr` binary, "
                 f"with their declared capability bits. For an interactive "
                 f"sortable/filterable view, open "
                 f"[`feature-matrix.html`](https://htmlpreview.github.io/?https://github.com/CrispStrobe/CrispASR/blob/main/docs/feature-matrix.html).")
    lines.append("")
    # Header
    headers = ["Backend"] + [label for _, label, _ in caps]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + [":-:" for _ in caps]) + "|")
    for b in backends:
        bc = set(b.get("caps", []))
        row = [f"`{b['name']}`"]
        for slug, _, _ in caps:
            row.append("✓" if slug in bc else "")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## What each capability means")
    lines.append("")
    lines.append("| Capability | Meaning |")
    lines.append("|---|---|")
    for _, label, desc in caps:
        if desc:
            lines.append(f"| {label} | {desc} |")
    lines.append("")
    lines.append("Regenerate with `python tools/gen-feature-matrix.py`.")
    lines.append("")
    out.write_text("\n".join(lines))


def write_html(backends: list[dict],
               caps: list[tuple[str, str]],
               out: Path) -> None:
    # Embed the data as a JSON island. The page is self-contained — no
    # external JS deps, no fetch, viewable offline / from disk.
    rows = []
    for b in backends:
        bc = set(b.get("caps", []))
        rows.append({
            "name": b["name"],
            "caps": {slug: (slug in bc) for slug, _, _ in caps},
        })
    payload = json.dumps({"caps": [list(c) for c in caps], "rows": rows},
                         separators=(",", ":"))

    # Vanilla JS — clickable column-header sort, top input filter,
    # cap-name filter pills, light/dark via prefers-color-scheme.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CrispASR feature matrix</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #ffffff; --fg: #1f2328; --muted: #6e7781;
    --border: #d0d7de; --hover: #f6f8fa; --accent: #0969da;
    --y-bg: #dafbe1; --y-fg: #1a7f37;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0d1117; --fg: #e6edf3; --muted: #7d8590;
      --border: #30363d; --hover: #161b22; --accent: #2f81f7;
      --y-bg: #1f3a26; --y-fg: #56d364;
    }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg); }}
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  header {{ padding: 24px 32px 8px; }}
  h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 600; }}
  .sub {{ color: var(--muted); font-size: 13px; }}
  .controls {{ padding: 16px 32px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; border-bottom: 1px solid var(--border); }}
  .controls input[type="text"] {{
    padding: 6px 10px; font: inherit; border: 1px solid var(--border);
    border-radius: 6px; background: var(--bg); color: var(--fg);
    min-width: 200px;
  }}
  .controls input[type="text"]:focus {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .pill {{
    padding: 4px 8px; font-size: 12px; border: 1px solid var(--border);
    border-radius: 12px; cursor: pointer; user-select: none;
    background: var(--bg); color: var(--fg);
  }}
  .pill.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
  .pill:hover {{ background: var(--hover); }}
  .pill.active:hover {{ background: var(--accent); opacity: 0.9; }}
  .stats {{ color: var(--muted); font-size: 12px; margin-left: auto; }}
  .table-wrap {{ overflow: auto; padding: 0 32px 32px; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  th, td {{
    padding: 6px 10px; border-bottom: 1px solid var(--border);
    text-align: center; white-space: nowrap;
  }}
  th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0; background: var(--bg); }}
  thead th {{
    position: sticky; top: 0; background: var(--bg); cursor: pointer;
    user-select: none; font-weight: 600; border-bottom: 2px solid var(--border);
    z-index: 1;
  }}
  thead th:first-child {{ z-index: 2; }}
  thead th:hover {{ background: var(--hover); }}
  thead th .arrow {{ color: var(--muted); margin-left: 4px; font-size: 11px; }}
  tbody tr:hover {{ background: var(--hover); }}
  tbody tr:hover td:first-child {{ background: var(--hover); }}
  td.y {{ background: var(--y-bg); color: var(--y-fg); font-weight: 600; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .meta {{ padding: 0 32px 24px; color: var(--muted); font-size: 12px; }}
  .meta a {{ color: var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>CrispASR feature matrix</h1>
  <div class="sub">Auto-generated from <code>crispasr --list-backends-json</code>. Click any column header to sort. Type to filter rows by backend name; click cap pills below to require that capability.</div>
</header>
<div class="controls">
  <input type="text" id="filter" placeholder="Filter by backend name…" autofocus>
  <div class="pills" id="pills"></div>
  <div class="stats" id="stats"></div>
</div>
<div class="table-wrap">
  <table id="matrix"><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table>
</div>
<div class="meta">
  <details><summary>What each capability means</summary>
    <table id="legend" style="margin-top:8px; font-size:12px;"></table>
  </details>
  Regenerate via <code>python tools/gen-feature-matrix.py</code>. Single source of truth: the C++ binary's
  <code>--list-backends-json</code> output (capability bits in <code>examples/cli/crispasr_backend.h</code>).
  Drift between the matrix and the binary is impossible by construction.
</div>
<script>
const DATA = {payload};
const $ = (id) => document.getElementById(id);
let sortKey = null, sortDesc = true;
const requiredCaps = new Set();

function render() {{
  const head = $("head"), body = $("body");
  head.innerHTML = "";
  const th0 = document.createElement("th");
  th0.textContent = "Backend";
  th0.dataset.key = "name";
  th0.onclick = () => clickSort("name");
  if (sortKey === "name") th0.innerHTML += " <span class='arrow'>" + (sortDesc ? "▼" : "▲") + "</span>";
  head.appendChild(th0);
  for (const [slug, label, desc] of DATA.caps) {{
    const th = document.createElement("th");
    th.textContent = label;
    th.dataset.key = slug;
    th.title = desc || slug;
    th.onclick = () => clickSort(slug);
    if (sortKey === slug) th.innerHTML += " <span class='arrow'>" + (sortDesc ? "▼" : "▲") + "</span>";
    head.appendChild(th);
  }}
  const filter = ($("filter").value || "").toLowerCase().trim();
  let rows = DATA.rows.filter(r => {{
    if (filter && !r.name.toLowerCase().includes(filter)) return false;
    for (const c of requiredCaps) {{
      if (!r.caps[c]) return false;
    }}
    return true;
  }});
  if (sortKey) {{
    rows = rows.slice().sort((a, b) => {{
      let va, vb;
      if (sortKey === "name") {{ va = a.name; vb = b.name; }}
      else {{ va = a.caps[sortKey] ? 1 : 0; vb = b.caps[sortKey] ? 1 : 0; }}
      if (va < vb) return sortDesc ? 1 : -1;
      if (va > vb) return sortDesc ? -1 : 1;
      // Tie-break by name (asc) so the result is stable.
      return a.name.localeCompare(b.name);
    }});
  }}
  body.innerHTML = "";
  for (const r of rows) {{
    const tr = document.createElement("tr");
    const td0 = document.createElement("td");
    const code = document.createElement("code");
    code.textContent = r.name;
    td0.appendChild(code);
    tr.appendChild(td0);
    for (const [slug] of DATA.caps) {{
      const td = document.createElement("td");
      if (r.caps[slug]) {{ td.className = "y"; td.textContent = "✓"; }}
      tr.appendChild(td);
    }}
    body.appendChild(tr);
  }}
  $("stats").textContent = rows.length + " of " + DATA.rows.length + " backends";
}}

function clickSort(key) {{
  if (sortKey === key) sortDesc = !sortDesc;
  else {{ sortKey = key; sortDesc = key !== "name"; }}
  render();
}}

function renderPills() {{
  const c = $("pills");
  c.innerHTML = "";
  for (const [slug, label, desc] of DATA.caps) {{
    const p = document.createElement("span");
    p.className = "pill" + (requiredCaps.has(slug) ? " active" : "");
    p.textContent = label;
    p.title = (desc ? desc + " — " : "") + "click to require " + slug;
    p.onclick = () => {{
      if (requiredCaps.has(slug)) requiredCaps.delete(slug);
      else requiredCaps.add(slug);
      renderPills();
      render();
    }};
    c.appendChild(p);
  }}
}}

function renderLegend() {{
  const t = $("legend");
  for (const [slug, label, desc] of DATA.caps) {{
    if (!desc) continue;
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    td1.style.paddingRight = "12px";
    td1.textContent = label;
    const td2 = document.createElement("td");
    td2.textContent = desc;
    tr.appendChild(td1); tr.appendChild(td2);
    t.appendChild(tr);
  }}
}}

$("filter").addEventListener("input", render);
renderPills();
renderLegend();
render();
</script>
</body>
</html>
"""
    out.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crispasr", type=Path, default=None,
                    help="Path to the crispasr binary (default: first of "
                         "build-ninja-compile/bin, build/bin, build-test/bin "
                         "under the repo root)")
    args = ap.parse_args()
    crispasr = args.crispasr or find_crispasr()
    if not crispasr or not crispasr.is_file():
        print("error: crispasr binary not found. Build first or pass --crispasr.",
              file=sys.stderr)
        return 1
    backends, caps = load_backend_data(crispasr)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = DOCS_DIR / "feature-matrix.md"
    html = DOCS_DIR / "feature-matrix.html"
    write_markdown(backends, caps, md)
    write_html(backends, caps, html)
    # Absolute paths: the files land in <repo>/docs/ regardless of cwd,
    # and a bare "docs\..." reads as cwd-relative on Windows (#407).
    print(f"wrote {md.resolve()}  ({md.stat().st_size:,} bytes)")
    print(f"wrote {html.resolve()} ({html.stat().st_size:,} bytes)")
    print(f"  {len(backends)} backends · {len(caps)} caps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
