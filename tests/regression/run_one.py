#!/usr/bin/env python3
"""Run one backend's regression against pinned HF artifacts.

For each backend in ``tests/regression/manifest.json``:

1. Download the GGUF under test at the pinned HF revision SHA.
2. Download the reference-dump archive at the pinned fixtures revision.
3. Run ``crispasr -m <gguf> -f <sample>`` and assert the transcript
   matches the pinned ``expected_transcript`` byte-for-byte.
4. Run ``crispasr-diff <backend_id> <gguf> <ref> <sample>`` and assert
   every stage's ``cos_min`` is at or above its pinned threshold.

Backends with ``"skip_diff": true`` in the manifest skip steps 2 and 4
(transcript-only check). This lets lightweight backends join the GH
Actions nightly matrix before their Kaggle-baked reference dumps are
ready. The GGUF + sample are still downloaded at pinned revisions so
upstream re-quantise is caught.

Pinned revisions guard against:

- Upstream re-quantise of the GGUF on HF silently changing what users
  download (the regression-from-ggml-assertion-hardening lesson).
- Drift in our own reference dumps between commits.

Exit code is the number of failures (0 on full success).

Usage:

  tests/regression/run_one.py parakeet-tdt-0.6b-ja
  BUILD_DIR=build-ninja-compile tests/regression/run_one.py parakeet-tdt-0.6b-ja
  CRISPASR_BIN=/path/crispasr DIFF_BIN=/path/crispasr-diff \\
    tests/regression/run_one.py parakeet-tdt-0.6b-ja

  # Dry-run: HEAD-check every pinned HF object referenced by the
  # manifest (or just the named backend). No downloads, no binary
  # needed. ~1 s for the seed manifest. Use as a fast PR-time gate.
  tests/regression/run_one.py --dry-run
  tests/regression/run_one.py --dry-run parakeet-tdt-0.6b-ja

Env:

  BUILD_DIR       Build directory containing bin/crispasr + bin/crispasr-diff
                  (default: build-regression).
  CRISPASR_BIN    Override the crispasr binary path entirely.
  DIFF_BIN        Override the crispasr-diff binary path entirely.
  WORK_DIR        Where to stage downloads (default: a tempdir; cleaned on
                  exit unless KEEP_WORK=1).
  KEEP_WORK       If set to 1, don't delete WORK_DIR on exit.
  REGRESSION_MANIFEST  Path to manifest.json (default:
                  tests/regression/manifest.json).
  REGRESSION_DIAGNOSTICS_DIR  Optional directory outside WORK_DIR for bounded
                  command logs and a failed TTS roundtrip's generated WAV.
  REGRESSION_KEEP_TTS_WAV  Set to 1 with the diagnostics directory to retain
                  a successful roundtrip's WAV as an explicit manual control.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_LIMIT = 512 * 1024
_WAV_LIMIT = 32 * 1024 * 1024


def _diagnostics_dir() -> Path | None:
    value = os.environ.get("REGRESSION_DIAGNOSTICS_DIR")
    if not value:
        return None
    path = Path(value).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bounded_output(value: str | bytes | None) -> dict:
    data = value.encode("utf-8", errors="replace") if isinstance(value, str) else value or b""
    text = data[-_LOG_LIMIT:].decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if len(encoded) > _LOG_LIMIT:
        # Replacing malformed bytes can expand the tail by up to 3x. Trim the
        # encoded suffix and drop only a partial codepoint at its new start.
        text = encoded[-_LOG_LIMIT:].decode("utf-8", errors="ignore")
    return {
        "bytes": len(data),
        "truncated": len(data) > _LOG_LIMIT or len(encoded) > _LOG_LIMIT,
        "text": text,
    }


def _run_captured(argv: list[str], *, check: bool, timeout: int, label: str) -> subprocess.CompletedProcess:
    """Run an unchanged CLI argv; persist bounded output when requested."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=check, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        diag = _diagnostics_dir()
        if diag is not None:
            (diag / f"{label}.json").write_text(json.dumps({
                "argv": argv,
                "returncode": exc.returncode if isinstance(exc, subprocess.CalledProcessError) else None,
                "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                "stdout": _bounded_output(exc.stdout),
                "stderr": _bounded_output(exc.stderr),
            }, indent=2) + "\n")
        raise
    diag = _diagnostics_dir()
    if diag is not None:
        (diag / f"{label}.json").write_text(json.dumps({
            "argv": argv,
            "returncode": proc.returncode,
            "timed_out": False,
            "stdout": _bounded_output(proc.stdout),
            "stderr": _bounded_output(proc.stderr),
        }, indent=2) + "\n")
    return proc


def _preserve_tts_wav(name: str, wav: Path) -> None:
    """Keep only a small generated WAV, never model or downloaded input files."""
    diag = _diagnostics_dir()
    if diag is None or not wav.is_file():
        return
    size = wav.stat().st_size
    if size > _WAV_LIMIT:
        (diag / "wav-skipped.json").write_text(json.dumps({
            "reason": "generated WAV exceeds diagnostics cap", "bytes": size,
            "limit": _WAV_LIMIT,
        }, indent=2) + "\n")
        return
    shutil.copyfile(wav, diag / f"tts_{name}.wav")


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"\033[31mERROR\033[0m  {msg}", file=sys.stderr)
    sys.exit(code)


def hf_download(repo: str, file_in_repo: str, revision: str, dest_dir: Path,
                repo_type: str = "model") -> Path:
    """Download one file from HF at a pinned revision; return its local path.

    Honours `HF_HOME` / `HUGGINGFACE_HUB_CACHE` from the environment if
    set (so a local dev run can share the on-disk HF cache and skip
    re-downloads). Falls back to a per-job cache under ``dest_dir`` —
    the right choice for CI where every job is a clean runner anyway.
    """
    from huggingface_hub import hf_hub_download
    import time as _time

    cache_dir = None
    if not (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        cache_dir = str(dest_dir / "hf_cache")

    print(f"  download  {repo}@{revision[:8]} :: {file_in_repo}", flush=True)
    # Retry with exponential backoff for HF 429 rate limiting.
    # With 29 concurrent CI jobs all hitting HF, anonymous rate limits
    # are easily exceeded. An HF_TOKEN secret helps; retries handle
    # the remaining transient 429s.
    for attempt in range(5):
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=file_in_repo,
                revision=revision,
                cache_dir=cache_dir,
                repo_type=repo_type,
            )
            return Path(local)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** (attempt + 1) + (hash(repo) % 5)  # stagger
                print(f"  429 rate limit — retry {attempt+1}/4 in {wait}s", flush=True)
                _time.sleep(wait)
            else:
                raise


def _levenshtein(a: list, b: list) -> int:
    """Standard Wagner-Fischer edit distance over two sequences.

    Sequences can be characters (for CER) or tokens (for WER). Uses
    O(min(len(a), len(b))) extra space.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                cur[j - 1] + 1,           # insertion
                prev[j] + 1,              # deletion
                prev[j - 1] + (ca != cb), # substitution
            )
        prev = cur
    return prev[-1]


def _normalize_for_wer(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Standard ASR WER normalization: ignore case + punctuation differences
    so the metric reflects semantic word substitution, not cosmetic drift.
    """
    import re
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def compute_transcript_metrics(expected: str, actual: str) -> tuple[float, float]:
    """Return (CER, WER) for an expected/actual transcript pair.

    * **CER** — character-level edit distance / len(expected). Counts
      punctuation, case, and spacing. CER = 0 means byte-equal.
    * **WER** — token-level edit distance / word_count(expected),
      after lowercasing and stripping ASCII punctuation. Reflects
      semantic word substitution.

    A degenerate empty expected string yields 0.0 if actual is empty
    and inf otherwise.
    """
    if not expected:
        return (0.0, 0.0) if not actual else (float("inf"), float("inf"))
    cer = _levenshtein(list(expected), list(actual)) / len(expected)
    e_words = _normalize_for_wer(expected)
    a_words = _normalize_for_wer(actual)
    wer = _levenshtein(e_words, a_words) / max(1, len(e_words))
    return cer, wer


# The spoken AI-disclosure the CLI prepends to voice-cloned TTS output
# (examples/cli/crispasr_tts_disclaimer.h → crispasr_disclaimer::text()).
# `--no-watermark` removes only the inaudible watermark, NOT this spoken
# sentence, so the roundtrip ASR always transcribes it ahead of the actual
# phrase. It must be stripped before WER or every voice-clone roundtrip eats
# a fixed ~7-word prefix penalty.
_AI_DISCLOSURE = _normalize_for_wer("This audio was generated by artificial intelligence.")


def strip_ai_disclaimer(transcript: str) -> str:
    """Drop a leading spoken AI-disclosure from a roundtrip transcript.

    Fuzzy-matches the first len(_AI_DISCLOSURE) normalized words against the
    known disclosure (≤2 word mismatches, to tolerate minor ASR slips). On a
    match, returns the remainder (normalized, space-joined); otherwise returns
    the transcript unchanged. Harmless for no-voice / preset backends whose
    output has no disclosure — the fuzzy guard simply won't fire.
    """
    words = _normalize_for_wer(transcript)
    n = len(_AI_DISCLOSURE)
    if len(words) < n:
        return transcript
    mismatches = sum(1 for a, b in zip(words[:n], _AI_DISCLOSURE) if a != b)
    if mismatches <= 2:
        return " ".join(words[n:])
    return transcript


def run_transcript(crispasr_bin: Path, gguf: Path, sample: Path, *, srt: bool = False,
                   label: str = "transcript", extra_args: list[str] | None = None) -> str:
    """Run `crispasr -m gguf -f sample`, return the transcript line.

    The CLI prints the transcript on its own line near the end, after
    the "transcribed N s audio in M s (Kx realtime)" status line.
    Grab the last non-empty non-status line.
    """
    if srt:
        # Bilingual backends emit several cues/lines. Checking the final stdout
        # line would silently discard earlier cues and every source transcript.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="regression-srt-", dir=gguf.parent) as temp:
            prefix = Path(temp) / "decoded"
            _run_captured([str(crispasr_bin), "-m", str(gguf), "-f", str(sample),
                           "-osrt", "-of", str(prefix)],
                          check=True, timeout=600, label=label)
            cues = prefix.with_suffix(".srt").read_text()
            return " ".join(line.strip() for line in cues.splitlines()
                            if line.strip() and not line.strip().isdigit() and "-->" not in line)
    proc = _run_captured(
        [str(crispasr_bin), "-m", str(gguf), "-f", str(sample)] + list(extra_args or []),
        check=True,
        timeout=600,
        label=label,
    )
    # The transcript is on stdout; status lines go to stderr. Take stdout's
    # last non-empty line as the transcript (the CLI prints a single text
    # line for non-streaming mode).
    text_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not text_lines:
        die(
            "crispasr produced no stdout; "
            f"stderr tail: ...{proc.stderr[-400:]}"
        )
    return text_lines[-1]


# Diff-harness output line format (examples/cli/crispasr_diff_main.cpp):
#   [PASS] encoder_output         shape=[1024,176]       cos_min=0.999594  cos_mean=...
#   [FAIL] mel_spectrogram        shape=[80,1407]        cos_min=0.951084  cos_mean=...
_DIFF_LINE = re.compile(
    r"^\[(PASS|FAIL)\]\s+(\S+)\s+shape=\[\S+\]\s+cos_min=([0-9.\-eE]+)\s+cos_mean=([0-9.\-eE]+)"
)



def run_lid(entry: dict, manifest: dict, lid_gguf: Path, sample: Path,
            work_dir: Path, crispasr_bin: Path) -> str:
    """Return the language label a LID model assigns, for `lid_mode` entries.

    LID models emit a label, not a transcript, so plain `crispasr -m lid.gguf`
    cannot drive them (it exits 3). Two modes:

    * ``text``  - text LID (CLD3 / fastText): ``crispasr-lid -m gguf --text
      <lid_text>``; first field of the first output line.
    * ``audio`` - audio LID (silero / ecapa / firered): run the pinned host ASR
      entry ``lid_host_asr`` with ``-dl --lid-backend <lid_backend> --lid-model
      gguf`` and parse the ``LID -> language = 'xx'`` status line, so the label
      comes from the model under test, never from the host ASR.
    """
    import re
    mode = entry["lid_mode"]
    if mode == "text":
        lid_bin = crispasr_bin.parent / "crispasr-lid"
        if not lid_bin.exists():
            die(f"crispasr-lid binary not found at {lid_bin}; build target crispasr-lid")
        proc = _run_captured([str(lid_bin), "-m", str(lid_gguf), "--text", entry["lid_text"],
                              "--quiet"], check=True, timeout=120, label="lid-text")
        lines = [ln.split() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            die(f"crispasr-lid produced no output; stderr tail: ...{proc.stderr[-400:]}")
        return lines[0][0]
    if mode != "audio":
        die(f"unknown lid_mode {mode!r}")
    host = next((b for b in manifest["backends"] if b["name"] == entry["lid_host_asr"]), None)
    if host is None:
        die(f"lid_host_asr={entry['lid_host_asr']!r} not in manifest.backends")
    host_gguf = hf_download(host["gguf"]["repo"], host["gguf"]["file"],
                            host["gguf"]["revision"], work_dir)
    for companion in host["gguf"].get("companion_files", []):
        c_local = hf_download(host["gguf"]["repo"], companion, host["gguf"]["revision"], work_dir)
        dest = host_gguf.parent / Path(companion).name
        if not dest.exists():
            shutil.copy2(c_local, dest)
    proc = _run_captured([str(crispasr_bin), "--backend", host["backend_id"], "-m", str(host_gguf),
                          "-f", str(sample), "-dl", "--lid-backend", entry["lid_backend"],
                          "--lid-model", str(lid_gguf)],
                         check=True, timeout=600, label="lid-audio")
    # Two status formats: the LID module's own "crispasr[lid]: detected 'xx'
    # (p=..) via <backend>" and the run loop's "LID -> language = 'xx' (<src>, p=..)".
    found = [(m.group(1), m.group(2)) for m in
             re.finditer(r"crispasr\[lid\]: detected '([^']*)' \(p=[^)]*\) via (\S+)", proc.stderr)]
    found += re.findall(r"LID -> language = '([^']*)' \(([^,]*),", proc.stderr)
    if not found:
        die(f"no LID result line from --lid-backend {entry['lid_backend']}; "
            f"stderr tail: ...{proc.stderr[-400:]}")
    label, source = found[0]
    print(f"  lid source: {source}")
    if entry["lid_backend"] not in source:
        die(f"LID label came from {source!r}, not --lid-backend {entry['lid_backend']}")
    return label


def run_probe(entry: dict, gguf: Path, sample: Path, crispasr_bin: Path) -> int:
    """Gate a model whose output is not text, via `crispasr-model-probe`.

    The probe prints one JSON object. Kinds:

    * ``embedding`` (titanet) - ``{"dim", "embedding"}``; gate cosine >=
      ``probe_cos_min`` against the pinned ``probe_expected_embedding``.
    * ``codes`` (mimo-tokenizer) - ``{"n_codes", "codes"}``; gate the fraction
      of codes equal to the pinned ``probe_expected_codes`` >= ``probe_match_min``
      (RVQ argmin can flip a rare near-tie across CPU ISAs).

    With no pinned value yet the probe prints CAPTURE and fails, so a missing
    pin can never pass silently.
    """
    probe_bin = crispasr_bin.parent / "crispasr-model-probe"
    if not probe_bin.exists():
        die(f"crispasr-model-probe binary not found at {probe_bin}; build target crispasr-model-probe")
    kind = entry["probe"]
    proc = _run_captured([str(probe_bin), kind, str(gguf), str(sample)],
                         check=True, timeout=900, label=f"probe-{kind}")
    out = json.loads(proc.stdout)
    if kind == "embedding":
        got = out["embedding"]
        want = entry.get("probe_expected_embedding")
        if not want:
            print(f"\033[31m  CAPTURE\033[0m  no pinned embedding; got {json.dumps(got)}")
            return 1
        if len(got) != len(want):
            print(f"\033[31m  FAIL\033[0m  dim {len(got)} != pinned {len(want)}")
            return 1
        dot = sum(a * b for a, b in zip(got, want))
        cos = dot / (math.sqrt(sum(a * a for a in got)) * math.sqrt(sum(b * b for b in want)) or 1.0)
        norm_ratio = math.sqrt(sum(a * a for a in got)) / (math.sqrt(sum(b * b for b in want)) or 1.0)
        ok = cos >= float(entry.get("probe_cos_min", 0.999)) and abs(norm_ratio - 1.0) <= 0.01
        print(f"\033[{32 if ok else 31}m  {'PASS' if ok else 'FAIL'}\033[0m  "
              f"cos={cos:.6f} (min {entry.get('probe_cos_min', 0.999)}) norm_ratio={norm_ratio:.5f}")
        return int(not ok)
    if kind == "codes":
        got = out["codes"]
        want_file = entry.get("probe_expected_codes")
        if not want_file:
            print(f"\033[31m  CAPTURE\033[0m  no pinned codes; got {json.dumps(out)}")
            return 1
        want = json.loads((REPO_ROOT / want_file).read_text())["codes"]
        if len(got) != len(want):
            print(f"\033[31m  FAIL\033[0m  {len(got)} codes != pinned {len(want)}")
            return 1
        match = sum(a == b for a, b in zip(got, want)) / max(1, len(want))
        ok = match >= float(entry.get("probe_match_min", 0.99))
        print(f"\033[{32 if ok else 31}m  {'PASS' if ok else 'FAIL'}\033[0m  "
              f"code match={match:.4f} over {len(want)} (min {entry.get('probe_match_min', 0.99)})")
        return int(not ok)
    die(f"unknown probe kind {kind!r}")

def parse_diff_stdout(stdout: str) -> dict[str, float]:
    """Parse a crispasr-diff stdout blob into {stage_name: cos_min}.

    Module-level so the smoke test (and the Kaggle multi-backend
    driver) can re-use it without spawning subprocesses.
    """
    result: dict[str, float] = {}
    for ln in stdout.splitlines():
        m = _DIFF_LINE.match(ln.strip())
        if not m:
            continue
        result[m.group(2)] = float(m.group(3))
    return result


def evaluate_stage_thresholds(
    stages: dict[str, float], thresholds: dict[str, float],
    default: float | None = None, advisory: tuple[str, ...] | list[str] = (),
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]], list[str], list[tuple[str, float]]]:
    """Apply manifest thresholds to parsed cos_min map.

    `default` (manifest `stage_threshold_default`) gates every captured stage
    that has no explicit threshold — per-layer / conv-snapshot stages used to be
    INFO-only and could drift arbitrarily without failing anything. Stages whose
    name starts with a prefix in `advisory` (manifest `advisory_stages`) stay
    ungated extras. Without `default`, all unlisted stages are extras (old
    behaviour).

    Returns (passes, fails, missing, extras):
      passes  — [(stage, cos_min, threshold)] above threshold
      fails   — [(stage, cos_min, threshold)] below threshold
      missing — [stage] in thresholds but not in stages
      extras  — [(stage, cos_min)] captured but not gated
    """
    passes: list[tuple[str, float, float]] = []
    fails: list[tuple[str, float, float]] = []
    missing: list[str] = []
    for stage, threshold in thresholds.items():
        if stage not in stages:
            missing.append(stage)
            continue
        v = stages[stage]
        (passes if v >= threshold else fails).append((stage, v, threshold))
    extras: list[tuple[str, float]] = []
    for s in sorted(set(stages) - set(thresholds)):
        v = stages[s]
        if default is None or s.startswith(tuple(advisory)):
            extras.append((s, v))
        else:
            (passes if v >= default else fails).append((s, v, default))
    return passes, fails, missing, extras


def stage_gate_kwargs(entry: dict) -> dict:
    """Manifest entry -> the default/advisory kwargs of evaluate_stage_thresholds."""
    d = entry.get("stage_threshold_default")
    return {"default": None if d is None else float(d), "advisory": tuple(entry.get("advisory_stages", []))}


def run_diff(diff_bin: Path, backend_id: str, gguf: Path, ref: Path, sample: Path) -> dict:
    """Run crispasr-diff and return {stage_name: cos_min}.

    crispasr-diff exits non-zero (the count of failed stages) when any
    captured stage falls below its built-in 0.999 cos_min threshold,
    even if we accept that under the manifest's per-stage threshold.
    We do our own pass/fail accounting against `diff_thresholds`, so
    `check=False` here is intentional — we parse stdout regardless of
    the harness's verdict and only fail if a stage is missing or the
    process crashed outright.
    """
    proc = _run_captured(
        [str(diff_bin), backend_id, str(gguf), str(ref), str(sample)],
        check=False,
        timeout=600,
        label="diff",
    )
    if proc.returncode < 0:
        die(f"crispasr-diff died from signal {-proc.returncode}\n"
            f"  stderr tail: {proc.stderr[-400:]}")
    # Index-Echo's harness also gates magnitude, exact prompt IDs and cached
    # greedy IDs. Those failures have no cosine row; ignoring its exit status
    # would silently drop the port's required acceptance gates.
    if backend_id == "index-echo" and proc.returncode != 0:
        die(f"Index-Echo mandatory diff gate failed (rc={proc.returncode})\n"
            f"  stdout tail: {proc.stdout[-1600:]}\n  stderr tail: {proc.stderr[-400:]}")
    # crispasr-diff prints summary lines on stdout. Parse the [PASS]/[FAIL]
    # lines; ignore the diff harness's own pass/fail verdict — we apply our
    # own per-stage thresholds from the manifest.
    result = parse_diff_stdout(proc.stdout)
    if not result:
        die(
            f"crispasr-diff produced no parseable stage lines.\n"
            f"  stdout tail: {proc.stdout[-400:]}\n"
            f"  stderr tail: {proc.stderr[-400:]}"
        )
    return result


def transcript_gate(entry: dict, actual: str) -> tuple[bool, list[str]]:
    """The transcript pass/fail rule, shared by run_one.py and the Kaggle suite
    (which used to compare byte-exact and so failed entries GH passes).

    DEFAULT is a zero-tolerance metric gate, not byte-equal: a space-delimited
    transcript that differs ONLY in punctuation/case has WER=0 and passes, even
    with no explicit tolerance block — otherwise every byte-equal backend flaps
    red the moment CI flips a comma (fastconformer / wav2vec2 / mini-omni2 flip
    run-to-run). An explicit transcript_tolerance widens the bound (e.g. TTS
    roundtrips). Gating metric by language:
      * space-delimited: WER (invariant to punctuation/case decode ties); CER
        is reported as advisory only. The diff-harness stages catch drift.
      * CJK / no-whitespace: CER (whitespace-split WER is degenerate there).
    Returns (ok, printable lines).
    """
    expected = entry["expected_transcript"]
    if actual == expected:
        return True, ["\033[32m  PASS\033[0m  (byte-equal)", f"    {actual!r}"]
    tol = entry.get("transcript_tolerance") or {"cer_max": 0.0, "wer_max": 0.0}
    cer, wer = compute_transcript_metrics(expected, actual)
    cer_max = float(tol.get("cer_max", 0.0))
    wer_max = float(tol.get("wer_max", 0.0))
    if " " in expected.strip():
        ok = wer <= wer_max
        gate = f"wer={wer:.4f} (max {wer_max}) [gate]  cer={cer:.4f} [advisory]"
    else:
        ok = cer <= cer_max
        gate = f"cer={cer:.4f} (max {cer_max}) [gate, CJK]  wer={wer:.4f} [advisory]"
    verdict = "\033[32m  PASS\033[0m" if ok else "\033[31m  FAIL\033[0m"
    suffix = " (within tolerance)" if ok else " (over tolerance)"
    return ok, [f"{verdict}  {gate}{suffix}", f"    expected: {expected!r}", f"    actual:   {actual!r}"]


def validate_exact_diff_transcript(entry: dict) -> None:
    """Reject incomplete full-precision controls before downloading models."""
    required = entry.get("require_exact_diff_transcript", False)
    if not isinstance(required, bool):
        raise ValueError("require_exact_diff_transcript must be boolean")
    if required and (not entry.get("diff_gguf") or entry.get("skip_diff", False)
                     or not entry.get("transcript_reference")):
        raise ValueError("Exact diff transcript requires diff_gguf, stage diff and independent transcript_reference")


def exact_diff_transcript_gate(entry: dict, actual: str) -> tuple[bool, list[str]]:
    """The F16 control always requires zero error against the original target."""
    return transcript_gate({**entry, "transcript_tolerance": {"cer_max": 0.0, "wer_max": 0.0}}, actual)


def verify_transcript_reference(entry: dict, reference: Path, sample: Path,
                                work_dir: Path) -> str:
    """Verify a transcript target against pinned, independent CTC logits.

    This checks provenance before native inference. It cannot silently bless a
    new native output by replacing the expected string. Human accuracy is
    reported separately because upstream-model parity can include model errors.
    """
    import hashlib
    from itertools import groupby
    import gguf
    import numpy as np

    source = entry["transcript_reference"]
    def check_hash(path: Path, expected: str):
        with path.open("rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
            actual = h.hexdigest()
        if actual != expected:
            raise ValueError(f"Transcript reference SHA256 mismatch: {path.name}")
    check_hash(reference, source["sha256"])
    check_hash(sample, source["sample_sha256"])
    pin = source["vocab"]
    vocab_path = hf_download(pin["repo"], pin["file"], pin["revision"], work_dir)
    check_hash(vocab_path, pin["sha256"])
    vocab = json.loads(vocab_path.read_text())
    inv = {i: text for text, i in vocab.items()}
    reader = gguf.GGUFReader(str(reference))
    logits = next(t.data for t in reader.tensors if t.name == source["stage"])
    if logits.ndim != 2 or logits.shape[1] != len(vocab) or not np.isfinite(logits).all():
        raise ValueError("Invalid independent CTC logit grid")
    blank = vocab[source["blank_token"]]
    ids = np.argmax(logits, axis=1).tolist()
    decoded = "".join(inv[i] for i, _ in groupby(ids) if i != blank)
    decoded = decoded.replace(source["word_delimiter_token"], " ").strip()
    if _normalize_for_wer(decoded) != _normalize_for_wer(entry["expected_transcript"]):
        raise ValueError(f"Transcript target disagrees with independent Python reference: {decoded!r}")
    print(f"[transcript-reference] SHA256 verified; independent Python decode: {decoded!r}")
    return decoded


def regression_for(name: str, manifest: dict, work_dir: Path,
                   crispasr_bin: Path, diff_bin: Path) -> int:
    """Run one backend's regression. Return number of failures."""
    entry = next((b for b in manifest["backends"] if b["name"] == name), None)
    if entry is None:
        die(f"backend '{name}' not in manifest")

    validate_exact_diff_transcript(entry)

    gguf_local = hf_download(
        entry["gguf"]["repo"],
        entry["gguf"]["file"],
        entry["gguf"]["revision"],
        work_dir,
    )
    # Some backends (e.g. moonshine) need companion files (tokenizer.bin)
    # placed in the same HF cache snapshot dir as the GGUF. The manifest
    # lists them under `gguf.companion_files`; each is downloaded from the
    # same repo+revision as the GGUF and symlinked/placed next to the GGUF.
    for companion in entry["gguf"].get("companion_files", []):
        companion_local = hf_download(
            entry["gguf"]["repo"],
            companion,
            entry["gguf"]["revision"],
            work_dir,
        )
        # Place companion in same directory as gguf_local so backends can
        # find it by relative path (e.g. tokenizer.bin next to moonshine.gguf).
        companion_dest = gguf_local.parent / Path(companion).name
        if not companion_dest.exists():
            import shutil
            shutil.copy2(companion_local, companion_dest)
    skip_diff = entry.get("skip_diff", False)
    ref_local: Path | None = None
    if not skip_diff:
        ref_local = hf_download(
            manifest["fixtures"]["repo"],
            entry["fixture_ref_path"],
            manifest["fixtures"]["revision"],
            work_dir,
        )
    # Sample WAVs live alongside the ref dumps in the fixtures HF repo
    # (samples/.gitignore in CrispASR excludes them from the source
    # tree to keep clones small). An in-tree fallback `sample` field
    # remains supported for legacy entries that lived in the repo.
    if "fixture_sample_path" in entry:
        sample = hf_download(
            manifest["fixtures"]["repo"],
            entry["fixture_sample_path"],
            manifest["fixtures"]["revision"],
            work_dir,
        )
    elif "sample_hf" in entry:
        # A sample pinned in its own repo/revision (e.g. a FLEURS clip added to
        # the fixtures repo after fixtures.revision), so the shared pin stays put.
        s = entry["sample_hf"]
        sample = hf_download(s["repo"], s["file"], s["revision"], work_dir, repo_type=s.get("repo_type", "model"))
    else:
        sample = REPO_ROOT / entry["sample"]
        if not sample.exists():
            die(f"sample WAV missing: {sample}")

    failures = 0
    skip_diff = entry.get("skip_diff", False)
    if "transcript_reference" in entry:
        if ref_local is None:
            raise ValueError("Transcript provenance requires an independent reference dump")
        verify_transcript_reference(entry, ref_local, sample, work_dir)

    # ----- 1. Transcript (or language-ID label for `lid_mode` entries) -----
    if "probe" in entry:
        print(f"\n[probe] {name}")
        return run_probe(entry, gguf_local, sample, crispasr_bin)
    # `asr_extra_files`: pinned files from any repo, each passed as `<arg> <path>`
    # (e.g. mimo-asr's audio tokenizer via --codec-model).
    extra_args: list[str] = []
    for extra in entry.get("asr_extra_files", []):
        local = hf_download(extra["repo"], extra["file"], extra["revision"], work_dir)
        extra_args += [extra["arg"], str(local)]
    if entry.get("lid_mode"):
        print(f"\n[lid] {name}")
        actual = run_lid(entry, manifest, gguf_local, sample, work_dir, crispasr_bin)
    else:
        print(f"\n[transcript] {name}")
        actual = run_transcript(crispasr_bin, gguf_local, sample,
                                srt=entry.get("transcript_format") == "srt",
                                extra_args=extra_args)
    # WER-normalised gate, shared with the Kaggle suite; see transcript_gate.
    ok, lines = transcript_gate(entry, actual)
    for ln in lines:
        print(ln)
    if not ok:
        failures += 1
    if "human_reference_transcript" in entry:
        _, human_wer = compute_transcript_metrics(entry["human_reference_transcript"], actual)
        print(f"[human-accuracy/advisory] WER={human_wer:.4f}; upstream parity is a separate gate")

    # ----- 2. Diff harness -----
    if skip_diff:
        print(f"\n[diff-harness] {name}  \033[33mSKIP\033[0m (skip_diff=true; no ref dump baked yet)")
        return failures

    # `diff_gguf` (optional): run the stage diff on a full-precision GGUF while
    # the transcript above stays on the shipped quant. A q4_k encoder sits at
    # cos ~0.95 against an F32 reference, so a gate on it can only be loose;
    # the F16 file of the same model matches at >= 0.99999 and can be gated tight.
    diff_gguf = gguf_local
    if entry.get("diff_gguf"):
        dg = entry["diff_gguf"]
        diff_gguf = hf_download(dg.get("repo", entry["gguf"]["repo"]), dg["file"],
                                dg.get("revision", entry["gguf"]["revision"]), work_dir)
    if entry.get("require_exact_diff_transcript", False):
        print(f"\n[transcript-f16] {name}")
        control = run_transcript(crispasr_bin, diff_gguf, sample,
                                 srt=entry.get("transcript_format") == "srt", label="transcript-f16")
        control_ok, control_lines = exact_diff_transcript_gate(entry, control)
        for line in control_lines:
            print(line)
        failures += int(not control_ok)
    print(f"\n[diff-harness] {name}  ({diff_gguf.name})")
    stages = run_diff(diff_bin, entry["backend_id"], diff_gguf, ref_local, sample)
    thresholds = entry["diff_thresholds"]
    passes, fails, missing, extras = evaluate_stage_thresholds(stages, thresholds, **stage_gate_kwargs(entry))
    for stage in missing:
        print(f"\033[31m  FAIL\033[0m {stage}  (required stage not captured by diff harness)")
    for stage, cos_min, threshold in passes:
        print(f"\033[32m  PASS\033[0m {stage:24s} cos_min={cos_min:.6f}  threshold={threshold}")
    for stage, cos_min, threshold in fails:
        print(f"\033[31m  FAIL\033[0m {stage:24s} cos_min={cos_min:.6f}  threshold={threshold}")
    failures += len(fails) + len(missing)
    # Captured but ungated: no stage_threshold_default, or listed in
    # advisory_stages. Add a threshold (or a default) if the stage is meant to hold.
    for stage, cos_min in extras:
        print(f"\033[33m  INFO\033[0m {stage} cos_min={cos_min:.6f} "
              f"(ungated: not in diff_thresholds / advisory_stages)")

    return failures

# ---------------------------------------------------------------------------
# Task backends (manifest.task_backends): models whose output is not a plain
# transcript — translation, source separation, music analysis, speech-to-speech,
# diarization, watermark detection. Every task is gated against a ground truth
# that exists independently of the model under test: a reference translation,
# the clean sources a mixture was built from, or the note/beat/chord grid a
# synthetic input was rendered from. Native-capture pins are used only where no
# such truth exists (none of the kinds below need one).
# ---------------------------------------------------------------------------

# Standard-tuning open-string MIDI notes, low E to high E (TabCNN string order).
_GUITAR_OPEN_MIDI = (40, 45, 50, 55, 59, 64)
_PITCH_CLASSES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
                  "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def _midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _write_wav(path: Path, x, sr: int) -> None:
    import numpy as np
    import wave
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767.0).round().astype("<i2").tobytes())


def _read_wav(path: Path):
    """Return (mono float64 samples, sample rate) for a PCM16/PCM32/float WAV."""
    import numpy as np
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width {sw}")
    except wave.Error:
        # WAVE_FORMAT_IEEE_FLOAT — parse the chunks by hand.
        import struct
        data = path.read_bytes()
        pos, fmt, payload = 12, None, None
        while pos + 8 <= len(data):
            cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
            elif cid == b"data":
                payload = data[pos + 8:pos + 8 + size]
            pos += 8 + size + (size & 1)
        if fmt is None or payload is None or fmt[0] != 3 or fmt[5] != 32:
            raise ValueError(f"unreadable WAV {path}")
        ch, sr = fmt[1], fmt[2]
        x = np.frombuffer(payload, "<f4").astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _resample(x, sr_in: int, sr_out: int):
    import numpy as np
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n_out) * (sr_in / sr_out), np.arange(len(x)), x)


def _render_notes(notes: list, sr: int, timbre: str, total_s: float):
    """Render [[onset_s, dur_s, midi], ...] to a mono signal.

    ``piano``: 8 decaying harmonics with a short attack (struck-string-like).
    ``pluck``: Karplus-Strong plucked string (guitar-like), deterministic seed.
    """
    import numpy as np
    out = np.zeros(int(total_s * sr) + 1)
    rng = np.random.default_rng(1234)
    for onset, dur, midi in notes:
        f0 = _midi_hz(midi)
        n = int(dur * sr)
        t = np.arange(n) / sr
        if timbre == "pluck":
            period = max(2, int(round(sr / f0)))
            buf = rng.uniform(-1, 1, period)
            y = np.empty(n)
            for i in range(n):
                y[i] = buf[i % period]
                buf[i % period] = 0.996 * 0.5 * (buf[i % period] + buf[(i + 1) % period])
            y *= np.minimum(1.0, (n - np.arange(n)) / (0.02 * sr))  # release
        else:
            y = sum((0.6 ** (k - 1)) * np.sin(2 * np.pi * k * f0 * t) * np.exp(-t * (1.5 + 0.7 * k))
                    for k in range(1, 9) if k * f0 < sr / 2)
            y *= np.minimum(1.0, t / 0.005) * np.minimum(1.0, (dur - t) / 0.03)
        s = int(onset * sr)
        out[s:s + n] += y[: len(out) - s]
    return out


def _render_clicks(bpm: float, total_s: float, sr: int, beats_per_bar: int = 4):
    """Kick on every beat (louder on the downbeat) plus hi-hat eighths."""
    import numpy as np
    out = np.zeros(int(total_s * sr) + 1)
    period = 60.0 / bpm
    rng = np.random.default_rng(99)
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
    hat = rng.uniform(-1, 1, int(0.04 * sr)) * np.exp(-np.arange(int(0.04 * sr)) / (0.006 * sr))
    i = 0
    while i * period / 2 < total_s - 0.3:
        s = int(i * period / 2 * sr)
        if i % 2 == 0:
            amp = 1.0 if (i // 2) % beats_per_bar == 0 else 0.7
            out[s:s + len(kick)] += amp * kick[: len(out) - s]
        out[s:s + len(hat)] += 0.15 * hat[: len(out) - s]
        i += 1
    return out


def _chord_notes(chords: list, block_s: float) -> list:
    """[[root, quality], ...] -> note list: root-position triad plus bass octave."""
    notes = []
    for i, (root, quality) in enumerate(chords):
        r = 48 + _PITCH_CLASSES[root]
        third = 3 if quality == "min" else 4
        for m in (r - 12, r, r + third, r + 7, r + 12):
            notes.append([i * block_s, block_s, m])
    return notes


def _synth_input(spec: dict, work_dir: Path, manifest: dict) -> tuple[Path, dict]:
    """Render a deterministic synthetic input; return (wav, ground truth)."""
    import numpy as np
    kind = spec["kind"]
    sr = int(spec.get("sr", 22050))
    path = work_dir / f"synth_{kind}.wav"
    truth: dict = {}
    if kind in ("notes", "pluck"):
        notes = spec["notes"]
        total = max(o + d for o, d, _ in notes) + 0.5
        x = _render_notes(notes, sr, "pluck" if kind == "pluck" else "piano", total)
        truth["notes"] = notes
    elif kind == "chords":
        notes = _chord_notes(spec["chords"], float(spec["block_s"]))
        total = len(spec["chords"]) * float(spec["block_s"])
        x = _render_notes(notes, sr, "piano", total)
        # Re-strike each block every second so the chord keeps sounding.
        for rep in np.arange(1.0, float(spec["block_s"]) - 0.2, 1.0):
            x += _render_notes([[o + rep, d - rep, m] for o, d, m in notes], sr, "piano", total)[: len(x)]
        truth["chords"] = spec["chords"]
    elif kind == "clicks":
        x = _render_clicks(float(spec["bpm"]), float(spec["duration_s"]), sr)
    elif kind == "speech_music":
        speech, ssr = _read_wav(REPO_ROOT / spec["speech_sample"])
        speech = _resample(speech, ssr, sr)
        total = len(speech) / sr
        music = _render_notes(_chord_notes(spec["chords"], total / len(spec["chords"])), sr, "piano", total)
        music += 0.8 * _render_clicks(float(spec.get("bpm", 100)), total, sr)
        music = music[: len(speech)]
        speech_rms = np.sqrt(np.mean(speech ** 2))
        music *= speech_rms / np.sqrt(np.mean(music ** 2)) * 10 ** (-float(spec["snr_db"]) / 20)
        x = speech + music
        peak = np.max(np.abs(x)) / 0.9
        truth["speech"], truth["music"], truth["sr"] = speech / peak, music / peak, sr
        x = x / peak
    else:
        die(f"unknown synth kind {kind!r}")
    x = x / max(1e-9, np.max(np.abs(x))) * 0.8 if kind != "speech_music" else x
    _write_wav(path, x, sr)
    return path, truth


def _si_sdr(est, ref) -> float:
    import numpy as np
    n = min(len(est), len(ref))
    est, ref = est[:n] - np.mean(est[:n]), ref[:n] - np.mean(ref[:n])
    proj = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    noise = est - proj
    return float(10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def _json_stdout(proc: subprocess.CompletedProcess) -> dict:
    text = proc.stdout
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON on stdout; stderr tail: ...{proc.stderr[-400:]}")
    return json.loads(text[start:])


def _report(ok: bool, line: str) -> int:
    print(f"\033[{32 if ok else 31}m  {'PASS' if ok else 'FAIL'}\033[0m  {line}")
    return int(not ok)


def _check_pitch(entry: dict, out: dict, truth: dict) -> int:
    import numpy as np
    frames = out["frames"]
    if not frames:
        return _report(False, "no pitch frames")
    keys = frames[0].keys()
    tkey = "time_ms" if "time_ms" in keys else "time"
    fkey = "f0_hz" if "f0_hz" in keys else "f0"
    pkey = next((k for k in keys if "prob" in k or "conf" in k), None)
    scale = 1e-3 if tkey == "time_ms" else 1.0
    tol, fails = float(entry["cents_tol"]), 0
    for onset, dur, midi in truth["notes"]:
        lo, hi = onset + 0.15 * dur, onset + 0.85 * dur
        seg = [f for f in frames if lo <= f[tkey] * scale <= hi]
        voiced = [f for f in seg if pkey is None or f[pkey] >= float(entry.get("voiced_min", 0.5))]
        if len(voiced) < 0.6 * max(1, len(seg)) or not voiced:
            fails += _report(False, f"note {midi}: only {len(voiced)}/{len(seg)} frames voiced")
            continue
        med = float(np.median([f[fkey] for f in voiced]))
        cents = 1200 * np.log2(max(med, 1e-3) / _midi_hz(midi))
        fails += _report(abs(cents) <= tol, f"note {midi} ({_midi_hz(midi):.1f} Hz): median f0 {med:.2f} Hz, "
                                            f"{cents:+.1f} cents (tol {tol}), {len(voiced)}/{len(seg)} voiced")
    return fails


def _check_beats(entry: dict, out: dict, spec: dict) -> int:
    import numpy as np
    beats = [b["time"] for b in out["beats"]]
    period = 60.0 / float(spec["bpm"])
    truth = np.arange(0.0, float(spec["duration_s"]) - 0.3, period)
    tol = float(entry.get("beat_tol_s", 0.07))
    hit = sum(1 for t in truth if beats and min(abs(b - t) for b in beats) <= tol)
    prec = sum(1 for b in beats if min(abs(b - t) for t in truth) <= tol) / max(1, len(beats))
    rec = hit / len(truth)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    tempo = float(out.get("tempo_bpm", 0.0))
    fails = _report(f1 >= float(entry["beat_f1_min"]),
                    f"beat F1={f1:.3f} (P={prec:.3f} R={rec:.3f}, {len(beats)} beats vs {len(truth)} true, "
                    f"tol {tol}s; min {entry['beat_f1_min']})")
    fails += _report(abs(tempo - float(spec["bpm"])) <= float(entry.get("tempo_tol_bpm", 3.0)),
                     f"tempo {tempo:.2f} BPM vs {spec['bpm']}")
    return fails


def _parse_chord(label: str) -> tuple[int, str] | None:
    """'C', 'C:maj', 'A:min', 'Am', 'F#:min7' -> (pitch class, 'maj'|'min'); None for N/X."""
    m = re.match(r"^([A-G][#b]?)(?::?(.*))?$", label.strip())
    if not m:
        return None
    rest = (m.group(2) or "").lower()
    quality = "min" if rest.startswith("min") or (rest.startswith("m") and not rest.startswith("maj")) else "maj"
    return _PITCH_CLASSES[m.group(1)], quality


def _check_chords(entry: dict, out: dict, spec: dict) -> int:
    block = float(spec["block_s"])
    correct, lines = 0, []
    for i, (root, quality) in enumerate(spec["chords"]):
        lo, hi = i * block + 0.25 * block, (i + 1) * block - 0.25 * block
        weight: dict = {}
        for c in out["chords"]:
            ov = min(hi, c["end_ms"] / 1000) - max(lo, c["start_ms"] / 1000)
            if ov > 0:
                weight[c["chord"]] = weight.get(c["chord"], 0.0) + ov
        top = max(weight, key=weight.get) if weight else "<none>"
        ok = _parse_chord(top) == (_PITCH_CLASSES[root], quality)
        correct += ok
        lines.append(f"{root}:{quality}->{top}{'' if ok else ' (WRONG)'}")
    frac = correct / len(spec["chords"])
    return _report(frac >= float(entry["chord_acc_min"]),
                   f"chord accuracy {correct}/{len(spec['chords'])} (min {entry['chord_acc_min']}): "
                   + ", ".join(lines))


def _check_piano(entry: dict, out: dict, truth: dict) -> int:
    notes = out["notes"]
    tol = float(entry.get("onset_tol_s", 0.1))
    used, hit = set(), 0
    for onset, _, midi in truth["notes"]:
        j = next((k for k, n in enumerate(notes) if k not in used and n["midi"] == midi
                  and abs(n["onset"] - onset) <= tol), None)
        if j is not None:
            used.add(j)
            hit += 1
    prec, rec = hit / max(1, len(notes)), hit / len(truth["notes"])
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    got = " ".join(f"{n['midi']}@{n['onset']:.2f}" for n in notes[:24])
    # Recall is the primary gate (every rendered note must be found at its exact
    # pitch and onset); precision has its own, lower floor because the additive
    # synthetic timbre legitimately excites some harmonic/re-onset detections,
    # while note spam or a pitch-shifted transcription still fails it.
    ok = rec >= float(entry["note_recall_min"]) and prec >= float(entry["note_precision_min"])
    return _report(ok, f"note recall={rec:.3f} (min {entry['note_recall_min']}) precision={prec:.3f} "
                       f"(min {entry['note_precision_min']}) F1={f1:.3f}; {len(notes)} detected vs "
                       f"{len(truth['notes'])} true; onset tol {tol}s; got: {got}")


def _check_tab(entry: dict, out: dict, truth: dict) -> int:
    frames, silent = out["frames"], out.get("silent_class")
    good = total = 0
    for onset, dur, midi in truth["notes"]:
        for f in frames:
            if onset + 0.1 * dur <= f["time"] <= onset + 0.7 * dur:
                total += 1
                # Each string is {"fret": n, "logp": x}; fret -1 (or the silent
                # class) means the string is not sounding.
                frets = [fr["fret"] if isinstance(fr, dict) else fr for fr in f["frets"]]
                pitches = {_GUITAR_OPEN_MIDI[s] + fr for s, fr in enumerate(frets)
                           if fr is not None and fr != silent and fr >= 0}
                good += midi in pitches
    frac = good / max(1, total)
    return _report(frac >= float(entry["frame_acc_min"]),
                   f"frames sounding the true pitch {good}/{total} = {frac:.3f} (min {entry['frame_acc_min']})")


def _check_separation(entry: dict, stems: dict, truth: dict) -> int:
    import numpy as np
    speech_name = entry["speech_stem"]
    if speech_name not in stems:
        return _report(False, f"speech stem {speech_name!r} not produced (got {sorted(stems)})")
    sr = truth["sr"]
    est = _resample(stems[speech_name][0], stems[speech_name][1], sr)
    mix_sdr = _si_sdr(truth["speech"] + truth["music"], truth["speech"])
    sdr = _si_sdr(est, truth["speech"])
    rms_ratio = float(np.sqrt(np.mean(est[: len(truth['speech'])] ** 2)) /
                      np.sqrt(np.mean(truth["speech"] ** 2)))
    fails = _report(sdr >= float(entry["si_sdr_min_db"]) and sdr - mix_sdr >= float(entry["si_sdr_gain_min_db"]),
                    f"{speech_name}: SI-SDR vs clean speech {sdr:.2f} dB (min {entry['si_sdr_min_db']}), "
                    f"mixture {mix_sdr:.2f} dB, gain {sdr - mix_sdr:+.2f} dB (min {entry['si_sdr_gain_min_db']})")
    fails += _report(0.5 <= rms_ratio <= 2.0, f"{speech_name}: RMS ratio vs clean speech {rms_ratio:.3f} (0.5..2.0)")
    return fails


def _entry_input(entry: dict, manifest: dict, work_dir: Path) -> tuple[Path, dict]:
    if "synth" in entry:
        return _synth_input(entry["synth"], work_dir, manifest)
    if "fixture_sample_path" in entry:
        return hf_download(manifest["fixtures"]["repo"], entry["fixture_sample_path"],
                           manifest["fixtures"]["revision"], work_dir), {}
    if "sample_hf" in entry:
        s = entry["sample_hf"]
        return hf_download(s["repo"], s["file"], s["revision"], work_dir, repo_type=s.get("repo_type", "model")), {}
    sample = REPO_ROOT / entry["sample"]
    if not sample.exists():
        die(f"sample missing: {sample}")
    return sample, {}


def _asr_text(manifest: dict, asr_name: str, wav: Path, work_dir: Path, crispasr_bin: Path,
              label: str) -> str:
    asr = next((b for b in manifest["backends"] if b["name"] == asr_name), None)
    if asr is None:
        die(f"ASR entry {asr_name!r} not in manifest.backends")
    g = hf_download(asr["gguf"]["repo"], asr["gguf"]["file"], asr["gguf"]["revision"], work_dir)
    for companion in asr["gguf"].get("companion_files", []):
        c = hf_download(asr["gguf"]["repo"], companion, asr["gguf"]["revision"], work_dir)
        if not (g.parent / Path(companion).name).exists():
            shutil.copy2(c, g.parent / Path(companion).name)
    proc = _run_captured([str(crispasr_bin), "--backend", asr["backend_id"], "-m", str(g), "-f", str(wav),
                          "--no-prints"] + list(asr.get("roundtrip_extra_args", [])),
                         check=True, timeout=600, label=label)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return " ".join(lines)


def task_for(name: str, manifest: dict, work_dir: Path, crispasr_bin: Path, diff_bin: Path) -> int:
    """Run one manifest.task_backends entry. Returns the number of failures."""
    entry = next((b for b in manifest.get("task_backends", []) if b["name"] == name), None)
    if entry is None:
        die(f"task backend '{name}' not in manifest.task_backends")
    task = entry["task"]
    g = entry["gguf"]
    gguf = hf_download(g["repo"], g["file"], g["revision"], work_dir)
    for companion in g.get("companion_files", []):
        c = hf_download(g["repo"], companion, g["revision"], work_dir)
        if not (gguf.parent / Path(companion).name).exists():
            shutil.copy2(c, gguf.parent / Path(companion).name)
    extra = list(entry.get("cli_extra_args", []))
    timeout = int(entry.get("timeout_s", 900))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[task:{task}] {name}  ({g['repo']}@{g['revision'][:8]}::{g['file']})")

    if task == "translate":
        fails = 0
        for i, case in enumerate(entry["cases"]):
            proc = _run_captured([str(crispasr_bin), "-m", str(gguf), "--text", case["text"],
                                  "-sl", case["sl"], "-tl", case["tl"], "--no-prints"] + extra,
                                 check=True, timeout=timeout, label=f"translate-{i}")
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            actual = lines[-1] if lines else ""
            ok, out = transcript_gate({"expected_transcript": case["expected"],
                                       "transcript_tolerance": entry["transcript_tolerance"]}, actual)
            print(f"  {case['sl']}->{case['tl']}: {case['text']!r}")
            for ln in out:
                print(ln)
            fails += int(not ok)
        return fails

    if task == "diff":
        sample, _ = _entry_input(entry, manifest, work_dir)
        ref = hf_download(manifest["fixtures"]["repo"], entry["fixture_ref_path"],
                          manifest["fixtures"]["revision"], work_dir)
        stages = run_diff(diff_bin, entry["backend_id"], gguf, ref, sample)
        passes, fails_, missing, extras = evaluate_stage_thresholds(stages, entry["diff_thresholds"],
                                                                    **stage_gate_kwargs(entry))
        for st in missing:
            print(f"\033[31m  FAIL\033[0m {st}  (required stage not captured by diff harness)")
        for st, v, th in passes:
            print(f"\033[32m  PASS\033[0m {st:24s} cos_min={v:.6f}  threshold={th}")
        for st, v, th in fails_:
            print(f"\033[31m  FAIL\033[0m {st:24s} cos_min={v:.6f}  threshold={th}")
        for st, v in extras:
            print(f"\033[33m  INFO\033[0m {st} cos_min={v:.6f} (ungated)")
        n = len(passes) + len(fails_)
        if n == 0:
            print("\033[31m  FAIL\033[0m  no gated stage was compared")
            return 1
        print(f"  {n} gated stage(s) compared")
        return len(fails_) + len(missing)

    sample, truth = _entry_input(entry, manifest, work_dir)
    verb = {"pitch": ["--pitch", "--pitch-format", "json"], "beats": ["--beats", "--beats-format", "json"],
            "chords": ["--chords", "--chords-format", "json"], "piano": ["--piano", "--piano-format", "json"],
            "tab": ["--tab", "--tab-format", "json"]}
    if task in verb:
        proc = _run_captured([str(crispasr_bin)] + verb[task] + ["-m", str(gguf), "-f", str(sample)] + extra,
                             check=True, timeout=timeout, label=task)
        out = _json_stdout(proc)
        if task == "pitch":
            return _check_pitch(entry, out, truth)
        if task == "beats":
            return _check_beats(entry, out, entry["synth"])
        if task == "chords":
            return _check_chords(entry, out, entry["synth"])
        if task == "piano":
            return _check_piano(entry, out, truth)
        return _check_tab(entry, out, truth)

    if task == "separate":
        out_dir = work_dir / "stems"
        out_dir.mkdir(exist_ok=True)
        _run_captured([str(crispasr_bin), "--separate", "-m", str(gguf), "-f", str(sample),
                       "--sep-output-dir", str(out_dir)] + extra, check=True, timeout=timeout, label="separate")
        prefix = sample.stem + "_"
        stems = {p.stem[len(prefix):]: _read_wav(p) for p in sorted(out_dir.glob(prefix + "*.wav"))}
        print(f"  stems: {sorted(stems)}")
        return _check_separation(entry, stems, truth)

    if task == "s2s_asr":
        out_wav = work_dir / f"s2s_{name}.wav"
        proc = _run_captured([str(crispasr_bin), "-m", str(gguf), "-f", str(sample), "--s2s",
                              "--s2s-output", str(out_wav), "--no-prints"] + extra,
                             check=False, timeout=timeout, label="s2s")
        if proc.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size < 1000:
            return _report(False, f"s2s produced no audio (rc={proc.returncode}); "
                                  f"stderr tail: ...{proc.stderr[-300:]}")
        x, sr = _read_wav(out_wav)
        print(f"  s2s output {len(x) / sr:.2f} s @ {sr} Hz")
        actual = strip_ai_disclaimer(_asr_text(manifest, entry["roundtrip_asr_backend"], out_wav, work_dir,
                                               crispasr_bin, "s2s-asr"))
        ok, out = transcript_gate(entry, actual)
        for ln in out:
            print(ln)
        return int(not ok)

    if task == "diarize":
        host = next(b for b in manifest["backends"] if b["name"] == entry["host_asr"])
        hg = hf_download(host["gguf"]["repo"], host["gguf"]["file"], host["gguf"]["revision"], work_dir)
        # Two real speakers back to back: the first sample, then the second.
        a, sra = _read_wav(sample)
        b, srb = _entry_input(entry["second_speaker"], manifest, work_dir)[0], None
        b, srb = _read_wav(b)
        import numpy as np
        b = _resample(b, srb, sra)
        gap = np.zeros(int(0.5 * sra))
        two = work_dir / "two_speakers.wav"
        _write_wav(two, np.concatenate([a, gap, b]), sra)
        boundary = (len(a) + len(gap) / 2) / sra
        prefix = work_dir / "diar"
        _run_captured([str(crispasr_bin), "--backend", host["backend_id"], "-m", str(hg), "-f", str(two),
                       "--diarize", "--diarize-method", entry["diarize_method"], "--diarize-embedder", str(gguf),
                       "-ojf", "-of", str(prefix), "--no-prints"] + extra,
                      check=True, timeout=timeout, label="diarize")
        data = json.loads(prefix.with_suffix(".json").read_text())
        segs = data.get("transcription") or data.get("segments") or []
        labelled = []
        for s in segs:
            spk = s.get("speaker")
            if spk is None:
                continue
            off = s.get("offsets") or {}
            t0 = (off.get("from", 0) / 1000.0) if off else float(s.get("start", 0))
            t1 = (off.get("to", 0) / 1000.0) if off else float(s.get("end", 0))
            labelled.append((t0, t1, str(spk)))
        if not labelled:
            return _report(False, f"no speaker-labelled segments in {prefix.with_suffix('.json').name}")
        first = {sp for t0, t1, sp in labelled if t1 <= boundary + 0.3}
        second = {sp for t0, t1, sp in labelled if t0 >= boundary - 0.3}
        print("  segments: " + "; ".join(f"{t0:.2f}-{t1:.2f} {sp}" for t0, t1, sp in labelled))
        ok = len(first) == 1 and len(second) == 1 and first != second
        return _report(ok, f"speakers before boundary {boundary:.2f}s: {sorted(first)}, after: {sorted(second)} "
                           "(want exactly one each, different)")

    if task == "watermark":
        fails = 0
        tts = next(b for b in manifest["tts_backends"] if b["name"] == entry["marking_tts"])
        tg = hf_download(tts["gguf"]["repo"], tts["gguf"]["file"], tts["gguf"]["revision"], work_dir)
        voice = hf_download(tts["voice"]["repo"], tts["voice"]["file"], tts["voice"]["revision"], work_dir)
        marked = work_dir / "marked.wav"
        _run_captured([str(crispasr_bin), "-m", str(tg), "--voice", str(voice), "--tts", entry["tts_phrase"],
                       "--tts-output", str(marked), "--seed", "1234", "--watermark-model", str(gguf),
                       "--no-prints"], check=True, timeout=timeout, label="watermark-embed")
        for label, wav, want in (("marked TTS", marked, True), ("unmarked sample", sample, False)):
            proc = _run_captured([str(crispasr_bin), "--detect-watermark", str(wav), "--watermark-model", str(gguf)],
                                 check=True, timeout=timeout, label=f"watermark-detect-{int(want)}")
            m = re.search(r"Watermark confidence: ([0-9.]+)", proc.stdout)
            detector = re.search(r"Detector: (.*)", proc.stdout)
            if not m or not detector or "AudioSeal" not in detector.group(1):
                fails += _report(False, f"{label}: no AudioSeal confidence line; stdout: {proc.stdout[-300:]!r}")
                continue
            conf = float(m.group(1))
            ok = conf >= float(entry["marked_min"]) if want else conf <= float(entry["unmarked_max"])
            fails += _report(ok, f"{label}: AudioSeal confidence {conf:.4f} "
                                 f"({'>= ' + str(entry['marked_min']) if want else '<= ' + str(entry['unmarked_max'])})")
        return fails

    if task == "truecase":
        # The host ASR emits lowercase text; the truecaser must restore the
        # reference casing. Gate on case accuracy over the words the ASR got
        # right (case-insensitively), and require a real gain over the
        # uncased baseline so a no-op truecaser cannot pass.
        import difflib
        host = next(b for b in manifest["backends"] if b["name"] == entry["host_asr"])
        hg = hf_download(host["gguf"]["repo"], host["gguf"]["file"], host["gguf"]["revision"], work_dir)
        for companion in host["gguf"].get("companion_files", []):
            c = hf_download(host["gguf"]["repo"], companion, host["gguf"]["revision"], work_dir)
            if not (hg.parent / Path(companion).name).exists():
                shutil.copy2(c, hg.parent / Path(companion).name)
        ref = re.sub(r"[^\w\s]", " ", entry["expected_transcript"]).split()

        def case_acc(text: str) -> tuple[float, int]:
            got = re.sub(r"[^\w\s]", " ", text).split()
            sm = difflib.SequenceMatcher(a=[w.lower() for w in ref], b=[w.lower() for w in got], autojunk=False)
            pairs = [(ref[i + k], got[j + k]) for i, j, n in sm.get_matching_blocks() for k in range(n)]
            return sum(a == b for a, b in pairs) / max(1, len(pairs)), len(pairs)

        texts = {}
        for label, args in (("baseline", []), ("truecased", ["--truecase-model", str(gguf)])):
            proc = _run_captured([str(crispasr_bin), "--backend", host["backend_id"], "-m", str(hg), "-f",
                                  str(sample), "--no-prints"] + args + extra,
                                 check=True, timeout=timeout, label=f"truecase-{label}")
            texts[label] = " ".join(ln.strip() for ln in proc.stdout.splitlines() if ln.strip())
        base, nb = case_acc(texts["baseline"])
        acc, na = case_acc(texts["truecased"])
        print(f"    baseline:  {texts['baseline']!r}")
        print(f"    truecased: {texts['truecased']!r}")
        ok = na >= 0.6 * len(ref) and acc >= float(entry["case_acc_min"]) and acc - base >= float(entry["case_gain_min"])
        return _report(ok, f"case accuracy {acc:.3f} over {na} aligned words (min {entry['case_acc_min']}); "
                           f"uncased baseline {base:.3f} over {nb}; gain {acc - base:+.3f} (min {entry['case_gain_min']})")

    die(f"unknown task {task!r}")


def tts_roundtrip_for(name: str, manifest: dict, work_dir: Path,
                      crispasr_bin: Path) -> int:
    """Run one TTS backend's TTS->ASR roundtrip regression.

    1. Download the TTS GGUF + companion voice file at pinned revision.
    2. Download the pinned ASR backend's GGUF (parakeet by default —
       it's the high-accuracy English ground-truth ASR the team's TTS
       validation policy uses).
    3. Synthesize the manifest's `tts_phrase` with the TTS backend.
    4. Transcribe the synth output with the ASR backend.
    5. Compute WER (lowercase + strip punctuation), assert below
       `wer_max`.

    Returns the number of failures (0 on success). Mirrors the cost
    accounting from `regression_for(...)` so the matrix log looks
    uniform.
    """
    entry = next((b for b in manifest.get("tts_backends", [])
                  if b["name"] == name), None)
    if entry is None:
        die(f"tts backend '{name}' not in manifest.tts_backends")

    # `advisory: true` — this roundtrip is reported but NEVER gates the run
    # (weak / high-variance TTS model; see the entry's advisory_note). That
    # covers not just an over-the-bar WER but also a synth/ASR timeout or crash
    # or empty output: on a slow CI runner those must not red the nightly for a
    # model we've explicitly decided not to gate. Emits ADVISORY and returns 0.
    advisory = bool(entry.get("advisory", False))

    def advisory_skip(reason: str) -> int:
        print(f"\033[33m  ADVISORY\033[0m  {name}: {reason} — non-gating, not failing the run")
        return 0

    # ---- 1. Download TTS GGUF + voice GGUF (or use a baked voice preset) ----
    gguf_local = hf_download(
        entry["gguf"]["repo"],
        entry["gguf"]["file"],
        entry["gguf"]["revision"],
        work_dir,
    )
    # Two voice paths:
    #   * `voice` block: download a separate voice GGUF (kokoro pattern —
    #     voices live in a sibling repo).
    #   * `voice_preset`: a name like "eric" baked into the TTS model
    #     (qwen3-tts-CustomVoice pattern — speaker embeds inside the
    #     model). Passed to `--voice <name>` without a path.
    # Exactly one of these must be set.
    voice_local: Path | None = None
    voice_preset: str | None = entry.get("voice_preset")
    if "voice" in entry:
        voice_local = hf_download(
            entry["voice"]["repo"],
            entry["voice"]["file"],
            entry["voice"]["revision"],
            work_dir,
        )

    # Optional `codec` block — separate codec/tokenizer GGUF passed via
    # `--codec-model` (qwen3-tts needs this; kokoro doesn't).
    codec_local: Path | None = None
    if "codec" in entry:
        codec_local = hf_download(
            entry["codec"]["repo"],
            entry["codec"]["file"],
            entry["codec"]["revision"],
            work_dir,
        )

    # Optional `companion_files` for backends whose voice / codec is in
    # the same repo as the main GGUF.
    for companion in entry["gguf"].get("companion_files", []):
        c_local = hf_download(
            entry["gguf"]["repo"],
            companion,
            entry["gguf"]["revision"],
            work_dir,
        )
        dest = gguf_local.parent / Path(companion).name
        if not dest.exists():
            shutil.copy2(c_local, dest)

    # `extra_files`: pinned files from OTHER repos (a shared codec / vocoder /
    # voice bank). With `arg` the path is passed as `<arg> <path>`; without it
    # the file is placed next to the TTS GGUF where the backend looks for it.
    extra_file_args: list[str] = []
    for extra in entry.get("extra_files", []):
        local = hf_download(extra["repo"], extra["file"], extra["revision"], work_dir)
        if extra.get("arg"):
            extra_file_args += [extra["arg"], str(local)]
        else:
            dest = gguf_local.parent / Path(extra["file"]).name
            if not dest.exists():
                shutil.copy2(local, dest)

    # `no_voice: true` — the TTS model has no voice-cloning / speaker input
    # (e.g. pocket-tts-english-NOVC). Synthesize with its baked default speaker
    # and pass NO --voice. Forcing a `--voice <name>` on such a model resolves to
    # a nonexistent WAV and yields silence (empty ASR → WER 1.0).
    no_voice = bool(entry.get("no_voice", False))
    if voice_local is None and not voice_preset and not no_voice:
        die(f"{name}: no voice resolved (entry needs a `voice` block, a "
            f"`voice_preset` name, or `no_voice: true`)")

    # ---- 2. Download ASR ground-truth model ----
    asr_name = entry["roundtrip_asr_backend"]
    asr_entry = next((b for b in manifest["backends"] if b["name"] == asr_name), None)
    if asr_entry is None:
        die(f"{name}: roundtrip_asr_backend={asr_name!r} not in manifest.backends")
    asr_gguf_local = hf_download(
        asr_entry["gguf"]["repo"],
        asr_entry["gguf"]["file"],
        asr_entry["gguf"]["revision"],
        work_dir,
    )

    # ---- 3. Synthesize ----
    tts_phrase = entry["tts_phrase"]
    out_wav = work_dir / f"tts_{name}.wav"
    print(f"\n[tts-roundtrip] {name}")
    print(f"  synth     {asr_name} -> {tts_phrase!r}")
    # Pin the RNG. Several TTS backends are stochastic — melotts is a VITS with
    # noise_scale/noise_w and seeds from std::random_device() when no seed is
    # given, so its output (and therefore its WER) differs run to run. Gating a
    # non-deterministic backend on WER makes the check flap, and worse, makes an
    # unrelated change look like a regression: a ggml-sync A/B read melotts as
    # 0.0 -> 0.1111 purely because "lazy dog" resampled to "lazy duck".
    # An entry can still override via tts_extra_args (appended after this).
    tts_seed = str(entry.get("tts_seed", 1234))
    syn_cmd = [
        str(crispasr_bin), "-m", str(gguf_local),
        "--tts", tts_phrase,
        "--tts-output", str(out_wav),
        "--seed", tts_seed,
        "--no-prints",
    ]
    # `--voice` takes either a path (voice GGUF) or a name (baked
    # preset like qwen3-tts's `eric`).
    if voice_preset:
        syn_cmd += ["--voice", voice_preset]
    elif voice_local is not None:
        syn_cmd += ["--voice", str(voice_local)]
    # `--codec-model` for backends with a separate tokenizer GGUF.
    if codec_local is not None:
        syn_cmd += ["--codec-model", str(codec_local)]
    syn_cmd += extra_file_args
    syn_cmd += list(entry.get("tts_extra_args", []))
    try:
        tts_timeout = int(entry.get("tts_timeout_s", 300))
        proc = _run_captured(syn_cmd, check=False, timeout=tts_timeout, label="tts-synth")
    except subprocess.TimeoutExpired:
        _preserve_tts_wav(name, out_wav)
        if advisory:
            return advisory_skip(f"TTS synth timed out after {tts_timeout} s")
        die(f"{name}: TTS synth timed out after {tts_timeout} s")
    if proc.returncode != 0:
        _preserve_tts_wav(name, out_wav)
        if advisory:
            return advisory_skip(f"TTS synth crashed (rc={proc.returncode})")
        print(f"\033[31m  FAIL\033[0m  TTS synth crashed (rc={proc.returncode})")
        print(f"    stderr tail: ...{(proc.stderr or '')[-300:]}")
        return 1
    if not out_wav.is_file() or out_wav.stat().st_size < 1000:
        _preserve_tts_wav(name, out_wav)
        if advisory:
            return advisory_skip(f"TTS produced no usable output: {out_wav}")
        print(f"\033[31m  FAIL\033[0m  TTS produced no usable output: {out_wav}")
        return 1

    # ---- 4. Transcribe with parakeet (or whichever ASR is pinned) ----
    asr_cmd = [str(crispasr_bin),
               "--backend", asr_entry["backend_id"],
               "-m", str(asr_gguf_local),
               "-f", str(out_wav),
               "--no-prints"]
    try:
        proc = _run_captured(asr_cmd, check=False, timeout=180, label="roundtrip-asr")
    except subprocess.TimeoutExpired:
        _preserve_tts_wav(name, out_wav)
        if advisory:
            return advisory_skip("ASR roundtrip timed out after 180 s")
        die(f"{name}: ASR roundtrip timed out after 180 s")
    if proc.returncode != 0:
        _preserve_tts_wav(name, out_wav)
        if advisory:
            return advisory_skip(f"ASR roundtrip crashed (rc={proc.returncode})")
        print(f"\033[31m  FAIL\033[0m  ASR roundtrip crashed (rc={proc.returncode})")
        print(f"    stderr tail: ...{(proc.stderr or '')[-300:]}")
        return 1
    text_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    raw_transcript = text_lines[-1] if text_lines else ""
    # Voice-cloned output carries a mandatory spoken AI-disclosure prefix that
    # `--no-watermark` does not remove; strip it before scoring.
    transcript = strip_ai_disclaimer(raw_transcript)

    # ---- 5. WER ----
    wer_max = float(entry["wer_max"])
    cer, wer = compute_transcript_metrics(tts_phrase, transcript)
    if " " not in tts_phrase.strip():
        # CJK phrase: whitespace-split WER is one token (0 or 1), so gate on a
        # punctuation-insensitive CER instead; `cer_max` is then required.
        p_chars = [c for c in tts_phrase if c.isalnum()]
        t_chars = [c for c in transcript if c.isalnum()]
        cer = _levenshtein(p_chars, t_chars) / max(1, len(p_chars))
        wer, wer_max = cer, float(entry["cer_max"])
        print(f"  (CJK phrase: gating character error rate, cer_max {wer_max})")
    ok = wer <= wer_max
    if not ok or os.environ.get("REGRESSION_KEEP_TTS_WAV") == "1":
        _preserve_tts_wav(name, out_wav)
    # `advisory: true` entries report their WER but never fail the run — for
    # legitimately weak / high-variance TTS models (e.g. bark-small) whose
    # roundtrip can't clear a tight cross-platform bar. NOT for masking a real
    # regression: the parakeet-ja gate stays hard precisely because a tight gate
    # surfaced a real bug there. See the entry's advisory_note.
    advisory = bool(entry.get("advisory", False))
    if ok:
        verdict = "\033[32m  PASS\033[0m"
    elif advisory:
        verdict = "\033[33m  ADVISORY\033[0m"  # over the bar, but non-gating
    else:
        verdict = "\033[31m  FAIL\033[0m"
    print(f"{verdict}  wer={wer:.4f} (max {wer_max}){' [advisory — not gating]' if advisory and not ok else ''}")
    print(f"    phrase:     {tts_phrase!r}")
    print(f"    transcript: {transcript!r}")
    if raw_transcript != transcript:
        print(f"    (raw incl. AI-disclosure: {raw_transcript!r})")
    return 0 if (ok or advisory) else 1


def compare_alignment(expected: list[dict], actual: list[dict], tolerance: float) -> tuple[bool, str, dict]:
    """Compare word alignments: identical normalized words, every start/end
    within ``tolerance`` seconds. Returns (ok, reason, stats)."""
    def norm(word: str) -> str:
        return re.sub(r"[^\w]", "", word.lower())

    if not expected:
        return False, "expected alignment is empty", {}
    if len(expected) != len(actual):
        return False, f"word count {len(actual)} != expected {len(expected)}", {}
    got = [norm(str(w.get("word", ""))) for w in actual]
    want = [norm(str(w["word"])) for w in expected]
    if got != want:
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        return False, f"word {first}: {got[first]!r} != expected {want[first]!r}", {}
    errors = []
    for a, b in zip(actual, expected):
        for key in ("start", "end"):
            try:
                value = float(a[key])
            except (KeyError, TypeError, ValueError):
                return False, f"missing or non-numeric {key!r} for {a.get('word')!r}", {}
            if not math.isfinite(value):
                return False, f"non-finite {key} for {a.get('word')!r}", {}
            errors.append(abs(value - float(b[key])))
    stats = {"words": len(actual), "max_abs_s": max(errors), "mean_abs_s": sum(errors) / len(errors)}
    if stats["max_abs_s"] > tolerance:
        return False, f"max timestamp error {stats['max_abs_s']:.3f}s > {tolerance}s", stats
    return True, "", stats


def align_for(name: str, manifest: dict, work_dir: Path, crispasr_bin: Path) -> int:
    """Run one forced-aligner regression (manifest.align_backends).

    Downloads the pinned aligner GGUF, runs ``crispasr --align-only`` on the
    entry's sample and reference text, and compares the JSON word alignment
    with the pinned expected words (``expected_source`` records where they came
    from, e.g. the upstream Python implementation). Returns failures (0/1).
    """
    entry = next((b for b in manifest.get("align_backends", []) if b["name"] == name), None)
    if entry is None:
        die(f"aligner '{name}' not in manifest.align_backends")
    g = entry["gguf"]
    print(f"\n[align] {name}  ({g['repo']}@{g['revision'][:8]}::{g['file']})")
    print(f"  expected  {entry['expected_source']}")
    # The Qwen3 forced-aligner path is selected by filename; keep the pinned name.
    gguf = hf_download(g["repo"], g["file"], g["revision"], work_dir)
    sample = REPO_ROOT / entry["sample"]
    work_dir.mkdir(parents=True, exist_ok=True)
    out_json = work_dir / f"align_{name}.json"
    cmd = [str(crispasr_bin), "--align-only", "-am", str(gguf), "-f", str(sample),
           "--ref-text", entry["ref_text"], "--align-format", "json", "--align-output", str(out_json),
           "-ng", "-t", str(entry.get("threads", 2))] + list(entry.get("extra_args", []))
    proc = _run_captured(cmd, check=False, timeout=int(entry.get("timeout_s", 900)), label="align")
    if proc.returncode != 0 or not out_json.exists():
        print(f"\033[31m  FAIL\033[0m  crispasr --align-only exited {proc.returncode}")
        print((proc.stderr or "")[-2000:])
        return 1
    try:
        actual = json.loads(out_json.read_text())
    except ValueError as exc:
        print(f"\033[31m  FAIL\033[0m  unreadable alignment JSON: {exc}")
        return 1
    ok, reason, stats = compare_alignment(entry["expected_words"], actual, float(entry["tolerance_s"]))
    if ok:
        print(f"\033[32m  PASS\033[0m  {stats['words']} words, max |dt| {stats['max_abs_s']:.3f}s "
              f"(tol {entry['tolerance_s']}s)")
        return 0
    print(f"\033[31m  FAIL\033[0m  {reason}")
    return 1


def dry_run(manifest: dict, backend_filter: str | None = None) -> int:
    """HEAD-check every pinned HF object referenced by the manifest
    (or just one backend's entries). No downloads, no binary needed.

    Catches the "added a backend, forgot to upload its fixtures" case
    in PR CI in ~1 second, well before the 25-minute Tier 1 run would
    discover it. Checks per backend:

      1. The GGUF file exists at the pinned revision SHA.
      2. The fixture ref.gguf exists at fixtures.revision
         (skipped for entries with ``skip_diff: true`` — they have no
         baked reference dump yet).
      3. The fixture audio.wav exists at fixtures.revision (if the
         entry uses `fixture_sample_path` rather than the legacy
         in-tree `sample`).

    Returns the count of failures (0 on full success). Output is
    one line per missing artifact, suitable for a CI log.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    # An EXPIRED token is worse than no token: huggingface_hub sends it and HF
    # answers 401 even for a PUBLIC repo, so a stale credential breaks a repo
    # that would have worked anonymously. cstr/crispasr-regression-fixtures is
    # public and ungated, and the 2026-09-16 nightly died on exactly that —
    # "Repository Not Found ... make sure you are authenticated" against a repo
    # needing no authentication at all.
    #
    # Fails OPEN on anything but an explicit 401/403: a timeout is not evidence
    # that a token is bad, and dropping a good one would break private fetches
    # for an unrelated reason.
    _tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if _tok:
        try:
            import urllib.error
            import urllib.request
            _rq = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {_tok}"})
            with urllib.request.urlopen(_rq, timeout=20) as _r:
                _rejected = _r.status in (401, 403)
        except urllib.error.HTTPError as _e:
            _rejected = _e.code in (401, 403)
        except Exception:
            _rejected = False
        if _rejected:
            print("\033[33mWARN\033[0m  HF_TOKEN is expired or revoked — dropping it "
                  "and continuing anonymously (the fixtures and gguf repos are public).")
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    api = HfApi()
    fx_repo = manifest["fixtures"]["repo"]
    fx_rev = manifest["fixtures"]["revision"]

    # One list_repo_files call covers every backend's fixture paths
    # (they all live in the same repo at the same revision). Far
    # cheaper than per-path file_exists() calls — and gives us a
    # nicer "did you mean ..." hint when a path is wrong.
    try:
        fx_files = set(api.list_repo_files(
            repo_id=fx_repo, repo_type="model", revision=fx_rev))
    except HfHubHTTPError as exc:
        die(f"fixtures repo {fx_repo}@{fx_rev[:8]} not reachable: {exc}")

    failures = 0
    backends = manifest["backends"]
    tts_backends = manifest.get("tts_backends", [])
    align_backends = manifest.get("align_backends", [])
    task_backends = manifest.get("task_backends", [])
    if backend_filter:
        align_backends = [b for b in align_backends if b["name"] == backend_filter]
        task_backends = [b for b in task_backends if b["name"] == backend_filter]
        filt_asr = [b for b in backends if b["name"] == backend_filter]
        filt_tts = [b for b in tts_backends if b["name"] == backend_filter]
        if not filt_asr and not filt_tts and not align_backends and not task_backends:
            die(f"backend '{backend_filter}' not in manifest")
        backends = filt_asr
        tts_backends = filt_tts

    print(f"dry-run: fixtures {fx_repo}@{fx_rev[:8]} has {len(fx_files)} files")
    print(f"dry-run: checking {len(backends)} ASR + {len(tts_backends)} TTS + "
          f"{len(align_backends)} aligner backend(s)")

    for entry in backends:
        validate_exact_diff_transcript(entry)
        name = entry["name"]
        # skip_preflight: true — GGUF repo temporarily inaccessible (e.g. HF
        # private-repo storage limit). Dry-run skips the artifact check so CI
        # doesn't fail on an infrastructure issue outside our control.
        if entry.get("skip_preflight", False):
            print(f"  \033[33mSKIP\033[0m {name}: skip_preflight=true (GGUF repo check disabled)")
            continue
        gguf_repo = entry["gguf"]["repo"]
        gguf_rev = entry["gguf"]["revision"]
        gguf_file = entry["gguf"]["file"]
        # GGUF — different repo per backend, so one call each.
        try:
            ok = api.file_exists(
                repo_id=gguf_repo, repo_type="model",
                revision=gguf_rev, filename=gguf_file)
        except HfHubHTTPError as exc:
            print(f"  \033[31mFAIL\033[0m {name}: gguf repo {gguf_repo}@"
                  f"{gguf_rev[:8]} not reachable: {exc}")
            failures += 1
            continue
        if not ok:
            print(f"  \033[31mFAIL\033[0m {name}: gguf {gguf_repo}@"
                  f"{gguf_rev[:8]}::{gguf_file} not found")
            failures += 1
            continue

        if "transcript_reference" in entry:
            pin = entry["transcript_reference"]["vocab"]
            try:
                vocab_ok = api.file_exists(repo_id=pin["repo"], repo_type="model",
                                           revision=pin["revision"], filename=pin["file"])
            except HfHubHTTPError as exc:
                vocab_ok = False
                print(f"  {exc}")
            if not vocab_ok:
                print(f"  FAIL {name}: pinned transcript-reference vocabulary missing")
                failures += 1

        dg = entry.get("diff_gguf")
        if dg and not entry.get("skip_diff", False):
            dg_repo, dg_rev = dg.get("repo", gguf_repo), dg.get("revision", gguf_rev)
            try:
                dg_ok = api.file_exists(repo_id=dg_repo, repo_type="model", revision=dg_rev, filename=dg["file"])
            except HfHubHTTPError as exc:
                dg_ok = False
                print(f"  {exc}")
            if not dg_ok:
                print(f"  \033[31mFAIL\033[0m {name}: diff_gguf {dg_repo}@{dg_rev[:8]}::{dg['file']} not found")
                failures += 1
                continue

        # Fixture ref.gguf — membership check against the listing.
        # Skipped for transcript-only entries (skip_diff: true).
        if not entry.get("skip_diff", False):
            ref_path = entry["fixture_ref_path"]
            if ref_path not in fx_files:
                print(f"  \033[31mFAIL\033[0m {name}: fixture ref {ref_path!r} "
                      f"not in {fx_repo}@{fx_rev[:8]}")
                failures += 1
                continue

        if "sample_hf" in entry:
            sp = entry["sample_hf"]
            try:
                sp_ok = api.file_exists(repo_id=sp["repo"], repo_type=sp.get("repo_type", "model"),
                                        revision=sp["revision"], filename=sp["file"])
            except HfHubHTTPError:
                sp_ok = False
            if not sp_ok:
                print(f"  \033[31mFAIL\033[0m {name}: sample {sp['repo']}@{sp['revision'][:8]}::{sp['file']} not found")
                failures += 1
                continue

        # Fixture audio.wav (new-style) — same membership check.
        if "fixture_sample_path" in entry:
            sp = entry["fixture_sample_path"]
            if sp not in fx_files:
                print(f"  \033[31mFAIL\033[0m {name}: fixture sample {sp!r} "
                      f"not in {fx_repo}@{fx_rev[:8]}")
                failures += 1
                continue

        print(f"  \033[32mPASS\033[0m {name}")

    # TTS backends — check the TTS GGUF + the voice GGUF (separate
    # repo/revision since kokoro voices live in cstr/kokoro-voices-GGUF
    # rather than the kokoro-82m repo) + optional companion files. The
    # `roundtrip_asr_backend` reference must resolve to a real ASR
    # entry in `manifest.backends`.
    def _check_pinned_file(label: str, name: str, repo: str, rev: str, f: str) -> bool:
        try:
            ok = api.file_exists(repo_id=repo, repo_type="model",
                                 revision=rev, filename=f)
        except HfHubHTTPError as exc:
            print(f"  \033[31mFAIL\033[0m {name}: {label} repo {repo}@"
                  f"{rev[:8]} not reachable: {exc}")
            return False
        if not ok:
            print(f"  \033[31mFAIL\033[0m {name}: {label} {repo}@"
                  f"{rev[:8]}::{f} not found")
            return False
        return True

    for entry in tts_backends:
        name = entry["name"]
        all_ok = True
        # TTS GGUF.
        all_ok &= _check_pinned_file(
            "tts gguf", name, entry["gguf"]["repo"],
            entry["gguf"]["revision"], entry["gguf"]["file"])
        # Voice is either a `voice` block (separate GGUF), a `voice_preset`
        # name (baked into the model), or `no_voice: true` (no-voice-clone
        # model, synth with its default speaker). Exactly one must be set.
        has_voice_block = "voice" in entry
        has_voice_preset = "voice_preset" in entry
        has_no_voice = bool(entry.get("no_voice", False))
        if (int(has_voice_block) + int(has_voice_preset) + int(has_no_voice)) != 1:
            print(f"  \033[31mFAIL\033[0m {name}: must set exactly one of "
                  f"`voice` (block), `voice_preset` (name), or `no_voice: true`; "
                  f"got voice={has_voice_block} voice_preset={has_voice_preset} "
                  f"no_voice={has_no_voice}")
            failures += 1
            continue
        if has_voice_block and all_ok:
            v = entry["voice"]
            all_ok &= _check_pinned_file(
                "voice gguf", name, v["repo"], v["revision"], v["file"])
        # Optional codec/tokenizer GGUF in its own pinned repo.
        if all_ok and "codec" in entry:
            c = entry["codec"]
            all_ok &= _check_pinned_file(
                "codec gguf", name, c["repo"], c["revision"], c["file"])
        for extra in entry.get("extra_files", []) if all_ok else []:
            all_ok &= _check_pinned_file("extra file", name, extra["repo"], extra["revision"], extra["file"])
        # Optional companion files in the same repo as the TTS GGUF.
        if all_ok:
            for cf in entry["gguf"].get("companion_files", []):
                if not _check_pinned_file(
                        "companion", name, entry["gguf"]["repo"],
                        entry["gguf"]["revision"], cf):
                    all_ok = False
                    break
        if not all_ok:
            failures += 1
            continue
        # roundtrip_asr_backend must point to a real ASR entry.
        asr_ref = entry["roundtrip_asr_backend"]
        if not any(b["name"] == asr_ref for b in manifest["backends"]):
            print(f"  \033[31mFAIL\033[0m {name}: roundtrip_asr_backend "
                  f"{asr_ref!r} not in manifest.backends")
            failures += 1
            continue
        print(f"  \033[32mPASS\033[0m {name}  (TTS roundtrip via {asr_ref})")

    for entry in align_backends:
        name = entry["name"]
        missing = [k for k in ("gguf", "sample", "ref_text", "expected_words", "expected_source", "tolerance_s")
                   if k not in entry]
        if missing:
            print(f"  \033[31mFAIL\033[0m {name}: align entry missing {missing}")
            failures += 1
            continue
        if not (REPO_ROOT / entry["sample"]).exists():
            print(f"  \033[31mFAIL\033[0m {name}: sample {entry['sample']} not in repo")
            failures += 1
            continue
        if not _check_pinned_file("align gguf", name, entry["gguf"]["repo"],
                                  entry["gguf"]["revision"], entry["gguf"]["file"]):
            failures += 1
            continue
        print(f"  \033[32mPASS\033[0m {name}  (aligner vs {entry['expected_source']})")

    for entry in task_backends:
        name = entry["name"]
        g = entry["gguf"]
        ok = _check_pinned_file("task gguf", name, g["repo"], g["revision"], g["file"])
        for cf in g.get("companion_files", []):
            ok = ok and _check_pinned_file("companion", name, g["repo"], g["revision"], cf)
        for key in ("sample_hf",):
            if ok and key in entry:
                sp = entry[key]
                try:
                    ok = api.file_exists(repo_id=sp["repo"], repo_type=sp.get("repo_type", "model"),
                                         revision=sp["revision"], filename=sp["file"])
                except HfHubHTTPError:
                    ok = False
                if not ok:
                    print(f"  \033[31mFAIL\033[0m {name}: sample {sp['repo']}::{sp['file']} not found")
        if ok and entry["task"] == "diff":
            for path in (entry["fixture_ref_path"], entry.get("fixture_sample_path")):
                if path and path not in fx_files:
                    print(f"  \033[31mFAIL\033[0m {name}: fixture {path!r} not in {fx_repo}@{fx_rev[:8]}")
                    ok = False
        if ok and "sample" in entry and not (REPO_ROOT / entry["sample"]).exists():
            print(f"  \033[31mFAIL\033[0m {name}: sample {entry['sample']} not in repo")
            ok = False
        if not ok:
            failures += 1
            continue
        print(f"  \033[32mPASS\033[0m {name}  (task {entry['task']})")

    total = len(backends) + len(tts_backends) + len(align_backends) + len(task_backends)
    if failures == 0:
        print(f"\n\033[32mOK\033[0m  dry-run: all {total} backend(s) "
              f"have their pinned artifacts on HF")
    else:
        print(f"\n\033[31mFAIL\033[0m  dry-run: {failures} missing artifact(s)")
    return failures


def main() -> int:
    # Tiny ad-hoc arg parser — argparse felt heavy for two real options
    # and the existing tests/regression/run_one.py wrapper invocations
    # are all "one positional + env vars".
    args = sys.argv[1:]
    do_dry_run = False
    if args and args[0] == "--dry-run":
        do_dry_run = True
        args = args[1:]
    backend_filter = args[0] if args else None
    if len(args) > 1:
        print(__doc__, file=sys.stderr)
        return 2

    manifest_path = Path(os.environ.get(
        "REGRESSION_MANIFEST",
        REPO_ROOT / "tests" / "regression" / "manifest.json",
    ))
    with manifest_path.open() as f:
        manifest = json.load(f)

    if do_dry_run:
        return dry_run(manifest, backend_filter)

    if not backend_filter:
        print(__doc__, file=sys.stderr)
        return 2
    backend_name = backend_filter

    build_dir = Path(os.environ.get("BUILD_DIR", "build-regression"))
    crispasr_bin = Path(os.environ.get(
        "CRISPASR_BIN", build_dir / "bin" / "crispasr"))
    diff_bin = Path(os.environ.get(
        "DIFF_BIN", build_dir / "bin" / "crispasr-diff"))
    if not crispasr_bin.exists():
        die(f"crispasr binary not found at {crispasr_bin}. "
            f"Build it first or set CRISPASR_BIN.")

    # `backend_name` can match either an ASR backend (manifest.backends)
    # or a TTS backend (manifest.tts_backends). Route on which list
    # holds it. ASR entries are looked up first (their schema is the
    # historical default).
    asr_entry = next((b for b in manifest["backends"] if b["name"] == backend_name), None)
    tts_entry = next((b for b in manifest.get("tts_backends", [])
                      if b["name"] == backend_name), None) if asr_entry is None else None
    align_entry = next((b for b in manifest.get("align_backends", [])
                        if b["name"] == backend_name), None) if asr_entry is None and tts_entry is None else None
    task_entry = next((b for b in manifest.get("task_backends", []) if b["name"] == backend_name), None) \
        if asr_entry is None and tts_entry is None and align_entry is None else None

    if asr_entry is None and tts_entry is None and align_entry is None and task_entry is None:
        die(f"backend '{backend_name}' not in manifest.backends, manifest.tts_backends, "
            f"manifest.align_backends or manifest.task_backends")
    if task_entry is not None and task_entry["task"] == "diff" and not diff_bin.exists():
        die(f"crispasr-diff binary not found at {diff_bin} (needed by task=diff)")

    # crispasr-diff is only needed for full ASR diff entries
    # (skip_diff=false). TTS roundtrip doesn't use it.
    if asr_entry and not asr_entry.get("skip_diff", False) and not diff_bin.exists():
        die(f"crispasr-diff binary not found at {diff_bin}. "
            f"Build it first or set DIFF_BIN. "
            f"(Not needed for skip_diff=true entries.)")

    work_root = Path(os.environ.get(
        "WORK_DIR",
        tempfile.mkdtemp(prefix="crispasr-regression-")))
    keep_work = os.environ.get("KEEP_WORK") == "1"

    try:
        diagnostics = os.environ.get("REGRESSION_DIAGNOSTICS_DIR")
        if diagnostics and Path(diagnostics).resolve().is_relative_to(work_root.resolve()):
            die("REGRESSION_DIAGNOSTICS_DIR must be outside WORK_DIR so cleanup preserves evidence")
        if task_entry is not None:
            failures = task_for(backend_name, manifest, work_root, crispasr_bin, diff_bin)
        elif align_entry is not None:
            failures = align_for(backend_name, manifest, work_root, crispasr_bin)
        elif tts_entry is not None:
            failures = tts_roundtrip_for(
                backend_name, manifest, work_root, crispasr_bin)
        else:
            failures = regression_for(
                backend_name, manifest, work_root, crispasr_bin, diff_bin)
        if failures == 0:
            print(f"\n\033[32mOK\033[0m  {backend_name}: all checks passed")
            return 0
        else:
            print(f"\n\033[31mFAIL\033[0m  {backend_name}: {failures} check(s) failed")
            return failures
    finally:
        if not keep_work and work_root.exists():
            print(f"\ncleanup  {work_root}", file=sys.stderr)
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
