# Per-backend regression tests

These tests guard against silent behavioural regressions by running
each registered backend's GGUF under test against:

1. A **pinned expected transcript** — byte-for-byte match against a
   string captured at a known-good commit. Catches greedy-decode
   divergences (e.g. issue #88), tokenizer changes, sample-rate
   handling drift.
2. The **per-stage diff harness** — cosine similarity of every
   diff-tested stage (encoder output, mel, intermediate captures)
   against a pre-computed Python-reference dump. Catches numerical
   regressions even when the transcript happens to land on the same
   string.

## Why pin everything

Two failure modes the regression infra protects against:

1. **HF upstream re-quantise.** A maintainer re-runs the quantiser
   on the source weights and pushes a new GGUF to the same repo
   path. Users who download fresh suddenly get a different file;
   the regression test silently changes verdict. **Fix:** each
   GGUF entry pins an HF revision SHA in `manifest.json`. The CI
   downloads exactly that revision.
2. **Reference-dump drift.** Re-running `tools/dump_reference.py`
   on a newer NeMo / transformers / torch can produce slightly
   different numbers (FP order-of-summation, casting heuristics).
   **Fix:** reference dumps live in `cstr/crispasr-regression-fixtures`,
   pinned to a specific revision in `manifest.json`'s `fixtures`
   block.

This was prompted by the ggml-assertion-hardening incident
(see LEARNINGS.md / issue history) where a silent upstream behaviour
change cascaded into user-visible regressions.

## Layout

```
tests/regression/
├── manifest.json    # per-backend pins: GGUF revision + fixture path + expected transcript + cos thresholds
├── run_one.py       # driver: downloads pinned GGUF + ref, runs crispasr + crispasr-diff, asserts
└── README.md
```

Sample WAVs live in `samples/` (already in-repo).

The reference-dump archives (`ref.gguf` containing encoder_output +
mel_spectrogram + per-layer captures) live in the HuggingFace repo
[`cstr/crispasr-regression-fixtures`](https://huggingface.co/cstr/crispasr-regression-fixtures)
under `<backend>/<sample-stem>/ref.gguf`.

## Running locally

Build first:

```bash
cmake -S . -B build-ninja-compile \
    -DCMAKE_BUILD_TYPE=Release \
    -DCRISPASR_BUILD_EXAMPLES=ON
cmake --build build-ninja-compile --target crispasr-lib crispasr-diff
```

Run one backend:

```bash
BUILD_DIR=build-ninja-compile tests/regression/run_one.py parakeet-tdt-0.6b-ja
```

Run all backends:

```bash
BUILD_DIR=build-ninja-compile \
  jq -r '.backends[].name' tests/regression/manifest.json | \
  xargs -I{} tests/regression/run_one.py {}
```

Env knobs (see `run_one.py` docstring for the full list):

- `WORK_DIR` — staging directory for HF downloads. Default: a
  tempdir, cleaned on exit. Set to a persistent path on
  `/Volumes/backups/...` to avoid re-downloading between runs.
- `KEEP_WORK=1` — keep the staging dir for debugging.
- `CRISPASR_BIN` / `DIFF_BIN` — override binary paths entirely.

## CI

`.github/workflows/regression.yml` runs nightly at 04:00 UTC
(`workflow_dispatch` also available). The matrix builds the CLI +
diff binary once per backend, downloads pinned artifacts from HF,
asserts both transcript and diff thresholds, and frees disk before
exiting. Each backend is its own matrix entry with
`fail-fast: false` so one bad backend doesn't mask the rest.

## Fast-track: transcript-only entry (`skip_diff`)

If you have a pinned GGUF and a known expected transcript but the Kaggle
rebake hasn't run yet, add the backend with `"skip_diff": true`:

```json
{
  "name": "my-backend",
  "backend_id": "my-backend",
  "skip_diff": true,
  "gguf": { "repo": "cstr/my-backend-GGUF", "revision": "<sha>", "file": "model.gguf", "approx_size_mb": 400 },
  "fixture_sample_path": "my-backend/sample/audio.wav",
  "expected_transcript": "The quick brown fox."
}
```

**Do not** include `fixture_ref_path` or `diff_thresholds` — the smoke
test enforces their absence. The nightly matrix will run the transcript
check immediately; once the Kaggle rebake produces a `ref.gguf`, remove
`skip_diff` and fill in the full fields.

## Transcript tolerance (`transcript_tolerance`) — opt-in WER/CER

By default the transcript check is byte-for-byte exact — no tolerance.
Most backends have deterministic greedy decode and reproduce their
captured transcript bit-perfectly across runner upgrades.

A few backends (currently `glm-asr-nano`) have LLM-style decoders that
tie on score-boundary punctuation/case choices — `you. Ask` vs
`you, ask` is the same English with a 1e-7 softmax-score difference
underneath. The encoder + mel diff checks both stay at `cos=1.000`
(model weights identical) but the decoded transcript shifts a comma.

For such backends, add a `transcript_tolerance` block to opt into a
relaxed metric:

```json
{
  "name": "glm-asr-nano",
  ...
  "expected_transcript": "And so, my fellow Americans, ask not ...",
  "transcript_tolerance": {
    "cer_max": 0.02,
    "wer_max": 0.05
  }
}
```

Semantics:

* **Byte-equal fast path runs first.** If `actual == expected`, the
  check passes with no metric computation. No change in behaviour for
  backends without the field.
* **On byte-equal failure**, if `transcript_tolerance` is set, compute
  CER (Levenshtein over characters / `len(expected)`) and WER
  (Levenshtein over whitespace-split tokens, after lowercasing +
  stripping ASCII punctuation, / word count). Pass if BOTH are at or
  under their maxes.
* **`cer_max`** counts every character difference (case, punctuation,
  spacing). Sensitive to cosmetic drift.
* **`wer_max`** is the industry-standard ASR metric: ignores
  punctuation + case so it reflects semantic word substitution.
  Catches real regressions (missing/wrong words, language confusion)
  without firing on commas.
* **Per-backend opt-in.** Backends without the field keep the strict
  byte-equal bar.

Recommended starting values: `cer_max: 0.02`, `wer_max: 0.05`. Tune
downward as the backend matures.

## TTS->ASR roundtrip (`tts_backends`)

The standard manifest backends are ASR-only. Silent TTS regressions
(e.g. the voxcpm2 BF16 noise generation bug in `a2324c59`) wouldn't
trip the encoder/mel diff harness — the audio comes out *different*,
not corrupted. To catch them, the `tts_backends` section adds a
TTS→ASR roundtrip:

```json
"tts_backends": [
  {
    "name": "kokoro-82m-en",
    "backend_id": "kokoro",
    "gguf": {
      "repo": "cstr/kokoro-82m-GGUF",
      "revision": "<sha>",
      "file": "kokoro-82m-q8_0.gguf",
      "approx_size_mb": 90
    },
    "voice": {
      "repo": "cstr/kokoro-voices-GGUF",
      "revision": "<sha>",
      "file": "kokoro-voice-af_heart.gguf"
    },
    "tts_phrase": "The quick brown fox jumps over the lazy dog",
    "roundtrip_asr_backend": "parakeet-tdt-0.6b-en",
    "wer_max": 0.10
  }
]
```

What `run_one.py` does for a TTS entry:

1. Download the TTS GGUF + the voice GGUF at their pinned revisions.
2. Download the ASR ground-truth model — `roundtrip_asr_backend`
   references an existing entry in `manifest.backends` (typically
   `parakeet-tdt-0.6b-en` — high-accuracy English ASR with
   ~0% baseline error on clean read speech).
3. Synthesize `tts_phrase` via the TTS backend's `--tts` flow.
4. Transcribe the synthesised WAV with the ASR backend.
5. Compute WER (lowercase + strip ASCII punctuation), assert
   `wer ≤ wer_max`.

`wer_max` recommendations (start tight, loosen if needed):

* `0.05` — high-quality English TTS on a clean pangram (kokoro, orpheus
  EN, qwen3-tts EN). Kokoro 82M Q8_0 routinely lands at WER 0.0.
* `0.10` — default headroom for prosody artefacts + parakeet's
  ~0% baseline drift across runner-image upgrades.
* `0.15-0.20` — heavier TTS backends, multilingual cases, or backends
  that emit additional cosmetic content (`.` insertions, prosody
  markers stripped post-ASR).

Why per-backend rather than a single global threshold? Voice quality
varies enormously across backends; a 5% bar that's strict for kokoro
would be impractical for vibevoice on long-form text.

Naming convention: `<tts-backend-id>-<voice-or-variant>`, e.g.
`kokoro-82m-en` (kokoro 82M + English voice), `orpheus-tara-en`.
`run_one.py` dispatches on the name across both lists, so TTS entries
must not collide with ASR entry names — `test_driver_smoke.py`
enforces this.

## Adding a new backend (full diff entry)

1. **Dump the reference** on a known-good commit:

   ```bash
   HF_HOME=/Volumes/backups/ai/huggingface-hub \
   TRANSFORMERS_OFFLINE=1 \
   python tools/dump_reference.py \
     --backend <name> \
     --model-dir <hf-id-or-local-path> \
     --audio samples/<sample>.wav \
     --output /Volumes/backups/ai/crispasr-regression/<backend>/<sample-stem>/ref.gguf
   ```

   `--backend <name>` must match an entry in
   `tools/dump_reference.py`'s `REGISTERED_BACKENDS`. The dump captures
   whatever stages the reference module declares (see
   `tools/reference_backends/<backend>.py`).

2. **Run crispasr** on the same `(GGUF, sample)` pair, manually
   sanity-check the transcript, and lock that string as the
   expected.

3. **Run crispasr-diff** and read the cos_min values for each
   stage. Set the per-stage threshold in `manifest.json` to the
   measured cos_min minus a small safety margin (`0.001` is
   reasonable for `cos_min ≥ 0.999`; looser for known-divergent
   stages like mel preprocessing).

4. **Upload the ref.gguf** to `cstr/crispasr-regression-fixtures`
   under `<backend>/<sample-stem>/ref.gguf`. Use
   `hf upload-large-folder` for resumable uploads (see
   `.claude/CLAUDE.md` for the playbook). Capture the resulting
   commit SHA and update `fixtures.revision` in `manifest.json`.

5. **Look up the GGUF revision SHA** for the model under test:

   ```bash
   curl -s "https://huggingface.co/api/models/<repo>" | \
     python -c "import json,sys; print(json.load(sys.stdin)['sha'])"
   ```

   Put it in the backend's `gguf.revision` field.

6. **Verify locally** with `tests/regression/run_one.py <name>` —
   should pass at the current commit. Commit `manifest.json` (and
   any new sample WAV in `samples/`) on the same branch.
