# Regression matrix — `tools/test-all-backends.py`

`tools/test-all-backends.py` is the pass / fail gate for the whole
backend matrix — sister to `tools/macbook-benchmark-all-backends.py`,
which is the perf benchmark.

Each backend declares advertised capabilities (`transcribe`,
`json-output`, `stream`, `beam`, `best-of-n`, `temperature`,
`word-timestamps`, `punctuation`, `vad`, `lid`, `tts-roundtrip`).
For each capability, three tiers exist: `ignore` / `smoke` / `full`.
Profiles set defaults; per-capability flags override.

```bash
# Quick smoke (transcribe-only on the JFK reference)
python tools/test-all-backends.py

# Full feature matrix (every advertised capability)
python tools/test-all-backends.py --profile=feature

# Subset by name or capability
python tools/test-all-backends.py --backends parakeet,kyutai-stt
python tools/test-all-backends.py --capabilities stream,beam

# Override one capability tier
python tools/test-all-backends.py --vad=full --transcribe=full
```

## Cache modes — `--cache-mode={keep,ephemeral}`

Two operational profiles for how the script handles model files:

| Mode | When | Behavior |
|---|---|---|
| `keep` (default) | local dev, bandwidth-limited | Downloads persist across runs in `--models PATH`. |
| `ephemeral` | tight-disk runners (Kaggle, cloud), bandwidth-cheap | After each backend's tier tests complete, deletes files we downloaded *this run*. Pre-existing files are never touched. Reference models (`whisper-tiny` for LID, `parakeet` for TTS-roundtrip ground truth) are pinned and kept until end of run. |

Kaggle invocation:

```bash
python tools/test-all-backends.py \
  --models /kaggle/working/models \
  --cache-mode=ephemeral \
  --profile=feature
```

The reference downloads (~500 MB total) stay resident; each big
backend (1–2 GB Q4_K) comes + goes one at a time, keeping peak disk
to ~2.5 GB even though the registry totals ~15 GB.

## Skip-missing — `--skip-missing`

Disables HF downloads entirely. Backends without a local model on
disk are reported as `NO_MODEL` (not `FAIL`). Useful for CI runners
that should fail on regression but not on missing models.

## Pre-download disk-space check

Each `Backend` declares an `approx_size_mb` hint. Before downloading,
the script checks `shutil.disk_usage(models_dir).free` against
`approx_size_mb + 2 GB margin`; if low, it skips the download
(reports `SKIP — need ~X MB, only Y MB free`) instead of half-filling
the disk.

## Companion benchmark suite

For perf rather than pass / fail, see
`tools/macbook-benchmark-all-backends.py`
(and `tools/kaggle-benchmark-all-backends.py` for the Kaggle T4
benchmark — results live in [`PERFORMANCE.md`](../PERFORMANCE.md)).
