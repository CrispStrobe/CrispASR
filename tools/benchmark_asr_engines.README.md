# benchmark_asr_engines.py

Head-to-head benchmark of `onnx-asr` (the `istupakov/onnx-asr` Python
library) vs CrispASR on the **same** Parakeet TDT 0.6B v3 model. Built
to reproduce the comparison from
[issue #81](https://github.com/CrispStrobe/CrispASR/issues/81).

**Read first**: `PERFORMANCE.md` § "onnx-asr cross-comparison —
issue #81 (2026-05-09)". That section documents the headline
conclusion (crispasr Metal Q8_0 wins TDT-vs-TDT by 1.32× on M1, and
CTC-vs-CTC by 1.58×; on Windows + RTX 4070 + DirectML the picture
inverts and ONNX wins because of dGPU EP coverage), pins the
upstream onnxruntime issue
[`microsoft/onnxruntime#26355`](https://github.com/microsoft/onnxruntime/issues/26355)
(closed *not planned*) for the external-data + CoreML loading
failure, and notes that on M1 the CoreML EP is *slower* than CPU EP
for parakeet-shaped graphs. This script is the **wider-matrix
follow-up** (5 quants × 2 modes × 2 audios + JSON output + Metal-
JIT prewarm + CUDA portability) — the per-cell numbers won't differ
much from PERFORMANCE.md, but the matrix lets you compare quants and
streaming windows directly.

The script runs both engines through the same audio under the same
window/warmup/runs settings and reports realtime factor (RTx),
mean/p50/p95 per-call latency, and (on the short clip) WER against the
JFK reference.

## What it measures

Two call modes — pick with `--mode {whole,chunked,both}` (default both):

- **whole** — feed the full audio in one transcribe call. Closest to
  `onnx-asr.recognize(path)`. Encoder cost amortizes over the whole
  utterance, so this is the fastest shape.
- **chunked** — split audio into N-second windows (default 4 s) and
  call transcribe on each chunk. This is the latency shape that matters
  for streaming ASR — it's what the issue #81 reporter measured.

Two CrispASR call paths — pick with `--crispasr-call {ctypes,cli,both}`
(default `ctypes`):

- **ctypes** — `ctypes.CDLL` against `libcrispasr.{dylib,so,dll}`,
  calling the public `crispasr_parakeet_*` C ABI. One process; the
  parakeet engine is loaded once and re-used across all runs. **This is
  the apples-to-apples path** and matches what the issue #81 reporter
  built.
- **cli** — `subprocess` `./build/bin/crispasr` per run. Includes
  process startup + any cold Metal kernel JIT, so subsequent runs after
  warmup are closer to steady-state but never as low as `ctypes`. Only
  whole-file is supported here (chunked needs `--stream`/stdin
  plumbing).

## Quick start (macOS / Metal)

```bash
# Default: both engines, both modes, both audios, all 5 quant cells.
# `--prewarm` is strongly recommended for stable numbers (see Caveat 2).
python tools/benchmark_asr_engines.py --prewarm --json /tmp/bench.json
```

The script will:

1. Auto-pick a sensible `onnxruntime` provider per host: `CUDA` /
   `TensorRT` / `DirectML` if available (the dGPU EPs that actually
   beat CrispASR on Windows), otherwise **`CPUExecutionProvider`**
   (including on macOS — the CoreML EP is slower than CPU EP for
   parakeet on M1, per PERFORMANCE.md). Override with
   `--providers CoreMLExecutionProvider,CPUExecutionProvider` if you
   want to measure CoreML explicitly.
2. Auto-find `libcrispasr.dylib` under `build-ninja-compile/src/`
   (override with `--crispasr-lib /path/to/lib`).
3. Auto-locate GGUF models under `/Volumes/backups/ai/crispasr/`,
   downloading from `cstr/parakeet-tdt-0.6b-v3-GGUF` if absent.
4. Auto-locate ONNX models under
   `/Volumes/backups/ai/huggingface-hub/parakeet-tdt-0.6b-v3-onnx/`,
   downloading from `istupakov/parakeet-tdt-0.6b-v3-onnx` if absent.
5. Generate `tests/fixtures/bench_long_60s.wav` deterministically
   from `samples/jfk.wav` (tiled to 60 s).

## Quick start (Windows / CUDA)

After building `libcrispasr` with CUDA on Windows:

```powershell
python tools\benchmark_asr_engines.py `
  --crispasr-lib build-cuda\src\crispasr.dll `
  --crispasr-bin build-cuda\bin\crispasr.exe `
  --providers CUDAExecutionProvider,CPUExecutionProvider `
  --gpu-backend cuda `
  --gguf-dir D:\models\crispasr `
  --onnx-dir D:\models\onnx-asr-parakeet
```

The script defaults already auto-pick `CUDAExecutionProvider` if it's
in `onnxruntime.get_available_providers()`, so on a CUDA-enabled host
you typically just need `--crispasr-lib` and `--gpu-backend cuda`.

## Common variations

```bash
# Just onnx-asr, fp32, on the 60 s clip:
python tools/benchmark_asr_engines.py --engine onnx --onnx-quants fp32 \
    --audio long --runs 5

# CrispASR Q8_0 only, ctypes, 2-second streaming chunks:
python tools/benchmark_asr_engines.py --engine crispasr \
    --gguf-quants q8_0 --mode chunked --window-s 2

# Add the CLI subprocess path to the matrix (slower, includes startup):
python tools/benchmark_asr_engines.py --crispasr-call both --mode whole

# CPU-only sanity check (both sides):
python tools/benchmark_asr_engines.py \
    --providers CPUExecutionProvider --gpu-backend cpu

# Force a specific 16 kHz mono wav:
python tools/benchmark_asr_engines.py --audio-path /path/to/clip.wav
```

## Output

Prints a markdown-friendly summary table to stdout. With `--json
PATH`, also writes raw timings and metadata for downstream plotting.

Columns:

| column | meaning |
|---|---|
| engine | `onnx-asr` / `crispasr-ctypes` / `crispasr-cli` |
| quant | GGUF: `q4_k`/`q8_0`/`f16`; ONNX: `int8`/`fp32` |
| mode | `whole` or `chunked` |
| audio | `short` (jfk.wav, 11 s) or `long` (60 s tiled JFK) |
| dur | audio duration |
| load | engine load / first-init wall time |
| mean run | mean across `--runs` (default 3) |
| RTx | audio_seconds / mean_run_s |
| p50 / p95 | per-call latency percentiles (chunked mode only) |
| calls | total per-call samples (= chunks × runs) |
| WER | only on short clip (long clip is tiled, so WER is uninformative) |
| sample | first 48 chars of transcript for visual sanity |

## Known caveats

1. **Long clip is tiled JFK** — the 60 s clip is `samples/jfk.wav`
   tiled 6× and trimmed. It's deterministic and dependency-free, but
   useful **only for speed**. WER on it is uninformative; the script
   suppresses it.
2. **First-cell variance is large without `--prewarm`.** Metal (and
   CUDA) pipeline kernels are shape-specialized: the first transcribe
   on a new `n_samples` hits 10–30 s of pipeline compile cost. Per-cell
   warmups absorb the SAME-shape JIT, but the first cell of the matrix
   still pays the full cold cost, and later cells benefit from kernels
   the earlier cells happened to compile. We've seen 3× variance for
   the same `(quant, mode, audio)` cell depending on its position in
   the matrix. **Always pass `--prewarm`** for headline numbers — it
   JITs every (quant × {whole, chunk}) shape once before the matrix
   begins, so every timed cell starts on a hot cache. Trade-off: adds
   roughly `n_quants × 2 × per-shape-JIT` to startup, which is amortized
   anyway because the OS keeps the Metal pipeline cache warm afterwards.
3. **M-series thermal throttling** affects sustained ML loads. If you
   run the matrix back-to-back, expect the second pass to be a bit
   slower. Let the GPU cool between matrix runs (or run with
   `--prewarm` once and leave a few seconds gap).
4. **First Metal init is slow** — pipeline JIT for kernel variants
   takes 10–30 s on first run, cached afterwards. The reported
   `load` column reflects this; `mean run` does not (warmups absorb
   it).
3. **CLI mode includes process startup** every run. The numbers will
   be inflated vs `ctypes`. This is intentional — it's a real cost
   for CLI-driven workflows. Compare like-for-like.
4. **`--crispasr-call cli` skips chunked mode**. The CLI doesn't
   expose per-chunk timing without `--stream` + stdin plumbing; we
   could add that later if useful.
5. **WER uses `jiwer`** (`pip install jiwer`). Without it, the WER
   column shows `—`.
6. **`onnx-asr` fp32 on macOS CoreML EP.** The fp32 ONNX encoder ships
   as `encoder-model.onnx + encoder-model.onnx.data` (external data).
   The onnxruntime CoreML EP can't load that combination: the
   external-data initializer + CoreML's subgraph split lose
   `model_path`; inlining hits protobuf's 2 GB ceiling. Tracked
   upstream as
   [`microsoft/onnxruntime#26355`](https://github.com/microsoft/onnxruntime/issues/26355)
   (closed *not planned*); see PERFORMANCE.md § "onnx-asr cross-
   comparison — issue #81" for the full table. **Default hides this**
   because we use CPU EP by default on macOS anyway (it's faster than
   CoreML for parakeet on M1). If you opt into CoreML via
   `--providers CoreMLExecutionProvider,CPUExecutionProvider`, the
   fp32 cell auto-falls-back to CPU EP and records the substitution
   in `extra.providers_used`. To force fp32 on CoreML, pre-merge once:
   `python -c "import onnx; m = onnx.load('encoder-model.onnx');
   onnx.save(m, 'encoder-model.onnx', save_as_external_data=False)"`
   (≈ 2.4 GB single-file rewrite). The int8 ONNX is a single file
   so it runs on CoreML without the workaround.

## Reproducing issue #81

The reporter's setup was:

- 60 s audio, 4 s window, 1 warmup + 3 runs (matches our defaults)
- ONNX precision: fp16/int8 (we pair with `int8` here; ONNX repo
  doesn't ship an fp16 file)
- CrispASR precision: Q8_0 GGUF
- DirectML EP on Windows for ONNX, CUDA backend for CrispASR

Equivalent on Windows:

```powershell
python tools\benchmark_asr_engines.py `
  --gguf-quants q8_0 --onnx-quants int8 `
  --mode chunked --window-s 4 --warmups 1 --runs 3 `
  --providers DmlExecutionProvider,CPUExecutionProvider `
  --gpu-backend cuda --audio long
```
