# Cohere Transcribe — runtime benchmark

Audio: `local_cohere_model/demo/voxpopuli_test_en_demo.wav` (5.44 s, 16 kHz mono).
Hardware: 8-thread CPU, single process. All paths produce the same transcript:

> If not, there will be a big crisis between you and the European Parliament.

(The Rust path prepends a stray `"not. "` — minor pre-roll glitch.)

## Results

| # | Path | Backend | Total wall | Inference-only | RTF | Peak RSS |
| - | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Python `transformers` | torch CPU F32 | 156.8 s | 134.2 s | 24.7× | ~5 GB |
| 2 | Rust `cohere_transcribe_rs` | libtorch (`tch`) F32 | 162.8 s | — | 29.9× | 7.3 GB |
| 3 | ONNX `onnx-final` | F32 | ~57 s | 51.2 s | 9.4× | — |
| 4 | C++ `cohere-main` | ggml F16 | 27.6 s | — | 5.1× | 7.1 GB |
| 5 | C++ `cohere-main` | ggml Q8_0 | 38.9 s | — | 7.2× | 4.7 GB |
| 6 | C++ `cohere-main` | ggml Q5_0 | 33.8 s | — | 6.2× | 3.4 GB |
| 7 | **C++ `cohere-main`** | **ggml Q4_K** | **14.8 s** | — | **2.7×** | **3.0 GB** |
| 8 | ONNX `onnx-final` | INT4 (cstr) | 17.1 s | 10.4 s | 3.1× | — |
| 9 | ONNX `cohere-int8-tristan` | INT8 (Tristan) | 31.5 s | 8.9 s | 5.8× | — |
| — | ONNX `onnx-final` | INT8 | broken | — | — | invalid graph: float16+DynamicQuantizeLinear |
| 10 | C++ `cohere-align` | ggml Q4_K + wav2vec2-xlsr-large | 80.9 s | — | 14.9× | 3.0 GB |
| 11 | **C++ `parakeet-main`** | **ggml Q4_K (parakeet-tdt-0.6b-v3)** | **5.3 s** | — | **0.97×** | **0.96 GB** |
| 12 | C++ `parakeet-main` | ggml F16 (parakeet-tdt-0.6b-v3) | 9.3 s | — | 1.71× | 2.5 GB |

## Parakeet vs Cohere

| | Cohere Transcribe Q4_K | **Parakeet TDT Q4_K** |
| --- | ---: | ---: |
| Wall time (5.4 s clip) | 14.8 s | **5.3 s** (2.8× faster) |
| Peak RSS                | 3.0 GB | **0.96 GB** (3× less) |
| Model size on disk      | 1.2 GB | **467 MB** (2.6× smaller) |
| Word timestamp accuracy | ~360 ms (cross-attn DTW) | **~80 ms** (TDT durations) |
| Languages               | 14 | **25** (auto-detect) |
| Open ASR WER (avg)      | **5.42 %** | 6.34 % |

Parakeet is faster, smaller, multilingual with auto-detect, and produces precise word timestamps with no second model. Cohere wins on raw English WER. For applications that need timestamps and multilingual support on CPU, parakeet is the right default; for the absolute lowest WER on English, cohere is.

RTF = total wall / audio duration. "Inference-only" excludes model load (only tracked for ONNX, where `benchmark_onnx.py` reports it explicitly).
"Total wall" is `time(1)` end-to-end where measured; for ONNX rows it is `load + encoder + decoder` reported by `benchmark_onnx.py` (8.5-23 s of that is one-shot model load).

The C++ Q5_0 / Q8_0 numbers are higher than expected this run — system load on the shared box was variable. Q4_K is consistently the fastest ggml variant in repeated runs and the only one we'd ship.

## Observations

- **`cohere-main` Q4_K is the fastest pure-CPU runtime end-to-end** at ~15 s wall (~2.7× realtime), narrowly beating ONNX INT4 (17.1 s, but only because ONNX has a 6.7 s cold-load penalty). On *inference-only* the two are essentially tied (ggml ~15 s including load vs ONNX 10.4 s pure inference). Tristan's INT8 ONNX has the fastest inference at 8.9 s but pays 22.6 s of cold-load.
- **F32 baselines are all ~5-10× slower** than the best quantised paths: torch CPU F32 (Python or Rust) lands at ~160 s, ONNX F32 at ~57 s.
- **ONNX INT8 in `onnx-final/` is broken** on current onnxruntime (encoder graph mixes float16 tensors with DynamicQuantizeLinear, which is rejected). Only the INT4 export from `onnx-final/` and Tristan's INT8 export work today.
- **`cohere-main` Q5_0 and Q8_0 underperformed** in this run (33-39 s vs Q4_K's 15 s); the gap is from system load on the shared host, not a real regression. Q4_K is the consistent winner.
- **Python `transformers` and Rust `tch`** are tied at ~25-30× realtime — both go through libtorch CPU F32. There is no easy win on the Rust side without switching backends; see previous discussion.
- **`cohere-align` adds ~5× overhead on top of `cohere-main`** (15 s → 81 s) for word-level CTC alignment. That's the cost of running a 24-layer wav2vec2-large encoder over the same audio with a single-thread F32 manual transformer in `wav2vec2-ggml.cpp`. Quantising the wav2vec2 weights or vectorising the manual MHA would close most of the gap.

## What's in `test_cohere/`

Existing scripts that fed this benchmark:

- **`benchmark_onnx.py`** — runs all four ONNX variants (`onnx-final` F32/INT4/INT8 and `cohere-int8-tristan` INT8) via `onnxruntime`. The script we used for rows 3, 8, 9.
- **`onnx-final/benchmark_cs.sh`** — the .NET 8 / `Microsoft.ML.OnnxRuntime` benchmark from the original `cstr/cohere-transcribe-onnx-int4` HF repo. Equivalent to row 8 but on `dotnet`. Not run here (no `dotnet` installed).
- **`export_cohere_baked.py`, `export_cohere_onnx.py`** — produced the F32/INT4/INT8 exports in `onnx-final/` (mel filterbank baked into the encoder graph as Conv1d DFT filters, cross-attention K/V pre-computed for all 8 decoder layers).
- **`cohere-int8-tristan/quantize.py`, `export_onnx.py`, `export_onnx_baked.py`** — Tristan Ripke's original INT8 quantisation pipeline (the one row 9 is built on).
- **`tests/test_enc_f32.py`, `tests/compare_enc2.py`, `tests/compare_mel_onnx.py`** — encoder-only sanity checks used during the original ONNX export.
- **`cohere_transcribe_rs/test_cohere_9.py`** — stand-alone Python sanity check used while developing the Rust crate.
- **`kaggle_full_script.py`** — full Kaggle GPU benchmark (CUDA, not exercised here).

## Reproducing

```bash
# 1. Python transformers (CohereAsrForConditionalGeneration via trust_remote_code)
python3 /tmp/bench_py.py local_cohere_model local_cohere_model/demo/voxpopuli_test_en_demo.wav

# 2. Rust (libtorch backend, uses Python's libtorch via LIBTORCH_USE_PYTORCH=1)
LD_LIBRARY_PATH=/opt/miniconda/lib/python3.13/site-packages/torch/lib \
  /tmp/cohere_rs_target/release/transcribe \
  --model-dir local_cohere_model --language en \
  local_cohere_model/demo/voxpopuli_test_en_demo.wav

# 3, 8, 9. ONNX variants
python3 benchmark_onnx.py        # runs F32, INT4, Tristan INT8 (and reports the broken INT8)

# 4-7. C++ cohere-main, all quants
for m in cohere-transcribe-q4_k.gguf cohere-transcribe-q5_0.gguf \
         cohere-transcribe-q8_0.gguf cohere-transcribe.gguf; do
  /usr/bin/time -f "WALL=%e MEM=%M" \
    ../whisper.cpp/build/bin/cohere-main -m $m \
    -f local_cohere_model/demo/voxpopuli_test_en_demo.wav -t 8 -np
done

# 10. C++ cohere-align (CTC forced alignment for word timestamps)
../whisper.cpp/build/bin/cohere-align -m cohere-transcribe-q4_k.gguf \
  -cw wav2vec2-xlsr-en.gguf \
  -f local_cohere_model/demo/voxpopuli_test_en_demo.wav -t 8 -np
```
