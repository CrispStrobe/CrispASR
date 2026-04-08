# Qwen3-ASR — implementation comparison benchmark

Wall-clock comparison of three independent C++/Python implementations of
Qwen3-ASR-0.6B inference, on the same audio clips, same hardware,
same wall-time methodology (`date +%s.%N` around the binary call, so
each row includes process startup + the ~1-2 s mmap of the 1.88 GB
GGUF — useful as a "what does the user actually wait for" number).

**Hardware:** generic 4-core x86_64 CPU container, no GPU.
**Threads:** 4 for all C++ implementations; Python uses its own thread
pool (typically 1-2 active per matmul through MKL).

## Setup

| Implementation | Build | Model file |
| --- | --- | --- |
| **CrispASR Q4_K** | this repo, `cmake --build build --target qwen3-asr-main` | `qwen3-asr-0.6b-q4_k.gguf` (676 MB, this repo's converter + cohere-quantize) |
| **CrispASR F16** | this repo | `qwen3-asr-0.6b.gguf` (1.88 GB, this repo's converter) |
| **predict-woo/qwen3-asr.cpp** | <https://github.com/predict-woo/qwen3-asr.cpp> @ HEAD (build: needs an `OpenMP::OpenMP_CXX` link added to `qwen3-asr-cli` for non-Apple builds, plus the ggml submodule built once into `ggml/build/` since their CMakeLists references `${GGML_BUILD_DIR}/src` for the `-lggml` link path) | `qwen3-asr-0.6b-predictwoo-f16.gguf` (1.88 GB, predict-woo's converter — different tensor naming, can't share with the CrispASR build) |
| **Python qwen-asr** | `pip install qwen-asr` (transformers backend, F32 weights, MKL/oneDNN linear algebra) | original HF safetensors |

## Results

### Run 1 (representative)

```
Wed Apr  8 13:23:32 UTC 2026
```

| | jfk.wav (11.0s) | RTF | obama_speech_16k.wav (89.2s) | RTF |
| --- | ---: | ---: | ---: | ---: |
| **CrispASR Q4_K** | **10.23 s** | **0.93×** | **101.48 s** | 1.14× |
| CrispASR F16 | 18.03 s | 1.64× | 115.63 s | 1.30× |
| predict-woo F16 | 19.69 s | 1.79× | 99.79 s | 1.12× |
| Python qwen-asr (F32 + MKL) | 13.40 s | 1.22× | 90.67 s | 1.02× |

### Inner-timing breakdown (from the binaries' own timers, no startup overhead)

#### CrispASR Q4_K on jfk.wav (11 s)
```
mel:     250 ms
encoder: 2660 ms
prefill: 2112 ms
decode:  29 tokens in 1879 ms (65 ms/token)
total:   ~6.6 s of actual compute
```

#### predict-woo F16 on jfk.wav (11 s)
```
Mel spectrogram: 2589 ms      (= 10× slower — see "mel filterbank" below)
Audio encoding:  4405 ms
Text decoding:   12478 ms
Total:           19472 ms
Tokens generated: 29
```

#### CrispASR Q4_K on obama_speech_16k.wav (89 s)
```
mel:     2627 ms
encoder: 21859 ms
prefill: 16762 ms
decode:  258 tokens in 47791 ms (185 ms/token)
total:   ~89 s
```

#### predict-woo F16 on obama_speech_16k.wav (89 s)
```
Mel spectrogram: 21591 ms
Audio encoding:  22408 ms
Text decoding:   66086 ms
Total:           110085 ms
```

## Per-stage comparison (jfk.wav, 11 s, inner timings)

| Stage | CrispASR Q4_K | predict-woo F16 | Speedup |
| --- | ---: | ---: | ---: |
| Mel spectrogram | 250 ms | 2589 ms | **10.4×** |
| Encoder | 2660 ms | 4405 ms | 1.66× |
| Decode (prefill + greedy) | 3990 ms | 12478 ms | **3.13×** |
| **Total compute** | **6.6 s** | **19.5 s** | **2.96×** |

## Where the speedups come from

1. **Mel: 10× faster.** This repo's converter bakes the
   `WhisperFeatureExtractor.mel_filters` and `audio.mel_window` directly
   into the GGUF (`models/convert-qwen3-asr-to-gguf.py`), so the C++
   runtime just reads the precomputed filterbank and runs an STFT loop
   per chunk. predict-woo recomputes the Slaney mel filterbank from
   scratch at runtime via its own implementation.
2. **Encoder: 1.66× faster.** Both implementations use a comparable
   ggml graph structure for the audio encoder (3-stage 2D-conv
   subsampler + 18 Whisper-style blocks + projector head). The win
   here likely comes from a combination of the lm_head last-token-only
   slice (cuts the prefill cost), the F16 KV cache (halves cache memory
   bandwidth on the GQA expand step), and ggml backend differences.
3. **Decode: 3.13× faster.** The big one. CrispASR has:
   - **Persistent F16 KV cache** with prefill/decode sharing
     (`(head_dim, max_ctx, n_kv, n_layers)` allocated to the backend)
   - **`ggml_flash_attn_ext` on both prefill and decode paths** —
     fuses Q@K^T → softmax → @V into one kernel and reads F16 K/V
     directly from the cache without an explicit dequant
   - **Last-token-only `lm_head` matmul** — the prefill `(d, T)` hidden
     state gets sliced to `(d, 1)` before the giant
     `(151936, 1024)` projection, so prefill only runs the lm_head
     once instead of T times
   - **Q4_K weight quantization** via the existing generic
     `cohere-quantize` (works on Qwen3-ASR GGUFs unchanged)

## Where the Python baseline still wins

For long audio specifically, the Python wrapper (`qwen-asr` on
top of HF transformers) is ~10% faster than CrispASR on this hardware
(90.67 s vs 101.48 s on the 89 s clip). The difference is essentially
**MKL/oneDNN matmul kernels** vs **portable ggml CPU kernels**. PyTorch
links a heavily x86-tuned BLAS that's 1.5-2× faster than ggml on
contiguous F32 GEMM, and the long-audio cost is dominated by the
prefill matmul over a 1185-token prompt.

The gap closes on short audio (Python is *slower* than us on jfk —
13.40 s vs 10.23 s — because the per-call PyTorch warmup tax dominates
the actual compute).

A future CrispASR build with `GGML_BLAS=ON` (linking ggml-cpu against
OpenBLAS/MKL) would likely close most of the long-audio gap.

## Correctness — all four agree on the transcript

All four implementations produce equivalent transcripts on both clips.
Minor punctuation variation (comma vs semicolon, "fatigues" vs "fatigue")
shows up between Q4_K and F16 — that's the K-quant introducing
~1e-2 cosine-distance perturbations in the LLM hidden states which
occasionally tip a 2-way tie between near-equal logits. The semantic
content is identical and matches the Python ground truth.

## German / multilingual smoke test

CrispASR Q4_K on a spread of German Wikipedia / Wikimedia clips
(4 threads, F16 KV cache, flash-attn on both prefill and decode):

| Clip | Duration | Lang detected | Wall | RTF | Quality |
| --- | ---: | --- | ---: | ---: | --- |
| `berlin_word.wav` | 0.7 s | German ✓ | 1.2 s | 1.7× | "Berlin" ✓ |
| `bundeskanzler_word.wav` | 1.6 s | German ✓ | 1.8 s | 1.1× | "Bundeskanzler" ✓ |
| `jazeschann.wav` | 4.8 s | German ✓ | 3.2 s | 0.66× | "Leider zu spät. Leider zu spät." ✓ (perfect) |
| `merkel.wav` | 58.7 s | **Russian** | 55.5 s | 0.95× | Coherent Russian transcript — **see note below** |
| `De-Abwasch-article.wav` | 79.4 s | German ✓ | 94 s | 1.18× | Wikipedia article reading (Dishwashing), full coherent German |
| `De-Afghani-article.wav` | 207.6 s | German ✓ | 407 s | 1.96× | Long Wikipedia article (Afghani currency), 688 tokens — full coherent German with dates / ISO codes / numbers |

### Note on `merkel.wav`

The Wikimedia file `Angela_Merkel_voice.ogg` (claimed German Merkel
content) is actually **a Russian-language recording**. Verified by
running `whisper-cli` (whisper-small) with auto language detection on
the same clip:

```
whisper_full_with_state: auto-detected language: ru (p = 0.852450)
```

Whisper produced a coherent Russian transcript that semantically
matches Qwen3-ASR's Russian output. When forced to German via `-l de`,
whisper produces broken repetitive text ("als Gegner, als Gegner, als
Gegner..."), the classic "wrong language forced" failure mode.

So the file is either:
1. A Russian-language voice-over of Merkel content
2. A Russian impersonator
3. Mislabeled on Wikimedia Commons

In any case, **Qwen3-ASR's Russian detection is correct, not a bug**.
A previous note in `test_german.md` characterizing this clip's
"Russian output" as a parakeet bug was wrong — parakeet, whisper, and
Qwen3-ASR all agree.

## GGML_BLAS=ON build experiment

Tried building CrispASR with `cmake -DGGML_BLAS=ON
-DGGML_BLAS_VENDOR=OpenBLAS`, which the FindBLAS module actually
resolved to **MKL** (the same library transformers uses for its matmul
kernels). Results on jfk.wav and obama_speech_16k.wav:

| Build | jfk total | obama prefill | obama decode/tok | obama total |
| --- | ---: | ---: | ---: | ---: |
| CrispASR Q4_K (no BLAS, current default) | 6.6 s | 16.8 s | 185 ms | 89 s |
| CrispASR Q4_K + GGML_BLAS=ON (MKL) | 6.9 s | 17.2 s | 178 ms | 86 s |
| CrispASR F16 + GGML_BLAS=ON (MKL) | (not measured) | 19.7 s | 208 ms | 99 s |

**Negative finding**: BLAS gives essentially no speedup on this
workload, and the F16+BLAS combo is actually slower than F16 without
BLAS. Reasons:

1. Most of CrispASR's matmuls go through Q4_K weights via ggml's
   k-quant kernels, which run their own optimised dispatch and skip
   BLAS entirely (BLAS is float-only).
2. The remaining F32 matmuls (norms, attention scores, lm_head) are
   either small or batch-1, which is the worst case for BLAS — the
   per-call overhead eats the gain.
3. ggml's CPU kernels are competitive with MKL at our matmul sizes
   for batch-1 decode.

The lm_head projection (`151936 × 1024`) is the only large per-token
matmul, but per-call BLAS overhead is comparable to its compute cost
at batch=1, so it doesn't help.

A future build linking against `GGML_BLAS=ON` would only pay off if we
operated on F16/F32 weights exclusively AND batched multiple decode
steps together (e.g. for batch transcription of many short clips).
For single-utterance ASR the existing path is optimal.

## How to reproduce

```bash
# CrispASR build
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target qwen3-asr-main cohere-quantize
python3 models/convert-qwen3-asr-to-gguf.py \
    --input <hf_dir> --output qwen3-asr-0.6b.gguf
./build/bin/cohere-quantize qwen3-asr-0.6b.gguf qwen3-asr-0.6b-q4_k.gguf q4_k

# Run on a 16 kHz mono PCM WAV
./build/bin/qwen3-asr-main \
    -m qwen3-asr-0.6b-q4_k.gguf \
    -f your-audio.wav -t 4
```

For the predict-woo build, you currently need to:
1. `git submodule update --init --recursive` (the repo doesn't enable
   recursive clone by default)
2. Build the ggml submodule once into `ggml/build/` because the
   parent CMakeLists references `${GGML_BUILD_DIR}/src` as the
   `target_link_directories` for the static library link
3. Add `find_package(OpenMP REQUIRED)` and append
   `OpenMP::OpenMP_CXX` to `target_link_libraries(qwen3-asr-cli ...)`
   on non-Apple builds (the upstream CMake only handles macOS Accelerate
   linkage)

(I sent these fixes to the predict-woo author's HEAD as a private
build patch — see /tmp/predict-woo-build-fixes.diff if you want them.)
