# Quantize

CrispASR ships a single, model-agnostic GGUF re-quantization tool,
**`crispasr-quantize`**, that works across all supported model
families: Whisper, Parakeet, Canary, Cohere, Voxtral, Qwen3, Granite,
Wav2Vec2, MiMo-ASR, GLM-ASR, Moonshine, VibeVoice, Kokoro, Qwen3-TTS,
and others. It iterates through the GGUF tensor list and re-quantizes
eligible 2D weight matrices while preserving metadata and
non-quantizable tensors (norms, positional embeddings, biases) in
their original types.

Replaces the legacy per-model tools (`cohere-quantize`,
`parakeet-quantize`, …) — those are no longer built. If a model card
references one of them, use `crispasr-quantize` instead with the same
arguments.

## Build

`crispasr-quantize` is built automatically as part of the default
build target. The shortest path:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

ls build/bin/crispasr-quantize    # confirm it exists
```

> If you previously built with `--target crispasr` (which only builds
> the main binary), re-run **without** `--target` or with
> `--target crispasr-quantize` to produce the quantize tool.

The `scripts/dev-build.sh` wrapper accepts `--target` directly:

```bash
scripts/dev-build.sh --target crispasr-quantize
```

> **glibc note.** Pre-built binaries on some HuggingFace model cards
> (e.g. `bin/cohere-quantize`) require glibc 2.38 and fail on Ubuntu
> 22.04 with:
> ```
> /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.38' not found
> ```
> Building from source (above) avoids this — CrispASR has no glibc
> minimum of its own and builds cleanly against whatever glibc your
> distro ships.

## Usage

```bash
./build/bin/crispasr-quantize <input.gguf> <output.gguf> <type>
```

`<type>` is one of the `ggml_ftype` names below.

### Supported types

| Type   | Description                                                          |
|--------|----------------------------------------------------------------------|
| `q4_0` | 4-bit (scale only)                                                   |
| `q4_1` | 4-bit (scale + minimum; slightly higher accuracy than q4_0)          |
| `q5_0` | 5-bit (scale only)                                                   |
| `q5_1` | 5-bit (scale + minimum; slightly higher accuracy than q5_0)          |
| `q8_0` | 8-bit (scale only) — usually indistinguishable from F16 quality      |
| `q2_k` | 2-bit K-quant (very lossy; use only when memory is critical)         |
| `q3_k` | 3-bit K-quant                                                        |
| `q4_k` | 4-bit K-quant (preferred over legacy q4_0/q4_1)                      |
| `q5_k` | 5-bit K-quant (preferred over legacy q5_0/q5_1)                      |
| `q6_k` | 6-bit K-quant (close to F16 quality, 60% of F16 size)                |

### Examples

```bash
# Whisper base.en F16 → Q4_K (small + fast)
./build/bin/crispasr-quantize ggml-base.en.bin ggml-base.en-q4_k.bin q4_k

# Parakeet TDT 0.6B F16 → Q4_K
./build/bin/crispasr-quantize parakeet-tdt-0.6b-f16.gguf parakeet-tdt-0.6b-q4_k.gguf q4_k

# Voxtral Mini 4B F16 → Q5_0
./build/bin/crispasr-quantize voxtral-mini-4b-realtime-f16.gguf \
                              voxtral-mini-4b-realtime-q5_0.gguf q5_0

# Canary 1B F16 → Q6_K (near-lossless)
./build/bin/crispasr-quantize canary-1b-v2-f16.gguf canary-1b-v2-q6_k.gguf q6_k
```

### Alignment fallback

K-quants (`q2_k` through `q6_k`) require tensor row sizes to be
multiples of 256. If a tensor doesn't meet this requirement (e.g. the
896-wide tensors in some Qwen3-ASR layers), the tool transparently
falls back to a compatible legacy quant (typically `q4_0` or `q8_0`)
for that tensor only, and the rest of the model still gets the
requested K-quant. The output GGUF is always fully quantized — there
is no half-quantized failure mode.

## Recommended quants per backend

These reflect the trade-offs we measured on the regression suite. F16
is always the reference; Q4_K is the daily-driver default for most
ASR backends.

| Backend             | F16   | Q8_0  | Q5_K  | Q4_K  | Notes                                                                    |
|---------------------|:-----:|:-----:|:-----:|:-----:|--------------------------------------------------------------------------|
| whisper             | ✓     | ✓     | ✓     | ✓     | All sizes work; Q4_K is upstream's recommended size/quality balance.     |
| parakeet            | ✓     | ✓     | ✓     | ✓     | Q4_K is the shipped default.                                             |
| canary              | ✓     | ✓     | ✓     | ✓     | Q4_K validated on test-all-backends.                                     |
| cohere              | ✓     | ✓     | ✓     | ✓     |                                                                          |
| voxtral / voxtral4b | ✓     | ✓     | ✓     | ✓     | Q4_K is the shipped HF default.                                          |
| qwen3-asr           | ✓     | ✓     | ✓     | ✓     | Some 896-wide tensors trigger the alignment fallback (still works).      |
| granite             | ✓     | ✓     | ✓     | ✓     |                                                                          |
| mimo-asr            | ✓     | ✓     | ✓     | ✓     | F16 + Q4_K shipped to HF; Q4_K validated.                                |
| glm-asr             | ✓     | ✓     | ✓     | ✓     |                                                                          |
| firered-asr         | ✓     | ✓     | ✓     | ✓     |                                                                          |
| moonshine           | ✓     | ✓     | ✓     | ✓     |                                                                          |
| wav2vec2 / fc-ctc   | ✓     | ✓     | ✓     | ✓     | CTC heads are small; Q4_K barely changes WER.                            |
| kokoro (TTS)        | ✓     | ✓     | —     | —     | Q5_K and below break the German backbone — ship F16 + Q8_0 only.         |
| qwen3-tts           | ✓     | ✓     | ✓     | ✓     |                                                                          |
| vibevoice (TTS)     | ✓     | ✓     | ✓     | ✓     | F16 + Q4_K shipped.                                                      |

> The cells marked `—` are not just "untested" — they have a known
> quality regression. See [`PERFORMANCE.md`](../PERFORMANCE.md) for
> the full benchmark numbers and the per-quant WER deltas.
