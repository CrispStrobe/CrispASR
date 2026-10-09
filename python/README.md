# crispasr

Python bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports the ASR backends compiled into the linked CrispASR library, including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, VibeVoice-ASR, and Index-Echo.

## Install

```bash
pip install crispasr
```

Platform wheels **bundle the native `libcrispasr`** — nothing else to install —
for Linux (x86_64, arm64), macOS (Apple Silicon, Metal-accelerated), and
Windows (x86_64).

### GPU wheels

CUDA and Vulkan builds are published to a separate index (llama-cpp-python
style — pass `--extra-index-url`):

```bash
# NVIDIA CUDA (Linux + Windows)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/cuda/
# Vulkan (Windows)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/vulkan/
```

### Other platforms / bring-your-own library

Where no prebuilt wheel matches, pip installs the pure-Python **sdist**, which
loads a `libcrispasr` you supply. Build/install it from source and, if it lands
in a non-standard location, point `CRISPASR_LIB_PATH` at the file:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR && cmake -B build && cmake --build build -j && sudo cmake --install build
export CRISPASR_LIB_PATH=/usr/local/lib/libcrispasr.so   # only if non-standard
```

## Quick start

```python
from crispasr import CrispASR

model = CrispASR("ggml-base.en.bin")
for seg in model.transcribe("audio.wav"):
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
model.close()
```

Or use the unified `Session` API for non-Whisper backends (Qwen3-ASR, FastConformer, Parakeet, …):

```python
from crispasr import Session

s = Session("qwen3-asr-0.6b-q4_k.gguf")
for seg in s.transcribe(pcm_f32, sample_rate=16000):
    print(seg.text)
```

Mono array inputs at other sample rates are converted to 16 kHz with a
band-limited Kaiser polyphase filter. Older libraries use the matching NumPy
fallback, without a SciPy dependency. Strided arrays are made contiguous before
native inference. See [audio resampling](../docs/python-audio-resampling.md)
for limits and regression checks.

Index-Echo uses the same `Session` API. Open the tower GGUF with its matching
decoder beside it, then call `s.set_target_language("en")` (also `ja` or `es`).
Each segment contains `transcript\ntranslation` and generated subtitle times.
Place Silero v6.2 beside the pair for the released full-file window recipe.

## API

- `CrispASR` — Whisper-compatible high-level API
- `Session` — unified API across all backends compiled into `libcrispasr`
- `ChatSession` — text → text chat over a GGUF chat model: one-shot and streaming generation, prompt-token counting, and cancellation through an abort predicate
- `align_words(...)` — word-level CTC alignment
- `diarize_segments(...)` — speaker diarization (energy / xcorr / vad-turns / pyannote / FoxNose)
- `diarize_segments_with_turns(...)` — label segments and return FoxNose's
  finer audio-derived speaker turns
- `SpeakerEmbedder(spec)` — pluggable embedder ("auto"/"titanet", "indextts"/"ecapa", or a `.gguf` path)
- `PyannoteCache(pcm, model)` — pre-computed pyannote-seg posteriors for cross-slice consistency
- `agglomerative_cluster(embeddings, ...)` — single-linkage cosine clustering for globally stable speaker IDs
- `TitaNet` / `SpeakerDB` — standalone speaker verification + closed-roster profile matching (consent-gated; requires `expected_names` + `consent=True`, see docs/diarization-speakers.md)
- `detect_language_pcm(...)` — language ID
- `registry_lookup(...)` — auto-download known models from the model hub
- `registry_default_bundle(...)` — enumerate the exact primary, companion, and extra files used by `-m auto`, including licence-acceptance policy

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, model registry, and CLI.

## License

MIT — see [LICENSE](LICENSE).
