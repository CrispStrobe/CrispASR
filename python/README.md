# crispasr

Python bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports the ASR backends compiled into the linked CrispASR library, including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, VibeVoice-ASR, and Index-Echo.

The package also exposes TTS, speech-to-speech, music analysis, ContentVec voice
conversion, alignment, diarization, VAD, text processors and model utilities.
See [the Python API inventory](../docs/python-api-coverage.md) for entry points,
result units, ownership, callback rules and verification limits. These additions
require a matching native build; current source coverage does not imply an
older published wheel contains the same APIs.

## Install

```bash
pip install crispasr
```

Platform wheels **bundle the native `libcrispasr`** — nothing else to install —
for Linux (x86_64, arm64), macOS (Apple Silicon, Metal-accelerated), and
Windows (x86_64).

Prebuilt Linux wheels use `manylinux_2_35` (glibc 2.35 or newer, with the
corresponding host C++ runtime). Their native libraries import GLIBC 2.34 and
GLIBCXX 3.4.30; the earlier `manylinux_2_28` label overstated compatibility.
The Apple Silicon wheel targets macOS 11.0, including its optional C helper.
Release packaging checks every included native library against these floors.
Installing the source distribution does not make an incompatible prebuilt
native library compatible; supply a native build suitable for your system.

### GPU wheels

CUDA, Vulkan and ROCm/HIP builds are published to a separate index
(llama-cpp-python style — pass `--extra-index-url`):

```bash
# NVIDIA CUDA (Linux x86_64 + Windows; Linux aarch64/SBSA from 0.8.47, not Jetson)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/cuda/
# Vulkan (Windows; Linux x86_64 from 0.8.47)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/vulkan/
# AMD ROCm/HIP (Linux x86_64, from 0.8.47)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/hip/
```

GPU wheels do not ship a GPU driver or vendor runtime. Linux `+vulkan` needs
the host's Vulkan loader (`libvulkan.so.1`) and a Vulkan driver; `+hip` needs a
ROCm 6.x install (HIP runtime, hipBLAS/rocBLAS and ROCm's `libomp`) on the
loader path, e.g. `/opt/rocm/lib` and `/opt/rocm/lib/llvm/lib`.

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

Session segments expose `tokens` (`SessionToken`: text, start/end seconds,
confidence, and lossless `raw_text` bytes). Missing word or token timestamps
are `None`; check `word.has_timestamps` before doing arithmetic. VibeVoice has
native segment times but no native word times. Use `align_words()` for word
alignment, then `free_aligner_cache()` or `session.free_aligner_cache()` to
release the process-wide aligner allocation. Older libraries return empty
token lists and report cache release as unavailable. See
[session metadata and compatibility](../docs/python-session-metadata.md). Parakeet
metadata is retained through Japanese slice/gap-fill and legacy overlap selection;
raw token blocks follow retained words even when timestamps are tied. These
main-branch changes require the matching native library and are not in v0.8.43.


## API

- `CrispASR` — Whisper-compatible high-level API
- `Session` — unified API across all backends compiled into `libcrispasr`
- `ChatSession` — text → text chat over a GGUF chat model: one-shot and streaming generation, prompt-token counting, and cancellation through an abort predicate
- `align_words(...)` — word-level CTC alignment
- `free_aligner_cache(...)` — release the process-wide forced-aligner cache
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
