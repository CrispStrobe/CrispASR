# crispasr

Python bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports 17 ASR backends including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

```bash
pip install crispasr
```

This wheel is **pure Python** and does **not** bundle the native library — install `libcrispasr` separately, the same way `crispasr`'s Python bindings work:

**macOS**
```bash
brew install crispasr        # once published; until then build from source
```

**Linux / Windows / from source**
```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build && cmake --build build -j
sudo cmake --install build   # installs libcrispasr.{so,dylib,dll}
```

If `libcrispasr` is in a non-standard location, set `CRISPASR_LIB_PATH`:

```bash
export CRISPASR_LIB_PATH=/path/to/libcrispasr.so
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
for seg in s.transcribe_pcm(pcm_f32, sample_rate=16000):
    print(seg.text)
```

## API

- `CrispASR` — Whisper-compatible high-level API
- `Session` — unified API across all 17 backends
- `align_words(...)` — word-level CTC alignment
- `diarize_segments(...)` — speaker diarization (pyannote-compatible)
- `detect_language_pcm(...)` — language ID
- `registry_lookup(...)` — auto-download known models from the model hub

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, model registry, and CLI.

## License

MIT — see [LICENSE](LICENSE).
