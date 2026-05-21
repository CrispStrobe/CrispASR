---
title: CrispASR
sdk: docker
app_port: 7860
pinned: false
---

# CrispASR Space

Gradio wrapper around the [CrispASR](https://github.com/CrispStrobe/CrispASR)
HTTP server, packaged for Hugging Face Docker Spaces. One container runs the
C++ inference engine on `:8080` and the Gradio UI on `:7860`.

## What's exposed

| Tab | Backend(s) | Endpoint hit | Approx. footprint |
|---|---|---|---|
| **Transcribe (ASR)** | whisper, parakeet, moonshine, moonshine-de, wav2vec2 (EN+DE), parakeet-ctc-0.6b, cohere, qwen3 | `POST /v1/audio/transcriptions` | 37 MB – 550 MB per model |
| **Speak (TTS)** | kokoro (82M, multilingual) | `POST /v1/audio/speech`, `GET /v1/voices` | ~85 MB |
| **Detect language (text)** | CLD3, GlotLID-V3, LID-176 | `crispasr-lid` subprocess | 440 KB – 250 MB |
| **About & backends** | static capability table | `GET /backends` + `/health` | — |

Models hot-swap through `POST /load` — only one model is resident at a time,
so switching backends triggers a download (first use) and a load (every use).
The first cold download for each backend lives in `/cache`.

The larger speech-LLM backends in CrispASR (Voxtral 2.5 GB, MiMo-ASR 4.5 GB,
Granite-4.1 3 GB, omniasr-llm) are deliberately omitted from this demo —
they exceed the free-tier (16 GB) RAM ceiling once Gradio + Python + KV
cache overhead is accounted for. To run them, build the image locally.

## Environment variables

- `CRISPASR_MODEL=/models/model.gguf`  (overridden by `/load` requests)
- `CRISPASR_BACKEND=whisper`           (initial backend; UI swaps later)
- `CRISPASR_LANGUAGE=auto`             (default language for transcription)
- `CRISPASR_AUTO_DOWNLOAD=1`           (1 → resolve `-m auto` from the registry)
- `CRISPASR_CACHE_DIR=/cache`          (auto-download landing zone)
- `CRISPASR_SAMPLES_DIR=/space/samples` (bundled `jfk.wav` etc.)
- `CRISPASR_API_KEYS=`                 (optional comma-separated keys; protects every `/v1/*`)
- `CRISPASR_EXTRA_ARGS=`               (extra CLI flags forwarded verbatim, e.g. `--vad --punc-model auto`)

## Local build / run

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .

docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_BACKEND=whisper \
  -e CRISPASR_AUTO_DOWNLOAD=1 \
  crispasr-hf-space
```

Persist the model cache between runs:

```bash
docker volume create crispasr-cache
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_AUTO_DOWNLOAD=1 \
  -v crispasr-cache:/cache \
  crispasr-hf-space
```

Adjust build parallelism with `--build-arg CRISPASR_BUILD_JOBS=8`.
The Dockerfile compiles two binaries from the CrispASR repo: `crispasr`
(server + ASR/TTS) and `crispasr-lid` (text language ID).

## Workflow notes

- **First transcription** of a chosen backend triggers an HF download into
  `/cache`. Whisper-base (~147 MB) and Moonshine-tiny (~37 MB) feel
  instant; Cohere (~550 MB) and Qwen3-ASR (~500 MB) take a minute.
- **Swap order matters**. Loading a TTS backend evicts the ASR backend and
  vice versa. Use the Transcribe tab's *Load model* button to swap back.
- **Voices**: Kokoro's built-in voicepacks (`af_heart`, `af_bella`,
  `am_michael`, `df_victoria`, …) are baked into the auto-downloaded GGUF.
  Drop extra `*.gguf` / `*.wav` files into `$VOICE_DIR` if you want
  `GET /v1/voices` to list them; the Space doesn't ship a voice dir by
  default.
- **Long audio**: enable VAD chunking with `CRISPASR_EXTRA_ARGS=--vad` so
  every backend processes minute-long files without truncation.
- **Word timestamps** on LLM-style backends (qwen3, cohere) need an
  external CTC aligner — outside the scope of this free-tier demo, see
  `docs/cli.md` in the main repo.
