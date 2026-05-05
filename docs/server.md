# Server mode (HTTP API)

`crispasr --server` starts a persistent HTTP server with the model
loaded once and reused across requests. Compatible with the OpenAI
audio-transcription protocol, so any tool that already speaks
OpenAI's API (LiteLLM, LangChain, custom clients) can point at
CrispASR with zero code changes.

## Quick start

```bash
# Start server with model loaded once
crispasr --server -m model.gguf --port 8080

# Transcribe via HTTP (model stays loaded between requests):
curl -F "file=@audio.wav" http://localhost:8080/inference
# {"text": "...", "segments": [...], "backend": "parakeet", "duration": 11.0}

# Hot-swap to a different model at runtime:
curl -F "model=path/to/other-model.gguf" http://localhost:8080/load

# Check server status:
curl http://localhost:8080/health
# {"status": "ok", "backend": "parakeet"}

# List available backends:
curl http://localhost:8080/backends
# {"backends": ["whisper","parakeet","canary",...], "active": "parakeet"}
```

The server loads the model once at startup and keeps it in memory.
Subsequent `/inference` requests reuse the loaded model with no reload
overhead. Requests are mutex-serialized. Use `--host 0.0.0.0` to
accept remote connections.

## API keys

To require API keys, set the `CRISPASR_API_KEYS` env var
(comma-separated). **Do not** pass keys as CLI arguments — they would
be visible in `ps` / `top`. Protected endpoints accept either
`Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health`
remains public for container health checks.

```bash
CRISPASR_API_KEYS=key-one,key-two crispasr --server -m model.gguf

curl -H "Authorization: Bearer key-one" \
  -F "file=@audio.wav" \
  http://localhost:8080/v1/audio/transcriptions
```

## OpenAI-compatible endpoint

`POST /v1/audio/transcriptions` is a drop-in replacement for the
[OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

```bash
# Same curl syntax as the OpenAI API:
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=json"
# {"text": "And so, my fellow Americans, ask not what your country can do for you..."}

# Verbose JSON with per-segment timestamps (matches OpenAI's format):
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
# {"task": "transcribe", "language": "en", "duration": 11.0, "text": "...", "segments": [...]}

# SRT subtitles:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt"

# Plain text:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**Supported form fields:**

| Field | Description |
|---|---|
| `file` | Audio file (required) |
| `model` | Ignored (uses the loaded model) |
| `language` | ISO-639-1 code (default: server's `-l` setting) |
| `prompt` | Initial prompt / context |
| `response_format` | `json` (default), `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | Sampling temperature (default: 0.0) |

`GET /v1/models` returns an OpenAI-compatible model list with the
currently loaded model.

## Text-to-speech endpoint

`POST /v1/audio/speech` is the OpenAI-compatible TTS counterpart to
`/v1/audio/transcriptions`. Available whenever the loaded backend
declares `CAP_TTS` (kokoro, qwen3-tts, vibevoice, orpheus,
chatterbox). Routes register on every backend; non-TTS backends
respond with a 400 pointing the caller at `POST /load`.

```bash
crispasr --server --backend qwen3-tts-customvoice \
  -m qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf \
  --codec-model qwen3-tts-tokenizer-12hz.gguf \
  --voice-dir ./voices \
  --port 8080

curl http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello there.", "voice": "vivian"}' \
  -o out.wav
```

**Body fields:**

| Field | Default | Description |
|---|---|---|
| `input` | (required) | Text to synthesize. Capped at `--tts-max-input-chars` (default 4096); set to 0 to disable the cap. Long input is automatically split on sentence boundaries before synthesis (see [long-form chunking](#long-form-chunking-for-v1audiospeech) below). |
| `model` | (ignored) | Read but not validated — we serve whatever was loaded via `-m` or `POST /load`. Surfaced in the synth log line. |
| `voice` | server's `--voice` | Passed through verbatim to the backend's `params.tts_voice`. Each backend interprets it on its own terms — qwen3-tts CustomVoice as a speaker name (`vivian`, `ryan`); qwen3-tts Base as a path or (with `--voice-dir`) a bare name resolving to `<voice-dir>/<name>.{wav,gguf}`; orpheus as a preset (`tara`, `leah`). |
| `instructions` | empty | Voice-direction prose for backends that support it (qwen3-tts VoiceDesign). Silently ignored on other backends so OpenAI clients targeting `gpt-4o-mini-tts` don't see 4xx errors. |
| `speed` | `1.0` | Tempo multiplier `0.25 .. 4.0` (OpenAI range). Applied as a post-synth linear resampler. Out-of-range returns 400 with `code=invalid_speed`. |
| `response_format` | `"wav"` | `wav` (16-bit PCM RIFF, 24 kHz mono — default), `pcm` (OpenAI spec: 24 kHz signed 16-bit LE raw, no header), or `f32` (crispasr-specific raw float32 for downstream DSP). |

**Returns:**

| Status | Content-Type | Body |
|---|---|---|
| 200 | `audio/wav` | RIFF WAV, 16-bit PCM int16, 24 kHz mono |
| 200 | `audio/pcm` | Raw int16 LE bytes (OpenAI `pcm`) |
| 200 | `application/octet-stream` | Raw float32 PCM (`f32`) |
| 400 | `application/json` | OpenAI error shape: `{"error": {"message", "type", "code", "param"}}`. Codes: `missing_required_field`, `input_too_long`, `invalid_json`, `invalid_speed`, `unsupported_response_format`. |
| 500 | `application/json` | Synthesis returned empty (e.g. unknown voice). `code=synthesis_failed`. |
| 503 | `application/json` | Model still loading. |

**Voice listing:**

```bash
curl http://localhost:8080/v1/voices
# {"voices": [{"name": "vivian", "format": "wav"}, {"name": "ryan", "format": "gguf"}]}
```

`GET /v1/voices` enumerates `*.wav` and `*.gguf` stems in `--voice-dir`.
The listing reflects the filesystem; whether a particular backend
actually accepts a given voice depends on the backend's own resolution
(e.g. CustomVoice models only accept names baked into the GGUF
metadata — the `<voice-dir>` files are irrelevant to those).

### Voice file conventions

```
voices/
  vivian.wav        # reference audio for runtime voice cloning
  vivian.txt        # transcription of vivian.wav (Qwen3-TTS ICL prefill)
  ryan.gguf         # baked voice pack
```

For Qwen3-TTS Base variant: when `voice` is a bare name (no path
separator, no extension), the backend looks for `<voice-dir>/<name>.wav`
+ `<voice-dir>/<name>.txt`, falling back to `<voice-dir>/<name>.gguf`.
The `.txt` companion is auto-loaded as `tts_ref_text` if the request
doesn't carry one. Path traversal in the name (`..`, `/`, `\\`, NUL) is
rejected before the filesystem is touched.

Other backends (kokoro, vibevoice, orpheus) interpret `voice` according
to their own conventions — see `docs/tts.md` for per-backend specifics.

### Long-form chunking for /v1/audio/speech

The talker LM in every TTS backend has a finite training horizon
(qwen3-tts-1.7b-base degrades past ~600 chars / 200 codec frames and
silently truncates trailing text at MAX_FRAMES). The route
auto-chunks `input` on sentence boundaries before dispatching to the
backend, then concatenates per-chunk PCM with a 200 ms silence pad.

- Recognises ASCII `.!?`, CJK ideographic full stop `。` (U+3002), and
  Devanagari danda `।` (U+0964).
- Decimal-aware: `1.5` stays intact (period is followed by a digit).
- Run-on input with no punctuation falls back to whitespace-boundary
  split at `--tts-max-input-chars`.
- Voice consistency holds across chunks because the talker re-prefills
  with the same ICL ref each call (and the per-call setup is amortised
  by qwen3-tts's `last_voice_key_` cache).
- Server log line reports `chunks=N` for observability.

Single-sentence input is a 1-element vector — per-call overhead is
one `std::vector<float>` move.

### CORS

Browser clients calling `/v1/*` from a different origin need the server
to opt in via `--cors-origin`:

```bash
crispasr --server --backend qwen3-tts-customvoice -m model.gguf \
  --cors-origin '*'   # any origin — for dev only
# or:
  --cors-origin 'https://app.example.com'   # specific origin — production
```

When set, every response carries `Access-Control-Allow-Origin`,
`-Methods`, and `-Headers`; preflight `OPTIONS` requests get a 204 with
the same. Default-empty stays default-locked.

### Speed

`speed` is applied at the server layer via linear-interpolation
resampling of the synthesised float32 buffer — backends produce at
their native rate, the server resamples before format dispatch.
Quality loss vs. a sinc resampler is minimal at modest speeds
(0.5x .. 2.0x) for speech. Backends that grow native duration knobs
will plumb `params.tts_speed` through directly and bypass this path
(future work).

### Python OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-anything")

resp = client.audio.speech.create(
    model="tts-1",          # ignored; we serve the loaded backend
    voice="vivian",
    input="The quick brown fox jumps over the lazy dog.",
    speed=1.0,
    response_format="wav",
)
resp.stream_to_file("out.wav")
```

### Deferred

| Feature | Status |
|---|---|
| Streaming response (chunked / SSE) | Pending — see PLAN §70 (couples with chunked-VAE for the full latency win). |
| `mp3` / `opus` / `aac` / `flac` encoding | Not implemented — needs lame/opusenc/etc. as build deps. |
| `POST /v1/voices` (multipart upload for runtime provisioning) | Pending — security review (size limits, content-type validation, disk quota). |
| `DELETE /v1/voices/{name}` | Pending alongside upload. |
| Native-backend `speed` (duration knobs vs server-side resample) | Pending — backend-by-backend. |

## Docker Compose

The repo includes a root-level
[`docker-compose.yml`](../docker-compose.yml) for running the
persistent HTTP server against a mounted model directory.

```bash
cp .env.example .env
# Edit CRISPASR_MODEL to point at a file mounted under ./models

docker compose up --build

# Health check
curl http://localhost:8080/health

# OpenAI-compatible transcription API
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $CRISPASR_API_KEY" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

By default the compose stack:
- builds from `.devops/main.Dockerfile`
- mounts `./models` into `/models`
- stores auto-downloaded models in the Docker-managed
  `crispasr-cache` volume at `/cache`
- serves on `http://localhost:8080`

If you want `/cache` to be a host directory instead, replace the
`crispasr-cache:/cache` volume with `./cache:/cache` and make it
writable by the container user before startup:

```bash
mkdir -p cache models
sudo chown -R "$(id -u):$(id -g)" cache models
```

You can cap or raise build parallelism with `CRISPASR_BUILD_JOBS`:

```bash
docker compose build --build-arg CRISPASR_BUILD_JOBS=8
```

For CUDA builds, use the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

## Prebuilt CUDA images — choosing a tag

We publish two CUDA tags on `ghcr.io/crispstrobe/crispasr`. Pick the
one that matches your host driver:

| Tag | CUDA | Min NVIDIA driver | Supported arches | Notes |
|---|---|---|---|---|
| `main-cuda` | 13.0 | **R535+** (R580+ for full features) | sm_75…sm_120 incl. RTX 50xx (Blackwell) | Default. Pull this on modern hosts. |
| `main-cuda-12` | 12.4 | **R510+** | sm_75…sm_90 (RTX 20/30/40-series, Hopper) | Legacy compat — use on RHEL 7/8, older Ubuntu LTS, or any host that hasn't updated drivers in a while. RTX 50xx is **not** supported here. |

Quick check: `nvidia-smi` shows your driver version in the top-right.
If it's R535 or higher, pull `main-cuda`. If it's R510–R534, pull
`main-cuda-12`. If it's older than R510, update your driver — neither
image will work.

```bash
docker pull ghcr.io/crispstrobe/crispasr:main-cuda      # modern hosts
docker pull ghcr.io/crispstrobe/crispasr:main-cuda-12   # legacy driver
```

## Hugging Face Space wrapper

There is also a Gradio-based Hugging Face Space wrapper under
[`hf-space/`](../hf-space/README.md). It starts the CrispASR HTTP
server inside the container and provides a small browser UI on top of
the OpenAI-compatible transcription endpoint.

Build it locally with:

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_MODEL=/models/ggml-base.en.bin \
  -v "$PWD/models:/models" \
  crispasr-hf-space
```

The compose files default to local image tags (`crispasr-local:*`)
so they don't depend on pulling a published registry image first.

## Environment overrides

You can override the loaded model and startup flags through `.env`:

| Variable | Purpose |
|---|---|
| `CRISPASR_MODEL` | Model path inside the container (e.g. `/models/parakeet-tdt-0.6b-v2.gguf`) |
| `CRISPASR_BACKEND` | Force a specific backend |
| `CRISPASR_LANGUAGE` | ISO-639-1 code or `auto` for LID |
| `CRISPASR_AUTO_DOWNLOAD` | Set to `1` to enable `-m auto` resolution |
| `CRISPASR_CACHE_DIR` | Where auto-downloaded models live (defaults to `/cache`) |
| `CRISPASR_API_KEYS` | Comma-separated API keys (see [API keys](#api-keys)) |
| `CRISPASR_EXTRA_ARGS` | Forwarded verbatim to the server CLI (e.g. `--no-punctuation`) |

The service is configured to avoid serving as root by default:
- `user: "${CRISPASR_UID:-1000}:${CRISPASR_GID:-1000}"`
- `security_opt: ["no-new-privileges:true"]`
