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
overhead. Requests are mutex-serialized by default (the HTTP layer accepts
connections concurrently, but inference runs one-at-a-time through the single
loaded model). To run transcriptions concurrently, or to scale out with
replicas / bulk process fan-out, see
**[Concurrency, parallelism & scaling](concurrency.md)** (`--server-workers N`).
Use `--host 0.0.0.0` to accept remote connections.

## API keys

To require API keys, set the `CRISPASR_API_KEYS` env var
(comma-separated). An `--api-keys` CLI flag also exists and its keys are
merged with the env var's, but **do not** use it — CLI arguments are
visible in `ps` / `top`. Protected endpoints accept either
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
| `response_format` | `json` (default), `verbose_json`, `diarized_json`, `text`, `srt`, `vtt` |
| `temperature` | Sampling temperature (default: 0.0) |
| `seed` | RNG seed for sampling (`0` = non-deterministic) |
| `max_tokens` | Generated-token cap for supported autoregressive ASR backends |
| `max_new_tokens` | Alias for `max_tokens` |
| `frequency_penalty` | Opt-in repeated generated-token penalty for supported autoregressive ASR backends (`0.0` disabled) |
| `offset_t_ms` | Start transcription this many ms into the audio (default: 0). Reported timestamps stay in original-audio time |
| `duration_ms` | Transcribe only this many ms from `offset_t_ms` (default: 0 = to end) |
| `translate` | `true`/`false` — translate to English (backends with `CAP_TRANSLATE`) |
| `source_lang` | Source language for AST backends (canary, cohere) |
| `target_lang` | Target language for AST backends |
| `punctuation` | `true`/`false` — enable/disable punctuation (default: `true`; `false` strips punctuation from output) |
| `diarize` | `true`/`false` — enable speaker diarization |
| `diarize_method` | `energy`, `xcorr`, `vad-turns`, `sherpa`, `pyannote`, `ecapa`, `foxnose` (default: `energy`) |
| `diarize_embedder` | Speaker-embedding model for cross-slice clustering (path or `auto`) |
| `diarize_cluster_threshold` | Cosine merge threshold for embedding clustering (default: 0.5) |
| `diarize_max_speakers` | Upper bound on speaker cluster count (default: 8) |
| `vad_export` | `true`/`false` — include the computed VAD/chunk boundaries in the JSON response under `vad_segments` (default: `false`) |
| `vad_import` | The `vad_segments` object from an earlier `vad_export` response. Reuses those boundaries and skips VAD entirely |
| `vad_export_raw` | `true`/`false` — with `vad_export`, return raw VAD speech segments (chunk-independent, re-chunked per request) instead of chunk boundaries (default: `false`) |
| `vad_import_strict` | `true`/`false` — with `vad_import`, reject (rather than warn) a chunk-boundary payload whose chunk length differs from this request (default: `false`) |
| `vad` | `true`/`false` — enable VAD pre-processing |
| `vad_threshold` | VAD speech probability threshold (default: 0.5) |
| `vad_min_speech_duration_ms` | Minimum speech segment duration in ms (default: 250) |
| `vad_min_silence_duration_ms` | Minimum silence gap to split on in ms (default: 100) |
| `vad_max_speech_duration_s` | Maximum speech segment duration in seconds |
| `vad_speech_pad_ms` | Padding around speech segments in ms (default: 30) |
| `hotwords` | Comma-separated hotword list for biased decoding |
| `hotwords_boost` | Log-prob boost per hotword token match (default: 2.0) |
| `suppress_regex` | Regex pattern to suppress from output |
| `suppress_nst` | `true`/`false` — suppress non-speech tokens |
| `grammar` | GBNF grammar string for constrained decoding |
| `grammar_rule` | Root rule name for the grammar |
| `best_of` | Whisper best-of-N sampling candidates |
| `beam_size` | Whisper beam search width |
| `return_logits` | `true`/`false` — for supported dense CTC backends, include `ctc_logits` in JSON responses (`n_frames`, `n_vocab`, frame-major `data`, optional `vocab`) |
| `sensitivity` | `conservative` / `balanced` / `aggressive` — the four threshold fields as one bundle. Applied before them, so an explicit `entropy_thold` etc. in the same request still wins. An unknown value is ignored with a stderr warning |
| `entropy_thold` | Entropy threshold for decoder fallback |
| `logprob_thold` | Log-probability threshold for decoder fallback |
| `no_speech_thold` | No-speech probability threshold |
| `temperature_inc` | Temperature increment for fallback retries |
| `no_fallback` | `true`/`false` — disable temperature fallback |
| `detect_language` | `true`/`false` — run language detection |
| `lid_backend` | Language-ID backend (`whisper`/`silero`/`firered`; `off`/`none` to disable) |
| `lid_model` | Optional language-ID model path |
| `no_timestamps` | `true`/`false` — omit timestamps from output |
| `split_on_word` | `true`/`false` — split segments on word boundaries |
| `max_len` | Maximum segment length in characters |
| `chunk_seconds` | Maximum chunk duration for long audio (default: 30) |
| `chunk_overlap` | Overlap context (seconds) around chunk boundaries |
| `split_on_punct` | `true`/`false` — split segments on punctuation (default: `false`) |
| `max_context` | Maximum decoder text-context tokens carried across chunks (default: `-1` = model default) |
| `audio_ctx` | Truncated audio context (whisper; default: `0` = full) |
| `word_thold` | Word-timestamp probability threshold (default: 0.01) |
| `carry_initial_prompt` | `true`/`false` — re-inject the initial prompt on every chunk (default: `false`) |
| `lcs_dedup` | NeMo-style LCS hypothesis stitching at chunk boundaries: `auto` (default, fires when chunking with overlap is active), `on`, `off` |
| `lcs_min_length` | Lower bound on LCS match length to act on (default: 1) |
| `parakeet_decoder` | Parakeet decode head: `tdt` (default) or `ctc` |
| `att_context` | Parakeet/canary local-attention window as `"L,R"` encoder frames (1 ≈ 80 ms) — NeMo `rel_pos_local_attn`, bounds long-audio encoder memory (#257). Unset = model default |
| `no_auto_aligner` | `true`/`false` — suppress the automatic forced-aligner pass (default: `false`) |
| `show_alternatives` | `true`/`false` — emit per-token alternative candidates (default: `false`) |
| `alt_n` | Number of alternatives to emit with `show_alternatives` (default: 3) |
| `stream` | `true`/`false` — for backends with `CAP_STREAMING`, stream partial text as Server-Sent Events (`data: {"text": "<delta>"}` … `data: [DONE]`, `Content-Type: text/event-stream`) instead of a buffered response. Ignored on backends without `CAP_STREAMING` |
| `strict_pipeline` | `true`/`false` — #311: fail the request (HTTP 400) if an explicitly-requested aux stage (VAD, forced aligner, punctuation) could not load or produce its output, instead of degrading silently (default: `false`) |
| `require_vad` | `true`/`false` — force the VAD-load-success requirement (needs `vad=true` in the request, or a server started with `--vad` / `-vm`; there is no per-request `vad_model` field) |
| `require_word_timestamps` | `true`/`false` — fail unless every non-empty segment carries word timestamps (native or aligned) |
| `require_punctuation` | `true`/`false` — fail unless a punctuation model is loaded (start the server with `--punc-model`) |

The `/inference` endpoint accepts the same CrispASR extension fields.

### Strict pipeline — fail on a required stage's failure (#311)

By default the server, like the CLI, **degrades gracefully**: a VAD, forced
aligner, or punctuation model that fails to load is skipped and the request
still returns `200`. Integrations that treat those stages as required task
properties can opt into strict semantics with the fields above — a required
stage that fails to load or produce its output then returns **HTTP 400** with
an `{"error": {...}}` body instead of a degraded `200`. A stage that ran and
legitimately produced nothing (VAD detected no speech) stays a success. This
mirrors the CLI's `--strict-pipeline` family (see
[`cli.md`](cli.md#strict-pipeline--require-aux-stages-to-succeed---strict-pipeline-311));
the strict decision is shared code (`crispasr_strict.h`), so the two front-ends
cannot drift.

```bash
# rc-style contract over HTTP: 200 ⟺ every required stage succeeded.
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://localhost:8080/v1/audio/transcriptions \
  -F file=@meeting.wav -F model=whisper \
  -F vad=true -F strict_pipeline=true    # 400 if VAD couldn't load; 200 otherwise
```

`require_punctuation` needs the server to have been started with `--punc-model`
(punctuation is a resident startup post-processor, not per-request) — otherwise
the request fails fast with a clear error.

### Reusing VAD boundaries across backends (#227)

VAD (or the fixed-chunk fallback) runs on every request. To transcribe the same
audio with several backends without paying it each time, ask for the boundaries
once and hand them back afterwards:

```bash
# 1. Transcribe + get the boundaries back.
curl -s -F file=@talk.wav -F vad=true -F vad_export=true \
     -F response_format=verbose_json \
     http://127.0.0.1:8080/v1/audio/transcriptions > first.json

# 2. Extract the vad_segments object.
jq '.vad_segments' first.json > vad.json

# 3. Reuse it on later requests — no VAD model is run.
curl -s -F file=@talk.wav -F "vad_import=<vad.json" \
     -F response_format=verbose_json \
     http://127.0.0.1:8080/v1/audio/transcriptions
```

`vad_segments` is the same wire format the CLI's `--vad-export` writes, so
boundaries are interchangeable between the two. Boundaries are clamped to the
audio of the request they're used on and out-of-range slices are dropped, so a
stale set can't read out of bounds; a malformed one returns
`invalid_request_error`. They are interpreted against the audio actually being
processed — i.e. after any `offset_t_ms`/`duration_ms` window is applied.

> **Parakeet segmentation (issue #257).** Backends that chunk internally
> (parakeet/canary — full-attention FastConformer) now receive the whole clip
> instead of dispatcher-pre-sliced pieces (per-slice transcribe corrupts this
> encoder). With an explicit `chunk_seconds=N`, the non-JA Parakeet response is
> split into **~N-second segments** of the complete transcript (each with
> `start`/`end`/`words`); without it, one coherent segment (or the silence-split
> longform above the memory cap). VAD, when requested, still provides
> silence-bounded slices. Matches the CLI's `--chunk-seconds` behaviour.

### Server startup flags (resident post-processors)

A few options are set once at launch rather than per request — the model is
loaded resident and applied to every transcription:

- `--punc-model auto|firered|fullstop|punctuate-all|pcs|<path>` — restore
  punctuation on backends that emit none (parakeet RNNT/CTC, etc.). Auto-enabled
  for non-PnC CTC backends, matching the CLI.
- `--truecase-model auto|crf|lstm|<path>` — truecasing applied after punctuation.
- `--no-warmup` (or `CRISPASR_NO_WARMUP=1`) — skip the startup warmup transcribe
  (workaround for GPU drivers that hang/crash in warmup; see #165).

### Diarization example

```bash
# Speaker diarization (energy method — works on stereo audio):
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "response_format=verbose_json" \
  -F "diarize=true"

# Pyannote diarization (works on mono, needs pyannote-seg GGUF):
crispasr --server -m model.gguf --diarize --diarize-method pyannote \
  --sherpa-segment-model pyannote-seg-3.0.gguf

curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "response_format=verbose_json" \
  -F "diarize=true" \
  -F "diarize_method=pyannote"
```

### Diarized JSON format (#206)

`response_format=diarized_json` returns OpenAI-compatible verbose JSON
extended with per-segment speaker labels. `(speaker N)` labels are
normalised to letters (`A`, `B`, `C`, … and `AA`, `AB`, … past 26); a
label in any other shape is passed through trimmed. Each segment includes a `type`
field for compatibility with transcription clients that expect the
diarized schema.

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@DIALOGUE.ogg" \
  -F "response_format=diarized_json" \
  -F "diarize=true" \
  -F "diarize_method=pyannote"
```

Response structure:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 245.029,
  "text": "Full transcript text...",
  "segments": [
    {
      "id": 0,
      "start": 0.00,
      "end": 26.10,
      "text": "I'll tell you basically what this is about...",
      "speaker": "A",
      "type": "transcript.text.segment"
    }
  ]
}
```

When `diarize` is not enabled, all segments default to speaker `"A"`.
Word-level timestamps are included in each segment when the backend
provides them.

### Translation example

```bash
# Translate non-English audio to English (whisper, canary, cohere):
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@german.wav" \
  -F "translate=true" \
  -F "response_format=json"
```

### Hotwords example

```bash
# Boost domain-specific terms:
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "hotwords=CrispASR,GGUF,parakeet" \
  -F "hotwords_boost=3.0"
```

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
| `voice` | server's `--voice` | Passed through verbatim to the backend's `params.tts_voice`. Each backend interprets it on its own terms — qwen3-tts CustomVoice as a speaker name (`vivian`, `ryan`); qwen3-tts Base as a path or (with `--voice-dir`) a bare name resolving to `<voice-dir>/<name>.{wav,gguf}`; orpheus as a preset (`tara`, `leah`); tada as a `tada-ref-*.gguf` path/name. For **tada the voice is switched per request without a restart** (#201): naming a different reference reloads it; `default`/`auto`/omitted keeps the currently-loaded voice. A `.wav` reference is cloned on the fly (needs `ref_text` + the encoder/aligner GGUFs) when the server runs with `CRISPASR_TADA_WAV_CLONE=1` — otherwise convert it to a `tada-ref.gguf` first. |
| `instructions` | empty | Voice-direction prose for backends that support it (qwen3-tts VoiceDesign). Silently ignored on other backends so OpenAI clients targeting `gpt-4o-mini-tts` don't see 4xx errors. |
| `seed` | `0` | RNG seed for sampling. `0` = non-deterministic. Same-seed + same-text produces bit-identical audio on all sampling-capable TTS backends (qwen3-tts, chatterbox, vibevoice, orpheus). |
| `temperature` | server's `--temperature` | Sampling temperature for AR TTS backends. `0` = greedy; backends apply their own default (e.g. 0.8 for qwen3-tts) when the global default of 0.0 is unchanged. |
| `max_new_tokens` | server's `--max-new-tokens` | AR token generation cap. `<= 0` clears the override and uses the backend default. |
| `max_speech_tokens` | backend default | MOSS-TTS / moss-tts-local hard cap on generated audio frames (~12.5 frames/sec). The decode loop is forced to end once this many frames are produced, so it bounds the worst-case synthesis length even when the model never emits its end token. Per request. |
| `min_speech_tokens` | backend default | MOSS-TTS / moss-tts-local minimum number of generated audio frames (~12.5 frames/sec). The decode loop is forbidden from ending until this many frames are produced. **Setting `min_speech_tokens == max_speech_tokens` yields exact-duration synthesis** — the model generates precisely that window of audio (within one frame ≈ 80 ms) with no post-hoc tempo change, which is what game-dubbing / lip-sync work needs. Per request. |
| `frequency_penalty` | `0.0` | Opt-in repeated generated-token penalty for AR TTS backends. `0.0` disabled. |
| `top_p` | backend default | Nucleus-sampling cutoff for AR TTS backends (tada, chatterbox). Applied per request; omit to keep the backend default. |
| `min_p` | backend default | Min-p sampling cutoff for AR TTS backends. Per request. |
| `top_k` | backend default | Top-k sampling cutoff (`0` = disabled). Honoured by tada. Per request. |
| `repetition_penalty` | backend default | Talker repetition penalty (`1.0` = none). Honoured by tada, chatterbox. Per request. |
| `do_sample` | backend default | `true`/`false` — enable/disable talker sampling (`false` = greedy). Honoured by tada. Per request. |
| `num_candidates` | backend default | tada per-token flow-matching candidates: higher = more robust timing, lower = faster (`1` = single noise draw). Per request. |
| `num_steps` | backend default | Flow-matching ODE steps. For tada this is the primary "quick and dirty" vs "slow and accurate" lever (Python `num_flow_matching_steps`, default 10): more steps = slower, higher acoustic fidelity. Also honoured by chatterbox/f5. Per request. |
| `cfg_scale` | backend default | Classifier-free-guidance scale. For tada the acoustic CFG (Python `acoustic_cfg`, default 1.6). Also chatterbox/f5. Per request. |
| `noise_temp` | backend default | tada flow-matching noise temperature (Python `noise_temp`, default 0.9). Per request. |
| `noise_scale` | backend default | piper VITS variance. Per request. |
| `noise_w` | backend default | piper stochastic duration-predictor noise. Per request. |
| `exaggeration` | backend default | chatterbox expressiveness. Per request. |
| `speaker_id` | backend default | Speaker index for a piper multi-speaker model. Per request. |
| `phonemes` | empty | #316: drive the acoustic model with these IPA phonemes, skipping the G2P. **kokoro and piper only** — any other backend is a `400` with `code=unsupported_phonemes` rather than a silent fallback to synthesizing `input`. |
| `pad_silence_ms` | `0` | Pad this many ms of silence at the start of the output (per-request form of `--tts-pad-silence-ms`). |
| `stream` | `false` | Stream the audio back with chunked transfer encoding instead of buffering the whole clip. See [Streaming synthesized audio](streaming.md#server-stream-true). Only the PCM formats are accepted (`wav`/`pcm`/`f32`); `mp3`/`aac`/`opus` return `400`. The body is always raw int16 LE PCM at the backend's native rate with `Content-Type: audio/pcm` — no RIFF header even when `response_format` was `wav`. |
| `speed` | `1.0` | Tempo multiplier `0.25 .. 4.0` (OpenAI range). Applied as a post-synth linear resampler. Out-of-range returns 400 with `code=invalid_speed`. |
| `response_format` | `"wav"` | `wav` (16-bit PCM RIFF, 24 kHz mono — default), `pcm` (OpenAI spec: 24 kHz signed 16-bit LE raw, no header), `f32` (crispasr-specific raw float32 for downstream DSP), or the compressed containers `mp3` / `aac` / `opus` — all encoded in-tree by [glint](https://github.com/CrispStrobe/glint), no build deps. `opus` returns a standard **Ogg Opus** file (`audio/ogg`); set `CRISPASR_OPUS_ENCODER=libopus` (build with libopus) to fall back to the legacy raw-packet framing (`audio/opus`) instead. |
| `consent_attestation` | empty | Required when `voice` is a **clone**: a `.wav` reference, or a `.gguf` pack stamped `crispasr.voice.cloned_from_recording` by the baker that derived it from a real recording. The name is resolved against `--voice-dir` first, so a bare `"victim"` is treated exactly like `"victim.wav"`. A free-text statement attesting speaker consent, e.g. `"I have the speaker's consent"`. Logged for audit, with the reason the voice was classified a clone. Preset packs (kokoro, vibevoice, `tada-ref-<lang>`, …) carry no stamp and need no attestation — see [`eu-ai-act.md` §6.2](eu-ai-act.md#62-art-504--deepfake-disclosure). Note that "no attestation" is not "nothing to disclose": a preset whose voice is a real person still owes the audible label, via `speaker_identity` below. |
| `ref_text` | empty | Transcript of the `.wav` clone reference, used by TADA on-the-fly cloning (#201). A companion `<name>.txt` in `--voice-dir` is used when omitted. Ignored by backends that clone from audio alone. |
| `language` | server's `-l` | The language to **speak** (alias: `target_lang`). ISO-639-1 (`de`) or an English name (`German`). cosyvoice3 compares it to the reference voice's language and switches to cross-lingual synthesis when they differ; qwen3-tts sets the talker's explicit `codec_language_id`; moss-tts fills its `- Language:` prompt field; kokoro/zonos/piper pick the eSpeak voice. Language-agnostic backends (voxcpm2, f5-tts, vibevoice) ignore it and read the script of `input` instead. |
| `source_lang` | empty | The language the **cloning reference** is spoken in (alias: `ref_lang`) — not the output language. Only cosyvoice3 acts on it, to decide whether the requested `language` needs cross-lingual synthesis. Optional: the backend otherwise infers it from the voice-bank entry or from `ref_text`. That inference declines rather than guesses on a short transcript, and when it declines the requested `language` has no effect and the clone keeps the reference's accent — set this to make it explicit (#329). |
| `speaker_identity` | empty (`unknown`) | Whose voice a **preset** voice is: `real_person`, `synthetic` or `unknown`. `real_person` adds the audible AI disclosure to non-cloned output — a preset shipped inside a model can be an identifiable individual, which makes the output a deep fake under Art. 3(60) even though nothing was cloned. It does **not** require `consent_attestation`: whether that donor agreed to the model being trained is settled upstream and you cannot attest to it. Outranks the pack's own `crispasr.voice.speaker_identity` stamp and the backend default. An unrecognised value is a `400` with `code=invalid_speaker_identity` rather than a silent downgrade. See [`eu-ai-act.md` §6.2a](eu-ai-act.md#62a-whose-voice-is-a-preset-voice-speaker_identity). |
| `spoken_disclaimer` | `true` | Set to `false` to skip the audible AI-disclosure prefix on voice-cloned output. Machine-readable provenance (watermark + C2PA) is always applied. When `false`, the caller assumes responsibility for providing appropriate AI-disclosure to end users — which is why the opt-out is only honoured when attested (see `marking_attestation`). |
| `marking_attestation` | empty | Required (since v0.8.22) to **honour** `"spoken_disclaimer": false` on a voice clone — a free-text affirmation that you accept the AI-content disclosure duty, e.g. `"I will disclose this is AI-generated"`. Logged for audit. A server launched with `--accept-marking-responsibility` has already accepted that duty for every response it serves and satisfies this per-request field. **Without it the opt-out is denied, not the request** (since v0.8.24): the response is still `200`, but it carries the spoken disclaimer and the headers below say so. Earlier v0.8.22/v0.8.23 servers returned `400 marking_attestation_required` instead. |

> **Watermarking.** Every response is watermarked by default. There is **no
> per-request watermark toggle** — the mark is disabled only at the process
> level by starting the server with `--no-watermark` (or `CRISPASR_NO_WATERMARK=1`),
> which turns it off for **all** responses. `--no-watermark` (like `--no-c2pa`)
> **requires `--accept-marking-responsibility`**: without it the server refuses
> to start rather than silently serving unmarked audio. With it, startup logs a
> `[MARKING] scope=server` line recording that the AI-content marking
> responsibility rests with the operator. See
> [`tts.md`](tts.md#disabling-the-watermark-operator-opt-out).
>
> **The floor that opt-out cannot cross.** A response may lose the watermark
> only if a C2PA manifest still marks it. `response_format` decides that, so the
> floor is applied per response: `wav` and `mp3` carry a manifest, and
> `pcm` / `f32` / `aac` / `opus` and the streaming path do not — for those the
> watermark is forced on even under `--no-watermark`. Before v0.8.26 the server
> had no floor at all and signed only `wav`, so `--no-watermark` plus
> `"response_format": "mp3"` returned entirely unmarked synthetic audio while
> the CLI in the same configuration forced the mark back on.

**Returns:**

| Status | Content-Type | Body |
|---|---|---|
| 200 | `audio/wav` | RIFF WAV, 16-bit PCM int16, 24 kHz mono |
| 200 | `audio/pcm` | Raw int16 LE bytes (OpenAI `pcm`) |
| 200 | `application/octet-stream` | Raw float32 PCM (`f32`) |
| 200 | `audio/mpeg` / `audio/aac` / `audio/ogg` | Compressed `mp3` / `aac` / `opus` (Ogg Opus) via glint |
| 400 | `application/json` | OpenAI error shape: `{"error": {"message", "type", "code", "param"}}`. Codes: `missing_required_field`, `input_too_long`, `invalid_json`, `invalid_speed`, `unsupported_response_format`, `consent_required`, `invalid_voice`, `invalid_speaker_identity`, `invalid_instructions`, `unsupported_phonemes`. Also plain (code-less) 400s for a backend without `CAP_TTS` and for `stream=true` with a compressed `response_format`. |
| 500 | `application/json` | Synthesis returned empty (e.g. unknown voice) — `code=synthesis_failed`; or the chosen container failed to encode — `code=encoding_failed`. |
| 503 | `application/json` | Model still loading. |

Every response carries `X-Crispasr-Request-Id` — the per-request correlation
id that also appears as `req=…` in the `[CONSENT]` / `[MARKING]` audit lines.
When a spoken disclosure was owed (a clone, or a `real_person` preset) the
response also carries `X-Crispasr-Spoken-Disclaimer: applied|skipped`, plus
`X-Crispasr-Marking-Warning` when an unattested opt-out was denied.

**Voice listing:**

```bash
curl http://localhost:8080/v1/voices
# {"voices": [{"name": "vivian", "format": "wav"}, {"name": "ryan", "format": "gguf"}]}
```

`GET /v1/voices` enumerates `*.wav` and `*.gguf` stems in `--voice-dir`.
It is also served on `GET /voices` and `GET /v1/audio/voices` as aliases for
llama-swap / OpenAI-client compatibility (#264).
The listing reflects the filesystem; whether a particular backend
actually accepts a given voice depends on the backend's own resolution
(e.g. CustomVoice models only accept names baked into the GGUF
metadata — the `<voice-dir>` files are irrelevant to those).

**Voice upload — requires a consent attestation:**

```bash
curl -X POST http://localhost:8080/v1/voices \
  -F voice=@speaker.wav \
  -F name=vivian \
  -F consent_attestation="I have the speaker's consent" \
  -F transcript="the reference transcript"   # optional, Qwen3-TTS ICL prefill
# 201 {"name": "vivian", "format": "wav", "size_bytes": 220544}
```

`consent_attestation` is **mandatory** — a missing or empty field is a `400`
with `code=consent_required`. Storing a recording as a reusable voiceprint *is*
the cloning step, the same act the CLI gates behind `--i-have-rights` and every
voice baker gates at bake time; taking the attestation only later, at
`/v1/audio/speech`, would mean a third party's voice could be enrolled by anyone
who could reach the endpoint. The upload emits an
`[CONSENT] scope=voice-upload …` audit line. See
[`eu-ai-act.md`](eu-ai-act.md) §6.2.

> **Breaking change.** This field is new; clients that provisioned voices
> without it get a `400` until they add it.

Requires `--voice-dir`; names must match `[a-zA-Z0-9_-]+`; an existing name
needs `?force=true`. `DELETE /v1/voices/{name}` removes one.

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

Some talker LMs have a finite training horizon and degrade (or silently
truncate at MAX_FRAMES) past a few hundred characters. For those backends the
route auto-chunks `input` on sentence boundaries before dispatching to the
backend, then concatenates per-chunk PCM with a 200 ms silence pad.

- Recognises ASCII `.!?`, CJK ideographic full stop `。` (U+3002), and
  Devanagari danda `।` (U+0964).
- Decimal-aware: `1.5` stays intact (period is followed by a digit).
- Run-on input with no punctuation falls back to a whitespace-boundary
  split at the chunker's own **600-char** cap (`crispasr_tts_split_sentences`),
  not at `--tts-max-input-chars` — the latter only decides whether the
  request is accepted at all.
- **Not every backend is chunked.** `vibevoice*`, `qwen3-tts*`, `tada*`,
  `dots-tts*` and `omnivoice*` are dispatched as **one** synthesis call
  whatever the length: they generate multi-sentence utterances in a single
  pass and rely on the continuous prompt / generated-text context to hold
  speaker identity, prosody and trailing-pause timing (splitting them
  re-runs the per-sentence pause over-prediction, and on a cloning request
  re-emits the spoken AI-disclaimer once per chunk). Set
  `CRISPASR_OMNIVOICE_CHUNK=1` to force the legacy sentence-split path on
  omnivoice.
- Where chunking does apply, voice consistency holds across chunks because
  the talker re-prefills with the same ICL ref each call.
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
`-Methods` (`GET, POST, OPTIONS, DELETE`), `-Headers`
(`Content-Type, Authorization, X-API-Key`) and `-Max-Age` (86400);
preflight `OPTIONS` requests get a 204 with the same. Default-empty
stays default-locked.

### Speed

`speed` is applied at the server layer via linear-interpolation
resampling of the synthesised float32 buffer — backends produce at
their native rate, the server resamples before format dispatch.
Quality loss vs. a sinc resampler is minimal at modest speeds
(0.5x .. 2.0x) for speech. Backends that declare `CAP_TTS_SPEED` (e.g. Kokoro)
plumb `params.tts_speed` through directly and bypass this path to preserve
fundamental frequency ($F_0$) and formants without pitch distortion.

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

## Speech-to-speech endpoint

`POST /v1/audio/speech-to-speech` runs end-to-end audio-in → audio-out
on S2S-capable backends (`lfm2-audio`, `mini-omni2`, `sidon`,
`voxcpm2-vae`). Non-S2S backends
return 400.

```bash
crispasr --server --backend lfm2-audio -m lfm2-audio-1.5b-q5_k.gguf
```

```bash
curl http://localhost:8080/v1/audio/speech-to-speech \
  -F "file=@input.wav" \
  -F "response_format=wav" \
  -o output.wav -D -
# X-Transcript header contains the intermediate ASR text (URL-encoded)
```

| Field | Default | Description |
|---|---|---|
| `file` | (required) | Audio file upload (multipart). Decoded to 16 kHz mono float32 internally. |
| `language` | `en` | Language hint passed to the backend. |
| `response_format` | `wav` | Output encoding: `wav`, `pcm` (int16 LE), `f32` (raw float32), `mp3`, `aac`, `opus` (Ogg Opus). Same per-response marking floor as `/v1/audio/speech`: containers that cannot carry a C2PA manifest get the watermark forced on. |

The intermediate ASR transcript (if the backend produces one) is
returned in the `X-Transcript` response header (URL-encoded). Output
audio is watermarked by default, same as TTS (process-level opt-out via
`--no-watermark` / `CRISPASR_NO_WATERMARK`).

### Deferred

| Feature | Status |
|---|---|
| Streaming response for `/v1/audio/speech` | **Done** — `"stream": true` returns chunked `audio/pcm`. On a `CAP_STREAMING` backend the chunks are the backend's own codec chunks (time-to-first-audio ≈ one chunk); otherwise one chunk per sentence. Sub-sentence chunked-VAE for the remaining backends is still pending — see PLAN §70. |
| Streaming response for `/v1/audio/speech-to-speech` | Pending — the S2S route is buffered end-to-end (couples with chunked-VAE for the full latency win, PLAN §70). |
| `mp3` / `aac` / `opus` encoding | **Done** — encoded in-tree by glint, no build deps (`response_format=mp3\|aac\|opus`; `opus` = Ogg Opus). `flac` output still pending. |
| `POST /v1/voices` (multipart upload for runtime provisioning) | **Done** — gated on `consent_attestation` (see above). Disk quota and content-type validation still pending. |
| `DELETE /v1/voices/{name}` | **Done**. |
| Native-backend `speed` (duration knobs vs server-side resample) | Pending — backend-by-backend. |

## Source separation endpoint (§381)

`POST /v1/audio/separation` runs source separation on uploaded audio and
returns one WAV per stem. Requires a secondary model loaded at startup via
`--separate-model` (independent of the primary ASR/TTS model).

```bash
crispasr --server -m parakeet-q4_k.gguf \
  --separate-model mel-band-roformer-vocals-f16.gguf
```

```bash
# Separate all stems:
curl http://localhost:8080/v1/audio/separation \
  -F "file=@song.wav" -o stems.bin -D -
# Content-Type: multipart/mixed (one WAV per stem)

# Select specific stems:
curl http://localhost:8080/v1/audio/separation \
  -F "file=@song.wav" -F "stems=vocals" -o vocals.wav
```

| Field | Default | Description |
|---|---|---|
| `file` | (required) | Audio file upload (multipart). Decoded to stereo at the model's native rate (44100 Hz). |
| `stems` | `all` | Comma-separated stem subset to return (e.g. `vocals,drums`). |

Response is `multipart/mixed` with one `audio/wav` part per selected stem
(stereo 16-bit PCM). Each part carries
`Content-Disposition: attachment; filename="<stem>.wav"`. No AI-provenance
tag — the audio is the user's own.

Returns `503` when `--separate-model` is not configured, `400` for
missing/invalid audio or a `stems` filter matching nothing, `500` on
backend failure.

## Translation endpoint

`POST /v1/translate` is the text-to-text translation counterpart — the HTTP
analogue of the CLI `--text` mode. Available whenever the loaded backend has
`CAP_TRANSLATE` (e.g. `m2m100`).

```bash
crispasr --server -m m2m100-418m-q8_0.gguf --backend m2m100 &

curl http://localhost:8080/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world, how are you today?", "source_lang": "en", "target_lang": "de"}'
# {"text": "Hallo Welt, wie bist du heute?"}
```

| Field | Meaning |
|---|---|
| `input` | Text to translate (required; `text` also accepted) |
| `source_lang` | Source language (falls back to the server's `--tr-sl`/`-sl` default) |
| `target_lang` | Target language (required unless a server default is set) |
| `max_tokens` | Optional output-token cap |

Returns `400` if the loaded backend lacks `CAP_TRANSLATE` or `input` is missing.

## Chat endpoint

`POST /v1/chat/completions` is an OpenAI-compatible text-LLM chat endpoint,
backed by a **second** GGUF loaded at startup with `--chat-model` — independent
of the primary ASR/TTS model, so one server can host ASR + chat off two
different files. Without `--chat-model` the route returns `503` with
`code=chat_disabled`.

```bash
crispasr --server -m parakeet-q4_k.gguf \
  --chat-model qwen3-1.7b-q4_k.gguf --chat-ctx 4096 --chat-gpu-layers 99
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarise the transcript."}]}'
# {"id": "chatcmpl-…", "object": "chat.completion", "created": …,
#  "model": "…", "choices": [{"index": 0,
#  "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}]}
```

| Field | Default | Description |
|---|---|---|
| `messages` | (required) | OpenAI message array. `content` may be a string or an array of content parts — multimodal arrays are collapsed to their joined text. |
| `model` | (ignored) | Read but not validated; the server answers with whatever `--chat-model` loaded. |
| `temperature` | `0.8` | Sampling temperature. |
| `top_p` | `0.95` | Nucleus-sampling cutoff. |
| `top_k` | `40` | Top-k cutoff (CrispASR extension). |
| `max_tokens` | `256` | Generated-token cap. |
| `seed` | `0` | RNG seed. |
| `stop` | — | A string or an array of strings. |
| `stream` | `false` | `true` → `text/event-stream` with `chat.completion.chunk` deltas terminated by `data: [DONE]`. Note the framing is SSE but the delivery is not yet incremental: generation runs to completion first and the assembled event stream is sent as one response, so `stream: true` buys correct parsing, not lower time-to-first-token. |

Each request is treated as a **stateless** conversation: the session's KV cache
is reset before generation, and the whole reset-then-generate transaction takes
one lock, so overlapping requests queue rather than interleave on the shared
session.

Every response (buffered and SSE) carries the EU AI Act Art. 50(2) synthetic-text
markers `X-Crispasr-Ai-Generated: true` and `X-Crispasr-Ai-Disclosure: <text>`.
A client that drops the headers publishes unmarked text — marking what you do
with it stays the deployer's duty (see [`eu-ai-act.md`](eu-ai-act.md) §6.6).

Error codes: `chat_disabled` (503), `chat_init_failed`, `chat_reset_failed`,
`chat_generate_failed`, `chat_stream_failed` (500), `invalid_json`,
`missing_required_field`, `invalid_message` (400).

## Progress endpoint (#408)

`GET /progress` lets a client poll the progress of the transcription job the
server is currently running, so a UI can show a real 0–100 bar instead of a
heartbeat while a long synchronous `POST` is in flight:

```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8080/progress
# idle:            {"busy": false, "progress": -1}
# job running:     {"busy": true,  "progress": 42}
# post-processing: {"busy": true,  "progress": 100}   (diarization/punctuation tail)
```

`progress` advances per chunk of the server's chunk loop, reaches 100 when the
last chunk has decoded, and stays there while post-steps (diarization,
punctuation, truecasing) finish; it returns to `-1` when the job completes.
Requests queued behind a running job do not disturb its reading. With
`--server-workers > 1`, `busy` counts every in-flight job but concurrent jobs
share the single `progress` slot (last writer wins) — per-request progress
would need job ids and is out of scope for now. The endpoint requires an API
key when keys are configured (only `/health` is public).

## Request limits & error handling

- **Upload cap: 512 MB.** A larger `Content-Length` is rejected with `413`
  before the body is buffered (so an oversized multipart cannot OOM the
  process ahead of auth/routing).
- **Chunked uploads are refused.** cpp-httplib's chunked reader does not honour
  the payload cap, so a `POST`/`PUT` carrying `Transfer-Encoding` gets `411
  Length Required` — resend with a `Content-Length`.
- An unmatched route returns `{"error": "not found. Use POST /v1/audio/transcriptions"}`;
  an exception inside a handler returns a structured `500` naming the reason
  rather than being mislabelled as a 404.
- `GET /health` answers `503 {"status": "loading"}` until the model is
  resident, then `200 {"status": "ok", "backend": "…"}`. It is the only
  endpoint exempt from API-key auth.
- `POST /load` returns `409` while a `--server-workers` pool is active
  (pooled workers would keep serving the old model — restart instead).
- Requests asking for `vad` or `diarize` on a host whose CPU lacks the
  instruction set this build was compiled for return `400` rather than
  SIGILL-ing the server (#261).

## Real-time streaming (WebSocket)

Pass `--ws-port N` to expose a real-time streaming-ASR WebSocket alongside the
HTTP server (`-1` = off, the default; `0` = HTTP port + 1; `N` = port N). This
is the server analogue of the CLI `--stream` path.

```bash
crispasr --server -m ggml-base.en.bin --backend whisper --ws-port 0
# → WS ws://127.0.0.1:8081
```

Clients connect, send binary frames of 16 kHz mono **float32** PCM, and receive
JSON updates as audio accumulates:

```json
{"text": "And so my fellow Americans", "t0": 0.0, "t1": 30.0, "counter": 1}
{"text": "...ask what you can do for your country", "t0": 0.0, "t1": 30.0, "counter": 3, "final": true}
```

Whisper-only today. (Each connection opens its own streaming session.)

### vLLM Realtime API (WebSocket)

When `--ws-port` is enabled, the server also exposes a **vLLM Realtime API** compatible WebSocket endpoint on `ws_port + 1`. This endpoint accepts standard JSON-encoded `input_audio_buffer.append` events (base64 PCM16) and streams back `conversation.item.input_audio_transcription.delta` events incrementally.

```bash
crispasr --server -m qwen3-asr.gguf --backend qwen3 --ws-port 8081
# → WS ws://127.0.0.1:8081 (Raw PCM)
# → WS ws://127.0.0.1:8082/v1/realtime (vLLM Realtime API)
```

The endpoint has a backend-owned realtime session when the loaded backend
supports one. Nemotron preserves its per-layer attention/convolution caches and
RNN-T predictor state across `input_audio_buffer.append` messages, processes
only newly stable encoder frames, and emits genuine pre-commit
`conversation.item.input_audio_transcription.delta` events. A commit flushes the
final short chunk and resets all per-turn state. Other backends use a safe
commit-only fallback; `session.created.partial_transcription` tells the client
which contract is active.

`--vad --vad-model MODEL` enables server-side turn detection on this endpoint.
PCM is held outside ASR until speech is detected, a bounded onset buffer protects
the first phoneme, `input_audio_buffer.speech_started` / `.speech_stopped` events
expose the boundary, and trailing silence auto-commits the turn. Without server VAD,
`session.created` reports `"turn_detection":"client_commit"` and clients should
gate appends themselves, retain pre-roll/trailing padding, and commit after
their silence threshold.

A 30-second safety cap force-commits forgotten turns without dropping overflow:
the remainder starts the next bounded turn. Completed events include
`audio_received_duration_ms`, `audio_processed_duration_ms`,
`queue_backlog_duration_ms`, `model_queue_wait_ms`, `processing_ms`,
`end_to_end_processing_ms`, and `realtime_factor`. Inference is synchronously
backpressured, so the internal queue backlog is zero rather than an unbounded
worker queue; contention for the shared model is visible separately as model
queue wait.

The raw WebSocket on `ws_port` remains Whisper-only. Nemotron clients should use
`ws_port + 1` (`/v1/realtime`) with explicit commits.

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
  -F "response_format=verbose_json" \
  -F "max_tokens=256" \
  -F "frequency_penalty=0.4"
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

## Wyoming protocol (Home Assistant Assist)

Pass `--wyoming-port N` to start a Wyoming peer-to-peer JSONL/TCP server
alongside the HTTP API. One CrispASR server instance then replaces both
`wyoming-faster-whisper` (STT) and `wyoming-piper` (TTS) in a Home Assistant
Assist pipeline — no extra containers needed.

```bash
# Start server with Wyoming on port 10300 (HA default)
crispasr --server -m model.gguf --port 8080 --wyoming-port 10300
```

> `--wyoming-port` belongs to the multi-backend `crispasr --server` binary.
> The separate legacy `crispasr-server` executable (built from
> `examples/server/server.cpp`) is whisper-only, serves just `/`,
> `/inference`, `/load` and `/health`, and does not accept this flag.

### Wire format

Each message is a JSON header line followed by an optional binary payload:

```
{"type":"...","data":{...},"payload_length":N}\n
<N bytes of binary payload>
```

### Events handled

| Incoming event | What CrispASR does |
|---|---|
| `describe` | Replies with `info` — advertises ASR + TTS capabilities |
| `transcribe` + `audio-start` + `audio-chunk` + `audio-stop` | Buffers int16 PCM chunks, resamples to 16 kHz float32 via linear interpolation after `audio-stop`, runs `backend->transcribe()`, returns `transcript` |
| `synthesize` | Calls `backend->synthesize()` under `model_mutex`, converts float32 → int16, streams back as `audio-start` / `audio-chunk*` / `audio-stop` at the model's native sample rate |

HA handles resampling from the model's native rate (e.g. 24 kHz for vibevoice)
to its playback device — no server-side downsampling needed.

### HA integration

In Home Assistant `configuration.yaml`:

```yaml
wyoming:
  - uri: tcp://<host>:10300
```

The server advertises both STT and TTS under the same URI. HA will automatically
use CrispASR for both directions once the integration is added.

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
| `CRISPASR_SERVER_WORKERS` | `N>1` loads N independent ASR backend instances so **pure-ASR** requests (explicit `language`, no aligner, no punctuation/truecaser) run concurrently instead of serializing on the single model. Costs N× model memory. Only a throughput win where a single request under-utilises the box (spare cores, a GPU not saturated by one stream, smaller models); a *net loss* on a saturated memory-bandwidth-bound CPU model, where the instances contend. Requests using shared LID/aligner/post-processing stay serialized. `/load` is disabled while a pool is active (restart to change models). Default `1` = single instance. Equivalent to the `--server-workers N` CLI flag (this env var overrides the flag when both are set). Full guidance: **[docs/concurrency.md](concurrency.md)**. |

Only the variables listed in the compose service's `environment:` block reach
the container. `CRISPASR_SERVER_WORKERS` is **not** one of them, so putting it
in `.env` alone has no effect under Docker — either add it to the `environment:`
block, or pass the equivalent flag through the variable that *is* forwarded:
`CRISPASR_EXTRA_ARGS=--server-workers 4`.

The service is configured to avoid serving as root by default:
- `user: "${CRISPASR_UID:-1000}:${CRISPASR_GID:-1000}"`
- `security_opt: ["no-new-privileges:true"]`
