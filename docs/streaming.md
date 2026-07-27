# Streaming & live transcription

CrispASR supports three streaming modes — pipe input, microphone
capture, and continuous live mode — and per-token confidence output.
All work with every supported backend.

> **Streaming TTS output** (the reverse direction) is documented in its own
> section at the bottom — [Streaming synthesized audio](#streaming-synthesized-audio-out).

> Over HTTP, the server exposes the same streaming decoder as a WebSocket
> endpoint — start it with `--ws-port` and send binary float32 PCM frames,
> or connect to `ws-port + 1` for the JSON-based **vLLM Realtime API** endpoint.
> See [`server.md`](server.md#vllm-realtime-api-websocket).

## Pipe mode (`--stream`)

```bash
# Pipe audio from ffmpeg, sox, or any tool that outputs raw PCM:
ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | \
    crispasr --stream -m model.gguf
```

Sliding-window chunking, default 10 s window with 3 s step and 200 ms
overlap. Tune via `--stream-step`, `--stream-length`, `--stream-keep`.

Quality-control flags supported in streaming mode:

- `--vad`, `--vad-model`, `--vad-threshold`, `--vad-min-speech-duration-ms`, `--vad-min-silence-duration-ms`, `--vad-speech-pad-ms`
- `--stream-vad-merge-gap-ms` for JSON streaming VAD close-gap tuning
- `--punc-model` and `--no-punctuation`

Notes:

- With VAD enabled, each streaming window is segmented before ASR. Silent windows are skipped instead of being decoded.
- `--punc-model` applies after streamed chunk transcription, matching file-mode post-processing.
- `--alt` / `--alt-n` are file-mode features. They currently do not print token alternatives from `--stream`, `--mic`, or `--live`.
- File-oriented output flags such as `-osrt`, `-ovtt`, `-oj`, and `-of` do not apply to `--stream` / `--mic` / `--live`; streaming writes transcripts to stdout (or JSON-Lines events with `--stream-json`).

## Structured streaming output (`--stream-json`)

For wrappers (browser bridges, live-translation pipelines, captioning
UIs) that need to distinguish a still-evolving partial from a
finalized utterance, pass `--stream-json`. CrispASR then emits one
JSON object per line on stdout — never plain text — and FireRed VAD
diagnostics stay off stderr unless you opt in with
`--firered-vad-debug`.

```bash
ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream --stream-json -m model.gguf \
      --vad --vad-model firered-vad.gguf \
      --stream-final-on-silence-ms 800
```

Event types:

| `type` | When | Fields |
|---|---|---|
| `partial` | A streaming step produced new text for the open utterance. At most one `partial` per `utterance_id` per step — multiple VAD slices belonging to the same utterance within a step are concatenated. | `utterance_id`, `text`, `t0`, `t1` |
| `final` | Trailing silence ≥ `--stream-final-on-silence-ms` (default `800`) after the last detected speech closed the open utterance. In the default `--stream-final-mode redecode` `text` is produced by re-running the backend on the buffered utterance PCM (covers `[t0..t1]`); in `prefix` mode `text` is a prefix accumulator stitched with the last partial. | `utterance_id`, `text`, `t0`, `t1`, *(optional)* `speaker` |
| `silence` | A streaming step produced no speech slices. Emitted regardless of whether an utterance is still open, so wrappers always see a timeline heartbeat. | `t` |

The optional `speaker` field on `final` events appears only with a backend that
populates the structured speaker label (`moss-diarize`; `vibevoice` from
v0.8.24; `granite` in speaker-aware `--diarize` mode) when the finalized
utterance is single-speaker; its ordinals are utterance-local. See
[Speaker diarization while streaming](#speaker-diarization-while-streaming).

Stream-contract guarantees:

- Once an `utterance_id` finalizes, its audio is bookmarked and never re-opens a later `utterance_id`. Earlier text will not reappear in later utterances' partials.
- Finalization fires as soon as `now - last_speech_end_sample ≥ --stream-final-on-silence-ms`, independent of the rolling-window length. A 260 ms silence threshold with `--stream-length 18000` finalizes ~260 ms after the speaker stops, not ~18 s later.
- `final.t1 = last_speech_end_sample / 16 kHz` and the redecode buffer is trimmed to `[utterance_start_sample, last_speech_end_sample]`, so `final.text` describes exactly the `[t0..t1]` interval (trailing silence past `t1` is not part of the decoded region).
- With `--stream-json --vad`, VAD post-merge only joins very close detector jitter gaps. `--stream-vad-merge-gap-ms` defaults to `250` and is clamped below `--stream-final-on-silence-ms`, so VAD merging cannot hide a gap that should finalize an utterance. The offline VAD short-slice merge policy is not used on this JSON streaming path.

Sample stream:

```json
{"type":"partial","utterance_id":1,"text":"is that they can be tuned and adjusted","t0":10.20,"t1":13.20}
{"type":"partial","utterance_id":1,"text":"is that they can be tuned and adjusted for a specific","t0":10.20,"t1":16.20}
{"type":"final","utterance_id":1,"text":"is that they can be tuned and adjusted for a specific hardware target.","t0":10.20,"t1":17.80}
{"type":"silence","t":18.60}
```

Live-translation wrappers can show `partial` events in a draft pane
and only ship `final` events to the translation API. Set
`--stream-final-on-silence-ms 0` to disable auto-finalization (useful
when the wrapper finalizes on its own signal — e.g., a UI button —
instead of trailing silence).

`t0` / `t1` are wall-clock seconds since stream start, derived from
the cumulative sample count, so they map to the same timeline as the
input PCM. `t0` marks where the **utterance** started (first VAD
speech frame, or first non-empty model decode in no-VAD mode); `t1`
marks the last detected speech frame for `final` events, or the
current decoder-step time for `partial`.

### Finalization timing

Finalization fires when there has been **`--stream-final-on-silence-ms`
worth of trailing silence after the last detected speech**, not when
the entire rolling window has decoded to empty. With VAD enabled the
silence detector uses each VAD slice's end time directly; without VAD
the fallback is "the model decoded nothing for that long."

The practical effect: a speaker who pauses mid-paragraph for ~800 ms
gets a `final` per natural pause, instead of one giant final at the
end of the recording. Set `--stream-final-on-silence-ms` higher
(e.g. `2000`) if you want fewer finalizations / longer-form chunks.

### How `final.text` is built — `--stream-final-mode`

Two modes; `redecode` is the default.

```bash
# Best quality — re-runs the backend on the buffered utterance PCM at
# finalize time. final.text is guaranteed to cover [t0..t1] regardless
# of how the rolling window evicted audio.
crispasr --stream --stream-json --stream-final-mode redecode ...

# Cheaper — no extra encoder pass. final.text is built from a
# longest-common-prefix accumulator across consecutive partials, with
# the last partial appended. Subject to text duplication when the
# rolling window evicts mid-utterance audio.
crispasr --stream --stream-json --stream-final-mode prefix ...
```

In `redecode` mode CrispASR buffers the speech-region PCM in memory
(capped at `--stream-utterance-max-sec`, default `60` s — about 4 MB
at 16 kHz mono float). When the cap is hit the current utterance
auto-finalizes and the next speech opens a new utterance with a
fresh `utterance_id`. For most live-captioning / translation use
cases the redecode path is what you want — its output covers the
whole utterance the way `t0`/`t1` advertise.

`prefix` mode preserves round-1 cost (no extra `transcribe()` call)
at the price of imperfect text reconstruction on long utterances.
Useful when the encoder is large and the per-chunk budget is tight.

**Short-utterance fallback.** Backends that use convolutional
encoders (moonshine, parakeet, voxtral, …) abort with `OW > 0` from
`ggml_im2col` when handed audio shorter than the encoder's first conv
kernel — about 2 s at 16 kHz. When `redecode` would hit that limit
(the VAD-trimmed `[t0..t1]` is under 2 s) CrispASR skips the extra
backend pass and falls back to the **`prefix`-mode stitcher** for
that one finalize. `final.text` is then the LCP-accumulated prefix
plus the last partial — the same content the wrapper has already
seen in `partial` events, never an empty string blanking a
previously-emitted partial. The fallback is internal; no flag, no
event change.

### Streaming punctuation (`--stream-punc`)

When `--stream-json --vad` is combined with `--punc-model`, FireRedPunc
can sit on either the partial path, the final path, both, or neither.
PR [#112](https://github.com/CrispStrobe/CrispASR/pull/112) introduced
the explicit knob; before that, partials and finals both ran through
FireRedPunc (equivalent to today's `--stream-punc partial`).

| Mode | Partials | Finals | Notes |
|---|---|---|---|
| `off` | ❌ | ❌ | FireRedPunc is bypassed entirely on the streaming path. **Both partials and finals come out unpunctuated** — `off` is the most permissive setting, not just "off for partials". |
| `final` *(default)* | ❌ | ✅ | Recommended realtime mode. Live partials stay cheap; finals get punctuation once per utterance via either `--stream-final-mode redecode` (segments are punc'd before stitching) or the stitched-fallback path (the final string is punc'd in place). |
| `partial` | ✅ | ✅ | Pre-#112 behaviour. Keep if every partial event needs punctuation downstream — the cost is one FireRedPunc forward per `--stream-step`. |

**Default change.** Before PR #112 the *de-facto* default was equivalent
to `partial` (no flag existed; every partial got punctuation). After
#112 the default is `final`. Wrappers that relied on punctuated
partials should pass `--stream-punc partial` explicitly to restore
the old behaviour; everyone else gets the better latency profile for
free.

Smoke results on 30 s of Cohere JA streaming
(`--stream-step 500 --stream-final-mode redecode`):

| mode | wall_sec | partials | finals |
|---:|---:|---:|---:|
| `off` | 35.5 | 36 | 11 |
| `final` | 44.3 | 36 | 11 |
| `partial` | 45.9 | 36 | 11 |

Event counts are identical across the three modes — the policy
controls *processing*, not emission. The ~10 s gap between `off`
and `final`/`partial` is FireRedPunc on the finals; the (smaller)
gap between `final` and `partial` is FireRedPunc on the 36 partials.
On longer audio or shorter `--stream-step` (more partials per second)
the `partial`-vs-`final` gap widens proportionally.

`--stream-punc` is a no-op without `--punc-model`. Combine with the
truecasers (`--truecase-model`, `--truecase-crf-model`,
`--truecase-lstm-model`) and PCS (`--pcs-model`) post-steps as usual
— those run on every mode (only the FireRedPunc step is gated).

## Microphone (`--mic`)

```bash
# Live microphone transcription (auto-detects arecord/sox/ffmpeg):
crispasr --mic -m model.gguf
```

CrispASR auto-detects whichever audio capture tool is on `$PATH`.

## Continuous live mode (`--live`)

```bash
# Continuous live mode (prints each chunk as a new line, never stops):
crispasr --live -m model.gguf

# With progress monitor symbols (▶ processing, ✓ got text, · silence):
crispasr --live --monitor -m model.gguf
```

`--live` runs indefinitely, emitting one transcript line per processed
chunk. `--monitor` adds visual feedback so you can tell processing
state at a glance.

## Per-token confidence

```bash
crispasr -m model.gguf -f audio.wav --alt
```

`--alt` prints alternative candidate tokens with probabilities — useful
for filtering low-confidence file transcriptions or for downstream
rescoring. Streaming modes do not currently emit this alternatives
block.

## Tuning the sliding window

| Flag | Default | Effect |
|---|---|---|
| `--stream-step N` | `3000` ms | Step between consecutive windows. Smaller = more frequent partial transcripts. |
| `--stream-length N` | `10000` ms | Rolling context window cap. The decode buffer accumulates audio up to this many ms, then drops the oldest samples from the front. Larger = better accuracy on long-form content but higher per-step cost. |
| `--stream-keep N` | `200` ms | Legacy — kept for compatibility, currently a no-op. The rolling buffer above subsumes it (see issue #84). |
| `--stream-partial-decode-ms N` | `0` ms | JSON+VAD only. Minimum interval between live partial ASR decodes. `0` preserves the previous behavior and decodes every `--stream-step`; larger values keep VAD/final timing at `--stream-step` while reducing partial ASR cadence. |

`--stream-vad-merge-gap-ms` defaults to `250` ms and applies only to
`--stream-json --vad`. It merges adjacent VAD slices only across gaps smaller
than that value. When `--stream-final-on-silence-ms` is enabled, the effective
merge gap is clamped below the finalization threshold. Set it to `0` to disable
this close-gap merge.

`--stream-partial-decode-ms` is useful when low-latency VAD/final timing is
desired but partial ASR decode is too expensive to run every step. For example,
`--stream-step 500 --stream-partial-decode-ms 750` keeps VAD and silence
finalization checks at 500 ms while allowing live partial ASR text at most every
750 ms. Steps that skip partial decode still keep VAD slice timing for the JSON
utterance state machine. When trailing silence has crossed the finalization
threshold, one step may bypass the partial-decode throttle before finalization
so short-utterance fallback finals can use a fresh normal partial.

The default value `0` means **"follow `--stream-step`"** — the throttle is
always conceptually present in the JSON+VAD path, but at `0` it locks to the
step cadence so every step decodes (matching the pre-#113 behaviour). It is
NOT "throttling disabled"; rather, the interval is set to one step's worth of
audio. Set `--stream-partial-decode-ms` to a value **larger than `--stream-step`**
to actually space out partial decodes. Setting it smaller than the step has no
effect — the gate only fires on stream-step boundaries, so the effective
minimum is one step regardless of what you pass.

The first step of a stream is always allowed (so the first partial fires
immediately), and `--stream-partial-decode-ms` is a no-op outside the
`--stream-json --vad` combination — non-JSON streaming always decodes every
step.

> **Note (issue #84).** Before May 2026, `--stream-length` was a
> *ceiling* on `keep + step` rather than a true rolling cap, so
> `--stream-length 18000 --stream-keep 200 --stream-step 3000`
> actually decoded ~3.4 s of audio per step instead of 18 s. The
> streaming loop was rewritten to accumulate up to `length_samples`
> and drop the oldest frame on overflow, which matches the documented
> behaviour. `--stream-keep` is now informational only.

### Per-token streaming backends

All autoregressive ASR backends implement `transcribe_streaming` and emit
tokens to the `--stream` callback as they are generated, without waiting for
the full decode to finish:

| Backend | Token decode type | Notes |
|---|---|---|
| `granite-speech` | LLM greedy (Granite LLM) | Standard `run_with_probs_cb` |
| `voxtral4b` | LLM greedy (Mistral LLM) | Per-step encoder-frame injection via `pre_hook` |
| `glm-asr` | LLM greedy (GLM BPE) | Adapter-side greedy loop using exported step APIs |
| `moss-audio` | LLM greedy (GPT-2 BPE) | Via `moss_audio_process_cb` |
| `moss-transcribe` | LLM greedy (GPT-2 BPE) | Via `moss_transcribe_transcribe_cb` |
| `gemma4-e2b` | LLM greedy (SentencePiece) | Via `gemma4_e2b_transcribe_cb`; control tokens filtered |
| `moonshine-streaming` | LLM greedy (SentencePiece) | Via `moonshine_streaming_transcribe_cb` |
| `kyutai-stt` | LLM greedy (SentencePiece) | Via `kyutai_stt_transcribe_cb`; padding tokens filtered in C lib |
| `mimo-asr` | LLM greedy (GPT-2 BPE) | Via `mimo_asr_transcribe_cb` |
| `nemotron` | RNN-T (per non-blank frame) | Via `nemotron_transcribe_cb`; fires per emitted frame |
| `qwen3-asr` | LLM greedy (Qwen3) | Native |
| `voxtral` | LLM greedy (Mistral LLM) | Native |

For these backends, `--stream` output grows one token at a time. For batch
backends (whisper, parakeet, canary, funasr, etc.), each full chunk produces
one update.

For native streaming-architecture backends (`voxtral4b`,
`moonshine-streaming`, `kyutai-stt`, `nemotron`), the encoder also runs
incrementally — the sliding window cost is lower than for batch backends.

### Nemotron streaming (cache-aware FastConformer)

`nemotron` supports true cache-aware streaming via the NeMo
`cache_last_channel` + `cache_last_time` architecture. Enable with:

```bash
CRISPASR_NEMOTRON_STREAMING=1 crispasr --backend nemotron -m model.gguf -f audio.wav
```

Four context presets trade latency for accuracy:

| Preset | Right-context | Chunk size | Approx latency | Published WER |
|--------|--------------|------------|----------------|---------------|
| 0      | 3 frames     | 4 frames   | ~160 ms        | 7.67 %        |
| 1      | 0 frames     | 1 frame    | ~80 ms         | 8.43 %        |
| 2      | 6 frames     | 7 frames   | ~560 ms        | 7.07 %        |
| 3      | 13 frames    | 14 frames  | ~1120 ms       | 6.93 %        |

Set via `CRISPASR_NEMOTRON_CONTEXT_PRESET=N` (default: 0).

## Speaker diarization while streaming

Short answer: streaming carries whatever speaker information a backend produces
per window/utterance, but **not** the cross-recording clustering pipeline or
named-voiceprint identification — those two are recorded-file (offline) features
by design. See [`diarization-speakers.md`](diarization-speakers.md) for the full
diarization model.

Both backends [issue #300](https://github.com/CrispStrobe/CrispASR/issues/300)
asked about produce speaker information while streaming, but by **two different
mechanisms** — worth understanding because they behave differently downstream:

| Backend | How speaker info is produced | In streaming you get |
|---|---|---|
| **`moss-diarize`** (MOSS-Transcribe-Diarize-0.9B, `cstr/MOSS-Transcribe-Diarize-GGUF`) | a **structured** per-segment speaker label (`seg.speaker`), parsed from the model's `[Sxx]` tags | inline `(Speaker N)` in plain `--stream`; a `"speaker"` field on `--stream-json` `final` events |
| **`vibevoice`** (VibeVoice-ASR, `cstr/vibevoice-asr-GGUF`) | a **structured** per-segment speaker label, parsed from the JSON array the model answers with (its prompt asks for "Start time, End time, Speaker ID, Content") — from v0.8.24; before that the blob was passed through as raw text | inline `(Speaker N)` in plain `--stream`; a `"speaker"` field on `--stream-json` `final` events. Set `CRISPASR_VIBEVOICE_RAW_TRANSCRIPT=1` for the old raw-blob behaviour |

Issue #300's change surfaces the **structured** `seg.speaker` field in streaming
— so it applies to `moss-diarize`, to `granite` in speaker-aware `--diarize`
mode, and to `vibevoice`.

> **`vibevoice` needs v0.8.24.** VibeVoice-ASR answers with a JSON array of
> utterances (`Start` / `End` / `Speaker` / `Content`), but until v0.8.24 the
> adapter handed that blob back as ONE segment's text — so the labels reached
> you as literal JSON, `seg.speaker` was never populated, and the `"speaker"`
> field below could not fire for this backend at all. v0.8.24 reads the answer:
> one segment per utterance, native per-utterance timings, and the speaker in
> the structured field like any other native diarizer.
> `CRISPASR_VIBEVOICE_RAW_TRANSCRIPT=1` restores the raw blob for callers that
> were parsing it themselves.

### What works in streaming

`moss-diarize` and `vibevoice` populate the structured field, so under
`--stream` / `--mic` / `--live` each decoded window carries its own speaker
labels (substitute `--backend vibevoice` in either recipe below):

```bash
# Plain streaming — labels are prefixed inline, exactly like file-mode text output:
ffmpeg -i meeting.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream -m auto --backend moss-diarize
# (Speaker 1) welcome everyone
# (Speaker 2) thanks, glad to be here
```

```bash
# Structured streaming — a `final` event gains a "speaker" field when the
# finalized utterance is single-speaker (text stays clean, no inline labels):
ffmpeg -i meeting.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream --stream-json -m auto --backend moss-diarize \
      --vad --vad-model auto --stream-final-on-silence-ms 800
# {"type":"partial","utterance_id":1,"text":"welcome everyone","t0":0.30,"t1":1.80}
# {"type":"final","utterance_id":1,"text":"welcome everyone.","speaker":"(Speaker 1)","t0":0.30,"t1":2.10}
```

The `speaker` field is **only present** on `final` events, and only when every
segment of that utterance shares one label (the common VAD-bounded case). A
`final` whose redecode spanned a mid-utterance speaker turn, and every
`partial`, omit the field — parse a missing `speaker` as "unlabeled", not as a
change of speaker. In plain (non-JSON) `--stream`, labels are prefixed inline
into the text instead, matching the file-mode `text`/`srt`/`vtt` convention.

> ⚠ **Speaker IDs are window/utterance-local, not globally stable.** No
> cross-window clustering runs in streaming mode, so `Speaker 1` in one step is
> not guaranteed to be the same physical voice as `Speaker 1` in a later step —
> the same caveat the diarized file-mode JSON documents for per-chunk labels.
> If you need recording-stable labels, transcribe the recorded file offline
> (below), where global clustering runs across the whole audio.

### What does NOT run in streaming (offline only)

- **`--diarize-speakers` / `--diarize` clustering** (pyannote segmenter +
  TitaNet embedder → globally-stable `(speaker N)` labels). Global clustering
  needs the whole recording to assign consistent labels, so it is applied as a
  file-mode post-processing stage and is a no-op on the streaming path.
- **`--speaker-db` / `--enroll-speaker` named identification.** These
  **hard-refuse** in streaming mode (`crispasr: error: --speaker-db/--enroll-speaker
  are not available in streaming mode`) — real-time biometric identification is
  deliberately unsupported (EU AI Act Art. 5(1)(h); see
  [`diarization-speakers.md`](diarization-speakers.md#2-named-voiceprint-profiles---speaker-db--deliberate-opt-in)).

### Recommended: near-real-time now, recording-stable labels offline

For live captioning where per-utterance labels are enough, stream a native
diarizer as above. For a transcript with **recording-stable** speaker labels
(the same person keeps the same number end to end), run the diarizer — or any
backend with `--diarize-speakers` — over the recorded file once capture ends:

```bash
crispasr -m auto --backend moss-diarize -f meeting.wav -ojf     # native labels + native timestamps
crispasr -m auto --backend cohere      -f meeting.wav --diarize-speakers -ojf   # any backend + clustering
```

## Streaming synthesized audio (out)

TTS output can be streamed progressively — audio starts flowing before the
whole clip is synthesized — from the CLI, the HTTP server, and the C ABI. All
three split the input into sentence chunks and emit each chunk as soon as it is
ready, so time-to-first-audio is one sentence, not the whole utterance.

### CLI (`--tts-stream`)

Emits raw **signed-16-bit little-endian mono PCM** to stdout at the backend's
sample rate; all logs stay on stderr, so stdout is a clean stream to pipe into a
player:

```bash
crispasr --backend irodori-tts -m model.gguf --codec-model dacvae-ja-32dim-f16.gguf \
    --tts "こんにちは。今日はいい天気ですね。" --tts-stream \
  | ffplay -f s16le -ar 48000 -nodisp -
```

The spoken AI-disclosure (voice-cloned output) is emitted first, each chunk is
watermarked before emit (unless disabled process-wide via `--no-watermark` /
`CRISPASR_NO_WATERMARK`), and a 200 ms gap separates chunks. Works with every
TTS backend.

### Server (`stream: true`)

`POST /v1/audio/speech` with `"stream": true` and a PCM `response_format`
(`pcm`, `wav`, or `f32`) streams each sentence as chunked transfer encoding:

```bash
curl -N http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts","input":"…","stream":true,"response_format":"pcm"}' > out.pcm
```

### C ABI (`crispasr_session_synthesize_streaming`)

For embedders/bindings — fires a callback per sentence chunk with that chunk's
watermarked PCM (backend-native sample rate, owned by the call). As with the
other bindings, this path watermarks unconditionally; the `--no-watermark` /
`CRISPASR_NO_WATERMARK` opt-out does not apply here (see
[`bindings.md`](bindings.md)):

```c
void on_chunk(const float* pcm, int n, int is_final, void* user) { /* play/queue */ }
crispasr_session_synthesize_streaming(session, "…", on_chunk, user);
```

Note: for diffusion backends (e.g. irodori) the per-*sentence* granularity above
is the real latency win — a diffusion utterance is generated in full before it is
decoded, so there is no sub-sentence audio to emit early.

### Source separation (htdemucs)

HTDemucs processes audio in overlapping chunks with cross-fading, but this is
internal chunking for memory management, not real-time streaming. The `streaming`
capability flag indicates support for chunked processing of long audio files.
