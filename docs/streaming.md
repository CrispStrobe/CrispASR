# Streaming & live transcription

CrispASR supports three streaming modes — pipe input, microphone
capture, and continuous live mode — and per-token confidence output.
All work with every supported backend.

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
| `final` | Trailing silence ≥ `--stream-final-on-silence-ms` (default `800`) after the last detected speech closed the open utterance. In the default `--stream-final-mode redecode` `text` is produced by re-running the backend on the buffered utterance PCM (covers `[t0..t1]`); in `prefix` mode `text` is a prefix accumulator stitched with the last partial. | `utterance_id`, `text`, `t0`, `t1` |
| `silence` | A streaming step produced no speech slices. Emitted regardless of whether an utterance is still open, so wrappers always see a timeline heartbeat. | `t` |

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

For native streaming-architecture backends (`voxtral4b`,
`moonshine-streaming`, `kyutai-stt`), the encoder runs incrementally —
the sliding window flags above still apply but the per-chunk cost is
lower than for batch backends.
