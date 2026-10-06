# Streaming & live transcription

CrispASR supports three streaming modes — pipe input, microphone
capture, and continuous live mode — and per-token confidence output.
All work with every supported backend.
A fourth, [live transcription + translation](#live-transcription--translation---live-translate),
adds a text translator behind the recogniser.

`--backend vibevoice-streaming -m auto` uses the model's native streaming
protocol: fixed 2.93-second chunks, 0.53-second lookahead, and one persistent
decoder cache for the whole session. `--stream-step`, `--stream-length`, and
`--stream-keep` therefore do not change its receptive field. A final flush
pads one last partial chunk and does not decode retained lookahead twice.

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

Sliding-window chunking, default 10 s rolling window with a 3 s step. Tune
via `--stream-step` and `--stream-length`; `--stream-keep` is still parsed
but is a no-op (see [the note on issue #84](#tuning-the-sliding-window)).

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

`--stream-punc` is a no-op without `--punc-model`, and it gates **only** the
FireRedPunc step. The truecasers (`--truecase-model auto|crf|lstm|<path>`)
and PCS run on every mode. Note that both variants are selected by the value
passed to a single flag — there are no separate `--truecase-crf-model`,
`--truecase-lstm-model` or `--pcs-model` flags, and PCS is
`--punc-model pcs`, which loads PCS *instead of* FireRedPunc (so on a PCS
server `--stream-punc` has nothing to gate).

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

## Live transcription + translation (`--live-translate`)

Listen, transcribe, and translate sentence by sentence — all local, two
models: a streaming recogniser and a text translator behind it.

```bash
# German speech in, German + English text out:
crispasr --live-translate -l de --tr-tl en \
    -m auto --backend parakeet --translate-model hy-mt2

# Same pipeline on a pipe instead of the microphone (a live feed, or a file
# replayed in real time). --stream-realtime tells it the input is live:
ffmpeg -re -i talk.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream --stream-realtime -l de --tr-tl en \
      -m auto --backend parakeet --translate-model hy-mt2
```

On a terminal, finished sentence pairs scroll up and the sentence still being
spoken is redrawn dimmed below them, with a draft translation:

```
de  Wir haben heute drei Punkte auf der Tagesordnung.
en  Today we have three points on the agenda.
de  Zuerst sprechen wir über die Ergebnisse des                    ← still open
en  First we talk about the results of                             ← draft
```

When stdout is not a terminal you get one `[de] …` / `[en] …` pair per
sentence. With `--stream-json` you get events instead (below).

`-l` is the spoken language and is required. `--tr-tl` is the language to
translate into (default: `en`, or `de` when the speech is English).
`--translate-model` takes `auto` (m2m100-418M, 100 languages, ~500 MB), a
registry name — `opus-mt-de-en` / `opus-mt-en-de` (Opus-MT, 84 MB, the
fastest), `hy-mt2` (Hy-MT2-1.8B, ~1.1 GB, the best speed-for-quality LLM) or
`index-translate` (Index-Translate-2B, ~1.3 GB) — or a translator GGUF. The
kind of translator is detected from the file
(`--translate-backend m2m100|marian|madlad|llm` overrides); see the table below.
Passing `--translate-model` on a plain `--stream` / `--mic` run enables
translation there too; `--live-translate` is the preset that also implies
`--mic`, `--vad`, and `--stream-realtime`.

### How it stays close to the speaker

Translating only when an utterance ends (a `final`) waits for a pause — on a
lecture that is tens of seconds. Translating every partial re-translates text
that is still changing. This mode commits **sentences**:

- A sentence is committed as soon as the recogniser has moved past it and its
  terminator has been written before — by the previous partial or the one
  before that (a recogniser undecided between "vorstellen. Er kommt" and
  "vorstellen, er kommt" flips between them on alternate partials). That second condition
  matters: recognisers put a period at the cut end of every partial
  (`Ich gehe.` → `Ich gehe nach Hause.`), and it must not split the sentence.
- A committed sentence is **final**. It is translated once and never revised —
  not by later partials and not when the utterance closes.
- The open remainder is re-translated as a dimmed draft
  (`--no-translate-drafts` turns that off). If it already contains a
  finished sentence that is only waiting to be confirmed, exactly that
  sentence is translated ahead of time, so its translation is ready the
  moment it commits (10 of 11 sentences on the test clip).
- Translation runs on its own thread, so a slow translator delays the
  translation line and never the recogniser.
- Each step decodes only the audio that is not yet committed. With a
  recogniser that has word timestamps (parakeet) the boundary is the end
  time of the last committed word and decoding starts 0.3 s before it;
  without them it is estimated from when each word first showed up, and
  decoding starts ~1.5 s earlier so the end of the committed text is in view
  to line up against. Either way a step no longer re-decodes the whole
  utterance (median decoded per step on the test clip: 5.5 s estimated,
  3.5 s with timestamps).
- A pause half as long as `--stream-final-on-silence-ms` commits the open text
  if it ends a sentence, instead of waiting for the next word or the final.
- If the open sentence grows past ~9 s of audio, its stable part is committed
  at a clause boundary so it stops being re-decoded. That piece is settled
  *text*, not a translation unit: it is held and translated together with
  the rest of its sentence (translated cold, "…auf den starken Export nach |
  Frankreich und Italien zurückzuführen." came out as "…due to strong
  exports." / "Caused by France and Italy.").
- With `--stream-realtime`, a step that ran long reads the whole backlog at
  once rather than working through it one `--stream-step` at a time. (The
  Windows side — `PeekNamedPipe`, and virtual-terminal mode for the live
  view — is written but has not been run on Windows.)

### Incremental sessions (`--stream-session`)

`--stream-session` hands the recogniser only each step's *new* audio through
its own stateful session (`nemotron`, `qwen3`, `vibevoice-streaming`) instead
of re-decoding the open speech. A turn ends when the speaker has been silent
for `--stream-final-on-silence-ms`, judged from the audio (VAD over the last
3 s); without a VAD model, when no text has arrived for 2.5 s.

For nemotron use the CPU (`-ng`): its chunks are 320 ms of audio, far too
small for a GPU to pay off. Since 2026-10-06 a session runs one graph per
chunk with its state kept in backend memory and caches the attention's K/V
projections and position table instead of recomputing them for all 56 cached
frames in every layer:

| nemotron q4_k, 50 s German clip, load ~8 | Compute | Per 320 ms chunk |
|---|---|---|
| CPU, before (one graph per layer, host caches) | 52.6 s | — |
| CPU, now | 24.5 s | 67 ms |
| GPU (Metal), now | not faster than before | 148 ms |

So it is real time on CPU with cores to spare it, and it follows the speaker
closely while it has them (translations 0.44–0.76 s after the deciding audio
in the first 22 s of a run). It is also CPU-bound: when unrelated jobs took
the cores mid-run it fell 17–25 s behind, where parakeet on the GPU held
~0.8 s at a load of 49. Transcript quality is below parakeet's ("bisher
hinfragen" for "bis hierhin Fragen", sentences run together). Recommendation
unchanged: parakeet for live use on a shared machine; nemotron on CPU when
the machine is yours and the lowest algorithmic latency matters.

### Which models

| Recogniser | Notes |
|---|---|
| `parakeet` (parakeet-tdt-0.6b-v3, 25 European languages) | **Use this.** Accurate, punctuated, stable partials. Not a streaming model — the open sentence is re-decoded each step — so on a laptop expect a second or two behind the speaker. |
| `moonshine-de` (61 M, German only, CC-BY-NC-SA) | Several times cheaper per step, so it keeps up on a busy machine. Partials are less stable (punctuation flips, so commits come later), and it **stops at the first longer pause in a clip** and drops the rest. |
| `nemotron` (39 languages, cache-aware) | A true streaming model whose cost follows the new audio only. Real time on CPU with `--stream-session -ng`; CPU-bound and less accurate than parakeet — see [Incremental sessions](#incremental-sessions---stream-session). |
| `canary` | Can translate speech directly (`-tl en`), but one decode gives the transcript *or* the translation, not both. |

| Translator | Per sentence (de→en) | Notes |
|---|---|---|
| **`opus-mt-de-en` / `opus-mt-en-de`: Opus-MT** (`marian` backend; ~75M parameters, q8_0 84 MB, CC-BY-4.0; [`cstr/opus-mt-de-en-GGUF`](https://huggingface.co/cstr/opus-mt-de-en-GGUF), [`cstr/opus-mt-en-de-GGUF`](https://huggingface.co/cstr/opus-mt-en-de-GGUF)) | **23–118 ms, median ~45** (load 7–10; m2m100 in the same interleaved runs: 125–922 ms, median 315–650) | **Fastest by a wide margin, and better text than m2m100** ("new colleague", "furniture packers"). One model per language pair: `--translate-backend marian` picks it from `-l` / `--tr-tl`; other pairs need `models/convert-marian-to-gguf.py`. At f16 the output equals the reference implementation exactly, greedy and with beam 4 (14/14 de→en, 8/8 en→de); the q8_0 file that is downloaded by default matches on 12/14 and 8/8 (the rest differ in wording). |
| `m2m100` (418M, the `auto` default) | 92–392 ms, median 210 | Fast, mediocre: dropped "Danach", wrote "colleagues" for one colleague. Greedy only (`--translate-beam 1`, the default here) — beam search is cached since 2026-10 but still costs several greedy decodes. |
| **`hy-mt2`: Hy-MT2-1.8B** (`tencent/Hy-MT2-1.8B-GGUF`, Q4_K_M 1.1 GB, Apache-2.0, 33 languages) | 370–910 ms, median 570 | **Best trade-off measured.** Clearly better translations, and the translation is shown as it is generated. `--translate-model hy-mt2`, or pass any GGUF: a file that is none of the built-in translators is run as a translation chat LLM. |
| **`index-translate`: Index-Translate-2B** (`IndexTeam/Index-Translate-2B-GGUF`, Q4_K_M 1.3 GB, Apache-2.0, 150 languages) | 476–1100 ms, median 692 | Best translations of the lot ("a warm welcome to today's meeting", "as early as 7 a.m."), a little slower than Hy-MT2. Needed a loader fix: its GGUF appends a multi-token-prediction block the vendored Qwen3.5 loader took for a recurrent layer. |
| `madlad` (MADLAD-400 3B, q4_k 2 GB) | 0.75–2.8 s (load ~6–8) | Good translations, too slow to keep up sentence by sentence; also slows the recogniser (same GPU). |
| `m2m100` with `wmt21-dense-24-wide-x-en` (4.7B, q4_k 2.7 GB) | 1–6.5 s, first two 13–15 s (load ~6–8) | Very good translations, not live: the display ran 20–35 s behind. |

The first three rows are from one session on 2026-10-06 at load average 4–5
(the quietest this machine got), same 50 s German clip, parakeet-v3
recogniser; every run kept real time (94–97 steps of 500 ms) and produced all
11 sentences. Time from the audio that decided a sentence to its translation
on screen: m2m100 0.32–1.7 s (median 0.95), Hy-MT2 0.62–1.7 s (median 1.1),
Index-Translate 0.73–1.7 s (median 1.1).

A translation LLM only works with the instruction it was trained on.
`--translate-prompt hy-mt2` and `--translate-prompt index-translate` are built
in (the second is chosen automatically for a file named like the model,
otherwise the first); anything else is taken as a template —
`--translate-prompt 'Translate from {src} to {tgt}:\n{text}'`. A reasoning
model's `<think>` block is removed from the output.
On Apple Silicon Opus-MT runs on the CPU and m2m100 on the GPU by default; the LLM
translators need the GPU (Hy-MT2 on CPU: ~8.5 s per sentence against ~0.6 s,
`CRISPASR_TRANSLATE_CPU=1`, load 12–27). `CRISPASR_M2M100_GPU=1` / `=0`
forces either device for m2m100 and Opus-MT; measured in interleaved pairs at load 10–50:

| | CPU | GPU |
|---|---|---|
| Opus-MT q8_0, translator alone (warm, 4 pairs) | median 81–128 ms | 60–76 ms |
| Opus-MT q8_0, in the live pipeline (3 pairs) | **median 38–69 ms** | 98–118 ms |
| m2m100 q8_0, translator alone (4 pairs) | median 307–476 ms | 203–288 ms |
| m2m100 q8_0, in the live pipeline (3 pairs) | median 195–251 ms, worst sentence 0.3–2.2 s | median 174–274 ms, worst sentence 0.5–0.7 s |

Tokens were identical on both devices in every isolated run. Alone, the GPU
wins because it is immune to the CPU contention on this machine. In the live
pipeline the recogniser already owns the GPU, and each of a sentence's ~20
single-token decoder steps queues behind it — for Opus-MT, whose step is
~4 ms of CPU work, that queueing costs more than the step. So Opus-MT stays
on the CPU. m2m100 was a draw on the median at that load, so it was
repeated at load 4–7: alone the GPU is ~20% faster (median 134–140 ms
against 158–186, 3 of 3 pairs), and in the live pipeline ~9% (summed
9.67 s against 10.59 s over 4 pairs, 3 of 4 in its favour), with identical
tokens for greedy and beam 5. m2m100 therefore defaults to the GPU;
`CRISPASR_M2M100_GPU=0` puts it back on the CPU.

Where an Opus-MT decoder step goes (`CRISPASR_M2M100_BENCH=1`, CPU): graph
build 0.11 ms, allocation 0.15 ms, compute 3.5–4 ms, read-back 0.04 ms.
The graph is rebuilt per token, but that is ~7% of the step; more than half
of the compute is the output projection over the 58k-word vocabulary.

**Quantisation of Opus-MT.** q8_0 differs from f16 on 2 of 14 German
sentences and q4_k on 6 of 14, which looks alarming next to the recognisers
and is not: the differences are "departs" / "leaves", "reduce taxes" /
"lower taxes", "on 3 October in Berlin" / "in Berlin on 3 October".
Translation has many near-tied continuations, so a small logit change picks
another valid sentence where a recogniser would still pick the same word.
Keeping the shared embedding (also the output projection) at full precision
was measured and does not pay: q8_0 unchanged at 12/14 for 111 MB instead of
84 MB, q4_k 10/14 instead of 8/14 for 89 MB instead of 47 MB. q8_0 is what
ships.
Drafts of the open sentence switch themselves off while committed sentences
take more than ~500 ms to translate — a slow translator would still be busy
with a draft when the next real sentence arrives.

Recogniser measurements, 2026-10-05/06 on an M1 (16 GB, Metal), the same clip.
**The machine was running unrelated heavy jobs for most of it (load average
4–66); nothing here was measured on an idle machine:**

- parakeet-v3 q4_k at load 4–5: median step cost ~300 ms for a 500 ms step
  (~240 ms recogniser for ~3.5 s of open audio with word-timestamp
  boundaries, ~65 ms VAD). At load 10–20 the same run falls 2–4 s behind and
  long sentences get committed at clause boundaries.
- parakeet decode cost: 32–75 ms per second of audio at load ~5 (q4_k and
  q8_0 the same), 100–480 ms under load.
- moonshine-de q4_k: translation 0.5–1.7 s after the deciding audio, median
  ~1.0 s (load 14–18). It transcribes up to the first longer pause of a clip
  and drops what follows (15 s clip: the sentence after a 0.7 s pause is
  missing; the same sentence alone transcribes fine), so it needs short
  decode regions — try `--stream-final-on-silence-ms 400`.
- nemotron q4_k: usable again on Metal after two fixes (the ggml fork had
  lost the `_hp` matmul kernel, which aborted every non-streaming run; the
  one-shot GPU stream cache resubmitted cached graphs and produced word
  salad after ~4.5 s). Speed is still not there: a 50 s clip took 57–106 s in
  streaming mode under load 7–19. Its `<de-DE>` language-tag tokens are now
  stripped from text, word lists and the session stream.

Other models people ask about (Hugging Face tags `speech-translation`,
`streaming-translation`, `simultaneous-translation`, looked at 2026-10-05,
none of these run here unless stated):

- `sbintuitions/hikari-medium` — causal-Whisper simultaneous speech
  translation, **English speech only** (→ ja/ru/de) plus English streaming
  ASR. The right kind of model for English→German; needs a backend port.
- `netease-youdao/Confucius4-T3PO` — append-only streaming text translation
  with KV-cache reuse, the ideal protocol for this mode; Qwen2.5-14B, zh↔en,
  smallest GGUF 10.5 GB.
- `febilly/Hy-MT2-1.8B-StreamRevise(-v4)` — Hy-MT2 fine-tuned to revise its
  own previous translation as the ASR hypothesis changes (no flicker);
  trained on zh/en/ja only.
- Index-Echo S2TT (`--backend index-echo`, already in this repo) — direct
  speech→text translation; upstream documents Chinese speech with
  English/Japanese/Spanish output.
- `unswnlporg/tor-simt-llama-3-8b-*-de-en` (8B), Phi-4-multimodal, GigaChat
  audio: too large for a laptop next to a recogniser.

`CRISPASR_STREAM_TIMING=1` prints one line per step (audio taken in, VAD
cost, decode cost, seconds decoded) — a step that costs more than the audio
it took in is a stream falling behind. The closing
`crispasr[translate]: …` line reports translate time, lag behind the audio,
and how many partials were discarded because they did not line up with the
committed text.

### Events (`--stream-json`)

The stream's own `partial` / `final` / `silence` events are still emitted,
plus:

| `type` | When | Fields |
|---|---|---|
| `sentence` | A sentence was committed. | `utterance_id`, `sentence_id`, `text`, `t` |
| `translation` | Its translation is ready. | `utterance_id`, `sentence_id`, `text`, `translation`, `source_lang`, `target_lang`, `t`, `mt_ms`, `lag_ms` |
| `translation_partial` | Draft translation of the open remainder: of the words the last two partials agree on, without the recogniser's provisional last word and full stop (`CRISPASR_LT_DRAFT_AGREE=0`: the whole remainder; that is rewritten 2–13× more often). May change or never be followed up. `stable` is the start of `translation` (whole words) that the previous draft of the same sentence agreed on; it is rewritten far less often (normalized erasure 0.16–0.18 against 1.17–1.48 for the whole draft, German→English with Opus-MT). Show only `stable` for a calm display, all of `translation` for the earliest one. The terminal view prints `stable` normally and the rest dimmed. | `utterance_id`, `text`, `translation`, `stable`, `t`, `mt_ms`, `lag_ms` |

`t` is the stream time of the step that decided the event; `lag_ms` is how
long after that audio arrived the translation was ready. `sentence_id` counts
up across the whole stream. Two things differ from plain `--stream-json`
while translating: `partial.text` covers only the region still being decoded
(not the whole utterance), and `final.text` is the utterance **as committed**
— its sentences joined — rather than a fresh re-decode, so `--stream-final-mode`
has no effect.

Not done here: `--punc-model` is switched to `--stream-punc partial`
automatically (sentences are found in partials); there is no speaker
labelling on `sentence` events. The first Ctrl+C ends the stream cleanly
(the sentence in progress is committed and translated); a second one kills
the process.

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
| `--stream-partial-tail-sec N` | `0` (off) | JSON+VAD only (#404). Cap each live partial decode to the last ~N seconds of the open utterance. Text decoded ahead of the moving anchor is kept as a committed prefix, so `partial.text` still covers the whole utterance, and `final.text` is untouched (redecode mode re-decodes the full utterance regardless). Cuts land on the quietest 100 ms, the same boundary policy as the long-audio chunker. Effective floor ~4 s. |

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

Related, and **on by default**: `CRISPASR_STREAM_SLICE_MEMO` memoizes each
VAD-closed slice's partial decode by its absolute sample range — a closed
slice keeps the same audio while it stays in the rolling window, so
re-decoding it every step repeated byte-identical work. Exact by decode
determinism (finals and partials byte-equal in the A/Bs; wall −12 % CPU /
−6 % GPU on an uncontended box); set `=0` to restore the old re-decode path.

`--stream-partial-tail-sec` attacks the other axis of partial cost: not how
*often* a partial decodes, but how much *audio* each one covers. Without it, the
partial decode of an open utterance re-encodes the whole utterance-so-far (up to
`--stream-length`) every time, so preview cost grows with utterance length even
though only the tail changes. Encoder-state reuse cannot fix this exactly — a
bidirectional encoder (e.g. cohere's Conformer, unmasked relative-position
attention over the whole window in every layer) makes every earlier frame's
encoding depend on later audio — so the incrementality lives at the *text*
level instead: the region behind the cap is decoded once at a quiet cut, its
text committed, and each subsequent partial decodes only `[cut, now]`. On
CPU, where every encoder pass pays a large weights-bandwidth constant, pair it
with `--stream-partial-decode-ms`; on GPU the per-decode saving dominates.
Finals are exact either way: `--stream-final-mode redecode` (the default)
re-decodes the buffered utterance PCM from scratch. Expect small cosmetic
seams in the stitched *partials* (a capital letter or period where two
independently-decoded regions join — e.g. "…that the Proposed…"); the final
replaces them with the seamless full-utterance text.

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
| `granite` (granite-speech) | LLM greedy (Granite LLM) | Standard `run_with_probs_cb` |
| `voxtral4b` | LLM greedy (Mistral LLM) | Per-step encoder-frame injection via `pre_hook` |
| `glm-asr` | LLM greedy (GLM BPE) | Adapter-side greedy loop using exported step APIs |
| `moss-audio` | LLM greedy (GPT-2 BPE) | Via `moss_audio_process_cb` |
| `moss-transcribe` | LLM greedy (GPT-2 BPE) | Via `moss_transcribe_transcribe_cb` |
| `gemma4-e2b` | LLM greedy (SentencePiece) | Via `gemma4_e2b_transcribe_cb`; control tokens filtered |
| `moonshine-streaming` | LLM greedy (SentencePiece) | Via `moonshine_streaming_transcribe_cb` |
| `kyutai-stt` | LLM greedy (SentencePiece) | Via `kyutai_stt_transcribe_cb`; padding tokens filtered in C lib |
| `mimo-asr` | LLM greedy (GPT-2 BPE) | Via `mimo_asr_transcribe_cb` |
| `nemotron` | RNN-T (per non-blank frame) | Via `nemotron_transcribe_cb`; fires per emitted frame |
| `qwen3` (Qwen3-ASR; alias `mega-asr`) | LLM greedy (Qwen3) | Native |
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
watermarked before emit, and a 200 ms gap separates chunks. Accepted on every
TTS backend, but the granularity is always **one sentence** — unlike the
server's `stream: true`, this path calls `synthesize()` per chunk and does not
use a backend's `CAP_STREAMING` sub-sentence emit. The backends the chunker
treats as single-shot (`vibevoice*`, `qwen3-tts*`, `tada*`, `dots-tts*`,
`omnivoice*` — see
[server.md](server.md#long-form-chunking-for-v1audiospeech)) therefore produce
exactly one chunk, i.e. no early audio at all.

Note that the watermark is **forced on** here regardless of `--no-watermark` /
`CRISPASR_NO_WATERMARK`: a raw PCM stream has no container, so no C2PA manifest
can ride along and the watermark is the only machine-readable mark available.

### Server (`stream: true`)

`POST /v1/audio/speech` with `"stream": true` and a PCM `response_format`
(`pcm`, `wav`, or `f32` — `mp3`/`aac`/`opus` return `400`) streams the audio
back with chunked transfer encoding. On a backend with `CAP_STREAMING` the
chunks are the backend's own codec chunks, so time-to-first-audio is roughly
one chunk; on every other backend it is one chunk per sentence. The body is
always raw **int16 LE mono PCM at the backend's native rate**, served as
`Content-Type: audio/pcm` — there is no RIFF header even when you asked for
`wav`, so the client must know the rate out-of-band:

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

This path has its own splitter, not the server/CLI one: it breaks on ASCII
`.`, `!`, `?` and **newline**, plus CJK `。`, `！`, `？` — no Devanagari danda,
no 600-char run-on cap, and no per-backend single-shot exemption. It also
inserts no silence between chunks (the caller concatenates), and applies
`--tts-pad-silence-ms` only to the first chunk.

Note: for diffusion backends (e.g. irodori) the per-*sentence* granularity above
is the real latency win — a diffusion utterance is generated in full before it is
decoded, so there is no sub-sentence audio to emit early.

### Source separation (htdemucs)

HTDemucs processes audio in overlapping chunks with cross-fading, but this is
internal chunking for memory management, not real-time streaming. The `streaming`
capability flag indicates support for chunked processing of long audio files.
