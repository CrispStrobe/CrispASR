# CLI reference

`crispasr` extends upstream whisper.cpp's argument set with a handful
of backend-dispatch flags. Every historical whisper flag still works —
when you don't pass `--backend`, the backend is auto-detected from the
model (filename heuristics first, then the GGUF `general.architecture`);
a `ggml-*.bin` resolves to whisper.

## Contents

- [Quick reference](#quick-reference) — most-used flag patterns
- [Core flags](#core) — model, backend, input, language, GPU backend
  - [Model resolution flags](#model-resolution-flags) — `--model-quant`, `--auto-download`, `--cache-dir`, `--hf-repo`
  - [TTS-specific flags](#tts-specific-flags) — voice, instruct, codec, steps, trim
- [Output formats](#output) — txt / srt / vtt / json / csv / lrc
- [Segmentation & chunking](#segmentation--chunking) — VAD, fixed chunks
- [Word-level timestamps via CTC alignment](#word-level-timestamps-via-ctc-alignment) — incl. standalone `--align-only`
- [Sampling / decoding](#sampling--decoding-whisper--llm-backends) — temperature, beam, grammar
- [Language detection (LID)](#language-detection-lid)
- [Diarization](#diarization) — `--diarize`, pyannote, embedder-based clustering
- [LLM-backend specific](#llm-backend-specific) — aligner, max-new-tokens
- [Multi-language / translation](#multi-language--translation)
- [Threading & processors](#threading--processors)
- [Whisper-only flags](#whisper-only-flags)
- [Pitch / F0 estimation (`--pitch`)](#pitch--f0-estimation---pitch) — CREPE pitch track
- [Piano transcription (`--piano`)](#piano-transcription---piano) — note events (onset/offset/pitch/velocity)
- [Chord recognition (`--chords`)](#chord-recognition---chords) — BTC chord timeline (**non-commercial weights**)
- [Auto-download (`-m auto`)](#auto-download--m-auto) — registry table
- [Audio formats](#audio-formats) — WAV / FLAC / MP3 / OGG / Opus / M4A
- [Memory footprint](#memory-footprint) — KV quant, mmap, recommended combos

## Quick reference

The 10 patterns that cover most usage:

```bash
# Auto-download + transcribe
crispasr -m auto --backend parakeet -f audio.wav

# VAD + SRT + sentence-split (best subtitle output)
crispasr -m parakeet.gguf -f long.wav --vad -osrt --split-on-punct

# Word timestamps via CTC aligner (LLM backends)
crispasr --backend voxtral -m auto -f jfk.wav -am canary-ctc-aligner.gguf -osrt -ml 1

# Auto language detection
crispasr -m auto --backend cohere -f audio.wav -l auto

# Translate to English
crispasr -m auto --backend whisper -f de.wav --translate

# Live mic
crispasr --mic -m auto --backend parakeet

# Text LID — auto-routes by GGUF arch, auto-downloads on first use
crispasr-lid -m auto --text "Bonjour le monde"           # cstr/cld3-GGUF (default)
crispasr-lid -m auto:glotlid --text "Bonjour le monde"   # 2102 ISO 639-3 + script
# Post-ASR text LID
crispasr -m parakeet.gguf -f speech.wav --lid-on-transcript auto

# JSON with word + token detail
crispasr -m auto --backend parakeet -f audio.wav -ojf

# Force GPU pick
crispasr --gpu-backend vulkan -dev 1 -m auto -f audio.wav

# Half-VRAM voxtral4b
CRISPASR_KV_QUANT=q4_0 CRISPASR_GGUF_MMAP=1 crispasr --backend voxtral4b -m auto -f audio.wav

# TTS — synthesize speech from text
crispasr --backend kokoro -m auto --tts "Hello, how are you?" -o output.wav

# S2S — speech-to-speech (audio in → audio out)
crispasr --backend lfm2-audio -m auto -f input.wav --s2s -o reply.wav

# Speech restoration — 16 kHz input, restored 48 kHz output
crispasr -m sidon-v0.1-f16.gguf -f input.wav --s2s --s2s-output restored.wav

# VoxCPM2 AudioVAE upscaling — 16 kHz input, upscaled 48 kHz output
crispasr -m voxcpm2-vae-f32.gguf -f input.wav --s2s --s2s-output upscaled.wav

# List every backend + capabilities
crispasr --list-backends
```

## Core

| Flag | Meaning |
|---|---|
| `-m FNAME`, `--model FNAME` | Path to a model file, or `auto` to download a default for the selected backend |
| `--backend NAME` | Force a specific backend. Default: auto-detected from GGUF metadata + filename heuristics |
| `-f FNAME`, `--file FNAME` | Input audio (can repeat; also accepts positional filenames) |
| `-t N`, `--threads N` | Thread count (default: `min(4, nproc)`) |
| `-l LANG`, `--language LANG` | ISO-639-1 code, or `auto` to detect (default: `auto`) |
| `--tts "TEXT"` | Synthesize speech from text (requires `CAP_TTS` backend). Output via `--tts-output` |
| `--tts-output FNAME` | Output path for the synthesized audio — `.wav`, `.mp3`, `.m4a`, `.mp4`, `.aac`, `.opus` (default: `tts_output.wav`) |
| `--tts-phonemes "IPA"` | Synthesize these phonemes verbatim, skipping the G2P — the seam for telling a G2P bug from a model bug (#316). `kokoro` and `piper` only; any other backend exits 2 rather than silently synthesizing `--tts` instead. See [tts.md](tts.md#driving-the-phonemes-directly---tts-phonemes) |
| `--tts-stream` | Stream s16le mono PCM to stdout per sentence (pipe to a player); logs stay on stderr. See [streaming.md](streaming.md#streaming-synthesized-audio-out) |
| `--s2s` | Speech-to-speech mode: audio in → audio out (requires `CAP_S2S` backend, e.g. `lfm2-audio`, `mini-omni2`, `sidon`, `voxcpm2-vae`) |
| `--s2s-output FNAME` | Output path for the S2S audio — same container list as `--tts-output` (default: `s2s_output.wav`) |
| `--voice PATH` | Voice reference for TTS: GGUF voice pack or reference WAV for cloning (`--i-have-rights` required for WAV cloning) |
| `--server` | Run as HTTP server with persistent model (see [`server.md`](server.md)) |
| `--server-workers N` | Server: `N>1` loads N model instances so pure-ASR requests run concurrently (N× memory; env `CRISPASR_SERVER_WORKERS` overrides). See [`concurrency.md`](concurrency.md) |
| `--ws-port N` | Server: real-time WebSocket ASR streaming port (`-1` off, `0` = HTTP port + 1) |
| `--wyoming-port N` | Server: Wyoming-protocol TCP port for Home Assistant Assist (`-1` off, default) |
| `--warmup` / `--no-warmup` | Run a short dummy transcribe after init to amortize first-call overhead / skip the always-on server warmup (workaround for GPU drivers that hang in warmup, #165) |
| `--list-backends` | Print the capability matrix and exit |
| `--list-backends-json` | Same as `--list-backends` but JSON-formatted, for tooling |
| `--version` | Print build info (version, git SHA, compiled backends) and exit |
| `--diagnostics`, `--diag` | Full diagnostics (build + env + GPU enumeration) and exit |
| `-v`, `--verbose` | Verbose debug: model/cache paths, device pick, per-stage timings |
| `--gpu-backend NAME` | Force GPU backend: `cuda`, `vulkan`, `metal`, or `cpu` (default: `auto`) |
| `--no-gpu` / `--device N` | Disable GPU entirely, or pin to GPU index N |
| `--return-logits` | For dense CTC backends, write `<audio>.ctc-logits.json` containing the frame-major CTC grid (`data[t * n_vocab + v]`) plus shape metadata and any exposed vocab |

### Model resolution flags

| Flag | Meaning |
|---|---|
| `-m auto` | Download the registry default for `--backend` on first use; subsequent runs are instant. `-m default` is an accepted synonym |
| `-m auto:Q` | Shorthand for `-m auto --model-quant Q` (e.g. `-m auto:q8_0`); an explicit `--model-quant` wins |
| `--model-quant Q` | Preferred quant for registry resolution; overrides the default. E.g. `--model-quant q8_0` to get Q8_0 instead of the default Q4_K. Also changes any companion model (voice pack, codec). |
| `--auto-download` | Explicitly allow auto-download of missing registry models. Implied by `-m auto` and `--hf-repo`. |
| `--cache-dir DIR` | Override the auto-download cache directory. Precedence: this flag, `$CRISPASR_CACHE_DIR`, `$CRISPASR_MODELS_DIR`, then `~/.cache/crispasr/`. |
| `-hfr REPO`, `--hf-repo OWNER/REPO[:FILE]` | Fetch model from an arbitrary HuggingFace repo. E.g. `--hf-repo cstr/parakeet-tdt-0.6b-v3-GGUF:parakeet-tdt-0.6b-v3-q4_k.gguf`. Implies `--auto-download`. |
| `-hff FNAME`, `--hf-file FNAME` | Filename within `--hf-repo` (alternative to the `OWNER/REPO:FILE` shorthand) |
| `--dry-run-resolve` | Print the resolved model path + companion paths and exit — does not load or synthesize |
| `--dry-run-ignore-cache` | As `--dry-run-resolve` but pretend the cache is empty (shows what would be downloaded) |

### TTS-specific flags

| Flag | Meaning |
|---|---|
| `--voice PATH` | GGUF voice pack or reference WAV. GGUF packs are used for VibeVoice/Qwen3-TTS/Orpheus style conditioning; WAV enables voice cloning (requires `--i-have-rights`) |
| `--voice-dir PATH` | Server: directory of `<name>.gguf` or `<name>.wav` voice profiles. Enables `/v1/voices` listing and name-based voice selection in `/v1/audio/speech` |
| `--ref-text "TEXT"` | Reference transcription for the ref audio (qwen3-tts, f5-tts). Auto-transcribed from `--voice <wav>` if omitted |
| `--ref-asr BACKEND` | ASR backend to auto-transcribe the ref audio (default: `whisper`) |
| `--instruct "TEXT"` | Natural-language voice/style description. For qwen3-tts: VoiceDesign mode (voice description) or CustomVoice mode (style control) |
| `--make-ref` | Create a TADA voice reference GGUF from `--voice <audio.wav>` + `--ref-text "transcript"`. Requires `--i-have-rights`. Pure C++, no Python. Auto-discovers `tada-encoder-f16.gguf` + `tada-aligner-<lang>.gguf` (falling back to `-en`) next to the model. Output path via `--make-ref-output` (default: `tada-ref-custom.gguf`) |
| `--make-ref-output PATH` | Output path for `--make-ref` (default: `tada-ref-custom.gguf`) |
| `--make-ref-encoder PATH` | Explicit path to the TADA encoder GGUF (auto-discovered if omitted) |
| `--make-ref-aligner PATH` | Explicit path to the TADA aligner GGUF (auto-discovered if omitted) |
| `--codec-model FNAME` | Explicit path to the codec/companion GGUF (e.g. Qwen3-TTS codec encoder). Defaults to sibling / cache / registry auto-discovery |
| `--codec-quant Q` | Preferred quant for registry companion resolution (codec model) |
| `--tts-steps N` | Diffusion / ODE step count (default 20). VibeVoice DPM-Solver++ takes 10–20; also read by irodori (40), chatterbox, f5-tts and tada |
| `--tts-cfg-scale X` | CFG guidance scale (vibevoice / chatterbox / f5-tts / tada / irodori). Unset = backend default |
| `--tts-speed X` | Speaking-rate multiplier (omnivoice / f5-tts / piper / melotts / fastpitch): `>1` faster, `<1` slower (default 1.0) |
| `--tts-min-speech-tokens N` | Floor on generated audio length in AR decode steps — one codec frame each (12.5 Hz ⇒ 80 ms, so `25` ≈ 2 s). moss-tts / moss-tts-local; `-1` = model default |
| `--tts-trim-silence` | Trim leading silence from TTS output |
| `--tts-pad-silence-ms N` | Prepend N ms of silence to the output (default 0; works around the VLC C2PA buffer drop) |
| `--tts-play` | Play the synthesized audio on the local default speaker (same watermarked PCM that is written to `--tts-output`) |
| `--tts-play-device N` | Speaker device index for `--tts-play` (`-1` = default device) |
| `--tts-max-input-chars N` | Server: cap on `/v1/audio/speech` `input` length in characters (default 4096; `0` = no cap) |

## Output

| Flag | Output |
|---|---|
| `-otxt` | Plain text to `<audio>.txt` |
| `-osrt` | SubRip (SRT) to `<audio>.srt` |
| `-ovtt` | WebVTT to `<audio>.vtt` |
| `-ocsv` | CSV (start, end, text) |
| `-oj`, `-ojf` | JSON (compact or full with word/token arrays) |
| `-olrc` | LRC lyrics format |
| `-of FNAME` | Output file base (no extension) |
| `-np` | No prints (suppress stderr progress) |
| `-pc` | Color-code output by token confidence (where supported) |
| `--no-timestamps` | Plain text only, no timing |
| `-ml N` | Max chars per display segment. `0`=unlimited, `1`=per-word, `N`=split at word boundaries |
| `-sp`, `--split-on-punct` | Split subtitle lines at sentence-ending punctuation (`. ! ?`). Creates readable subtitles even for CTC models that produce long segments |

### JSON layout

CrispASR writes outputs side-by-side with the input audio (e.g.
`jfk.wav` → `jfk.srt`, `jfk.vtt`, `jfk.json`):

```json
{
  "crispasr": {
    "backend": "parakeet",
    "model":   "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "language":"en"
  },
  "transcription": [
    {
      "timestamps": { "from": "00:00:00,240", "to": "00:00:10,880" },
      "offsets":    { "from": 240, "to": 10880 },
      "text":       "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    }
  ]
}
```

Add `-ojf` (`--output-json-full`) to include per-word `words[]` and
per-token `tokens[]` arrays when the backend populates them:

```json
      "words": [
        { "text": "And", "t0": 24, "t1": 52, "offsets": { "from": 240, "to": 520 } }
      ]
```

**Time units.** Segment `offsets` (and each word/token `offsets`) are in
**milliseconds**. The per-word / per-token `t0`/`t1` fields are in
**centiseconds** — this is the library's internal timebase (see the C ABI's
`t0_cs`/`t1_cs`) and is retained for backwards compatibility. Prefer the
`offsets` object for a unit that is consistent across every level of the
document (issue #228).

## Segmentation / chunking

| Flag | Meaning |
|---|---|
| `--vad` | Enable Silero VAD. Auto-downloads `ggml-silero-v6.2.0.bin` (~885 KB) to `~/.cache/crispasr/` on first use |
| `-vm FNAME`, `--vad-model FNAME` | Override the VAD model: a path, or one of `auto`, `silero`, `firered`, `marblenet`, `webrtc`, `whisper-vad` (default: auto) |
| `-vt F` | VAD threshold (default 0.5) |
| `-vspd N` | VAD min speech duration (ms, default 250) |
| `-vsd N` | VAD min silence duration (ms, default 100) |
| `-vmsd F`, `--vad-max-speech-duration-s F` | VAD max speech duration in seconds; longer runs are auto-split (default: unlimited / `FLT_MAX`) |
| `-vp N`, `--vad-speech-pad-ms N` | Padding added to each side of a VAD segment (ms, default 30) |
| `-vo F`, `--vad-samples-overlap F` | Overlap between consecutive VAD segments (seconds, default 0.1) |
| `--vad-stitch` | Opt in to concatenating all VAD segments into one buffer for a single decode (non-whisper backends). Preserves cross-segment context but collapses the result into one output segment, so it breaks SRT/VTT. Off by default — the per-slice path keeps one segment per VAD region |
| `--vad-export FILE` | Compute VAD boundaries and write them to `FILE` as JSON, then exit. Implies `--vad`. No ASR model needed — standalone verb |
| `--vad-import FILE` | Read segment boundaries from `FILE` instead of running VAD — reuse boundaries across backends without recomputing VAD (issue #227). Implies `--vad` |
| `--vad-import-strict` | With `--vad-import`, refuse (rather than warn) if the file's chunk length differs from this run |
| `--vad-export-raw FILE` | Like `--vad-export`, but writes raw VAD **speech segments** (chunk-length-independent). Imports at any `--chunk-seconds`, re-chunked per run. Implies `--vad` |
| `-ck N`, `--chunk-seconds N` | Fallback chunk size when VAD is off (default 30 s). Backends declaring `CAP_UNBOUNDED_INPUT` / `CAP_INTERNAL_CHUNKING` handle long audio themselves and ignore it unless you pass it explicitly |
| `--chunk-overlap F` | Overlap context (seconds) at chunk boundaries (default 3.0) |
| `--lcs-dedup auto\|on\|off` | NeMo-style sub-word LCS dedup across chunk boundaries (default `auto` — fires when chunking with overlap) |
| `--lcs-min-length N` | Minimum LCS length to act on (default 1; raise to 3-4 on long-silence audio where blank tokens dominate boundaries) |
| `--parakeet-decoder ctc\|tdt\|maes` | Select decode strategy: `ctc` (CTC head), `tdt` (TDT greedy/beam, default), `maes` (MAES beam search — requires `-bs N` with N>1) |
| `-bs N`, `--beam-size N` | Parakeet TDT/RNNT beam search width (default: unset = greedy). `2`–`4` recommended with hotwords or MAES. CTC decode is frame-synchronous and always greedy |
| `--sensitivity conservative\|balanced\|aggressive` | Named bundle of the four whisper fallback thresholds (`-et`, `-lpt`, `-nth`, temperature step). `balanced` is the shipped default and always a no-op. See below |

#### `--sensitivity` — the four decode thresholds as one knob

`entropy_thold`, `logprob_thold`, `no_speech_thold` and the temperature step
interact: a decode is only rejected and retried when the logprob **and**
no-speech bars are both crossed, so moving one alone produces a combination
that does not mean what you intended. `--sensitivity` moves them as a set.

| Preset | Behaviour |
|---|---|
| `conservative` | Tighter entropy/logprob bars and a **lower** no-speech bar, so the "this is silence" verdict is reached sooner and borderline segments are discarded rather than guessed at. Fewer hallucinations, some marginal speech lost. Use on noisy or music-heavy material where a confident wrong sentence is worse than a missing one |
| `balanced` | The shipped defaults (2.40 / -1.00 / 0.60). Naming it changes nothing |
| `aggressive` | Looser bars and a **higher** no-speech bar, so quiet or whispered audio still produces text instead of being written off as silence. Use when the audio is quiet by nature and you would rather post-filter than lose content |

`strict`, `default` and `loose` are accepted aliases. An unrecognised name is
rejected with a non-zero exit rather than silently treated as `balanced` — a
mistyped preset decoding at the wrong thresholds is exactly what the flag
exists to prevent.

Last flag wins, so an explicit threshold after the preset overrides it:

```bash
crispasr -m model.bin -f noisy.wav --sensitivity conservative
crispasr -m model.bin -f quiet.wav --sensitivity aggressive -nth 0.9   # preset, then override
```

Same bundles from the session API: `crispasr_session_set_sensitivity()` in C,
`Session.set_sensitivity("conservative")` in Python.

#### Reusing VAD boundaries across backends (#227)

`--vad-export` is a standalone verb: it runs Silero VAD on the audio,
writes the boundaries, and exits — no ASR model download or load needed.
To compare several backends on the same audio without recomputing VAD:

```bash
# Step 1: export VAD boundaries (no model needed, ~300 ms).
crispasr -f talk.wav --vad-export talk.vad.json

# Step 2: transcribe with multiple backends using the same boundaries.
crispasr -m parakeet.gguf -f talk.wav --vad-import talk.vad.json
crispasr -m moonshine.gguf -f talk.wav --vad-import talk.vad.json
```

Both `--vad-export` and `--vad-import` imply `--vad`, so the separate
`--vad` flag is not required. The JSON records each segment's sample
offsets plus absolute centisecond timestamps; boundaries are clamped to
the imported audio and rescaled if the sample rate differs.

**Two export forms.** `--vad-export` writes **chunk boundaries** (a `"kind":
"chunks"` file): the VAD segments already split to the run's `--chunk-seconds`.
These are only valid for that chunk length — importing them under a different
`--chunk-seconds` warns (and, with `--vad-import-strict`, refuses), because the
chunking would not match.

`--vad-export-raw` writes the raw VAD **speech segments** (`"kind":
"vad_segments"`), before any chunking. One raw file imports cleanly at *any*
`--chunk-seconds` — it is re-chunked for each run exactly as a fresh VAD pass
would be — so it is the form to keep when the same audio will be transcribed by
backends that want different chunk sizes:

```bash
crispasr -f talk.wav --vad-export-raw talk.vad.json          # once
crispasr -m parakeet.gguf -f talk.wav --vad-import talk.vad.json --chunk-seconds 12
crispasr -m whisper.gguf  -f talk.wav --vad-import talk.vad.json --chunk-seconds 30
```

Files written before `"kind"` existed are read as `chunks` (the historical
behaviour), so older exports keep working.

### Strict pipeline — require aux stages to succeed (`--strict-pipeline`, #311)

By default CrispASR **degrades gracefully**: a VAD, forced-aligner, or
punctuation model that fails to load is skipped with a stderr warning and the
command still exits `0`. That is convenient interactively but hides failures
from automation — a zero exit and a valid `-ojf` do **not** prove the requested
stages ran. For integrations that treat those stages as *required task
properties*, opt into strict semantics:

| Flag | Effect |
|---|---|
| `--strict-pipeline` | Require every stage **explicitly requested** on this command line: VAD if `--vad`/`-vm`, word timestamps if `-am`/`--force-aligner`, punctuation if `--punc-model`. |
| `--require-vad` | Force the VAD requirement (needs `--vad`/`-vm`). |
| `--require-word-timestamps` | Every non-empty output segment must carry word timestamps (native **or** aligned). |
| `--require-punctuation` | Force the punctuation-model requirement (needs `--punc-model`). |

Under strict semantics a required stage that **fails to load or produce its
output** returns a **non-zero exit** (and the output file is not written), while
a stage that **ran and legitimately found nothing** (VAD detected no speech)
stays a success. Nothing depends on parsing stderr, and a direct local `-m`/
`-vm`/`-am`/`--punc-model` path is used as-is — strict mode never falls back to
auto-download.

Distinct exit codes let a caller tell *which* stage failed:

| Exit | Meaning |
|---|---|
| `2` | Config error — a `--require-*` whose stage was never requested |
| `30` | Required VAD model failed to load |
| `31` | Required word timestamps missing (aligner failed to load, or no native/aligned words on a non-empty segment) |
| `32` | Required punctuation model failed to load |

```bash
# The integration's contract: rc 0 ⟺ every required stage succeeded.
crispasr --backend parakeet -m asr.gguf -f in.wav -l zh \
    --vad -vm vad.gguf -am aligner.gguf --force-aligner \
    --punc-model punc.gguf --strict-pipeline -ojf -of result
echo "rc=$?"   # 0 = VAD ran + words present + punctuation applied; 30/31/32 = that stage failed
```

Strict requirements route through the unified backend dispatch, so pass an
explicit `--backend` (any backend, including `--backend whisper`) rather than
relying on the legacy whisper-only path.

**Across surfaces.** The CLI and the **HTTP server** both auto-run these stages
and both honour strict semantics (the server via `strict_pipeline` /
`require_*` form fields, returning HTTP 400 — see
[`server.md`](server.md#strict-pipeline--fail-on-a-required-stages-failure-311)).
The decision logic is shared (`crispasr_strict.h`) so they can't drift. The
**session C-ABI** (`crispasr_session_*`, used by the language bindings) is a
lower-level, caller-driven primitive — it does not auto-orchestrate VAD +
alignment + punctuation, so there is no silent degradation to guard: a binding
caller invokes each stage explicitly (`crispasr_session_transcribe_vad`, the
aligner, the punc setter) and already sees each one's result directly. Strict
mode is therefore a property of the two orchestrating front-ends (CLI, server).

#### Transcribing a time window (`--offset-t` / `--duration`, #91)

Restrict processing to a slice of the input instead of the whole file:

```bash
crispasr -m parakeet.gguf -f talk.wav --offset-t 60000              # skip the first 60 s
crispasr -m parakeet.gguf -f talk.wav --offset-t 60000 --duration 30000   # only 60 s → 90 s
```

| Flag | Meaning |
|---|---|
| `-ot MS`, `--offset-t MS` | Start transcription `MS` milliseconds into the audio |
| `-d MS`, `--duration MS` | Process only `MS` milliseconds from the offset (0 = to end) |

The window is applied to the decoded audio before VAD/chunking, and reported
timestamps (segment, word, token) are shifted back so they stay in
original-audio time. These flags previously affected only the `whisper`
backend's internal seek; they now work for every backend. An offset past the
end of the file is reported and exits cleanly.

### MAES beam search (§134)

MAES (Modified Adaptive Expansion Search) is a transducer-specific beam
search that processes one encoder frame at a time with adaptive expansion.
More efficient than the label-looping beam for transducers.

```bash
# Via --parakeet-decoder maes:
crispasr -m parakeet-tdt-0.6b-v3.gguf -f audio.wav --parakeet-decoder maes --beam-size 4

# Via env var:
CRISPASR_PARAKEET_MAES=1 crispasr -m model.gguf -f audio.wav --beam-size 4
```

**Tuning env vars:**

| Variable | Default | Description |
|---|---|---|
| `CRISPASR_PARAKEET_MAES` | 0 | Set to 1 to enable MAES (alternative to `--parakeet-decoder maes`) |
| `CRISPASR_MAES_NUM_STEPS` | 2 | Max non-blank expansions per frame |
| `CRISPASR_MAES_GAMMA` | 2.3 | Pruning threshold (lower = more aggressive) |
| `CRISPASR_MAES_BETA` | 2 | Extra candidates beyond beam_size |

Works with both TDT and RNNT parakeet models, and nemotron (via
`CRISPASR_NEMOTRON_MAES=1`). CTC beam search is separate (activated by
`--beam-size N` with CTC models, uses shared
`core_ctc::prefix_beam_search`).

**Nemotron MAES** (§167b): same algorithm ported to the nemotron RNNT
decoder. On JFK 11s, MAES removes spurious `<en-US>` language tags and
produces proper punctuation vs standard beam.

```bash
CRISPASR_NEMOTRON_MAES=1 crispasr -m nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
  --backend nemotron --beam-size 4 -f audio.wav
```

### Parakeet long-form encoding (issue #89 / §216)

Parakeet's long-audio path is **model-dependent** — the JA-only model and the
multilingual / v3 / EN models behave very differently:

- **Non-JA (v3 / multilingual / EN, vocab > 4096):** a single full-attention
  pass is byte-for-byte identical to upstream NeMo (verified 100 % word match
  vs `nvidia/parakeet-tdt-0.6b-v3`, 30 s → 5 min). The backend advertises
  internal chunking so the dispatcher hands it the whole clip, then:
  - **≤ cap** (default 300 s): one full-attention pass — NeMo-exact.
  - **> cap:** split at silence into ≤cap single-pass pieces, each transcribed
    with ±2 s acoustic context and committed by word timestamp at a single
    shared cut — no overlap, no gap, **no boundary duplicates**, memory bounded
    by the cap. (Full attention is O(T²): ~5 min is safe on 16 GB, ~28 min
    OOMs, hence the cap + silence-split.)

  This replaced the old forced `streamed` path, which *collapsed* on v3
  (5 min → 75 words), and the dispatcher chunk-30 + LCS-merge fallback, which
  duplicated a phrase at every 30 s boundary (token-id-exact LCS can't cancel
  the divergent re-transcription of the overlap).

  On some recordings a stretch of speech in the middle of a long file can
  still come out missing from the transcript, with no error (#350). To catch
  this, any stretch of 3 s or more where nothing was transcribed is
  automatically transcribed again and the recovered words merged back in. When
  the transcript was already complete this changes nothing and adds no time.
  `CRISPASR_GAP_FILL=0` turns it off; `CRISPASR_GAP_FILL_MIN_CS` tunes it
  (see below).
- **JA-only (vocab ≤ 4096):** the bidirectional encoder is numerically unstable
  when attention spans the whole utterance — codec-level perturbations as small
  as 0.3 % RMS flipped the encoder output std by ~14 % on the #89 clip, driving
  the TDT decoder into emit-blank-forever past ~20 s. (Upstream NeMo fails the
  same way on the same audio — plain, local-attention, and buffered TDT
  inference all score 1–51 % content recall vs a whisper-large-v3 reference.)
  JA therefore uses **auto-VAD + a 12 s slice cap + single-pass per slice**:
  VAD finds speech, slices longer than 12 s (continuous speech merges far past
  the encoder's ~12 s safe window) are re-split at energy minima, and each
  slice gets one NeMo-exact full-attention pass. A **gap-fill second pass**
  then re-transcribes any span ≥1 s the first pass left empty inside a slice
  (the encoder sometimes blanks an utterance whenever enough context follows
  it, even though the same span transcribes verbatim in isolation) and merges
  the recovered words back. Measured on the issue #89 reporter's clips
  (phonetic char-bigram recall vs whisper-large-v3-turbo): 60 s **97.2 %**,
  120 s **96.9 %**, 300 s **95.9 %** — up from 61–64 % with the previous
  streamed default; NeMo's best long-form mode scores ~51 % raw recall on the
  same audio, and an independent SenseVoice-small run tops out at the same
  ~97 % (the inter-model agreement ceiling — the rest is kana/kanji hearing
  variants, not missing content).

**Env vars for tuning (all override the per-model defaults):**

| Variable | Default | Meaning |
|---|---|---|
| `CRISPASR_PARAKEET_STREAM_THRESHOLD` | non-JA 300, JA 12 | Single-pass cap (seconds). Audio ≤ this gets one full-attention pass; `0` disables single-pass entirely (always streamed). |
| `CRISPASR_PARAKEET_VAD_SLICE_CAP` | non-JA 0, JA 12 | Max VAD slice duration (seconds); longer slices are re-split at energy minima before decoding. `0` = no cap. |
| `CRISPASR_PARAKEET_ATT_CONTEXT` | unset | `"L,R"` switches the encoder to rel_pos_local_attn with that window (encoder frames, 1 = 80 ms) — NeMo's `change_attention_model` equivalent. `"-1,-1"` forces full attention. |
| `CRISPASR_GAP_FILL` | 1 | Automatically re-transcribe any stretch of audio the transcript left empty and merge the recovered words back. `0` disables. |
| `CRISPASR_GAP_FILL_MIN_CS` | non-JA 300, JA 100 | Smallest empty stretch that triggers gap-fill, in centiseconds (300 = 3.0 s). Lower values catch shorter drops but may re-transcribe normal pauses, occasionally adding filler words. |
| `CRISPASR_PARAKEET_LONGFORM` | non-JA 1, JA 0 | `1` = silence-split single-pass above the cap; `0` = streamed fallback above the cap. |
| `CRISPASR_PARAKEET_INTERNAL_CHUNKING` | non-JA on, JA off | `0` = revert to the dispatcher's chunk-30 + overlap-save + LCS-merge path (A/B). |
| `CRISPASR_PARAKEET_STREAM_CHUNK` | 0 (auto: 8 JA / 30 non-JA) | Streamed-path encoder chunk size (seconds). |
| `CRISPASR_PARAKEET_STREAM_OVERLAP` | 2 | Streamed-path encoder overlap (seconds). |
| `CRISPASR_PARAKEET_VRAM_BUDGET_MB` | 0 (off) | Proactive memory policy: if single-pass full attention's estimated O(T²) rel-pos bias exceeds this, use the streamed (bounded-window) encoder *before* allocating — avoids the OOM spike on small GPUs. 0 = disabled (single-pass as before; the reactive OOM fallback still backstops). |
| `CRISPASR_PARAKEET_MEM_POLICY` | `auto` | `auto` honours the VRAM budget; `single`/`streamed` force that path; `off` disables the proactive check (reactive-only). |
| `CRISPASR_PARAKEET_MEM_COEFF` | 8.0 | O(T²) estimate coefficient. Default calibrated so a ~4 min clip (T≈2800, 8 heads) estimates ~1.9 GiB, matching a measured CUDA allocation. |
| `CRISPASR_SESSION_UNIFIED_DISPATCH` | 1 | No effect on the CLI. `0` makes the session/bindings API use its older parakeet code path instead of the one shared with the CLI — for troubleshooting comparisons only. |

CLI escape hatches (no env needed): `--chunk-seconds N` forces the dispatcher's
N-second chunk + merge; `--vad` forces the VAD path.

For **parakeet** (non-JA, `CAP_INTERNAL_CHUNKING`), `--chunk-seconds N` does *not*
go through the dispatcher's per-slice merge (which corrupts this full-attention
FastConformer). Instead it runs one coherent internal-streamed decode — encoded
at the model's quality window (30 s, bounded VRAM), so the full transcript is
preserved — and then groups the resulting words into **~N-second output
segments** with per-segment `offsets`/`words`/`tokens` (issue #257). So
`--chunk-seconds 7` yields ~7-second segments of the *complete* transcript, not a
truncated single blob. The encoder window can be overridden independently with
`CRISPASR_PARAKEET_STREAM_CHUNK`. (For bounded-VRAM *single-pass* long audio with
no segmentation, use `--att-context L,R` instead.)

**Examples:**

```bash
# Non-JA (v3): default is NeMo-exact single-pass / silence-split longform —
# no flags needed, any length:
crispasr -m parakeet-tdt-0.6b-v3.gguf -f long_de.wav -osrt

# JA model — auto-VAD + 12 s slice cap + per-slice single-pass by default
# (whole-clip single-pass collapses past ~12 s), no flags needed:
crispasr -m parakeet-tdt-0.6b-ja.gguf -f podcast_ja.wav -osrt

# Force the old dispatcher chunk+merge path for comparison:
CRISPASR_PARAKEET_INTERNAL_CHUNKING=0 \
  crispasr -m parakeet-tdt-0.6b-v3.gguf -f long_de.wav -osrt

# Lower the single-pass cap on a memory-constrained box (splits sooner):
CRISPASR_PARAKEET_STREAM_THRESHOLD=120 \
  crispasr -m parakeet-tdt-0.6b-v3.gguf -f long_de.wav -osrt
```

### How VAD works

Every non-whisper backend uses the Silero VAD model to segment long
audio into speech regions, transcribes **one slice at a time**, and
remaps timestamps back to original-audio positions — so each VAD
region becomes its own transcript segment with correct timing, which
is what SRT/VTT need. Short VAD segments (< 3 s) are auto-merged, and
oversized segments are split at `--chunk-seconds` boundaries. Whisper
handles VAD internally via `wparams.vad`.

`--vad-stitch` opts into the alternative: concatenate every slice into
one contiguous buffer (with silence gaps) and transcribe in a single
pass. That preserves cross-segment context but collapses the result
into one output segment, so it is off by default.

```bash
# Just pass --vad — the model is auto-downloaded on first use
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad -osrt

# Or point at an existing GGUF
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad-model ~/models/ggml-silero-v6.2.0.bin -osrt
```

The cached model lives at `~/.cache/crispasr/ggml-silero-v6.2.0.bin`
(~885 KB). If you don't pass `--vad`, whisper falls back to fixed
30-second chunking (`-ck 30`). Backends with `CAP_UNBOUNDED_INPUT`
(parakeet, canary, canary-qwen, nemotron, wav2vec2, firered-asr,
fastconformer-ctc, granite-nar, voxtral, higgs-stt, ark-asr,
lfm2-audio) process the full audio in one encoder pass by default
because their non-autoregressive encoders lose context at fixed chunk
boundaries (#89). The remaining LLM-based backends (cohere, moonshine,
granite, qwen3, etc.) still chunk at 30 s to avoid OOM from growing
KV caches. Pass `--chunk-seconds N` explicitly to force or override
chunking for any backend.

### Recommended for subtitles

```bash
crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad -osrt --split-on-punct
```

- **Best timing quality:** **parakeet**. Native TDT timestamps are
  more accurate and natural than the forced-aligner fallback used by
  LLM backends.
- **Best default subtitle flags:** `--vad --split-on-punct`. VAD
  segments at natural speech pauses, then CrispASR stitches and
  remaps timestamps back to the original timeline. Avoids the
  mid-sentence boundary problems of fixed 30-second chunking.
- **For backends without native timestamps** (`cohere`, `granite`,
  `voxtral`, `voxtral4b`, `qwen3`): use a CTC aligner together with
  `--vad`. Without VAD, leading silence can throw off sentence
  starts, especially for the qwen3 forced aligner.
- **Any length (non-JA v3 / multilingual / EN):** a single full-attention
  pass is NeMo-exact and the default; clips over the 300 s cap are
  silence-split into single-pass pieces with no boundary duplicates. No
  manual `--chunk-seconds` needed. JA-only models use the streamed path
  instead. See the "Parakeet long-form encoding" section above.
- **If parakeet OOMs on very long audio:** lower the single-pass cap with
  `CRISPASR_PARAKEET_STREAM_THRESHOLD=120` so the silence-split longform
  kicks in sooner (each piece is then ≤120 s of full attention).
- **Hybrid TDT+CTC models** (e.g. `parakeet-tdt_ctc-0.6b-ja`): pass
  `--parakeet-decoder ctc` to use the CTC head. CTC decode is
  frame-synchronous and avoids TDT emission-frame-shift artifacts
  at chunk boundaries.

## Word-level timestamps via CTC alignment

The LLM-based backends (`qwen3`, `voxtral`, `voxtral4b`, `granite`)
don't emit timestamps natively. CrispASR supports a second-pass
forced alignment via NVIDIA's canary-ctc-aligner — a 600 M-param
FastConformer + CTC head that works on any transcript + audio pair
in 25+ European languages.

```bash
# Auto-download the aligner — Q4_K (~442 MB) lives in the registry.
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav \
    -am auto -osrt -ml 1
# [00:00:00.240 --> 00:00:00.640]  And
# [00:00:00.640 --> 00:00:00.880]  so,
# [00:00:00.880 --> 00:00:01.040]  my

# …or grab it once manually (Q4_K / Q5_0 / Q8_0 / F16 all on the same repo):
curl -L -o canary-ctc-aligner.gguf \
    https://huggingface.co/cstr/canary-ctc-aligner-GGUF/resolve/main/canary-ctc-aligner-q4_k.gguf
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav \
    -am canary-ctc-aligner.gguf -osrt -ml 1
```

Alignment granularity is one encoder frame (~80 ms).

### Language-specific wav2vec2 aligners (WhisperX parity)

For non-English audio, use a language-matched wav2vec2 CTC aligner
instead of the multilingual canary-ctc model. These are the same models
WhisperX uses for word alignment, converted to GGUF and available via
auto-download:

| Alias | Language | Source model |
|---|---|---|
| `-am wav2vec2-aligner` | English (default) | wav2vec2-xlsr-en |
| `-am wav2vec2-aligner-de` | German | jonatasgrosman/xlsr-53-german |
| `-am wav2vec2-aligner-fr` | French | jonatasgrosman/xlsr-53-french |
| `-am wav2vec2-aligner-es` | Spanish | jonatasgrosman/xlsr-53-spanish |
| `-am wav2vec2-aligner-it` | Italian | jonatasgrosman/xlsr-53-italian |
| `-am wav2vec2-aligner-ja` | Japanese | jonatasgrosman/xlsr-53-japanese |
| `-am wav2vec2-aligner-zh` | Chinese | jonatasgrosman/xlsr-53-chinese-zh-cn |
| `-am wav2vec2-aligner-nl` | Dutch | jonatasgrosman/xlsr-53-dutch |
| `-am wav2vec2-aligner-pt` | Portuguese | jonatasgrosman/xlsr-53-portuguese |
| `-am wav2vec2-aligner-ar` | Arabic | jonatasgrosman/xlsr-53-arabic |
| `-am wav2vec2-aligner-uk` | Ukrainian | Yehor/xls-r-300m-uk-with-small-lm |
| `-am wav2vec2-aligner-cs` | Czech | comodoro/xls-r-300m-cs-250 |

```bash
# Japanese word timestamps with the JA-specific aligner:
crispasr --backend cohere -m cohere.gguf -f japanese.wav \
    -am wav2vec2-aligner-ja --auto-download -osrt -ml 1

# French with Voxtral:
crispasr --backend voxtral -m auto -f french.wav \
    -am wav2vec2-aligner-fr --auto-download -osrt -ml 1
```

All models are Q4_K-quantized (~200 MB each) and auto-download on first
use. The canary-ctc aligner remains the default (`-am auto`) because
it covers 25+ languages in one model.

### Language-specific FastConformer-CTC aligners

A second aligner family (NVIDIA FastConformer-Hybrid-CTC, CC-BY-4.0) covers
languages the wav2vec2 set doesn't — Slavic, Caucasian, Central-Asian. Same
usage, all auto-download:

`-am fastconformer-aligner-<code>` where `<code>` is one of: `en` /
`en-pc` (English, +punctuation/caps) · `de` · `es` · `fr` · `it` · `nl` ·
`pl` (Polish) · `ru` (Russian) · `ua` (Ukrainian) · `hr` (Croatian) ·
`be` (Belarusian) · `ar` (Arabic) · `fa` (Persian) · `ka` (Georgian) ·
`hy` (Armenian) · `uz` (Uzbek) · `kk-ru` (Kazakh+Russian). Bare
`-am fastconformer-aligner` = English.

```bash
# Polish word timestamps (no wav2vec2 PL aligner exists):
crispasr --backend voxtral -m auto -f polish.wav \
    -am fastconformer-aligner-pl --auto-download -osrt -ml 1
```

For LLM backends with a native timestamp head there is also
`-am qwen3-forced-aligner` (multilingual, ~500 MB) — note it is more
sensitive to leading silence than the CTC aligners.

For subtitle output, prefer adding `--vad --split-on-punct`:

```bash
./build/bin/crispasr --backend cohere -m cohere.gguf -f talk.wav \
    -am canary-ctc-aligner.gguf --vad -osrt --split-on-punct
```

### `--force-aligner` / `-falign` — override native timestamps (issue #62)

By default the CTC aligner is a **fallback** — it only runs when the
backend doesn't already produce word-level timestamps natively.
`--force-aligner` flips that: even when the backend has native
timing (whisper, parakeet, canary, cohere, kyutai-stt), the aligner
runs and replaces the native words with CTC-derived ones. Useful
when the user trusts the aligner's word-onset accuracy more than
the backend's native timing.

```bash
# Parakeet's native word ts replaced with canary-ctc-aligner output:
./build/bin/crispasr --backend parakeet -m auto -f samples/jfk.wav \
    -am auto --force-aligner -ojf
```

The flag also lifts the `CAP_TIMESTAMPS_CTC` capability gate, so
backends that don't formally advertise CTC alignment compatibility
(whisper, parakeet, kyutai-stt — they only declare native timing)
can still use it once the user explicitly asks. Short alias is
`-falign`, NOT `-fa` (already taken by `--flash-attn`).

Notes:
- Without `--force-aligner`, the aligner path is a fallback for
  backends that lack native timestamps.
- `qwen3-forced-aligner` is more sensitive to leading silence;
  `--vad` is strongly recommended with it.
- Parakeet remains the better choice when timestamp quality is the
  top priority and you don't want to pay for a second forward pass.

### Canary auto-aligner default — `--no-auto-aligner` (SubtitleEdit #10775)

Canary's native word timing is cross-attention DTW on the encoder–
decoder, which has a measured **~414 ms MAE on word boundaries**
(`src/canary.cpp:1377-1390`). The official NVIDIA companion model
**`canary-ctc-aligner`** (separate 600 M FastConformer + CTC head,
shipped inside the same `.nemo` tarball as the main canary weights;
`hf_readmes/canary-ctc-aligner-GGUF.md`) gets ~78 ms MAE — **5.3×
tighter**.

Because the gap is so large and the aligner is curated/registered,
`--backend canary` now defaults to `-am auto --force-aligner` whenever
the requested output benefits from word-level timestamps (`-osrt`,
`-ovtt`, `-ojf`, `-owts`, `--max-len > 0`, `--split-on-punct`,
`--print-colors`). This auto-downloads `canary-ctc-aligner-q4_k.gguf`
(~442 MB) into the crispasr cache the first time and reuses it
afterwards. Stream / mic / server / `--text` / `--tts` modes are
exempt.

```bash
# v0.7.0+: equivalent to passing `-am auto --force-aligner` automatically
./build/bin/crispasr --backend canary -m auto -f samples/jfk.wav \
    --max-len 50 --split-on-punct -osrt
```

To opt out (e.g., to keep the old DTW path because you don't want
the ~442 MB download or the second forward pass), add
`--no-auto-aligner`:

```bash
./build/bin/crispasr --backend canary -m auto -f samples/jfk.wav \
    --max-len 50 --split-on-punct -osrt --no-auto-aligner
```

The implicit-enable line goes to stderr (suppressed under
`--no-prints`) so it doesn't perturb stdout subtitle parsing in
upstream tools like SubtitleEdit.

### Standalone alignment — `--align-only` (issue #217)

Aligns pre-existing text against audio without running ASR first.
Accepts plain text (via `--ref-text` or `--text-file file.txt`), an
unaligned `.srt` file, or a CrispASR `--output-json` transcription
(`.json`). `--text-file -` reads from stdin and sniffs the format
(#317). Works with all three aligner families: canary-ctc,
wav2vec2/hubert, qwen3-forced-aligner.

For `.srt` input the cue structure is preserved: the cue texts are
aligned as one transcript and each cue is re-emitted with corrected
timings (first/last aligned word of that cue). So a mistimed subtitle
file goes in, and the same subtitles come out re-timed.

```bash
# Re-time an unaligned/mistimed SRT (same cues, corrected timestamps):
crispasr --align-only -am auto --auto-download -f audio.wav \
    --text-file subtitles.srt --align-output retimed.srt

# Align a transcript against audio, output word-level SRT:
crispasr --align-only -am auto --auto-download -f audio.wav \
    --ref-text "And so my fellow Americans ask not what your country can do for you"

# Align a plain .txt (one subtitle line per text line) into cue-level SRT:
crispasr --align-only -am auto --auto-download -f audio.wav \
    --text-file transcript.txt --align-granularity segment \
    --align-output aligned.srt

# JSON with per-segment + nested per-word timings:
crispasr --align-only -am canary-ctc-aligner-q4_k.gguf -f audio.wav \
    --text-file subtitles.srt --align-format json
```

No ASR model (`-m`) is required. Output formats: `srt` (default),
`json` (start/end in seconds), `plain` (tab-separated). Destination:
stdout (default) or `--align-output <path>`.

Granularity is controlled by `--align-granularity`:

| Value | Meaning |
|---|---|
| `auto` (default) | `segment` for `.srt` / `.json` input, `word` otherwise |
| `segment` | one output entry per input SRT cue / JSON segment / non-empty `.txt` line, re-timed from the word alignment; JSON nests the per-word timings under each segment |
| `word` | one output entry per aligned word (the pre-0.8.9 behaviour, also for `.srt` input) |

### Aligner model options

Any of these resolves via `-am <name> --auto-download`. All are
permissively licensed (no NC restriction):

| `-am` name | Family | Size (q4_k) | Languages | Upstream license |
|---|---|---|---|---|
| `canary-ctc-aligner` (= `auto`) | FastConformer-CTC (canary-1b-v2 aux) | ~442 MB | 25 European | CC-BY-4.0 |
| `fastconformer-aligner[-en]` / `fastconformer-ctc` | FastConformer-CTC standalone | ~83 MB | en | CC-BY-4.0 |
| `fastconformer-{aligner,ctc}-{en-pc,es,fr,it,nl,pl,ru,ua,hr,be,ar,fa,ka,hy,uz,kk-ru,de}` | FastConformer hybrid CTC branch | ~82 MB | per-language (+punct/caps except fa, kk-ru) | CC-BY-4.0 |
| `parakeet-ctc-0.6b` / `parakeet-ctc-1.1b` | FastConformer-CTC | ~455 / ~795 MB | en | CC-BY-4.0 |
| `wav2vec2-aligner-{en,de,fr,es,it,ja,zh,nl,uk,…}` | wav2vec2/XLSR CTC | ~212–300 MB | per-language (incl. CJK) | Apache-2.0 |
| `qwen3-forced-aligner` / `qwen3-fa` | Qwen3 timestamp head | ~500 MB | multilingual | Apache-2.0 |

Everything with GGUF arch `canary-ctc` (the whole NeMo
`stt_*_fastconformer_ctc_*` standalone family, plus CTC branches
extracted from the `stt_*_fastconformer_hybrid_large_pc` hybrids via
`models/convert-stt-fastconformer-ctc-to-gguf.py`) loads through the
default aligner dispatch — the ~80 MB FastConformer models are the
smallest/fastest aligner option. The popular MMS-based aligners are
deliberately absent (weights CC-BY-**NC**-4.0), as is the Portuguese
hybrid `stt_pt_fastconformer_hybrid_large_pc` — the only NC release in
NVIDIA's per-language hybrid fleet.

### Granite word-level timestamps and `--max-len` (#205)

`--backend granite` (Granite Speech 4.1 PLUS variant) now requests
word-level timestamps from the model whenever the output format needs
them: `--max-len`, `-osrt`, `-ovtt`, or `--split-on-punct`. Previously,
word timestamps were only generated for `--output-wts` and `-ojf`,
causing `--max-len` to silently have no effect.

```bash
./build/bin/crispasr --backend granite \
    -m granite-speech-4.1-2b-plus-q4_k.gguf \
    -f audio.wav --max-len 50 -osrt
```

> **Note:** Qwen3 does not support word-level timestamps, so `--max-len`
> segment splitting is not available with that backend.

## Sampling / decoding (whisper + LLM backends)

| Flag | Meaning |
|---|---|
| `-tp F`, `--temperature F` | Sampling temperature. `0` = pure argmax (default, bit-identical). `> 0` enables multinomial sampling for whisper, voxtral, voxtral4b, qwen3, granite |
| `--seed N` | RNG seed for sampling. `0` = non-deterministic. Used by temperature-sampling ASR backends and TTS backends that sample; CLI values override backend-specific env seeds |
| `-bo N`, `--best-of N` | Number of best candidates to keep when temperature > 0 (whisper + some AR backends) |
| `-bs N`, `--beam-size N` | Beam search width. Unset means greedy; whisper substitutes 5 when beam search is engaged. Supported on: whisper, parakeet, nemotron, canary, canary-qwen, cohere, granite, qwen3, voxtral, voxtral4b, glm-asr, kyutai-stt, moonshine, moonshine-streaming, firered-asr, omniasr, gemma4-e2b, funasr, sensevoice, granite-nle, moss-audio, moss-transcribe, moss-diarize, higgs-stt, ark-asr, mimo-asr, m2m100, madlad/t5. Also lfm2-audio (stub). Not applicable to paraformer (NAR) |
| `-tpi F`, `--temperature-inc F` | Whisper temperature-fallback increment |
| `-nf`, `--no-fallback` | Disable temperature fallback (equivalent to `--temperature-inc 0`) |
| `--frequency-penalty F` | Opt-in repeated generated-token penalty for autoregressive ASR backends (`0.0` disabled). Applied to generated output tokens before greedy/sampling selection. |
| `--grammar FNAME` | GBNF grammar file for constrained whisper decoding |
| `--grammar-rule NAME` | Top-level rule name in the grammar. No default — grammar-constrained decoding only engages when both `--grammar` and `--grammar-rule` are given |
| `--grammar-penalty F` | Scales down logits of tokens that violate the grammar (default: `100.0`) |
| `--alt` | Show alternative token candidates with per-token probabilities (whisper + any backend that emits token alternatives) |
| `--alt-n N` | Number of alternative token candidates per step (default: `3`) |
| `--print-confidence` | After the transcript, print each segment's tokens with an inline confidence annotation (`word[NN%]`). Works for any backend that emits per-token confidence (whisper, parakeet, moonshine, …); backends without token info print plain text |
| `--prompt STR` | Initial prompt for whisper |

## Hotwords / contextual biasing

Supply domain-specific vocabulary that the ASR should prefer when in
doubt — names, jargon, product terms, place names. Works by boosting
the log-probability of tokens that continue a matching hotword prefix
during CTC/TDT decoding (shallow fusion via an Aho-Corasick trie).

| Flag | Meaning |
|---|---|
| `--hotwords "A,B,C"` | Comma-separated hotword list |
| `--hotwords-file FILE` | One hotword per line |
| `--hotwords-boost F` | Per-token log-prob boost (default: `2.0`) |

Per-word boost suffix: `"Berenz^5.0,NVIDIA^3.0,plain"`.

### Backend coverage

| Mechanism | Backends |
|---|---|
| **CTC-WS trie (Phase A)** — token-level logit boost during CTC/TDT decode | parakeet (CTC + TDT) |
| **LLM prompt injection (Phase B)** — hotwords appended to the system/instruction prompt | qwen3-asr, voxtral |
| **Free-form prompt injection** — separate `--context` flag, not `--hotwords` (see below) | vibevoice |
| Not applicable | voxtral4b (fixed streaming prompt), granite-nle (NAR, no text prompt), funasr (hardcoded prompt), whisper (use `--prompt` instead) |

### VibeVoice-ASR: `--context` (free-form prompt injection)

VibeVoice-ASR uses its own flag, `--context "TEXT"`, rather than
`--hotwords` — the model's prompt format accepts free-form prose/metadata
(names, organizations, topic context) instead of a structured hotword
list with per-word boost weights. The text is spliced directly into the
model's instruction prompt, matching the `context_info` parameter in
microsoft/VibeVoice's `vibevoice_asr_processor.py`. Empty or
whitespace-only `--context` behaves identically to omitting the flag
(same prompt, byte-for-byte).

```bash
crispasr --backend vibevoice -m auto -f meeting.wav \
    --context "ACME Corp, John Smith, Q3 earnings"
```

### Example

```bash
# Boost rare names during parakeet CTC decode
crispasr --backend parakeet -m auto -f meeting.wav \
    --parakeet-decoder ctc \
    --hotwords "Berenz,Acme Corp,GPU-PB"

# Same hotwords for qwen3-asr (injected into LLM system prompt)
crispasr --backend qwen3 -m auto -f meeting.wav \
    --hotwords "Berenz,Acme Corp,GPU-PB"

# Boost from a file with custom boost values
crispasr --backend parakeet -m auto -f meeting.wav \
    --hotwords-file names.txt --hotwords-boost 3.0
```

### Beam search + hotwords (recommended for domain vocabulary)

With greedy decode, a boosted token that loses to blank or a common
word at one frame is gone forever. Beam search (`-bs 2` or higher)
keeps alternative hypotheses alive, so a boosted rare term can survive
in a lower-ranked beam and win later when more confirming audio
arrives. The combination is especially effective for medical, legal,
or brand-name vocabulary where greedy hotword boosting alone is
insufficient.

```bash
# Beam search + hotwords for medical transcription
crispasr --backend parakeet -m auto -f consultation.wav \
    --hotwords "metformin,lisinopril,atorvastatin" -bs 4
```

The overhead is negligible: parakeet's TDT beam search only multiplies
the cheap predictor+joint steps (~10 KB LSTM state per beam), while
the encoder dominates wall time. Measured overhead: ~3 % at beam=2,
~12 % at beam=4 on 60 s audio.

### How it works (CTC-WS)

An Aho-Corasick multi-pattern trie is built from the hotword strings
by tokenizing each word through the backend's SentencePiece vocab.
Before each frame's argmax (or each beam expansion step), the trie
checks which tokens continue an active hotword prefix match and adds
the boost to their logits. After token selection, the trie state
advances based on the emitted token. Each beam hypothesis maintains
its own trie position. The overhead is O(1) per frame — no measurable
impact on throughput.

## Language detection (LID)

CrispASR has **two distinct LID paths**: audio-LID (decides what language
the audio is in, before/during ASR) and text-LID (decides the language of
a transcript or arbitrary UTF-8 string).

### Audio LID (pre-/in-ASR)

| Flag | Meaning |
|---|---|
| `-l auto`, `--detect-language` | Auto-detect the input language. Backends without native lang-detect (cohere, canary, granite, voxtral, voxtral4b) get it via the LID pre-step |
| `--lid-backend NAME` | Audio-LID provider: `whisper` (default), `silero` (95 langs, 16 MB), `ecapa` (107 or 45 langs, 40-43 MB), `firered` (120 langs, 544 MB), `probe` (ask the ASR model itself — cohere; no second model, and it can only return a language that model supports), or `off` / `none` |
| `--lid-model FNAME` | Override the audio-LID model path (default: auto-downloads `ggml-tiny.bin` ~75 MB on first use) |

**Silero evidence gate (#409).** On hard inputs — heavily compressed audio
(MP3/OGG artifacts), very short or band-limited speech — the 16 MB
silero-lang95 classifier's whole logit vector deflates and the softmax then
renormalizes noise into a confident-looking wrong answer (`yo` at softmax
p=0.58 on an MP3-coded JFK clip; `be` on clearly French audio in #409 — the
upstream ONNX reference produces the same garbage, so this is the model, not
the port). The separating signal is the RAW top-logit magnitude (free-energy
OOD scoring): correct detections sit at ~`-1.1`, every observed failure at
`-3.35` or below. A silero answer whose top logit is below `-2.0` is treated
as **inconclusive**: discarded with a stderr note, and LID falls back to
whisper-tiny, which is markedly more robust on such audio. Tune or disable
with `CRISPASR_SILERO_LID_MIN_LOGIT` (e.g. `-3.0` to relax, `-999` to accept
everything).

### Text LID (post-ASR / standalone)

Runs on a transcript or any UTF-8 string. The dispatcher in
`src/text_lid_dispatch.{h,cpp}` peeks the GGUF's `general.architecture`
and picks fastText (GlotLID-V3 / LID-176) or Google CLD3 — same flag,
same binary, any text-LID GGUF.

| Flag | Meaning |
|---|---|
| `--lid-on-transcript FNAME` | After ASR, run a text-LID GGUF on the assembled transcript and emit `lang=<code>\tconf=<score>\tbackend=<name>` to stderr. Accepts a path or `auto[:cld3\|glotlid\|lid-fasttext176]` (default `cld3`, auto-downloaded). Errors are logged but never fail the run |

Standalone CLI: `crispasr-lid` (separate binary, ships with every build):

| Flag | Meaning |
|---|---|
| `-m`, `--model PATH\|auto[:variant]` | Path or auto-download key. `auto` / `auto:cld3` → `cstr/cld3-GGUF` (default). `auto:glotlid` → `cstr/glotlid-GGUF`. `auto:lid-fasttext176` → `cstr/fasttext-lid176-GGUF`. A bare canonical filename (e.g. `cld3-f16.gguf`) is also looked up in the registry and downloaded if missing |
| `--text STR` | Input text (otherwise stdin) |
| `-k`, `--topk N` | Top-k predictions, one `label\tscore` per line (default 1) |
| `--quiet` | Suppress the trailing `backend=… variant=… dim=… N labels` summary |

First-use download lands in `~/.cache/crispasr/` (or
`$CRISPASR_CACHE_DIR` if set). Subsequent runs are instant.

Example — same input, three different label spaces:

```bash
$ crispasr-lid -m cld3-f16.gguf --text "Bonjour le monde, comment allez-vous?"
fr	0.999983
crispasr-lid: backend=lid-cld3 variant=Google CLD3 dim=80 109 labels

$ crispasr-lid -m lid-glotlid-f16.gguf --text "Bonjour le monde, comment allez-vous?"
fra_Latn	0.983436
crispasr-lid: backend=lid-fasttext variant=glotlid-v3 dim=256 2102 labels

$ crispasr-lid -m lid-fasttext176-f16.gguf --text "Bonjour le monde, comment allez-vous?"
fr	0.958174
crispasr-lid: backend=lid-fasttext variant=fasttext-lid176 dim=16 176 labels
```

CLD3 is the smallest+fastest option (440 KB F16, 109 langs, Apache-2.0)
but inherits CLD3's known short-input quirks (`"Hello world"` lands on
`ky 0.72` consistently — too short to disambiguate; the C++ port
faithfully reproduces upstream's `pycld3` behaviour). GlotLID-V3 covers
the most languages (2102 ISO 639-3 + script). LID-176 is **CC-BY-NC-4.0
(non-commercial)** — pick CLD3 or GlotLID for commercial distribution.

## Diarization

Diarization assigns a speaker label to every transcribed segment. Every method
below works with every ASR backend.

**Start with `foxnose`** unless you have a reason not to: it is the most
accurate in-tree path, needs no Python and no external binary, and estimates
the number of speakers instead of requiring you to know it.

Measured on 8 VoxConverse dev files against human labels (whisper-tiny
segments, 0.25 s collar, `tools/der_score.py`):

| Method | Mean DER |
|---|---|
| `foxnose` + WeSpeaker | **7.3 %** |
| `pyannote` + TitaNet | 7.8 % |

Scored the way the upstream reference scores itself — on diarization turns
rather than ASR segments — `foxnose` is at parity with it: 3.18 % against
3.07 %.

```bash
# Recommended: foxnose (auto-downloads the 24 MB WeSpeaker GGUF)
crispasr -m auto --backend cohere -f podcast.wav \
    --diarize --diarize-method foxnose --diarize-embedder auto -ojf

# Pin the speaker count when you know it (skips estimation entirely)
crispasr -m auto --backend cohere -f podcast.wav \
    --diarize --diarize-method foxnose --diarize-embedder auto \
    --diarize-num-speakers 2 -ojf

# Native GGUF pyannote (no Python, no sherpa-onnx)
crispasr -m auto --backend cohere -f podcast.wav \
    --diarize --diarize-method pyannote --sherpa-segment-model auto -ojf

# Same + embedder-based global speaker IDs (recommended for >2 speakers
# or long-form audio where pyannote local-track IDs drift)
crispasr -m auto --backend cohere -f podcast.wav \
    --diarize --diarize-method pyannote --sherpa-segment-model auto \
    --diarize-embedder auto -ojf
```

### `--diarize-method NAME`

| Method | Stereo / mono | What it does |
|---|---|---|
| `energy` | stereo | `|L|` vs `|R|` per segment; the louder channel wins (1.1× margin) |
| `xcorr` | stereo | TDOA via cross-correlation, ±5 ms search window |
| `vad-turns` | mono | Alternates 0/1 every >600 ms gap (mono-friendly proxy) |
| `foxnose` | mono | **Recommended.** WeSpeaker ResNet34-LM embeddings over sliding windows -> PCA + full-covariance GMM/BIC speaker counting -> Ng-Jordan-Weiss spectral clustering -> Viterbi temporal smoothing. Estimates the speaker count; `--diarize-num-speakers N` pins it. Needs `--diarize-embedder auto` (or a WeSpeaker GGUF path). Weights are CC-BY-4.0 — see `THIRD_PARTY_NOTICES.txt` |
| `pyannote` | mono | Native GGUF pyannote-seg-3.0; runs once globally over the full audio, splits ASR segments at speaker-turn boundaries when per-word timestamps exist. Auto-downloads the GGUF via `--sherpa-segment-model auto` |
| `sherpa` / `ecapa` | mono | External `sherpa-onnx` subprocess with segmentation + speaker-embedding model. Since #110, runs once globally over the full audio (not per-slice), producing consistent speaker IDs across the whole file. Splits ASR segments at speaker-turn boundaries when per-word timestamps exist. Requires `--sherpa-bin`, `--sherpa-segment-model`, `--sherpa-embedding-model` |

Bare `--diarize` (no `--diarize-method`) defaults to `energy` for stereo
input and `vad-turns` for mono — the historical behaviour.

Supporting flags:

| Flag | Meaning |
|---|---|
| `--diarize-embedder MODEL` | Speaker-embedding model (`auto` / `titanet` / `indextts` / `ecapa` / path). Empty = off |
| `--diarize-cluster-threshold F` | Cosine merge threshold (default 0.5) — consulted only when passed explicitly (#326) |
| `--diarize-max-speakers N` | Hard cap on the cluster count (default 8) |
| `--diarize-num-speakers N` | Pin the speaker count outright, skipping estimation (`foxnose`; `0` = estimate) |
| `--sherpa-bin PATH` | `sherpa-onnx-offline-speaker-diarization` binary (default: found on `PATH`) |
| `--sherpa-segment-model PATH` | Segmentation model for the `sherpa`/`pyannote` paths (`auto` downloads the GGUF) |
| `--sherpa-embedding-model PATH` | Speaker-embedding ONNX for the `sherpa` path |
| `--sherpa-num-clusters N` | Sherpa cluster count (`0` = auto, the default) |
| `--speaker-db DIR`, `--enroll-speaker NAME` | Persistent named-profile path — see [diarization-speakers.md](diarization-speakers.md) |
| `--speaker-db-consent` | REQUIRED for the named-profile path (GDPR Art. 9 attestation) |
| `--expect-speakers "A,B"` | Closed roster REQUIRED with `--speaker-db`; matching is a claimed-participant confirmation, never an open search |
| `--speaker-threshold F`, `-st F` | Cosine threshold for cluster→profile matching (default 0.7); below it a cluster stays anonymous |
| `--titanet-model PATH`, `--spk-model PATH` | TitaNet GGUF used for enrollment / identification embeddings (default: auto-download) |

> **Note on global execution (issue #110).** Both `pyannote` and
> `sherpa`/`ecapa` now run once on the full audio before any VAD/ASR
> slicing. The global speaker-turn timeline is then used to assign
> speakers to each per-slice ASR segment, ensuring speaker IDs are
> consistent across the entire file. Before #110, `sherpa`/`ecapa`
> ran per-slice, producing local IDs that could reset between slices.

#### Trading accuracy for throughput (`CRISPASR_DIARIZE_SPAN_EMBED=1`)

`foxnose` slides a 1.2 s window at a 0.6 s hop, so every sample goes through the
embedding network twice. Setting `CRISPASR_DIARIZE_SPAN_EMBED=1` runs one
network pass per *span* of 32 windows and takes each window as a slice of it.

Measured on the VoxConverse dev shard: **1.78x less diarization CPU** (66.0 s ->
37.0 s on an 85 s file), for **+0.30 mean DER** (7.32% -> 7.62%). Six of eight
files come out identical; one borderline file loses a speaker, because the
slightly different embeddings flip the speaker-count estimate from 4 to 3.

Off by default — accuracy is the better default for a diarizer. Turn it on when
you are throughput-bound and can accept that. Span size
(`CRISPASR_DIARIZE_SPAN_WINDOWS`, default 32) does **not** affect the accuracy
cost — it is identical from N=2 to N=32 — so there is nothing to tune: larger is
simply faster.

### `--diarize-embedder MODEL` — globally stable speaker IDs

The pyannote method's per-pass local tracks (spk0 / spk1 / spk2) are
not anchored to physical speakers across an entire file. Pass
`--diarize-embedder` to extract a speaker embedding per segment and
cluster on cosine similarity, producing IDs that are consistent across
the whole audio.

| Alias | Backend | Dim |
|---|---|---|
| `auto`, `titanet` | TitaNet-Large | 192 |
| `indextts`, `indextts-bigvgan`, `ecapa` | IndexTTS-BigVGAN ECAPA-TDNN | 512 |
| `<path>.gguf` | Dispatched by filename (`indextts` substring -> IndexTTS, otherwise TitaNet) | — |

The interface is pluggable: add a new adapter by subclassing
`CrispasrSpeakerEmbedder` in `src/crispasr_speaker_embedder.cpp` and
extending the factory's dispatch.

`--diarize-max-speakers N` (default 8) bounds the search. `--diarize-num-speakers N`
pins the count outright and skips estimation.

> **`--diarize-cluster-threshold` only applies if you pass it (#326).** The
> default path estimates the speaker count with the same spectral clusterer
> `foxnose` uses, and a cosine merge threshold means nothing to it. It used to
> be consulted always — single-linkage agglomerative at a fixed 0.5 — and
> because single linkage chains, the merge loop never reached the threshold and
> `--diarize-max-speakers` silently became the answer rather than a ceiling:
> on 4 of 8 VoxConverse dev files it returned exactly 8 speakers against 4-7
> real ones. Passing the flag explicitly still selects the old agglomerative
> path for anyone who tuned it; leaving it alone gets the estimator.

This clustering is **session-scoped**: embeddings are computed per
recording and discarded, labels are anonymous `(speaker N)`, and nothing
is persisted — no voiceprint database, no names. It identifies no one; it
only makes the labels stable within one file.

### `--diarize-speakers` — opt-in convenience alias

`--diarize-speakers` is shorthand for
`--diarize --diarize-method pyannote --diarize-embedder auto` (each part is
only filled in if you did not set it yourself).
Use it when you just want stable per-recording speaker labels without
remembering the flag combination:

```bash
crispasr -m auto --backend cohere -f meeting.wav --diarize-speakers -ojf
```

> **Named profiles are a separate, deliberately opt-in feature.** The
> `--enroll-speaker` / `--speaker-db` flags build a *persistent* database
> of voiceprints linked to real names — biometric special-category data
> under GDPR Art. 9, with consent/transparency obligations. Matching is a
> **closed-roster confirmation** applied per global speaker cluster: it
> requires `--speaker-db-consent` AND `--expect-speakers "NameA,NameB"`
> (the enrolled participants you assert are present), only runs on
> recorded files (never streaming), and unmatched clusters keep their
> anonymous `(speaker N)` labels. An open "identify anyone in the
> database" 1:N scan is deliberately unsupported (EU AI Act, Annex III
> 1(a)). The session-scoped clustering above does **not** persist
> anything and needs no consent flag. See
> [diarization-speakers.md](diarization-speakers.md) for the full
> comparison and the legal/privacy posture.

### Output shape

Each segment carries the label as the string `"(speaker N) "` in
`crispasr_segment.speaker`. Output writers surface it as:

* **txt / wts**: prefixed inline: `(speaker 0) hello world`
* **srt**: prefixed inline
* **vtt**: `<v Speaker 0>` markup
* **json**: per-segment `"speaker": "0"` (stripped of the `(speaker )` wrapper)

When the embedder is enabled the labels are global cluster IDs;
otherwise they are pyannote-local track IDs.

### What changed in 0.6.6+ (issue [#107](https://github.com/CrispStrobe/CrispASR/issues/107))

* `--diarize-method pyannote` is now actually correct end-to-end:
  pyannote-seg runs ONCE over the full audio (cross-slice consistent
  IDs), overlap classes contribute to both involved speakers, ASR
  segments split at speaker-turn boundaries when word timestamps
  exist.
* `--diarize-method X` now also works with the whisper backend
  (previously silently ignored — only the upstream stereo-energy
  diarize ran).
* Output writers prefer the unified `segs[i].speaker` over the legacy
  stereo-only energy estimator. Mono input now gets a `speaker` field
  in JSON when an explicit method is set.

## LLM-backend specific

| Flag | Meaning |
|---|---|
| `-am FNAME`, `--aligner-model FNAME` | CTC aligner GGUF for word-level timestamps |
| `-n N`, `--max-new-tokens N` | Max tokens the LLM may generate (default 512) |
| `--frequency-penalty F` | Penalize repeated generated token IDs on supported autoregressive backends. Useful with `-n` as a retry knob after cap-triggered degeneration. |
| `--ask "TEXT"` | Replace the transcription instruction with a free-form question about the audio (audio-QA). Honoured by ark-asr, glm-asr, granite, higgs-stt, mimo-asr, voxtral |
| `--prefix-text "TEXT"` | Seed the assistant turn with an already-decoded transcript so the model continues from it instead of re-decoding; the output is the continuation only (granite-speech, #205) |
| `--context "TEXT"` | Free-form prompt/context injection — vibevoice-asr only; see [Hotwords](#hotwords--contextual-biasing) |

## Multi-language / translation

There are **three distinct translation paths**, each with its own
flags. Pick by what you have for input and what you need out:

| You have | You want | Use |
|---|---|---|
| Audio in language X | Translated text in English | `-tr` / `--translate` (audio→EN-text on whisper, canary, granite, voxtral, qwen3) |
| Audio in language X | Translated text in language Y | `-sl X -tl Y` (audio AST on canary, granite-4.1, qwen3) |
| Plain text in language X | Translated text in language Y | `--text "..." -sl X -tl Y --backend m2m100` (text→text only) |

### Audio-side translate (`--translate`, `-sl`/`-tl`)

| Flag | Meaning |
|---|---|
| `-sl LANG`, `--source-lang LANG` | Source language (canary AST source; explicit pin overrides LID) |
| `-tl LANG`, `--target-lang LANG` | Target language (canary AST; set different from `-sl` for X→Y translation) |
| `-tr`, `--translate` | Translate to English (whisper, canary) — boolean toggle, no string arg |

On a **TTS** backend the same two flags mean the synthesis languages: `-tl` is
the language to SPEAK (overriding `-l`), and `-sl` is the language the
`--voice` cloning reference is spoken in, which cosyvoice3 uses to decide
whether to switch to cross-lingual synthesis. See
[`docs/tts.md`](tts.md#output-language-and-cross-lingual-cloning--tl---sl).

### Punctuation / truecasing / output flushing

| Flag | Meaning |
|---|---|
| `--no-punctuation` | Disable punctuation in the output. Native for cohere/canary, post-processed for everyone else |
| `--punc-model FNAME` | Punctuation-restoration GGUF: `auto`, `firered`, `fullstop`, `punctuate-all`, or a path |
| `--truecase-model FNAME` | Truecaser: `auto` (German) or a path to a `.bin` |
| `--flush-after N` | Flush SRT to stdout every N segments (`0` = all at the end, the default) |

### Text-to-text translate (m2m100, WMT21, MADLAD-400)

Three text-to-text translation backends, all driven by `--text "..."
-sl <src> -tl <tgt>`:

| Backend | Model | Languages | Status |
|---|---|---|---|
| `m2m100` | [`facebook/m2m100_418M`](https://huggingface.co/cstr/m2m100-418m-GGUF) — 12L+12L transformer, ~502 MB Q8_0 | 100, any-to-any | ✓ production-ready (en→de exact match to Python ref) |
| `m2m100-wmt21` | [`facebook/wmt21-dense-24-wide-en-x`](https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-GGUF) + [`facebook/wmt21-dense-24-wide-x-en`](https://huggingface.co/cstr/wmt21-dense-24-wide-x-en-GGUF) — 24L+24L wider, ~2.5 GB Q4_K each | English ↔ 7 languages (separate `en-x` / `x-en` checkpoints) | ✓ runs on m2m100 runtime; vocab fix in 7f48bad |
| `madlad` (alias `t5`) | [`google/madlad400-3b-mt`](https://huggingface.co/cstr/madlad400-3b-mt-GGUF) — T5 12L+12L, ~1.9 GB Q4_K | 419 | ✓ tokens match Python SP bit-by-bit; outputs match HF reference |

```bash
# m2m100 base — production
./build/bin/crispasr --backend m2m100 -m auto \
    --text "Hello world, how are you today?" \
    -sl en -tl de
# → Hallo Welt, wie bist du heute?

# WMT21 (4.7B, English-to-X, auto-downloads ~2.5 GB)
./build/bin/crispasr --backend m2m100-wmt21 -m auto \
    --text "The president said he would not attend." \
    -sl en -tl de

# MADLAD-400 (419 languages — output matches Python SP)
./build/bin/crispasr --backend madlad -m auto \
    --text "Hello world." \
    -sl en -tl ta
```

For MADLAD-400 the source-language tag is informational (T5 encoders
are language-agnostic); the adapter synthesises the `<2xx>` target-
language prefix from `-tl` automatically. m2m100 / WMT21 use both
`-sl` and `-tl`.

| Flag | Meaning |
|---|---|
| `--text "TEXT"` | Plain text to translate. Output goes to stdout. |
| `-sl LANG`, `-tl LANG` | Source / target — same flags as audio AST; ISO-639-1 codes. |
| `--translate-max-tokens N` | Max output tokens (default 256). |
| `--tr-sl LANG`, `--tr-tl LANG` (long: `--translate-source-lang` / `--translate-target-lang`) | Translator-stage source/target. Falls back to `-sl`/`-tl`. Only matters in 2-stage pipelines where the primary backend's `-sl`/`-tl` mean something else (the primary's AST source/target). 2-stage piping (ASR → m2m100) needs `--translate-model PATH` — that's a follow-up; the override flags are plumbed but the standalone path is what's exercised today. |

## Threading / processors

| Flag | Meaning |
|---|---|
| `-t N`, `--threads N` | Threads per inference call (default `min(4, nproc)`) |
| `-p N`, `--processors N` | Run N parallel decoder states (whisper only — uses `whisper_full_parallel`) |
| `--no-gpu` / `--device N` | Disable GPU or pin to GPU N |

## Whisper-only flags

These work both with the historical default whisper code path AND
with `--backend whisper`. The historical path retains a few extras
unique to it (`-owts` karaoke, full-mode JSON DTW tokens) — pass a
`ggml-*.bin` model without `--backend` to get them.

| Flag | Meaning |
|---|---|
| `--diarize` | Generic diarization post-step. Stereo defaults to `energy`, mono to `vad-turns`. Pair with `--diarize-method` for pyannote / sherpa / etc. — see [Diarization](#diarization). |
| `-tdrz`, `--tinydiarize` | TinyDiarize speaker turn detection (upstream whisper feature, separate from `--diarize`) |
| `--carry-initial-prompt` | Forward `--prompt` across audio chunks |
| `-dtw MODEL`, `--dtw MODEL` | Output DTW token-level timing in `-ojf` JSON |
| `-fa`/`--flash-attn`, `-nfa`/`--no-flash-attn` | Force flash-attn on / off (default: on) |
| `--suppress-regex REGEX` | Suppress tokens whose detokenized text matches the regex |
| `-sns`, `--suppress-nst` | Suppress non-speech tokens |
| `-owts`, `--output-words` | Karaoke-style word-timestamp WTS output |
| `-fp`, `--font-path` | Monospace font for the karaoke video script (default: the macOS Courier New Bold path) |
| `-mc N`, `--max-context N` | Max text-context tokens carried between segments (`-1` = model default) |
| `-ac N`, `--audio-ctx N` | Audio context size (`0` = all) |
| `-wt F`, `--word-thold F` | Word-timestamp probability threshold (default 0.01) |
| `-on N`, `--offset-n N` | Segment index offset |
| `-sow`, `--split-on-word` | Split on word rather than on token |
| `-dl`, `--detect-language` | Detect the language and exit |
| `-ps`, `--print-special` | Print special tokens |
| `-pp`, `--print-progress` | Print progress |
| `-ls`, `--log-score` | Log the best decoder token scores |
| `-oved D`, `--ov-e-device D` | OpenVINO device used for encode inference (default `CPU`) |

For the full list of upstream whisper flags see `crispasr --help`
when invoked with a `ggml-*.bin` model loaded.

## Pitch / F0 estimation (`--pitch`)

Pitch tracking is its own task, like `--separate`: audio goes in, a pitch
track comes out — not transcript segments. `--pitch` therefore routes to its
own dispatcher before any ASR backend is constructed.

```bash
# Auto-download the default model (crepe tiny) and print a pitch track
crispasr --pitch -m auto --auto-download -f samples/jfk.wav

# Explicit model, JSON output, 20 ms hop
crispasr --pitch -m crepe-full-f16.gguf --pitch-format json --pitch-hop-ms 20 -f audio.wav
```

The backend (`crepe`) is auto-detected from the GGUF `general.architecture`,
so you never name it; input is decoded to CREPE's native 16 kHz mono
automatically.

| Flag | Meaning |
|---|---|
| `--pitch` | Enable the pitch task |
| `--pitch-format FMT` | `text` (default) or `json` |
| `--pitch-hop-ms MS` | Analysis hop in milliseconds (default 10, CREPE's reference) |

Default output is one tab-separated line per frame — `time_ms`, `f0_hz`,
`voiced_prob` — so it pipes straight into `awk` / `cut`:

```
0.0	123.456	0.9312
10.0	123.501	0.9370
```

`--pitch-format json` emits `{file, model, n_frames, frames: [...]}` with the
same three fields per frame.

**Models** (`cstr/crepe-GGUF`, MIT):

| Registry key | File | Size | Notes |
|---|---|---|---|
| `crepe` / `crepe-tiny` | `crepe-tiny-f16.gguf` | ~1.0 MB | **default** — RTF 0.28 on Metal |
| `crepe-full` | `crepe-full-f16.gguf` | ~44.5 MB | RTF 2.0 on Metal; offline use |

CREPE is compute-heavy per frame (282 GFLOP per second of audio for `full`,
7.3 for `tiny`), so tiny is the shipping default and the GPU path is not
optional — on CPU even tiny runs at roughly RTF 2.4.

## Beat tracking (`--beats`)

A standalone task: audio in, a beat and downbeat grid out. Like `--chords`,
`--pitch` and `--separate` it routes to its own dispatcher before any ASR
backend is built.

> **No DBN, and that is the point.** Nearly every published beat tracker
> post-processes its framewise logits with madmom's Dynamic Bayesian Network,
> which is Böck-patented and licensed non-commercially. **Beat This!** (CPJKU,
> ISMIR 2024) reaches state-of-the-art *without* one, and its postprocessing
> here is peak-picking only. Both the code and the published weights are MIT,
> and the dependency list contains no part of madmom — so unlike `--chords`
> there is no licence gate on this task.

```bash
# Explicit model
crispasr --beats -m beat-this-f16.gguf -f song.wav

# JSON, including the median-interval tempo estimate
crispasr --beats -m beat-this-f16.gguf --beats-format json -f song.wav
```

The backend (`beat-this`) is auto-detected from the GGUF
`general.architecture`; input is decoded to the model's native 22.05 kHz mono
automatically, and long files are internally split into 1500-frame chunks with
a 6-frame border overlap, matching upstream so results do not drift at seams.

| Flag | Meaning |
|---|---|
| `--beats` | Enable the beat task |
| `--beats-format FMT` | `text` (default) or `json` |

Default output is one tab-separated line per beat — `time_sec` and either
`beat` or `downbeat` — which is the `.beats` layout the MIR beat datasets
(Ballroom, GTZAN-Rhythm) use, so it drops straight into `mir_eval.beat`:

```
0.000	downbeat
0.540	beat
1.020	beat
1.520	beat
2.020	downbeat
```

**Every downbeat is also emitted as a beat.** The postprocessor snaps each
downbeat onto its nearest detected beat, so downbeats are a strict subset of
beats and you never have to merge two lists to reconstruct the grid.

## Piano transcription (`--piano`)

Another standalone task: audio in, **note events** out — onset, offset, MIDI
pitch, name and velocity. Routes to its own dispatcher before any ASR backend
is built, like `--pitch` / `--chords` / `--separate`. Three backends share the
surface, selected by GGUF architecture: **`piano-transcription`** (ByteDance
CRNN, 88-key piano, the higher-accuracy piano specialist), **`basic-pitch`**
(Spotify, ~110 KB, polyphonic any-instrument, #250 — no pedal events), and
**`mt3`** (Google Magenta, ~96 MB, MULTI-INSTRUMENT — every note carries a
General MIDI program and drums are a separate class; alias
`music-transcription`).

`--piano-format` takes `text` (default), `json`, or **`midi`**. The MIDI form
writes a format-1 Standard MIDI File — one track per program, drums on GM
channel 10 — to the matching `-of` name, or the input path with its extension
replaced. That is the only form that reaches a DAW or notation editor; text and
JSON are for inspection and diffing.

MT3's per-note program survives into `--piano-format json` (`program`,
`instrument`, `is_drum` keys) and adds one trailing column to the text form
(`prog=<n>` or `drum`) after the existing five. It is NOT carried by the
session C ABI, whose note quad is fixed-width.

```bash
crispasr --piano -m auto --auto-download -f piano.wav
crispasr --piano --piano-format json -m piano-transcription-f16.gguf -f piano.wav
# any-instrument polyphonic, tiny model:
crispasr --piano --backend basic-pitch -m auto --auto-download -f guitar.wav
# multi-instrument, straight to a MIDI file:
crispasr --piano --backend mt3 -m auto --auto-download -f band.wav --piano-format midi
```

Default output is one tab-separated line per note — `onset_sec`, `offset_sec`,
`midi`, `name`, `velocity`:

```
1.406	1.455	60	C4	32
2.597	4.990	42	F#2	74
```

Deliberately **not** the `.lab` layout `--chords` uses: a chord timeline is
contiguous non-overlapping spans, whereas piano notes overlap freely, so
reusing that shape would imply a structure the data does not have.
`--piano-format json` adds pedal events.

| Flag | Meaning |
|---|---|
| `--piano` | Enable the piano-transcription task |
| `--piano-format FMT` | `text` (default) or `json` |

Before this verb existed the model was reachable only as `--backend
piano-transcription` through `transcribe()`, which rendered each note as a
segment whose text read `"C4 v=80"` — parsing that back is lossy, and it was
never the intended seam. The structured session API
(`crispasr_session_piano_notes`) already existed for bindings; `--piano` gives
the CLI the same fidelity. The old path still works.

**Model** (`cstr/piano-transcription-GGUF`, Apache-2.0): ByteDance/Kong
high-resolution piano transcription, 16 kHz mono, CRNN + BiGRU with four heads
(frame/onset/offset/velocity) plus onset- and frame-refinement GRUs.

## Guitar tablature (`--tab`)

Per-frame, per-string fret **scores** from guitar audio, via **TabCNN**
(Wiggins & Kim, ISMIR 2019). Backend key `tabcnn`; the architecture is
auto-detected from the GGUF, so `-m <file>` alone is enough.

```bash
crispasr --tab -m tabcnn-f16.gguf -f guitar.wav
crispasr --tab -m auto --auto-download -f guitar.wav      # fetches cstr/tabcnn-GGUF
crispasr --tab -m tabcnn-f16.gguf -f guitar.wav --tab-format json
```

Text output is one line per frame, tab-separated: the frame time in seconds,
then one column per string from low E to high E, with `-` for a string that is
not played.

```
0.023   1   -   -   -   -   -
0.046   1   -   -   -   -   -
```

`--tab-format json` adds the per-string log-probability of each displayed fret,
plus `frame_period_sec`, `n_strings`, `n_classes` and `silent_class`.

### ⚠️ These are emission scores, not a tablature

The model has **no decoder**: six independent softmaxes per frame, no
inter-string coupling, no temporal model, no search. The frets the CLI prints
are a plain `argmax` — they exist so the CLI has something to show and they
ignore every playability constraint (one note per string, fret range, capo, hand
span). Because the strings are scored independently, the grid can contain
physically impossible combinations.

For real use, take the log-probabilities through the C ABI and run your own
constrained Viterbi/DP:

```c
int n = crispasr_session_tab(s, pcm, n_samples, sample_rate);
int frames, strings, classes;
const float* emissions =
    crispasr_session_tab_emissions(s, &frames, &strings, &classes);
int silent = crispasr_session_tab_silent_class(s);   // read it, never assume
float hop  = crispasr_session_tab_frame_period(s);
int open0  = crispasr_session_tab_string_open_midi(s, 0);  // for capo/transpose
```

`emissions` is `[frame][string][class]` log-probabilities, frame-major, valid
until the next call or session close.

### Model and licence

#### Managed model downloads

CrispASR records the upstream licence for every managed registry artifact and
prints it when the artifact is loaded. Restricted terms (including Llama,
Gemma, FunASR, CC-BY-NC/SA, Pocket TTS and other custom licences) require an
exact acceptance tag before auto-download:

```bash
crispasr --backend tada -m auto --auto-download \
  --accept-license llama3.2 -f input.wav
```

`--accept-license all` is also supported. This is an explicit acknowledgement
gate, not a technical enforcement mechanism: downstream distributors remain
responsible for carrying the upstream licence, attribution, acceptable-use and
non-commercial terms. Explicit local paths and caller-supplied URLs are not
managed downloads and are left under the caller's control.

Weights are **CC BY 4.0** from the EGSet12 record
(<https://zenodo.org/records/11406378>) — commercial use permitted,
**attribution required**. This is the GuitarProFX-augmented variant; the
baseline collapses from tablature F1 0.748 to 0.447 on real electric guitar,
while the augmented one recovers to 0.585 (DAFx-24). Cite:

> Pedroza HE, Abreu W, Corey R, Roman IR. "Leveraging real electric guitar tones
> and effects to improve robustness in guitar tablature transcription modeling."
> DAFx, 2024.

| variant | size | tablature F1 |
|---|---|---|
| `tabcnn-f16.gguf` | 1.78 MB | 0.7732 (default) |
| `tabcnn-q8_0.gguf` | 1.10 MB | 0.7749 |
| `tabcnn-q4_k.gguf` | 0.72 MB | 0.7749 |

Quantization preserves `head.weight` at full precision — quantizing the output
layer costs 5.8 F1 points at Q4_0, while `dense0` costs nothing.

### Limitations

- Solo guitar; mixed music is out of domain.
- Not a streaming surface. The CQT is normalised against the **whole clip's**
  maximum (`amplitude_to_db(ref=np.max)`), so features cannot be computed
  chunked without changing them — `--tab` is two-pass by construction.
- GuitarSet-reported numbers overstate real-world accuracy; EGSet12 is the
  honest reference point.

## Chord recognition (`--chords`)

Another standalone task: audio in, a chord timeline out. Like `--pitch` and
`--separate` it routes to its own dispatcher before any ASR backend is built.

> **⚠ The weights are non-commercial.** CrispASR is MIT and the upstream BTC
> code (jayg996/BTC-ISMIR19) is MIT, but the *checkpoints* are CC-BY-NC-SA:
> they were trained on the Isophonics / Robbie Williams / UsPop2002 chord
> annotations, whose licences forbid commercial use. The registry refuses to
> download them unless you explicitly accept that licence, and a commercial
> product must train or supply its own weights. See
> [docs/music-transcription/PLAN.md](music-transcription/PLAN.md).

```bash
# Accept the non-commercial licence and auto-download the default model
crispasr --chords -m auto --auto-download --accept-license cc-by-nc-sa-4.0 -f song.wav

# Explicit model, JSON output
crispasr --chords -m btc-chords-large-f32.gguf --chords-format json -f song.wav

# Collapse the 170-class output to plain maj/min
CRISPASR_BTC_MAJ_MIN=1 crispasr --chords -m btc-chords-large-f32.gguf -f song.wav
```

The backend (`btc`) is auto-detected from the GGUF `general.architecture`;
input is decoded to BTC's native 22.05 kHz mono automatically.

| Flag | Meaning |
|---|---|
| `--chords` | Enable the chord task |
| `--chords-format FMT` | `text` (default) or `json` |
| `--accept-license SPDX` | Required to download the CC-BY-NC-SA weights (or `all`) |

Default output is one tab-separated line per span — `start_sec`, `end_sec`,
`chord` — which is exactly the `.lab` layout the chord datasets (Isophonics et
al.) use, so it drops straight into `mir_eval`:

```
0.000	1.950	C
1.950	3.901	G
3.901	8.081	N
```

`N` means "no chord". `--chords-format json` emits
`{file, vocabulary, n_spans, chords: [...]}` with a `confidence` per span.

**Models** (`cstr/btc-chords-GGUF`, weights CC-BY-NC-SA, code MIT):

| Registry key | File | Size | Notes |
|---|---|---|---|
| `btc-chords`, `btc-chords-large` | `btc-chords-large-f16.gguf` | 5.6 MB | **default** — 170-class vocabulary |
| `btc-chords-majmin` | `btc-chords-f16.gguf` | 5.6 MB | 25-class (maj/min + N) |

f32 variants (11.7 MB) are also published. Both dtypes pass the per-stage diff
harness at cos 1.000000 on all 13 stages, so f16 is the shipping default.

The 170-class model is the default deliberately: it reduces to maj/min with
`CRISPASR_BTC_MAJ_MIN=1`, whereas a 25-class model can never be expanded.

## Auto-download (`-m auto`)

When you pass `-m auto` (or `-m default`), CrispASR downloads the
default quantized model for the selected backend into
`~/.cache/crispasr/` on first use. The registry (kept in sync with
`src/crispasr_model_registry.cpp`):

**ASR backends**

| Backend | Default download | Approx size |
|---|---|---|
| whisper | `ggerganov/whisper.cpp/ggml-base.bin` | ~147 MB |
| parakeet | `cstr/parakeet-tdt-0.6b-v3-GGUF` | ~467 MB |
| canary | `cstr/canary-1b-v2-GGUF` | ~600 MB |
| voxtral | `cstr/voxtral-mini-3b-2507-GGUF` | ~2.5 GB |
| voxtral4b | `cstr/voxtral-mini-4b-realtime-GGUF` | ~3.3 GB |
| granite | `cstr/granite-speech-4.0-1b-GGUF` | ~2.94 GB |
| granite-4.1 | `cstr/granite-speech-4.1-2b-GGUF` | ~2.94 GB |
| granite-4.1-plus | `cstr/granite-speech-4.1-2b-plus-GGUF` | ~2.96 GB |
| granite-4.1-nar | `cstr/granite-speech-4.1-2b-nar-GGUF` | ~3.2 GB (Q4_K default) |
| qwen3 | `cstr/qwen3-asr-0.6b-GGUF` | ~500 MB |
| cohere | `cstr/cohere-transcribe-03-2026-GGUF` | ~550 MB |
| wav2vec2 | `cstr/wav2vec2-large-xlsr-53-english-GGUF` | ~212 MB |
| omniasr | `cstr/omniASR-CTC-1B-v2-GGUF` | ~658 MB |
| omniasr-llm | `cstr/omniasr-llm-300m-v2-GGUF` | ~1019 MB |
| hubert | `cstr/hubert-large-ls960-ft-GGUF` | ~200 MB |
| data2vec | `cstr/data2vec-audio-960h-GGUF` | ~60 MB |

**TTS backends** — all auto-download the model + a default voice pack:

| Backend | Default download | Approx size | Notes |
|---|---|---|---|
| vibevoice-tts | `cstr/vibevoice-realtime-0.5b-GGUF` (Q4_K) + `vibevoice-voice-emma.gguf` | ~636 MB + ~3 MB | `--model-quant q8_0` → ~1.1 GB higher-quality variant |
| vibevoice | `cstr/vibevoice-asr-GGUF` (Q4_K) | ~4.5 GB | ASR + TTS combo model |
| vibevoice-1.5b | `cstr/vibevoice-1.5b-GGUF` (Q4_K) | ~1.6 GB | Base model, runs without a voice pack |
| kokoro | `cstr/kokoro-82m-GGUF` (Q8_0) + `kokoro-voice-af_heart.gguf` | ~135 MB + ~4 MB | German routes automatically when the `kokoro-de-hui-base` backbone sits next to the model and you pass `-l de` |
| qwen3-tts | `cstr/qwen3-tts-0.6b-base-GGUF` (Q8_0) + 12 Hz tokenizer | ~986 MB + ~60 MB | Streaming-capable; tokenizer/codec auto-discovered |
| qwen3-tts-1.7b-base | `cstr/qwen3-tts-1.7b-base-GGUF` (Q8_0) + 12 Hz tokenizer | ~1.9 GB + ~60 MB | Higher quality |
| orpheus | `cstr/orpheus-3b-0.1-ft-GGUF` (Q8_0) + SNAC codec | ~3.7 GB + ~80 MB | Llama-3 based; US-English |
| chatterbox | `cstr/chatterbox-GGUF` (Q8_0 T3 + Q8_0 S3Gen) | ~610 MB + ~349 MB | S3Gen + T3; multilingual |
| chatterbox-nano | `cstr/chatterbox-nano-GGUF` (Q8_0 T3) + Turbo S3Gen companion | ~345 MB + ~627 MB | GPT2-small T3 on the Turbo S3Gen (#382); English |
| raon | `cstr/raon-opentts-0.3b-GGUF` (F16, single GGUF) | ~959 MB | KRAFTON Raon-OpenTTS 0.3B on the F5 runtime; EN zero-shot cloning; **CC-BY-NC-4.0** (non-commercial, license gate) |
| chatterbox-finnish-nano | `JJarvinen/chatterbox-finnish-nano-GGUF` (Q8_0 T3) + Turbo S3Gen companion | ~329 MB + ~627 MB | Finnish-only Nano v0.1.3; MIT |
| piper | `cstr/piper-en_US-lessac-medium-GGUF` (F16) | ~30 MB | Lightweight, many voices via `--voice` |
| tada-1b | `cstr/tada-tts-1b-GGUF` (Q4_K + F16 codec) | ~1.7 GB + ~250 MB | English-only; `--voice tada-ref.gguf` |
| tada / tada-3b-ml | `cstr/tada-tts-3b-ml-GGUF` (F16 + F16 codec) | ~6.6 GB + ~250 MB | 9 languages; `-l fr` auto-downloads `tada-ref-fr.gguf` — see [tts.md §TADA](tts.md#tada--multilingual-and-voice-cloning) |

Downloads go through `curl` (preferred) with a `wget` fallback — **no
Python, no libcurl link dependency**. Works identically on Linux,
macOS, and Windows 10+ where `curl` ships in the base system. Models
are cached by filename; re-running is a single `stat()` check. The
same registry + cache helpers are reachable from the language
bindings (see [bindings.md](bindings.md)) so Python/Rust/Dart callers
can drive `-m auto`-style resolution without re-implementing it.

## Audio formats

CrispASR decodes most common formats natively — no ffmpeg required — and
auto-resamples to 16 kHz mono. The decoders are permissive-licensed and either
embedded or linked:

- **[miniaudio](https://miniaud.io/)** (MIT-0) — WAV (any bit depth: 16/24/32
  PCM, IEEE float, A-law, μ-law, ADPCM), FLAC, MP3, and the WAV family **AIFF /
  W64 / RF64** (via its bundled `dr_wav`)
- **[stb_vorbis](https://github.com/nothings/stb)** (public domain) — OGG Vorbis
- **[glint](https://github.com/CrispStrobe/glint)** (MIT) — in-tree clean-room
  decoder for **raw ADTS `.aac`** (AAC-LC) and **Ogg `.opus`** (RFC-conformant:
  all 12 RFC 6716/8251 vectors, SILK byte-identical to libopus). Cross-platform,
  always available with no runtime library, so it is the **default** for both
  `.aac` and `.opus` via the library `crispasr_audio_load` API — `.opus` decodes
  even in builds without libopus. Pin the previous decoder with
  `CRISPASR_AAC_DECODER=fdk`/`coreaudio` or `CRISPASR_OPUS_DECODER=libopus`
  (`=glint`/`auto` = default); `CRISPASR_{AAC,OPUS}_DEBUG=1` prints a one-line
  decode summary. Opus is glint by default **everywhere it appears** — bare
  `.opus`, the stereo loader, and **WebM/Matroska Opus** — so a build with
  `CRISPASR_OPUS=OFF` (no libopus at all) still decodes them.
- **libopus + opusfile** (BSD-3) — optional fallback for Opus (`.opus` + WebM),
  selected via `CRISPASR_OPUS_DECODER=libopus`. Built by default (`CRISPASR_OPUS`,
  on when system `opusfile` is found; `CRISPASR_OPUS_FETCH=ON` builds it statically
  for platforms without system libs). No longer required for any input format.
- **AudioToolbox** (Apple system framework) — container **`.m4a` / `.alac` /
  `.caf`** (and `.aac`) on macOS/iOS, no extra dependency; **fdk-aac** via
  `dlopen` provides the same container support on Linux/Windows when installed.

| Format | Linux/other | Apple (macOS/iOS) | `CRISPASR_FFMPEG=ON` |
|---|:---:|:---:|:---:|
| WAV / FLAC / MP3 / OGG Vorbis / AIFF / W64 / RF64 | ✔ | ✔ | ✔ |
| `.opus` | ✔ | ✔ | ✔ |
| `.aac` (raw ADTS, AAC-LC) | ✔ (glint) | ✔ (glint) | ✔ |
| `.m4a` / `.alac` / `.caf` (container) | ✔¹ | ✔ (AudioToolbox) | ✔ |
| `.webm` (Opus/Vorbis, Matroska) | ✔ (glint/stb_vorbis) | ✔ | ✔ |
| `.wma` / `.amr` / raw PCM | ✗ | ✗ | ✔ / pre-convert |

¹ Container AAC (`.m4a`) needs libfdk-aac installed (loaded via `dlopen`) or
`CRISPASR_FFMPEG=ON`; only **raw ADTS `.aac`** is covered dependency-free by the
in-tree glint decoder. On Apple all AAC/ALAC/CAF is handled natively
(AudioToolbox).

The `crispasr` CLI decodes all of the above through the same library loader
(`crispasr_audio_load`), so `-f in.opus` / `in.aac` / `in.m4a` / `in.webm` work
with **no ffmpeg subprocess** — the ffmpeg fallback is only reached for formats
the native decoders can't handle. For those, build with `CRISPASR_FFMPEG=ON` (an
optional, dynamically-linked fallback — see [install.md](install.md)) or
pre-convert: `ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

## Memory footprint

Three runtime knobs control how much RAM / VRAM the binary uses.
All are env vars (no CLI flags — these are rarely-changed deployment
settings, not per-invocation switches).

### `CRISPASR_KV_QUANT={f16,q8_0,q4_0}` — KV cache dtype

The default `f16` KV cache is the highest-quality option but the
biggest VRAM consumer. `q8_0` halves it; `q4_0` quarters it. Quality
drift is <0.1 % WER on validated backends; for long-audio chunked
work on a VRAM-tight host, this is the cheapest knob you can turn.

```bash
CRISPASR_KV_QUANT=q8_0 ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
```

Per-backend coverage:

| Backend | Honors `CRISPASR_KV_QUANT`? |
|---|:-:|
| voxtral / voxtral4b | ✔ |
| qwen3-asr | ✔ |
| granite / granite-4.1 / granite-4.1-plus | ✔ |
| glm-asr | ✔ |
| mimo-asr | ✔ |
| omniasr-llm | ✔ |
| gemma4-e2b | ✔ |
| canary | ✔ (cast-on-read fallback for read path; flash_attn_ext on for self-attn since §73) |
| cohere | ✔ (same — §73 flash_attn_ext shipped; +11 % regression on JFK quant K/V vs cast, see PERFORMANCE.md) |
| kyutai-stt | ✔ (native flash_attn_ext, quant-safe by construction) |
| orpheus | ✔ |
| qwen3-tts | ✔ (talker only) |
| chatterbox / chatterbox-turbo / chatterbox-nano / chatterbox-finnish-nano / kartoffelbox-turbo / lahgtna-chatterbox | ✔ (T3 LM side; S3Gen Conformer attention is F32 by design) |
| vibevoice | F16-only — flag is read but the σ-VAE attention path uses `ggml_cpy(K_perm, view)` write that's incompatible with quant K/V. Migration recipe is the canary/cohere flash_attn_ext port (see PERFORMANCE.md "Where the gaps are"). |
| granite-4.1-nar | — (non-autoregressive variant, no LLM decode path) |
| whisper / parakeet / fc-ctc / wav2vec2 / hubert / data2vec / firered-asr / moonshine / moonshine-streaming / omniasr-CTC | — (no KV cache: CTC / transducer / encoder-only) |
| kokoro | — (single-shot StyleTTS2 / iSTFTNet, no KV cache) |

The flag is read once per session via
`core_attn::kv_dtype_from_env(<backend_name>)`; subsequent
`session_transcribe` calls reuse the dtype from session open. Set
the env before launching `crispasr` (or before opening the session
in Python / Rust / Dart).

### `CRISPASR_KV_QUANT_K` / `CRISPASR_KV_QUANT_V` — asymmetric K vs V

The two halves of the KV cache have very different sensitivity
profiles: V is forgiving (errors get averaged inside the
post-softmax weighted sum), K is fragile (errors distort which
positions get attended to). llama.cpp exposes `--cache-type-k` /
`--cache-type-v` for this; CrispASR does the same via two env
vars that override `CRISPASR_KV_QUANT` per half.

```bash
# Common llama.cpp recipe — ~40 % more KV memory savings vs symmetric
# Q8_0/Q8_0, with PPL barely moved on Llama-class models.
CRISPASR_KV_QUANT_K=q8_0 CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
```

Both halves fall through to `CRISPASR_KV_QUANT` when their
type-specific var is unset, so the legacy single-knob configuration
keeps working unchanged.

Same per-backend coverage as the table above — the asymmetric
plumbing was added to every backend that honored `CRISPASR_KV_QUANT`
(voxtral, voxtral4b, omniasr, qwen3_asr, granite_speech, orpheus,
glm_asr, gemma4_e2b, mimo_asr, qwen3_tts).

### `CRISPASR_KV_ON_CPU=1` — spill KV cache to system RAM

For users with very long context where even `CRISPASR_KV_QUANT=q4_0` won't
fit in VRAM. Allocates the KV cache on the CPU backend instead of
the GPU backend, even when model weights are active on GPU.

```bash
# Long-context fallback when VRAM is exhausted
CRISPASR_KV_ON_CPU=1 ./build/bin/crispasr --backend voxtral4b -m auto -f long-audio.wav

# Stacks with CRISPASR_KV_QUANT_K/_V — minimum-memory KV path
CRISPASR_KV_ON_CPU=1 CRISPASR_KV_QUANT_K=q8_0 CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f long-audio.wav
```

**Try `CRISPASR_KV_QUANT` first.** The expensive part isn't the alloc —
every attention step copies the KV slice GPU↔CPU↔GPU. The
PCIe / unified-memory traffic is typically slower than just running
with quantised KV in VRAM. Reach for `CRISPASR_KV_ON_CPU` only when
quantisation alone can't fit the context.

The verbose log line shows `(on cpu)` vs `(on gpu)` so you can
confirm where the cache landed:

```
voxtral4b: kv cache 169 MiB k=q8_0 v=q4_0 (on cpu, ...)
```

Same per-backend coverage as `CRISPASR_KV_QUANT` (voxtral, voxtral4b,
omniasr, qwen3_asr, granite_speech, orpheus, glm_asr, gemma4_e2b,
mimo_asr, qwen3_tts).

### `CRISPASR_N_GPU_LAYERS=N` — layer-residency offload

llama.cpp `--n-gpu-layers` parity. Default `-1` keeps legacy
single-backend behaviour (everything on GPU, or CPU if `-ng`).
Setting `N` in `[0, total_layers)` puts the first N transformer
blocks on GPU and spills the rest to system RAM, so models larger
than VRAM can still run end-to-end.

```bash
# Voxtral4b has 26 transformer blocks. Half on GPU, half on CPU.
CRISPASR_N_GPU_LAYERS=13 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav

# Tight VRAM: only audio encoder + projection + embeddings on GPU,
# all 26 transformer blocks on CPU.
CRISPASR_N_GPU_LAYERS=0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
```

Verbose log shows weight residency and the layer split:

```
voxtral4b: weight residency: gpu=1585 MiB (571 tensors), cpu=821 MiB (143 tensors)
voxtral4b: layer offload: gpu=[0,13), cpu=[13,26) (CRISPASR_N_GPU_LAYERS=13)
```

Coverage (10 LLM-decode backends): voxtral, voxtral4b, qwen3_asr,
granite_speech, glm_asr, orpheus, omniasr-llm, gemma4_e2b, mimo_asr,
vibevoice. Vibevoice is dual-mode — ASR-only files split the
28-layer `lm.layers.<N>.*` path; TTS-enabled files (`tts_n_layers > 0`)
split the dominant 20-layer `tts_lm.layers.<N>.*` path while the
4-layer base LM stays on GPU. Encoder-decoder ASR (canary, cohere,
kyutai-stt) is not yet covered — cross-attention layout has no
`<prefix><N>.*` block-tagged tensors and needs a bespoke predicate.

**Stacks with `CRISPASR_KV_ON_CPU` and `CRISPASR_KV_QUANT_K/_V`** — set all three for
the most aggressive memory footprint reduction. `CRISPASR_KV_QUANT` is
cheaper than layer offload; reach for `CRISPASR_N_GPU_LAYERS` only when the
*model* doesn't fit, not the cache.

### `CRISPASR_GGUF_MMAP` — zero-copy weight load (default **on**)

Map the GGUF file directly into the model's backend buffer instead
of read-and-copy. Saves one full copy of the GGUF on load: a 14.9 GB
F16 model goes from "load + 14.9 GB peak RSS" to "mmap +
~working-set RSS." No quality impact; pure load-time + RAM win.

Default-on since 0.6.7 (issue #94 — chatterbox-turbo slow / failing
init on macOS, where the legacy alloc+copy path took 30-60 s for
the 658 MB T3 GGUF). Opt out with `CRISPASR_GGUF_MMAP=0` if your
model files live on volumes that may disappear mid-run — mmap-backed
weights SIGBUS if the underlying file vanishes (network mounts,
removable disks).

```bash
# Default — mmap is on, no env var needed
./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav

# Opt out for removable media
CRISPASR_GGUF_MMAP=0 ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
```

Honored by every backend that uses `core_gguf::load_weights()` —
all non-whisper backends. Whisper itself uses upstream's loader and
isn't affected.

### `CRISPASR_GGUF_PRELOAD=1` — page-walk on load

When mmap is enabled, this triggers a one-byte read on every page
to force the working set resident before returning. Trades cold-
start *load* time for cold-start *prefill* time. Useful for servers
that will do many short generations after one-time load and don't
want the first request to pay the page-fault tax.

```bash
CRISPASR_GGUF_MMAP=1 CRISPASR_GGUF_PRELOAD=1 ./build/bin/crispasr ...
```

### Recommended combos for VRAM-constrained voxtral4b

In order of cost — try the cheapest first:

```bash
# 1. Cheapest — half the KV. ~0.05 % WER drift on validated suite.
CRISPASR_KV_QUANT=q8_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav

# 2. Aggressive — quarter the KV. ~0.2 % WER drift.
CRISPASR_KV_QUANT=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav

# 3. Plus mmap so the load doesn't double-allocate the model weights.
#    Useful when you're loading a multi-GB F16 model and the host has
#    less RAM than 2× model size.
CRISPASR_KV_QUANT=q4_0 CRISPASR_GGUF_MMAP=1 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
```

See the `CRISPASR_N_GPU_LAYERS` and `CRISPASR_KV_ON_CPU` sections above
for the full layer-offload and KV-spill knobs — both are supported.

### TTS provenance & watermarking flags

All TTS output is watermarked by default. Additional flags control
the neural watermark, C2PA signing, voice-cloning consent, and the opt-out:

| Flag | Description |
|------|-------------|
| `--watermark-model PATH\|auto` | Load an AudioSeal GGUF for neural watermarking (`auto` downloads it; upgrades the built-in spread-spectrum) |
| `--no-watermark` | Disable the AI-content watermark on TTS output. Equivalent to the `CRISPASR_NO_WATERMARK` env var; both emit a one-time stderr warning and shift the AI-content marking responsibility onto the operator (see below). **Requires `--accept-marking-responsibility`** — without it the run is refused with exit `12`. Honored only while the output still carries a C2PA manifest (WAV/MP3/M4A/MP4); for raw `.aac` / `.opus` and `--tts-stream` the watermark is kept anyway |
| `--no-c2pa` | Disable C2PA Content Credentials signing on synthesized output. Also requires `--accept-marking-responsibility`; on the CLI the audio watermark is then forced on so no output is ever fully unmarked |
| `--accept-marking-responsibility` | Explicit attestation, REQUIRED to honor any provenance opt-out (`--no-watermark` / `--no-spoken-disclaimer` / `--no-c2pa`). Logs a `[MARKING]` audit line when an opt-out is honored |
| `--consent-log PATH` | Append every `[CONSENT]` audit record to `PATH` as JSON Lines, in addition to stderr (also `CRISPASR_CONSENT_LOG`). Off by default; tamper-resistance is the storage's job |
| `--detect-watermark PATH` | Read a WAV file, run watermark detection, print detector / duration / confidence + verdict, then exit. Built-in per-frame statistic (the default): `≥0.65` detected, `0.5–0.65` inconclusive, `≤0.5` not detected. With `--watermark-model` (AudioSeal) the score is a probability and the bar is `>0.5`. The legacy bin-sign statistic (`CRISPASR_WATERMARK_DETECT=sign`) reports a p-value instead: `p<0.01` detected, `p<0.20` inconclusive |
| `--i-have-rights` | Required for voice cloning (`--voice <file.wav>`); attests speaker consent |
| `--print-speaker-identity FILE` | Standalone verb: print whose voice a model or voice pack produces (`real_person` / `synthetic` / `unknown`) and exit. Uses the same resolution the Art. 50(4) disclosure gate uses — stamp inside the file first, then the researched table — so a script never has to restate a verdict. Exits 3 when the answer is unknown |
| `--speaker-identity VALUE` | Whose voice a **preset** voice is: `real_person`, `synthetic` or `unknown` (default). `real_person` adds the audible AI disclosure to non-cloned output — a stock voice can be an identifiable individual, which makes the output a deep fake under Art. 3(60) without any cloning. It does **not** require `--i-have-rights`. Backends default to `unknown` and warn once per model until someone answers; don't silence that with `synthetic` unless you've read the model card. See [`eu-ai-act.md` §6.2a](eu-ai-act.md#62a-whose-voice-is-a-preset-voice-speaker_identity) |
| `--no-spoken-disclaimer` | Skip the audible AI-disclosure prefix on voice-cloned output (watermark + C2PA still applied; caller assumes disclosure responsibility). Requires `--accept-marking-responsibility` |
| `--g2p-dict SOURCE` | G2P pronunciation dictionary: `olaph` (MIT, default), `open-dict` (CC-BY-SA), or path to a custom dict file. Auto-downloads on first use. See [`tts.md`](tts.md) for details. |
| `--c2pa-cert PATH` | X.509 certificate for C2PA Content Credentials signing |
| `--c2pa-key PATH` | Private key for C2PA signing (generate both with `scripts/generate-c2pa-cert.sh`) |

**Disabling the watermark.** `--no-watermark` and `CRISPASR_NO_WATERMARK=1` are
equal-status opt-outs (neither is more "official"). The *flag* additionally
requires `--accept-marking-responsibility` — passing it alone exits `12`. Either
one turns the mark off for the whole process and logs, once:

```
crispasr: warning: watermarking disabled. AI usage marking responsibility rests with the operator.
```

The message is deliberately jurisdiction-neutral — no statute is named at
runtime. Turning the mark off does not remove any legal AI-disclosure obligation
that may apply to the output; it transfers responsibility for meeting it to
whoever runs the binary. See [`tts.md`](tts.md) for the full rationale.

Debug env vars:
- `CRISPASR_AUDIOSEAL_DEBUG=1` — print AudioSeal tensor shapes during graph build
- `CRISPASR_AUDIOSEAL_DUMP_STAGES=1` — dump per-stage binary tensors to `/tmp/`

See [`tts.md`](tts.md) for full watermarking documentation.

### TTS-side env vars

For TTS-specific deployment knobs (codec backend selection, graph
reuse, etc.) see [`tts.md`](tts.md):
- `CRISPASR_QWEN3_TTS_CODEC_GPU` — clean codec-on-GPU path (CUDA / Vulkan)
- `CRISPASR_QWEN3_TTS_O15` — code-predictor graph reuse (CPU/Metal opt-in)
- `CRISPASR_KOKORO_GEN_GPU` — generator on GPU (CUDA / Vulkan)
- `CRISPASR_COSYVOICE3_FLOW_STEPS=N` — CosyVoice3 flow Euler steps (`1..100`;
  model default `10`). Lower values reduce flow latency approximately
  linearly at a possible quality cost; `5` is a practical fast mode.
- `CRISPASR_COSYVOICE3_BENCH=1` — print CosyVoice3 per-stage timings.
- `CRISPASR_COSYVOICE3_CFG_BATCH=0` — compatibility fallback to two separate flow
  forwards per Euler step. The default batch-2 path matches upstream and is
  faster while producing identical output on validated CPU and Metal runs.
- `CRISPASR_COSYVOICE3_KV_BUCKET=0` — compatibility fallback that exposes the full KV
  allocation to every AR step instead of the default 256-token active buckets.
- `CRISPASR_TADA_NUM_CANDIDATES=N` — TADA flow-matching duration candidates per token,
  ranked by reconstruction likelihood (CLI default `4`). The duration head is
  noise-sensitive, so a single draw (`N=1`, fastest) can occasionally collapse
  timing into rushed/garbled speech; `4`–`8` make it robust. All candidates
  for a step solve in one batched forward, so higher `N` adds little wall-clock.
  Default `4` also applies through the C ABI / bindings / server; override there
  at runtime with `set_tts_num_candidates(n)`.
  See [`tts.md`](tts.md#timing-quality-crispasr_tada_num_candidates).
- `CRISPASR_TADA_DO_SAMPLE`, `CRISPASR_TADA_TEMPERATURE`, `CRISPASR_TADA_TOP_P`, `CRISPASR_TADA_TOP_K`,
  `CRISPASR_TADA_REPETITION_PENALTY` — TADA **talker** text-decoder sampling, matching
  upstream `InferenceOptions` defaults (do_sample=1, temp=0.6, top_p=0.9,
  top_k=0, rep_penalty=1.1). The talker samples by default; greedy decoding
  (`CRISPASR_TADA_DO_SAMPLE=0`) has no repetition control and loops/cuts off words on
  harder or non-English text. Honoured by the CLI, C ABI, bindings and server;
  `set_temperature` / `set_top_p` / `set_repetition_penalty` also reach TADA at
  runtime.
- `CRISPASR_VIBEVOICE_VAE_BACKEND={auto,cpu,gpu}` — VAE decoder placement
- `CRISPASR_VIBEVOICE_TTS_FLASH_ATTN={1,0}` — TTS LM attention: `1` (default)
  uses fused `ggml_flash_attn_ext`; `0` uses an explicit
  `softmax(QKᵀ)·V` path. Set `0` if VibeVoice TTS garbles, mixes
  voices, or repeats on a GPU whose fused flash-attention shader is
  buggy — notably **AMD RDNA4 (RX 9700 XT) on Vulkan**, whose coopmat2
  FA shader produces wrong hidden states (issue #171). The
  no-rebuild equivalent is `GGML_VK_DISABLE_COOPMAT2=1`. This knob and
  `CRISPASR_VIBEVOICE_VAE_BACKEND` bisect the TTS GPU graph (LM attention vs.
  the conv/col2im VAE) to localise a bad kernel.

CosyVoice3 performance notes:

- Keep GPU execution enabled. The 10-step DiT flow is substantially slower
  on CPU; `--gpu-backend metal` selects Metal explicitly on macOS.
- `-n/--max-new-tokens` is also the AR KV-cache sizing bound. A realistic
  cap reduces per-token work, but a value that is too low truncates speech.
- `CRISPASR_COSYVOICE3_FLOW_STEPS=N` sets the CFM Euler step count (default `10`). Flow
  work is ~linear in `N` and flow is ~48 % of the wall. M1 sweep (`--seed 42`,
  log-mel-spectrogram corr vs the 10-step output — ASR roundtrip is verbatim at
  8/6 and cannot distinguish steps): `8`→0.9948, `6`→0.9925, `4`→0.9895 with a
  one-word ASR slip. **`6` is the perceptual sweet spot** (~−40 % flow work,
  ~−19 % of the wall) and matches the chatterbox default; `4` starts to show
  audible artifacts. `6` was confirmed across short / numbers / long sentences
  (6-vs-10 mel-corr 0.9953 / 0.9930 / 0.9938). `10` is the conservative
  upstream default; drop to `6` for a large speedup at near-identical quality.
- An external model directory affects cold startup, not steady-state
  synthesis. For repeated requests, use server mode so the ~1.2 GB model set
  remains resident.
- Baked voices do not load the S3 tokenizer or CAMPPlus encoder. Those
  companions add roughly 475 MiB and are loaded lazily on the first `.wav`
  cloning request, including a later request to a resident server.
- On an Apple M1 test using the Q4_K LLM and Q8_0 flow, request-sized KV
  caching plus active KV buckets reduced a 17-token LM decode from about
  6.8 s to 0.5 s. Batched CFG reduced the 10-step flow from about 4.6 s to
  1.9–2.3 s. The final WAV remained byte-identical. Hardware and prompt length
  change results.

### Comparison with llama.cpp

For users coming from `llama.cpp`, here's how the equivalent knobs
map:

| Concern | llama.cpp | CrispASR |
|---|---|---|
| KV cache dtype | `--type-k q8_0 --type-v q8_0` (CLI flag, separate K/V) | `CRISPASR_KV_QUANT=q8_0` for symmetric, or `CRISPASR_KV_QUANT_K` / `_V` per half |
| mmap weights | `--no-mmap` (mmap is default **on**) | `CRISPASR_GGUF_MMAP=0` (mmap is default **on** since 0.6.7) |
| Lock pages in RAM | `--mlock` | (not supported — `mmap+preload` is the closest analogue) |
| GPU layer count | `--n-gpu-layers N` / `-ngl N` (CLI flag) | `CRISPASR_N_GPU_LAYERS=N` env var — 10 LLM backends |
| KV-on-CPU-only | `--no-kv-offload` | `CRISPASR_KV_ON_CPU=1` env var |
| Flash attention | `--flash-attn` / `-fa` | always-on where the backend's `capabilities()` declares `CAP_FLASH_ATTN` |
| Threads | `--threads N` / `-t N` | `--threads N` / `-t N` (matched) |
| Force CPU | `--gpu-layers 0` | `--no-gpu` / `--gpu-backend cpu` |

Differences worth flagging:

1. **mmap default.** Both projects now default mmap **on**. CrispASR
   flipped from opt-in to default-on in 0.6.7 after issue #94 (slow /
   failing chatterbox-turbo init on macOS — the legacy alloc+copy
   path took 30-60 s for the 658 MB T3 GGUF). Set
   `CRISPASR_GGUF_MMAP=0` to opt out (matches llama.cpp's
   `--no-mmap`).
2. **K/V dtype unified.** llama.cpp lets you set `--type-k` and
   `--type-v` independently (rare scenario: quantize K but keep V
   at f16). CrispASR uses a single `CRISPASR_KV_QUANT` for both.
   The split would be a small change if anyone needs it; file an
   issue with a use case.
3. **CLI flags vs env vars.** llama.cpp surfaces every memory knob
   as a CLI flag; CrispASR uses env vars for them on the assumption
   that they're rarely-changed deployment settings. If you want flag
   parity, see open issue / PR — converting the env vars to flags
   is mechanical (`-DCRISPASR_KV_QUANT=val` style) but adds CLI
   surface area.
4. **`CRISPASR_N_GPU_LAYERS=N`.** Equivalent to `--n-gpu-layers N`.
   Supported on 10 LLM-decode backends (voxtral, voxtral4b, qwen3-asr,
   granite, glm-asr, orpheus, omniasr-llm, gemma4-e2b, mimo-asr,
   vibevoice). See the section above for details.

---

## See also

- [`docs/streaming.md`](streaming.md) — `--stream`, `--mic`, `--live`,
  sliding-window flags, per-token confidence
- [`docs/tts.md`](tts.md) — Kokoro / Qwen3-TTS / VibeVoice / Orpheus / Chatterbox
  + every TTS-side env var (`CRISPASR_QWEN3_TTS_CODEC_GPU`,
  `CRISPASR_QWEN3_TTS_SKIP_REF_DECODE`, `CRISPASR_QWEN3_TTS_O15`, `CRISPASR_KOKORO_GEN_GPU`,
  `CRISPASR_VIBEVOICE_VAE_BACKEND`, …)
- [`docs/server.md`](server.md) — HTTP `/inference`, OpenAI-compat
  `/v1/audio/transcriptions`, `/v1/audio/speech` (TTS),
  `/v1/audio/speech-to-speech` (S2S)
- [`docs/bindings.md`](bindings.md) — Python / Rust / Dart / Go / Java
  / Ruby — every CLI feature is reachable through the C-ABI
- [`docs/install.md`](install.md) — full build options, GPU backends,
  ffmpeg ingestion, glibc compatibility
- [`docs/quantize.md`](quantize.md) — `crispasr-quantize` per-backend
  recommended quants
- [`docs/architecture.md`](architecture.md) — internals: `src/core/`
  primitives, per-backend graph survey
- [`docs/contributing.md`](contributing.md) — adding a new backend,
  PyTorch-vs-C++ stage diff workflow
- [`docs/regression-matrix.md`](regression-matrix.md) —
  `tools/test-all-backends.py` capability tiers
