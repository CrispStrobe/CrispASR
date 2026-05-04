# CLI reference

`crispasr` extends upstream whisper.cpp's argument set with a handful
of backend-dispatch flags. Every historical whisper flag still works —
when you don't pass `--backend`, whisper is the default.

## Contents

- [Quick reference](#quick-reference) — most-used flag patterns
- [Core flags](#core) — model, backend, input, language
- [Output formats](#output) — txt / srt / vtt / json / csv / lrc
- [Segmentation & chunking](#segmentation--chunking) — VAD, fixed chunks
- [Word-level timestamps via CTC alignment](#word-level-timestamps-via-ctc-alignment)
- [Sampling / decoding](#sampling--decoding-whisper--llm-backends) — temperature, beam, grammar
- [Language detection (LID)](#language-detection-lid)
- [LLM-backend specific](#llm-backend-specific) — aligner, max-new-tokens
- [Multi-language / translation](#multi-language--translation)
- [Threading & processors](#threading--processors)
- [Whisper-only flags](#whisper-only-flags)
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

# JSON with word + token detail
crispasr -m auto --backend parakeet -f audio.wav -ojf

# Force GPU pick
crispasr --gpu-backend vulkan -dev 1 -m auto -f audio.wav

# Half-VRAM voxtral4b
CRISPASR_KV_QUANT=q4_0 CRISPASR_GGUF_MMAP=1 crispasr --backend voxtral4b -m auto -f audio.wav

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
| `-l LANG`, `--language LANG` | ISO-639-1 code (default: `en`) |
| `--list-backends` | Print the capability matrix and exit |

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
per-token `tokens[]` arrays when the backend populates them.

## Segmentation / chunking

| Flag | Meaning |
|---|---|
| `--vad` | Enable Silero VAD. Auto-downloads `ggml-silero-v5.1.2.bin` (~885 KB) to `~/.cache/crispasr/` on first use |
| `--vad-model FNAME` | Override the VAD model path (default: auto) |
| `-vt F` | VAD threshold (default 0.5) |
| `-vspd N` | VAD min speech duration (ms, default 250) |
| `-vsd N` | VAD min silence duration (ms, default 100) |
| `-ck N`, `--chunk-seconds N` | Fallback chunk size when VAD is off (default 30 s) |

### How VAD works

Every non-whisper backend uses the Silero VAD model to segment long
audio into speech regions, **stitches them into a single contiguous
buffer** (with 0.1 s silence gaps), transcribes in one pass, and
remaps timestamps back to original-audio positions. This preserves
cross-segment context and avoids boundary artifacts. Short VAD
segments (< 3 s) are auto-merged, and oversized segments are split at
`--chunk-seconds` boundaries. Whisper handles VAD internally via
`wparams.vad`.

```bash
# Just pass --vad — the model is auto-downloaded on first use
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad -osrt

# Or point at an existing GGUF
./build/bin/crispasr --backend parakeet -m parakeet.gguf -f long_audio.wav \
    --vad-model ~/models/ggml-silero-v5.1.2.bin -osrt
```

The cached model lives at `~/.cache/crispasr/ggml-silero-v5.1.2.bin`
(~885 KB). If you don't pass `--vad`, CrispASR falls back to fixed
30-second chunking (`-ck`). Encoder cost is O(T²) in the frame count,
so for multi-minute audio you really want VAD.

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
- **If parakeet is too heavy for very long audio:** keep parakeet for
  timing quality, but cap memory with fixed chunking
  (`--chunk-seconds 180`).

## Word-level timestamps via CTC alignment

The LLM-based backends (`qwen3`, `voxtral`, `voxtral4b`, `granite`)
don't emit timestamps natively. CrispASR supports a second-pass
forced alignment via NVIDIA's canary-ctc-aligner — a 600 M-param
FastConformer + CTC head that works on any transcript + audio pair
in 25+ European languages.

```bash
# Grab the aligner once (~400 MB)
curl -L -o canary-ctc-aligner.gguf \
    https://huggingface.co/cstr/canary-ctc-aligner-GGUF/resolve/main/canary-ctc-aligner-q5_0.gguf

# Now any LLM backend can produce word-level SRT output
./build/bin/crispasr --backend voxtral -m auto -f samples/jfk.wav \
    -am canary-ctc-aligner.gguf -osrt -ml 1
# [00:00:00.240 --> 00:00:00.640]  And
# [00:00:00.640 --> 00:00:00.880]  so,
# [00:00:00.880 --> 00:00:01.040]  my
```

Alignment granularity is one encoder frame (~80 ms).

For subtitle output, prefer adding `--vad --split-on-punct`:

```bash
./build/bin/crispasr --backend cohere -m cohere.gguf -f talk.wav \
    -am canary-ctc-aligner.gguf --vad -osrt --split-on-punct
```

Notes:
- The aligner path is a fallback for backends that lack native
  timestamps.
- `qwen3-forced-aligner` is more sensitive to leading silence;
  `--vad` is strongly recommended with it.
- Parakeet remains the better choice when timestamp quality is the
  top priority.

## Sampling / decoding (whisper + LLM backends)

| Flag | Meaning |
|---|---|
| `-tp F`, `--temperature F` | Sampling temperature. `0` = pure argmax (default, bit-identical). `> 0` enables multinomial sampling for whisper, voxtral, voxtral4b, qwen3, granite |
| `-bs N`, `--beam-size N` | Beam search width (whisper only) |
| `-tpi F`, `--temperature-inc F` | Whisper temperature-fallback increment |
| `--grammar FNAME` | GBNF grammar file (whisper only, including `--backend whisper`) |
| `--grammar-rule NAME` | Top-level rule name in the grammar |
| `--prompt STR` | Initial prompt for whisper |

## Language detection (LID)

| Flag | Meaning |
|---|---|
| `-l auto`, `--detect-language` | Auto-detect the input language. Backends without native lang-detect (cohere, canary, granite, voxtral, voxtral4b) get it via the LID pre-step |
| `--lid-backend NAME` | LID provider: `whisper` (default), `silero` (95 langs, 16 MB), `ecapa` (107 or 45 langs, 40-43 MB), `firered` (120 langs, 544 MB), or `off` |
| `--lid-model FNAME` | Override the LID model path (default: auto-downloads `ggml-tiny.bin` ~75 MB on first use) |

## LLM-backend specific

| Flag | Meaning |
|---|---|
| `-am FNAME`, `--aligner-model FNAME` | CTC aligner GGUF for word-level timestamps |
| `-n N`, `--max-new-tokens N` | Max tokens the LLM may generate (default 512) |

## Multi-language / translation

| Flag | Meaning |
|---|---|
| `-sl LANG`, `--source-lang LANG` | Source language (canary) |
| `-tl LANG`, `--target-lang LANG` | Target language (canary; set different from `-sl` for X→Y translation) |
| `-tr`, `--translate` | Translate to English (whisper, canary) |
| `--no-punctuation` | Disable punctuation in the output. Native for cohere/canary, post-processed for everyone else |

## Threading / processors

| Flag | Meaning |
|---|---|
| `-t N`, `--threads N` | Threads per inference call (default `min(4, nproc)`) |
| `-p N`, `--processors N` | Run N parallel decoder states (whisper only — uses `whisper_full_parallel`) |
| `--no-gpu` / `--device N` | Disable GPU or pin to GPU N |

## Whisper-only flags

These work both with the historical default whisper code path AND
with `--backend whisper`. The historical path retains a few extras
unique to it (`-owts` karaoke, full-mode JSON DTW tokens, `-di`
stereo diarize) — pass a `ggml-*.bin` model without `--backend` to
get them.

| Flag | Meaning |
|---|---|
| `--diarize` | Whisper-internal stereo diarization (upstream feature) |
| `-tdrz`, `--tinydiarize` | TinyDiarize speaker turn detection |
| `--carry-initial-prompt` | Forward `--prompt` across audio chunks |
| `-dtw` | Output DTW token-level timing in `-ojf` JSON |
| `-fa`, `-nfa` | Force flash-attn on / off |
| `-suppress-regex` | Suppress tokens whose detokenized text matches the regex |
| `-suppress-nst` | Suppress non-speech tokens |
| `-owts` | Karaoke-style word-timestamp WTS output |

For the full list of upstream whisper flags see `crispasr --help`
when invoked with a `ggml-*.bin` model loaded.

## Auto-download (`-m auto`)

When you pass `-m auto` (or `-m default`), CrispASR downloads the
default quantized model for the selected backend into
`~/.cache/crispasr/` on first use. The registry (kept in sync with
`src/crispasr_model_registry.cpp`):

| Backend | Download | Approx size |
|---|---|---|
| whisper | `ggerganov/whisper.cpp/ggml-base.en.bin` | ~147 MB |
| parakeet | `cstr/parakeet-tdt-0.6b-v3-GGUF` | ~467 MB |
| canary | `cstr/canary-1b-v2-GGUF` | ~600 MB |
| voxtral | `cstr/voxtral-mini-3b-2507-GGUF` | ~2.5 GB |
| voxtral4b | `cstr/voxtral-mini-4b-realtime-GGUF` | ~3.3 GB |
| granite | `cstr/granite-speech-4.0-1b-GGUF` | ~2.94 GB |
| granite-4.1 | `cstr/granite-speech-4.1-2b-GGUF` | ~2.94 GB |
| granite-4.1-plus | `cstr/granite-speech-4.1-2b-plus-GGUF` | ~5.6 GB |
| granite-4.1-nar | `cstr/granite-speech-4.1-2b-nar-GGUF` | ~5.4 GB (F16) / ~3.2 GB (Q4_K) |
| qwen3 | `cstr/qwen3-asr-0.6b-GGUF` | ~500 MB |
| cohere | `cstr/cohere-transcribe-03-2026-GGUF` | ~550 MB |
| wav2vec2 | `cstr/wav2vec2-large-xlsr-53-english-GGUF` | ~212 MB |
| omniasr | `cstr/omniASR-CTC-1B-GGUF` | ~551 MB |
| omniasr-llm | `cstr/omniasr-llm-300m-v2-GGUF` | ~580 MB |
| hubert | `cstr/hubert-large-ls960-ft-GGUF` | ~200 MB |
| data2vec | `cstr/data2vec-audio-960h-GGUF` | ~60 MB |

Downloads go through `curl` (preferred) with a `wget` fallback — **no
Python, no libcurl link dependency**. Works identically on Linux,
macOS, and Windows 10+ where `curl` ships in the base system. Models
are cached by filename; re-running is a single `stat()` check. The
same registry + cache helpers are reachable from the language
bindings (see [bindings.md](bindings.md)) so Python/Rust/Dart callers
can drive `-m auto`-style resolution without re-implementing it.

## Audio formats

Every audio path goes through `read_audio_data()` inherited from
upstream whisper.cpp. Two single-header decoders are embedded:

- **[miniaudio](https://miniaud.io/)** — WAV (any bit depth: 16/24/32
  PCM, IEEE float, A-law, μ-law, ADPCM), FLAC, MP3
- **[stb_vorbis](https://github.com/nothings/stb)** — OGG Vorbis

Out of the box, CrispASR accepts **WAV / FLAC / MP3 / OGG Vorbis** at
any bit depth and any sample rate (auto-resampled to 16 kHz), mono or
stereo (auto-mixed to mono).

| Format | Default build | `CRISPASR_FFMPEG=ON` |
|---|:---:|:---:|
| WAV / FLAC / MP3 / OGG | ✔ | ✔ |
| `.opus` | ✗ | ✔ |
| `.m4a` / `.mp4` / `.webm` | ✗ | ⚠ upstream crash, pre-convert |
| `.aiff` / `.wma` / raw PCM | ✗ | pre-convert |

For anything in the bottom half, the reliable path is
`ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav` then pass the
WAV. To enable `CRISPASR_FFMPEG=ON`, see [install.md](install.md).

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

| Backend | Honors `KV_QUANT`? |
|---|:-:|
| voxtral / voxtral4b | ✔ |
| qwen3-asr | ✔ |
| granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar | ✔ |
| glm-asr | ✔ |
| mimo-asr | ✔ |
| omniasr-llm | ✔ |
| gemma4-e2b | ✔ |
| orpheus | ✔ |
| qwen3-tts | ✔ (talker only) |
| whisper / parakeet / canary / cohere / fc-ctc / wav2vec2 / firered-asr / moonshine / kyutai-stt | — (no KV cache or model-specific path) |

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
(voxtral, voxtral4b, omniasr, qwen3_asr, granite_speech, orpheus).

### `CRISPASR_GGUF_MMAP=1` — zero-copy weight load

Map the GGUF file directly into the model's backend buffer instead
of read-and-copy. Saves one full copy of the GGUF on load: a 14.9 GB
F16 model goes from "load + 14.9 GB peak RSS" to "mmap +
~working-set RSS." No quality impact; pure load-time + RAM win.

```bash
CRISPASR_GGUF_MMAP=1 ./build/bin/crispasr --backend voxtral4b -m auto -f audio.wav
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

**Not yet supported:** N-layer CPU offload (`--n-gpu-layers N` style)
and KV-on-CPU-only modes — both tracked as PLAN #69 for future
implementation. For most voxtral4b VRAM-pressure cases the
`KV_QUANT=q4_0 + MMAP=1` combo above is sufficient; the layer-split
features are only needed when even that doesn't fit.

### TTS-side env vars

For TTS-specific deployment knobs (codec backend selection, graph
reuse, etc.) see [`tts.md`](tts.md):
- `QWEN3_TTS_CODEC_GPU` — clean codec-on-GPU path (CUDA / Vulkan)
- `QWEN3_TTS_O15` — code-predictor graph reuse (CPU/Metal opt-in)
- `KOKORO_GEN_GPU` — generator on GPU (CUDA / Vulkan)
- `VIBEVOICE_VAE_BACKEND={auto,cpu,gpu}` — VAE decoder placement

### Comparison with llama.cpp

For users coming from `llama.cpp`, here's how the equivalent knobs
map:

| Concern | llama.cpp | CrispASR |
|---|---|---|
| KV cache dtype | `--type-k q8_0 --type-v q8_0` (CLI flag, separate K/V) | `CRISPASR_KV_QUANT=q8_0` for symmetric, or `CRISPASR_KV_QUANT_K` / `_V` per half |
| mmap weights | `--no-mmap` (mmap is default **on**) | `CRISPASR_GGUF_MMAP=1` (mmap is default **off**) |
| Lock pages in RAM | `--mlock` | (not supported — `mmap+preload` is the closest analogue) |
| GPU layer count | `--n-gpu-layers N` / `-ngl N` (CLI flag) | not supported yet — see [PLAN #69a](https://github.com/CrispStrobe/CrispASR/blob/main/PLAN.md) |
| KV-on-CPU-only | `--no-kv-offload` | not supported yet — see [PLAN #69b](https://github.com/CrispStrobe/CrispASR/blob/main/PLAN.md) |
| Flash attention | `--flash-attn` / `-fa` | always-on where the backend's `capabilities()` declares `CAP_FLASH_ATTN` |
| Threads | `--threads N` / `-t N` | `--threads N` / `-t N` (matched) |
| Force CPU | `--gpu-layers 0` | `--no-gpu` / `--gpu-backend cpu` |

Differences worth flagging:

1. **mmap default.** llama.cpp defaults mmap **on**, CrispASR defaults
   it **off** (PLAN #51a flipped this opt-in pending wider RSS
   measurements). On hosts with plenty of RAM, the default-off
   behavior pays a copy that mmap would skip — set
   `CRISPASR_GGUF_MMAP=1` to match llama.cpp's behavior.
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
4. **No `--n-gpu-layers` yet.** This is the biggest missing knob
   for VRAM-constrained hosts. Tracked as PLAN #69a, prioritised by
   external request on issue #60. Today the workaround is the
   `KV_QUANT=q4_0 + MMAP=1` combo above, which usually clears
   enough headroom for voxtral4b-class models.

---

## See also

- [`docs/streaming.md`](streaming.md) — `--stream`, `--mic`, `--live`,
  sliding-window flags, per-token confidence
- [`docs/tts.md`](tts.md) — Kokoro / Qwen3-TTS / VibeVoice / Orpheus
  + every TTS-side env var (`QWEN3_TTS_CODEC_GPU`,
  `QWEN3_TTS_SKIP_REF_DECODE`, `QWEN3_TTS_O15`, `KOKORO_GEN_GPU`,
  `VIBEVOICE_VAE_BACKEND`, …)
- [`docs/server.md`](server.md) — HTTP `/inference` and OpenAI-compat
  `/v1/audio/transcriptions`
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
