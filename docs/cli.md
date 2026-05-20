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
- [Diarization](#diarization) — `--diarize`, pyannote, embedder-based clustering
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
| `-ck N`, `--chunk-seconds N` | Fallback chunk size when VAD is off (default: 30 s for whisper, disabled for other backends) |
| `--chunk-overlap F` | Overlap context (seconds) at chunk boundaries (default 3.0) |
| `--lcs-dedup auto\|on\|off` | NeMo-style sub-word LCS dedup across chunk boundaries (default `auto` — fires when chunking with overlap) |
| `--lcs-min-length N` | Minimum LCS length to act on (default 1; raise to 3-4 on long-silence audio where blank tokens dominate boundaries) |
| `--parakeet-decoder ctc\|tdt` | Select CTC or TDT decode head for hybrid parakeet models |

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
(~885 KB). If you don't pass `--vad`, whisper falls back to fixed
30-second chunking (`-ck 30`). Backends with `CAP_UNBOUNDED_INPUT`
(parakeet, canary, wav2vec2, firered-asr, fastconformer-ctc,
granite-nar) process the full audio in one encoder pass by default
because their non-autoregressive encoders lose context at fixed chunk
boundaries (#89). LLM-based backends (cohere, moonshine, voxtral,
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
- **If parakeet OOMs on very long audio:** cap memory with explicit
  chunking (`--chunk-seconds 60` or `120` recommended). When chunking
  is active, each chunk is extended by `--chunk-overlap` seconds
  (default 3.0) on each side for encoder context, and a NeMo-style
  sub-word LCS pass (`--lcs-dedup auto`) removes any duplicate token
  runs that survive the boundary trim. Avoid `--chunk-seconds 30` —
  benchmarks show it is an anomalous worst case (93% of full quality)
  while 60/120/180 stay within ±1%. To disable the LCS pass for A/B
  testing pass `--lcs-dedup off`; to raise the match-length floor on
  audio with long silence regions pass `--lcs-min-length 3` or `4`.
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

CrispASR has **two distinct LID paths**: audio-LID (decides what language
the audio is in, before/during ASR) and text-LID (decides the language of
a transcript or arbitrary UTF-8 string).

### Audio LID (pre-/in-ASR)

| Flag | Meaning |
|---|---|
| `-l auto`, `--detect-language` | Auto-detect the input language. Backends without native lang-detect (cohere, canary, granite, voxtral, voxtral4b) get it via the LID pre-step |
| `--lid-backend NAME` | Audio-LID provider: `whisper` (default), `silero` (95 langs, 16 MB), `ecapa` (107 or 45 langs, 40-43 MB), `firered` (120 langs, 544 MB), or `off` |
| `--lid-model FNAME` | Override the audio-LID model path (default: auto-downloads `ggml-tiny.bin` ~75 MB on first use) |

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
the most languages (2102 ISO 639-3 + script). LID-176 is **CC-BY-SA-3.0
(viral)** — pick CLD3 or GlotLID for non-SA distribution.

## Diarization

Diarization assigns a speaker label to every transcribed segment. Two
high-level paths, both work with every ASR backend:

```bash
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
| `pyannote` | mono | Native GGUF pyannote-seg-3.0; runs once globally over the full audio, splits ASR segments at speaker-turn boundaries when per-word timestamps exist. Auto-downloads the GGUF via `--sherpa-segment-model auto` |
| `sherpa` / `ecapa` | mono | External `sherpa-onnx` subprocess with segmentation + speaker-embedding model. Requires `--sherpa-bin`, `--sherpa-segment-model`, `--sherpa-embedding-model` |

Bare `--diarize` (no `--diarize-method`) defaults to `energy` for stereo
input and `vad-turns` for mono — the historical behaviour.

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
extending the factory's dispatch. Tune clustering with
`--diarize-cluster-threshold X` (default 0.5; higher = more clusters)
and `--diarize-max-speakers N` (default 8 — hard cap).

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
| `--no-punctuation` | Disable punctuation in the output. Native for cohere/canary, post-processed for everyone else |

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
| chatterbox / chatterbox-turbo / kartoffelbox-turbo / lahgtna-chatterbox | ✔ (T3 LM side; S3Gen Conformer attention is F32 by design) |
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

For users with very long context where even `KV_QUANT=q4_0` won't
fit in VRAM. Allocates the KV cache on the CPU backend instead of
the GPU backend, even when model weights are active on GPU.

```bash
# Long-context fallback when VRAM is exhausted
CRISPASR_KV_ON_CPU=1 ./build/bin/crispasr --backend voxtral4b -m auto -f long-audio.wav

# Stacks with KV_QUANT_K/_V — minimum-memory KV path
CRISPASR_KV_ON_CPU=1 CRISPASR_KV_QUANT_K=q8_0 CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f long-audio.wav
```

**Try `KV_QUANT` first.** The expensive part isn't the alloc —
every attention step copies the KV slice GPU↔CPU↔GPU. The
PCIe / unified-memory traffic is typically slower than just running
with quantised KV in VRAM. Reach for `KV_ON_CPU` only when
quantisation alone can't fit the context.

The verbose log line shows `(on cpu)` vs `(on gpu)` so you can
confirm where the cache landed:

```
voxtral4b: kv cache 169 MiB k=q8_0 v=q4_0 (on cpu, ...)
```

Same per-backend coverage as `KV_QUANT` (voxtral, voxtral4b,
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

**Stacks with `KV_ON_CPU` and `KV_QUANT_K/_V`** — set all three for
the most aggressive memory footprint reduction. `KV_QUANT` is
cheaper than layer offload; reach for `N_GPU_LAYERS` only when the
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
| mmap weights | `--no-mmap` (mmap is default **on**) | `CRISPASR_GGUF_MMAP=0` (mmap is default **on** since 0.6.7) |
| Lock pages in RAM | `--mlock` | (not supported — `mmap+preload` is the closest analogue) |
| GPU layer count | `--n-gpu-layers N` / `-ngl N` (CLI flag) | not supported yet — see [PLAN #69a](https://github.com/CrispStrobe/CrispASR/blob/main/PLAN.md) |
| KV-on-CPU-only | `--no-kv-offload` | not supported yet — see [PLAN #69b](https://github.com/CrispStrobe/CrispASR/blob/main/PLAN.md) |
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
4. **No `--n-gpu-layers` yet.** This is the biggest missing knob
   for VRAM-constrained hosts. Tracked as PLAN #69a, prioritised by
   external request on issue #60. Today the workaround is the
   `KV_QUANT=q4_0 + MMAP=1` combo above, which usually clears
   enough headroom for voxtral4b-class models.

---

## See also

- [`docs/streaming.md`](streaming.md) — `--stream`, `--mic`, `--live`,
  sliding-window flags, per-token confidence
- [`docs/tts.md`](tts.md) — Kokoro / Qwen3-TTS / VibeVoice / Orpheus / Chatterbox
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
