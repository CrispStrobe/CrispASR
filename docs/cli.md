# CLI reference

`crispasr` extends upstream whisper.cpp's argument set with a handful
of backend-dispatch flags. Every historical whisper flag still works —
when you don't pass `--backend`, whisper is the default.

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

`--diarize`, `-tdrz` / `--tinydiarize`, `--carry-initial-prompt`,
`-dtw`, `-fa` / `-nfa`, `-suppress-regex`, `-suppress-nst`, and the
full upstream `crispasr --help` list.

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
