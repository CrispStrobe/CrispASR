# Environment variables

CrispASR exposes a large number of environment variables — tuning knobs, device
placement overrides, benchmark timers, and debug/dump switches. This page is the
single reference for all of them: the naming convention, the cross-cutting
variables, the per-backend variables, and the ones CrispASR deliberately does
**not** own.

> This document was introduced for issue #265 ("consistency and documentation").
> The same change standardized every backend-owned variable onto one prefix and
> unified the reference-voice cache location (see below).

## Naming convention

Every CrispASR-owned variable is named:

```
CRISPASR_<BACKEND>_<FEATURE>
```

for example `CRISPASR_OMNIVOICE_CODEC_GPU`, `CRISPASR_IRODORI_CODEC_GPU`,
`CRISPASR_PARAKEET_BENCH`. Process-global variables that are not tied to one
backend drop the `<BACKEND>` segment (`CRISPASR_MODELS_DIR`, `CRISPASR_KV_ON_CPU`).

### Legacy aliases and deprecation warnings

Historically many backends used a bare prefix (`OMNIVOICE_CODEC_GPU`,
`QWEN3_TTS_BENCH`, `CHATTERBOX_DEBUG`, …). Those bare names **still work** as
legacy aliases so existing scripts, notebooks, and Kaggle A/B kernels keep
running. The alias is derived automatically by stripping the `CRISPASR_` prefix,
so for any `CRISPASR_FOO_BAR` documented here the bare `FOO_BAR` is also honored.

The first time a run reads a value from a **legacy** name, CrispASR prints a
one-time notice to stderr:

```
[crispasr] warning: environment variable 'OMNIVOICE_CODEC_GPU' is deprecated and
will be removed in a future release; use 'CRISPASR_OMNIVOICE_CODEC_GPU' instead
(the old name still works for now). ...
```

Migrate to the canonical `CRISPASR_`-prefixed name at your convenience. To
silence the warnings (e.g. for a pipeline that scrapes stderr and cannot migrate
yet), set:

```
CRISPASR_SUPPRESS_ENV_DEPRECATION=1
```

The lookup + aliasing is implemented once in `src/core/crispasr_env.h`
(`crispasr_env::get / truthy / present`); new backends should read env vars
through that helper with a canonical `CRISPASR_`-prefixed name.

### Value/truthiness conventions

Most variables are read with one of two conventions — check the surrounding
suffix legend, but as a rule:

- **Flag (opt-in)** — set to `1` (or any non-empty, non-`0` value) to enable.
  Some legacy flags treat *being set at all* as on, so prefer `=1` / unset it
  entirely rather than `=0` when a flag's exact semantics matter.
- **Path / value** — the variable's value is a filesystem path, integer, or
  float used directly.

### Common suffix legend

The same feature suffixes recur across nearly every backend. Rather than repeat
them per backend below, they mean:

| Suffix | Meaning |
|--------|---------|
| `_BENCH` | Print per-stage wall-clock timings for that engine. |
| `_DEBUG` / `_DIAG` | Verbose debug logging. |
| `_DUMP_DIR` / `_DUMP` / `_DUMP_*` | Write intermediate tensors / stages to disk (diagnostics, diff-harness). |
| `_FORCE_SCALAR` | Disable SIMD kernels (numeric-parity debugging). |
| `_GPU` / `_USE_GPU` / `_FORCE_METAL` | Force that stage/codec onto the GPU. |
| `_CPU` / `_CPU_ONLY` / `_FORCE_CPU` | Force that stage/codec onto the CPU. |
| `_NO_FA` / `_FLASH_ATTN` | Toggle flash-attention. |
| `_FASTCONV` (`_DEBUG`) | Enable the baked-F32 / matmul conv fast path for a codec/vocoder. |
| `_CFG_INTERVAL` (`_DEBUG`) | Interval-CFG cadence for a flow-matching decoder. |
| `_SEED` | RNG seed for reproducible generation. |
| `_*_FILE` / `_*_PATH` / `_*_GGUF` | Override an asset/model/fixture path (mostly dev/test). |

## Global / cross-cutting variables

These are not tied to a single backend.

### Paths, cache, and models

| Variable | Purpose |
|----------|---------|
| `CRISPASR_MODELS_DIR` | Directory searched for GGUF models (also the auto-download target root). |
| `CRISPASR_CACHE_DIR` | Base cache directory for auto-downloaded models/assets. Resolution order is `CRISPASR_CACHE_DIR` → `CRISPASR_MODELS_DIR` → platform default (`$HOME/.cache/crispasr`; `%USERPROFILE%\.cache\crispasr` on Windows). |
| `CRISPASR_SCRATCH_DIR` | Scratch directory for temporary run artifacts. |
| `CRISPASR_DUMP_DIR` | Global tensor-dump directory (diagnostics). |
| `CRISPASR_GGUF_MMAP` / `CRISPASR_GGUF_PRELOAD` | Control GGUF mmap vs. preload-into-RAM loading. |
| `CRISPASR_GGUF_MAX_ALLOC_CHUNK` | Lower the per-buffer allocation cap in bytes (default 1.5 GiB) used to split a model across several backend buffers on drivers with a small `maxMemoryAllocationSize` (#276). |
| `CRISPASR_MLOCK` | mlock model weights into RAM. |
| `CRISPASR_IGNORE_CPU_ISA` | `=1` exactly (no other value works) continues past the startup build-vs-host CPU instruction-set check (#380) instead of exiting; the process will SIGILL at the first compute op the CPU can't run. |

### GPU / device placement

| Variable | Purpose |
|----------|---------|
| `CRISPASR_N_GPU_LAYERS` | Number of transformer layers to offload to the GPU. |
| `CRISPASR_ARG_DEVICE` | Default device selection for the CLI. |
| `CRISPASR_KV_ON_CPU` | Keep the KV cache on the CPU. |
| `CRISPASR_KV_QUANT` / `_KV_QUANT_K` / `_KV_QUANT_V` / `_KV_READ_F32` | KV-cache quantization / read format. |
| `CRISPASR_GPU_PREF_CPU_LEGACY` | `1` restores the pre-T18 behaviour where `--gpu-backend cpu` fell through to the best GPU. Off by default: the fall-through silently constructed a Metal device the flag exists to avoid. |
| `CRISPASR_METAL_PIPELINE_CACHE_MAX_MB` | Size cap in MiB (default `64`) above which the on-disk Metal pipeline-cache archive is skipped at init — opening it costs ~1 ms/MB. `0` = uncapped (always use it). macOS only. |

> Device *selection* across compiled backends also honors the standard ggml /
> CUDA variables `CUDA_VISIBLE_DEVICES` and `GGML_VK_VISIBLE_DEVICES` — see
> "Variables CrispASR does not own" below.

### Session / long-audio chunking

| Variable | Purpose |
|----------|---------|
| `CRISPASR_SESSION_AUTOCHUNK` | Enable auto-chunking in the session API for long audio. |
| `CRISPASR_SESSION_CHUNK_SECONDS` | Chunk length (seconds) for session auto-chunking. |
| `CRISPASR_SESSION_PERBACKEND_CHUNK` | Use per-backend chunk-window tuning instead of a flat window. |
| `CRISPASR_SESSION_UNIFIED_DISPATCH` | Route surfaces through the unified library dispatch path. |
| `CRISPASR_SLICE_PIPELINE` | Force the CLI's encode ∥ decode slice pipeline on/off. The override may only turn it OFF, or ON where it is *already* safe — it can never switch off one of the safety conditions (`-p N`, `--return-logits`, gap-fill re-entry, single slice). |

### Post-decode hygiene (PLAN.md §W2–W7)

All OFF unless set. Each of these can delete or alter user-visible text, so
none of them switches on by surprise; a wrong deletion is worse than a
surviving artifact. Applied on both the CLI and the session C-ABI.

| Variable | Purpose |
|----------|---------|
| `CRISPASR_SEG_MAX_CHARS` | Truncate any segment longer than N **code points**, backing up to the last `。．.！!？?、,` but never below 75% of the cap. A line past the cap is almost always a repetition hallucination that survived the n-gram collapse. |
| `CRISPASR_SEG_DROP_NONVERBAL` | Drop segments that are entirely a non-verbal marker — `[Music]`, `(applause)`, `（喘ぎ声）`, `♪`. Running speech merely *containing* such a word is never dropped. |
| `CRISPASR_SEG_LOGPROB_THOLD` | Drop segments whose average log-probability is below this. Post-hoc, on top of the decoder's own fallback gate. |
| `CRISPASR_SEG_LOGPROB_MARGIN` | Loosen that threshold by this much for segments ≤1.6 s — a short segment's mean logprob is noisier, so it gets more room, not less. |
| `CRISPASR_SEG_MERGE_REPEATS` | Collapse runs of near-identical adjacent segments into one spanning the whole run. Catches a phrase repeating *across* segment boundaries, which per-segment loop fixes cannot see. |
| `CRISPASR_SEG_MERGE_SIMILARITY` | Similarity bar for the above (default 0.90; LCS over code points). |
| `CRISPASR_SEG_MERGE_GAP_CS` | Never merge across a gap wider than this many centiseconds (default 200). Two identical lines a minute apart are two real utterances. |
| `CRISPASR_SEG_MERGE_MIN_RUN` | Minimum consecutive similar segments before merging (default 3), so an ordinary repeated "yes." pair survives. |

### Alignment and VAD sanity checks

| Variable | Purpose |
|----------|---------|
| `CRISPASR_ALIGN_SENTINEL` | `0` disables the forced-alignment collapse check. On by default, **detect + warn only**. Catches `ctc_forced_align()` returning words at `t0 == t1 == 0` — its two silent-zero paths (characters absent from the CTC vocab, or a word the Viterbi path never visited) produce garbage timestamps inside a *successful* return. |
| `CRISPASR_ALIGN_SENTINEL_REDISTRIBUTE` | `1` opts into repair: respace the words across the clip in proportion to character count. Off by default — a wrong auto-repair would be just as invisible as the collapse. |
| `CRISPASR_VAD_FAILOVER` | `0` disables the VAD sanity check. On by default: if a clip over 120 s comes back with under 1% speech coverage (or a couple of segments covering under 10% of a very long clip), the VAD is wrong and the run falls back to fixed full-clip chunks rather than losing the transcript. |
| `CRISPASR_NGRAM_LOOPFIX_OFF` | `1` disables the repeated-n-gram collapse entirely, exposing the RAW decoded text. Diagnostic: for telling whether a loop originates in the decode itself or is merely being masked. |
| `CRISPASR_ORDER_WARN` | `0` disables the one-shot "segment timestamps go backwards" warning. On by default; detect + warn only. Cues that merely *overlap* are deliberately not flagged (gap-fill jitter). |
| `CRISPASR_ALIGN_NO_ROMANIZE` | `1` passes non-Latin reference text through raw instead of auto-romanizing it for a CTC aligner with a Latin vocabulary (#252). Since #419 the romanization is only the aligner's internal label — aligned words hand back the original script, so srt/vtt/`-sp`/`-sow` output no longer flips Cyrillic/CJK transcripts to transliteration. |
| `CRISPASR_ALIGN_DEBUG` | `1` prints the romanized reference transcript the aligner actually used. |

### Decoding / beam search (shared)

| Variable | Purpose |
|----------|---------|
| `CRISPASR_MAES_BETA` / `_MAES_GAMMA` / `_MAES_NUM_STEPS` | MAES beam-search parameters. |
| `CRISPASR_TDT_BATCH` / `CRISPASR_RNNT_BATCH` | Batch the TDT / RNNT joint decode. |
| `CRISPASR_RNNT_GGML_PERSTEP` | Per-step (vs. persistent-graph) ggml RNNT decode. |
| `CRISPASR_NGRAM_LOOPFIX_OFF` | Disable the n-gram decode-loop breaker. |
| `CRISPASR_STREAM_SLICE_MEMO` | Memoize per-slice streaming partial decodes by absolute sample range (#404). **Default ON** — finals byte-equal, wall −12 % CPU / −6 % GPU in the quiet-box A/B; `=0` re-decodes closed slices every step. |
| `CRISPASR_GAP_FILL` / `_GAP_FILL_MIN_CS` | Re-transcribe spans a first pass left empty (long audio); on by default for parakeet, threshold non-JA 300 cs / JA 100 cs. |

### G2P / phonemizer

| Variable | Purpose |
|----------|---------|
| `CRISPASR_CMUDICT_PATH` | Path to the CMUdict pronunciation dictionary. |
| `CRISPASR_DE_DICT_PATH` / `_FR_DICT_PATH` / `_ES_DICT_PATH` | Language-specific pronunciation dictionaries. |
| `CRISPASR_G2P_DICT_SOURCE` / `_G2P_MODEL_PATH` | G2P dictionary source / neural G2P model path. |
| `CRISPASR_ESPEAK_DATA_PATH` | eSpeak-NG data directory. |
| `CRISPASR_MISAKI_DICT_PATH` | Path to the misaki US contextual-word dictionary (default `~/.cache/crispasr/misaki-us.txt`) used by the English misaki G2P (#316). |
| `CRISPASR_KOKORO_G2P` | Kokoro G2P backend selection. |
| `CRISPASR_KOKORO_MISAKI_IPA` | `0` disables the espeak-IPA → misaki-alphabet conversion Kokoro needs (#316), restoring the raw G2P spelling for A/B. On by default. |
| `CRISPASR_G2P_DE_UNSTRESS` | `1` reads the German closed class the way espeak reads it in a SENTENCE (`sie` → `ziː`) instead of the citation form our per-word dictionary stores (`zˈiː`). Off by default: it takes phoneme agreement with espeak from 45.9% to 87.1%, but the ASR round-trip could not resolve a difference, and that metric measures intelligibility rather than naturalness (#316). |
| `CRISPASR_KOKORO_DE_MISAKI_ALPHABET` | `1` applies misaki's tied-sequence collapse for German (`tsvˈaɪ` → `ʦvˈI`), which is what the published training recipe does. Off by default: it made the ASR round-trip worse on the `kokoro-de-hui-base` model we ship, which appears to predate that part of the recipe (#316). The `ʏ`→`y` vocabulary fixup is applied either way. |
| `CRISPASR_T5_REPEAT_BREAK` | `0` disables the decode-loop break for madlad/T5 translation, restoring exact PyTorch-blueprint behaviour. On by default: MADLAD greedy-decodes into a repeated token cycle on some short inputs and burns the whole token budget on it. The blueprint does the same — this is a deliberate improvement on it, not a parity fix (#333). |
| `CRISPASR_T5_NO_KV_REUSE` | `1` re-forwards the whole decoder prefix each step instead of appending to the KV cache. Same output, much slower; an A/B lever for isolating cache bugs (#333). |
| `CRISPASR_T5_KEEP_EMBED` | `1` keeps `shared.embed.*` and `lm_head.*` at source precision when quantizing a T5 model. **Off by default because it was measured and loses**: on madlad400 it makes q8_0 3.38→3.62 GB and q4_k 2.04→2.41 GB for a worst-stage cosine that does not improve (0.999922→0.999920, 0.992929→0.992606). The Q4_K loss accumulates through the 32 encoder blocks, not in the embedding lookup (#333). |
| `CRISPASR_KOKORO_PUNCT` | `0` drops punctuation from the phoneme string for the German/French/Spanish built-in G2Ps, restoring pre-0.8.26 behaviour for A/B. On by default: Kokoro's vocabulary contains `,.;:!?` and they are how it pauses (#316). English is not gated — it is settled against misaki. |

### Watermark / provenance

| Variable | Purpose |
|----------|---------|
| `CRISPASR_NO_WATERMARK` | Disable the audio watermark. |
| `CRISPASR_WATERMARK_LEGACY` | Use the legacy watermark path. |
| `CRISPASR_WATERMARK_DETECT` | Which statistic `--detect-watermark` uses: `frames` (per-frame *t* + decoy specificity) or `sign` / `0` (the older averaged-spectrum sign test, kept for A/B and for re-reading an older release's score). |
| `CRISPASR_NO_C2PA_REMUX` | Skip the C2PA MP4 remux step. |
| `CRISPASR_CONSENT_LOG` | Path to a JSON-Lines sink for voice-cloning consent records. Without it the records only go to stderr, which is interleaved with model-load noise and so a poor evidential artefact. |

### Quantization / diff-harness / misc

| Variable | Purpose |
|----------|---------|
| `CRISPASR_QUANT_LMHEAD` | Quantize the LM head during `crispasr-quantize`. |
| `CRISPASR_IMATRIX_OUT` | Importance-matrix output path. |
| `CRISPASR_ACTDUMP_OUT` / `_ACTDUMP_TENSOR` | Activation dump output / target tensor. |
| `CRISPASR_DIFF_NO_GPU` / `_DIFF_USE_GPU` / `_DIFF_SLICES` / `_DIFF_STAGES` | `crispasr-diff` harness controls. |
| `CRISPASR_MEL_SERIAL` | Force the serial STFT. The parallel mel/STFT path is DEFAULT ON since #305; this is the opt-out. (The older opt-in `CRISPASR_MEL_PARALLEL` is no longer read.) |
| `CRISPASR_MEL_TIMING` | Print mel/STFT stage timings. |
| `CRISPASR_HQ_RESAMPLE` | `0` selects the cheap linear resampler for CLI input decoding instead of the high-quality one. |
| `CRISPASR_CORE_ATTN_EAGER_F32` | Force the shared attention helper's eager (non-flash) path to F32 accumulation. |
| `CRISPASR_CORE_ATTN_DUMP_FA_LAYER` | Dump the shared attention helper's flash-attn inputs/outputs for one layer index. |
| `CRISPASR_VERBOSE` | Global verbose output. |
| `CRISPASR_NO_WARMUP` / `CRISPASR_WARMUP` | Skip / force the model warmup pass. |
| `CRISPASR_SERVER_WORKERS` / `CRISPASR_API_KEYS` | HTTP server worker count / API keys. |
| `CRISPASR_TEST_STREAM_THROW` | Test-only: lets the server's streaming worker throw on the magic input `__throw_test__` (both the variable *and* the input are required, so it cannot fire in production). |

### Container / launcher

Read by the Docker images and `.devops/run-server.sh`, not by the C++ itself —
`crispasr-diagnostics` echoes them so a support dump shows how the container was
started.

| Variable | Purpose |
|----------|---------|
| `CRISPASR_BACKEND` | Backend the container's entrypoint should select (`docker-compose*.yml`, `.env.example`). |
| `CRISPASR_USE_CUDA_COMPAT` | `1` prepends `/usr/local/cuda/compat` to `LD_LIBRARY_PATH` in the CUDA images — for hosts whose driver is older than the image's CUDA runtime. |

### Vendored ggml (CrispASR-added)

CrispASR's in-tree `ggml/` carries a few CrispASR-prefixed knobs on top of
upstream's `GGML_*` set. They are read by `getenv` directly (no legacy alias).

| Variable | Purpose |
|----------|---------|
| `CRISPASR_GGML_ALLOC_TRACE` | Trace `ggml-alloc` graph-allocation decisions. |
| `CRISPASR_GGML_ALLOC_TRACE_MAX_PASSES` | Number of allocation passes that trace prints (default: unlimited once the trace is on). |
| `CRISPASR_METAL_N_CB` | Override the Metal backend's command-buffer count (#83). |
| `CRISPASR_METAL_PROFILE` | `1` whole-graph host/GPU split, `2` per-op breakdown, `3` per-op plus a per-node trace announced *before* each encode — with `3` the last line names the node an encode faulted on. |
| `CRISPASR_METAL_STRICT_FP` | Compile the Metal kernels with fast-math OFF (#83). Costs throughput; buys bit-identical CPU/GPU output where operand reordering was downconverting F32 intermediates. |
| `CRISPASR_METAL_FORCE_BARRIER` | `1` forces a memory barrier before every Metal op (concurrency-hazard bisection, #83). |
| `CRISPASR_METAL_IM2COL_FLAT` | `0` restores the legacy IM2COL Metal kernel. The flat kernel (one thread per dst element) is the default. |
| `CRISPASR_FORCE_BLIT_COPY` | Use the blit-encoder copy path even for a shared (unified-memory) Metal buffer (#83). |
| `CRISPASR_FORCE_DMB` | Insert a full memory barrier after the host memcpy into a shared Metal buffer (#83). |

## Reference-voice cache (voice cloning)

Encoding a reference clip for voice cloning (a codec encoder, a Conformer /
Perceiver, or an ASR pass) is slow and produces a small, reusable blob. Every
voice-cloning TTS backend caches that blob through **one shared mechanism**
(`src/core/tts_ref_cache.h`), so the location and disable switch are identical
across backends (issue #265 unified OmniVoice — which previously used a bespoke
`~/.cache/crispasr` cache — onto this path):

| Variable | Purpose |
|----------|---------|
| `CRISPASR_TTS_REF_CACHE` | Set to `0` to disable reference-voice caching everywhere. |
| `CRISPASR_TTS_REF_CACHE_DIR` | Override the cache directory (default `<TMPDIR>/crispasr-tts-refcache`). |

Content-addressed entries are keyed by a hash of the raw reference (plus, for
some backends, an encoder-weight fingerprint) and tagged per backend
(`irodori-latent`, `openvoice2-se`, `omnivoice-voice`, `f5-reftext`, …) so no two
backends read each other's blob. OmniVoice additionally honors the legacy
`CRISPASR_OMNIVOICE_VOICE_CACHE=0` as an alias for the shared disable switch.

## Variables CrispASR does not own

These are read by CrispASR but are **OS / third-party conventions** and are
intentionally *not* renamed to the `CRISPASR_` prefix:

| Variable | Origin |
|----------|--------|
| `HOME`, `USERPROFILE`, `LOCALAPPDATA`, `XDG_CACHE_HOME` | OS home / cache dirs. |
| `TMPDIR`, `TEMP`, `TMP` | OS temp dir (also the default ref-cache root). |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | HuggingFace auth (Hub convention). |
| `CUDA_VISIBLE_DEVICES` | CUDA device selection. |
| `GGML_VK_VISIBLE_DEVICES` | ggml Vulkan device selection. |
| `LLAMA_*` | Vendored `llama.cpp` (talk-llama example). |

### Sibling modular libraries (own conventions)

The in-tree modular libraries keep their own established prefixes and are synced
from their sibling repos, so their variables are **not** part of the `CRISPASR_`
scheme:

- **`crisp_audio/`** — audio tower: `CRISP_AUDIO_DUMP_STAGES`, `CRISP_AUDIO_KEEP_PAD_FRAMES`, `CRISP_AUDIO_WINDOWED_ATTN`.
- **`glint/`** — clean-room MP3/AAC codec: `GLINT_*`, `AACDBG`.
- **`crisp_lid/` · `crisp_punc/` · `crisp_truecase/`** — standalone LID / punctuation / truecasing libraries; they and their `src/` counterparts follow each library's own naming (`LID_*`, `FIREREDPUNC_*`, `PCS_*`, `TRUECASER_*`). `FIREREDPUNC_DEBUG=1` prints each restore pass as `[PUNCDBG] in=<…>` / `out=<…>`, which is the quickest way to see a backend's *true* model output — `--no-punctuation` is not, because it strips punctuation after the fact and so hides text the model punctuated itself.
  > ⚠ **These libraries are built by the main target, and each has a second copy under `src/`.** `src/CMakeLists.txt` prefers `crisp_punc/` (etc.) and falls back to the `src/` copy only when the sibling directory is missing from a checkout — so the `crisp_punc/` copy is what normally links, and **a change must be applied to both**. #308's capitalisation fix went into `src/fireredpunc.cpp` alone and was dead code for months while the shipping copy kept the bug. `tests/test-punc-copies-in-sync.cpp` now fails if they diverge.

## Test fixtures

The live/integration test suite reads a few `CRISPASR_`-prefixed fixture
variables (model/audio paths); `tests/env-live-tests.sh` sets sensible defaults.
Like every other variable here, the pre-standardization bare names are still
honored as deprecated aliases:

| Variable | Purpose |
|----------|---------|
| `CRISPASR_PARAFORMER_MODEL` / `_PARAFORMER_MODEL_Q4K` / `_PARAFORMER_AUDIO_ZH` / `_PARAFORMER_AUDIO_EN` | Paraformer test model / audio paths. |
| `CRISPASR_AUDIOSEAL_GGUF` | AudioSeal test model path. |
| `CRISPASR_PIPER_TEST_MODEL` | Piper phonemize test model path. |
| `CRISPASR_OV2_DUMP_DIR` | OpenVoice2 HiFi test dump directory. |
| `CRISPASR_MODEL_BTC_CHORDS` | BTC chord-recognition test model path. Defaults to `$CRISPASR_MODELS_DIR/btc-chords-large-f32.gguf`. |

---

# Per-backend variables

Every variable below is the canonical `CRISPASR_`-prefixed name; the bare form
(without `CRISPASR_`) is accepted as a deprecated alias. See the **suffix
legend** above for the meaning of the common `_BENCH` / `_DEBUG` / `_DUMP_*` /
`_GPU` / `_CPU` / `_FORCE_SCALAR` / `_FASTCONV` / `_CFG_INTERVAL` / `_SEED`
suffixes.

### AAC codec

- `CRISPASR_AAC_DEBUG`
- `CRISPASR_AAC_DECODER`

### ARK-ASR

- `CRISPASR_ARKASR_BLOCK_FROM_ID`
- `CRISPASR_ARKASR_CPU`
- `CRISPASR_ARKASR_DEBUG_GEN`
- `CRISPASR_ARKASR_GPU`
- `CRISPASR_ARKASR_INSTRUCTION`
- `CRISPASR_ARKASR_MAX_SINGLE_PASS_S`
- `CRISPASR_ARKASR_NO_CHUNK_CONTEXT`
- `CRISPASR_ARKASR_NO_EOS_SUPPRESS`
- `CRISPASR_ARKASR_NO_SPECIAL_SUPPRESS`
- `CRISPASR_ARKASR_TIMING`

### AudioSeal watermark

- `CRISPASR_AUDIOSEAL_BENCH`
- `CRISPASR_AUDIOSEAL_DEBUG`
- `CRISPASR_AUDIOSEAL_DUMP_STAGES`

### BananaMind TTS

- `CRISPASR_BANANAMIND_DEBUG`
- `CRISPASR_BANANAMIND_TTS_BENCH`

### Bark

- `CRISPASR_BARK_BENCH`
- `CRISPASR_BARK_DECODE_CODES`
- `CRISPASR_BARK_DUMP_DIR`

### Beat-This (beat tracking)

- `CRISPASR_BEAT_THIS_DEBUG`

### BERT encoder

- `CRISPASR_BERT_ENCODER_BENCH`

### BTC chord recognition

- `CRISPASR_BTC_DEBUG`
- `CRISPASR_BTC_DUMP_FEAT`
- `CRISPASR_BTC_MAJ_MIN` — collapse the 170-class chord output to the 25-class
  maj/min vocabulary. Default off (full 170-class output): 170 reduces to
  maj/min at runtime, but a 25-class model can never be expanded, so the
  170-class checkpoint is the shipping default.

> The shipped BTC weights are **CC-BY-NC-SA** (trained on Isophonics / Robbie
> Williams / UsPop2002 chord annotations) even though the upstream BTC code and
> CrispASR itself are MIT. The registry refuses to download them without
> `--accept-license cc-by-nc-sa-4.0` (or `CRISPASR_ACCEPT_LICENSE`). A
> commercial product must supply its own weights.

### Canary / Canary-CTC / Canary-Qwen

- `CRISPASR_CANARY_BENCH`
- `CRISPASR_CANARY_CTC_BENCH`
- `CRISPASR_CANARY_QWEN_BENCH`
- `CRISPASR_CANARY_QWEN_DEBUG`
- `CRISPASR_CANARY_QWEN_MIN_ENC_FRAMES`
- `CRISPASR_CANARY_QWEN_NO_ECHO_STRIP`
- `CRISPASR_CANARY_LEGACY_STREAM`
- `CRISPASR_CANARY_SEAM_DEDUP`
- `CRISPASR_CANARY_STREAM_THRESHOLD_S`

### Chatterbox

- `CRISPASR_CHATTERBOX_BENCH`
- `CRISPASR_CHATTERBOX_BENCH_B2`
- `CRISPASR_CHATTERBOX_DEBUG`
- `CRISPASR_CHATTERBOX_DUMP_ATTN_AT`
- `CRISPASR_CHATTERBOX_DUMP_FFN_AT`
- `CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS`
- `CRISPASR_CHATTERBOX_DUMP_KPROJ_AT`
- `CRISPASR_CHATTERBOX_DUMP_KROPE_AT`
- `CRISPASR_CHATTERBOX_DUMP_KV_AT`
- `CRISPASR_CHATTERBOX_DUMP_KV_LAYER`
- `CRISPASR_CHATTERBOX_DUMP_LAYER`
- `CRISPASR_CHATTERBOX_DUMP_LOGITS_AT`
- `CRISPASR_CHATTERBOX_DUMP_NORM_AT`
- `CRISPASR_CHATTERBOX_DUMP_QPROJ_AT`
- `CRISPASR_CHATTERBOX_DUMP_VPROJ_AT`
- `CRISPASR_CHATTERBOX_DUMP_WK`
- `CRISPASR_CHATTERBOX_FLASH_ATTN` — force `ggml_flash_attn_ext` for the T3
  GPT-2 (turbo/nano) attention even on Vulkan, where naive attention is the
  default since issue #402 (RADV 780M crashes in the Vulkan FLASH_ATTN_EXT
  pipeline; the explicit softmax(QK^T)V path is verified working there).
- `CRISPASR_CHATTERBOX_FORCE_GPU`
- `CRISPASR_CHATTERBOX_FULL_CPU`
- `CRISPASR_CHATTERBOX_LANG`
- `CRISPASR_CHATTERBOX_KV_CONT` — materialize GPT-2 K/V layer views before
  flash attention, reproducing the pre-PR-410 path for correctness/performance
  A/Bs. The default passes the views directly; naive attention still
  materializes them because its matrix operations require that layout.
- `CRISPASR_CHATTERBOX_NAIVE_ATTN` — force the explicit softmax(QK^T)V T3
  attention on every backend (debug gate; outranks `_FLASH_ATTN`).
- `CRISPASR_CHATTERBOX_S3GEN_CPU`
- `CRISPASR_CHATTERBOX_T3_CPU_S3GEN_GPU`
- `CRISPASR_CHATTERBOX_T3_GPU`
- `CRISPASR_CHATTERBOX_SEED`
- `CRISPASR_CHATTERBOX_SYN_TEXT`
- `CRISPASR_CHATTERBOX_T3_BUCKET_REUSE`
- `CRISPASR_CHATTERBOX_T3_CFG_B2`
- `CRISPASR_CHATTERBOX_T3_CFG_BUCKET`
- `CRISPASR_CHATTERBOX_T3_SEED`
- `CRISPASR_CHATTERBOX_TEMP`
- `CRISPASR_CHATTERBOX_THREADS`

### Chatterbox S3Gen

- `CRISPASR_S3GEN_CFG_INTERVAL`
- `CRISPASR_S3GEN_DUMP`
- `CRISPASR_S3GEN_DUMP_UNET`
- `CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK`
- `CRISPASR_S3GEN_ENCODER_CPU`
- `CRISPASR_S3GEN_FASTCONV`
- `CRISPASR_S3GEN_FASTCONV_DEBUG`
- \`CRISPASR_S3GEN_SIMDCONV\` — opt into the CPU HiFT packed SIMD path for all 72 ResBlock Conv1d kernels; default off. Stays OPT-IN deliberately: the Kaggle A/B measured a 5.4 % s3gen REGRESSION on Xeon avx512f against the contributor's 1.25x win on Zen 4 — micro-arch dependent; A/B on your own hardware before enabling.
- `CRISPASR_S3GEN_SIMDCONV_DEBUG` — print pack count, selected ISA, and CPU/GPU fallback status.
- `CRISPASR_S3GEN_RC_AS_MUL_MAT`
- `CRISPASR_S3GEN_VOCODER_CPU`
- `CRISPASR_S3GEN_UNET_CFG_SINGLE`
- `CRISPASR_S3GEN_UNET_CPU`
- `CRISPASR_S3GEN_UNET_GALLOCR`
- `CRISPASR_S3GEN_UNET_KEEP_GPU_OP`
- `CRISPASR_S3GEN_UNET_MARK_DB_OUT`
- `CRISPASR_S3GEN_UNET_MARK_DB_RESNET`
- `CRISPASR_S3GEN_UNET_MARK_DB_TB`
- `CRISPASR_S3GEN_UNET_MARK_MB_OUT`
- `CRISPASR_S3GEN_UNET_MARK_MB_OUT_INDEX`
- `CRISPASR_S3GEN_UNET_MARK_MB_OUT_MAX`
- `CRISPASR_S3GEN_UNET_MARK_MB_RESNET`
- `CRISPASR_S3GEN_UNET_PIN_CPU_OP`
- `CRISPASR_S3GEN_UNET_PRESERVE_INTERMEDIATES`
- `CRISPASR_S3GEN_UNET_PROBE_BLOCK1`
- `CRISPASR_S3GEN_UNET_PROBE_DENOISER_OUT`
- `CRISPASR_S3GEN_UNET_PROBE_INPUT_SNAPSHOT`
- `CRISPASR_S3GEN_UNET_PROBE_RC_OUT`

### Chatterbox sub-modules

- `CRISPASR_CB_CAMPPLUS_BENCH`
- `CRISPASR_CB_S3GEN_BENCH`
- `CRISPASR_CB_S3TOK_BENCH`
- `CRISPASR_CB_VE_BENCH`

### Cohere

- `CRISPASR_COHERE_BENCH`
- `CRISPASR_COHERE_DEBUG`
- `CRISPASR_COHERE_DEVICE`
- `CRISPASR_COHERE_DUMP_ATTN`
- `CRISPASR_COHERE_DUMP_ENCOUT`
- `CRISPASR_COHERE_DUMP_MEL`
- `CRISPASR_COHERE_DUMP_STAGES`
- `CRISPASR_COHERE_FLASH`
- `CRISPASR_COHERE_GAPS`
- `CRISPASR_COHERE_LANGS`
- `CRISPASR_COHERE_LEGACY_SA`
- `CRISPASR_COHERE_PROBE_MAX_LANGS`
- `CRISPASR_COHERE_PROBE_REUSE_ENC`
- `CRISPASR_COHERE_PROBE_TEXTLID`
- `CRISPASR_COHERE_PROF`
- `CRISPASR_COHERE_SILENCE_GATE`
- `CRISPASR_COHERE_THREADS`

### Confucius4 TTS

- `CRISPASR_CONFUCIUS4_BEAMS`
- `CRISPASR_CONFUCIUS4_CFG_FUSE`
- `CRISPASR_CONFUCIUS4_CFG_RATE`
- `CRISPASR_CONFUCIUS4_COND_DIR`
- `CRISPASR_CONFUCIUS4_COND_PYEMB`
- `CRISPASR_CONFUCIUS4_DUMP_S2A`
- `CRISPASR_CONFUCIUS4_GRAPH_EMBED`
- `CRISPASR_CONFUCIUS4_LR_LEGACY`
- `CRISPASR_CONFUCIUS4_MAX_LAYERS`
- `CRISPASR_CONFUCIUS4_PERSIST`
- `CRISPASR_CONFUCIUS4_REP_PEN`
- `CRISPASR_CONFUCIUS4_S2A_TEMP`
- `CRISPASR_CONFUCIUS4_SCHED`
- `CRISPASR_CONFUCIUS4_TEXT_IDS`
- `CRISPASR_CONFUCIUS4_T_SCHEDULE`

### CosyVoice3

- `CRISPASR_COSYVOICE3_BENCH`
- `CRISPASR_COSYVOICE3_CAMPPLUS_PATH`
- `CRISPASR_COSYVOICE3_CFG_BATCH`
- `CRISPASR_COSYVOICE3_CFG_INTERVAL`
- `CRISPASR_COSYVOICE3_CFG_INTERVAL_DEBUG`
- `CRISPASR_COSYVOICE3_DUMP_HIFT`
- `CRISPASR_COSYVOICE3_DUMP_MEL`
- `CRISPASR_COSYVOICE3_DUMP_TOKENS`
- `CRISPASR_COSYVOICE3_FASTCONV`
- `CRISPASR_COSYVOICE3_FASTCONV_DEBUG`
- `CRISPASR_COSYVOICE3_SIMDCONV` — CPU-only direct SIMD Conv1d path for the 72 HiFT ResBlock convolutions; **default ON** since the Kaggle quiet-box A/B (1.07x on Xeon avx512f, 1.34x on Zen 4, output 1-LSB-equal, roundtrip exact). Set `=0` for the ggml path.
- `CRISPASR_COSYVOICE3_SIMDCONV_DEBUG` — print pack count, selected ISA, and GPU fallback status.
- `CRISPASR_COSYVOICE3_FLOW_STEPS`
- `CRISPASR_COSYVOICE3_FORCE_GALLOCR`
- `CRISPASR_COSYVOICE3_GREEDY`
- `CRISPASR_COSYVOICE3_HIFT_ON_GPU`
- `CRISPASR_COSYVOICE3_HIFT_PATH`
- `CRISPASR_COSYVOICE3_KV_BUCKET`
- `CRISPASR_COSYVOICE3_NO_CLONE_CACHE` — re-extract the `--voice ref.wav`
  speaker (s3tokenizer + CAMPPlus + prompt mel) on every synthesis instead of
  once per reference. Output-identical; the cached path is ~30% faster on a
  multi-sentence `--tts` (#334).
- `CRISPASR_COSYVOICE3_NO_MIN_LEN` — drop the decode's minimum-length floor
  (2 speech tokens per target text token, upstream's `min_token_text_ratio`).
  Without the floor a single unlucky sample at step 0 ends the decode with no
  audio at all (#334).
- `CRISPASR_COSYVOICE3_UPSTREAM_DIR`
- `CRISPASR_COSYVOICE3_VOICES_PATH`
- `CRISPASR_COSYVOICE3_VULKAN_NATIVE`

### CosyVoice3 (diff-harness assets)

- `CRISPASR_CV3_CAMPPLUS_GGUF`
- `CRISPASR_CV3_FLOW_GGUF`
- `CRISPASR_CV3_HIFT_GGUF`
- `CRISPASR_CV3_S3TOK_GGUF`

### CREPE (pitch)

- `CRISPASR_CREPE_BATCH`
- `CRISPASR_CREPE_DEBUG`
- `CRISPASR_CREPE_NO_BAKE_F32`
- `CRISPASR_CREPE_NO_GPU`

### CSM TTS

- `CRISPASR_CSM_BENCH`
- `CRISPASR_CSM_WAV_FRAMES`
- `CRISPASR_CSM_WAV_OUT`
- `CRISPASR_CSM_WAV_TEMP`
- `CRISPASR_CSM_WAV_TEXT`

### Dia TTS

- `CRISPASR_DIA_BENCH`
- `CRISPASR_DIA_DECODE_CODES`
- `CRISPASR_DIA_DUMP_DIR`
- `CRISPASR_DIA_DUMP_STEPLOGITS`
- `CRISPASR_DIA_DUMP_TOKENS`
- `CRISPASR_DIA_FORCE_TOKENS`
- `CRISPASR_DIA_GREEDY`
- `CRISPASR_DIA_MAX_STEPS`
- `CRISPASR_DIA_TTS_GPU`

### dots.tts

- `CRISPASR_DOTS_CFG_INTERVAL`
- `CRISPASR_DOTS_CFG_INTERVAL_DEBUG`
- `CRISPASR_DOTS_DIFF_GPU`
- `CRISPASR_DOTS_DIT_DEBUG`
- `CRISPASR_DOTS_EOS_THRESHOLD`
- `CRISPASR_DOTS_FAST`
- `CRISPASR_DOTS_FM_AB`
- `CRISPASR_DOTS_FM_DUMP`
- `CRISPASR_DOTS_FUSED_STEP`
- `CRISPASR_DOTS_MAX_PATCHES`
- `CRISPASR_DOTS_ODE_STEPS`
- `CRISPASR_DOTS_PENC_VERIFY`
- `CRISPASR_DOTS_TTS_BENCH`
- `CRISPASR_DOTS_TTS_CPU`
- `CRISPASR_DOTS_TTS_DEBUG`

### ECAPA (LID / speaker)

- `CRISPASR_ECAPA_ASP_CPU`
- `CRISPASR_ECAPA_ASP_GGML`
- `CRISPASR_ECAPA_FORCE_SCALAR`
- `CRISPASR_ECAPA_LID_BENCH`
- `CRISPASR_ECAPA_REF_FBANK`
- `CRISPASR_ECAPA_TIMING`

### F5-TTS

- `CRISPASR_F5_BATCH_CFG`
- `CRISPASR_F5_BENCH`
- `CRISPASR_F5_CFG_INTERVAL`
- `CRISPASR_F5_DIT_SKIP`
- `CRISPASR_F5_DURATION_CLAMP` — clamp the per-char speech rate into a sane English band so a reference whose audio/transcript lengths are mismatched can't truncate (or balloon) the output (#294). Default on; set `0` to restore the exact upstream `ref_T / ref_text_len * gen_text_len / speed` estimate.
- `CRISPASR_F5_EMBED_GPU`
- `CRISPASR_F5_F16_ACT`
- `CRISPASR_F5_HIFIGAN_CPU` — `1` restores the pre-`a72fb66d` CPU-loop
  HiFi-GAN decode (A/B fallback; the default ggml `core_hifigan` graph path
  is ~250x faster on GPU and cosine-1.000000 identical).
- `CRISPASR_F5_VOCODE_MEL` — debug probe: vocode this mel dump directly,
  bypassing the DiT (used for the CPU-vs-graph vocoder parity A/B).
- `CRISPASR_F5_FORCE_SCALAR`
- `CRISPASR_F5_REF_MAX_SEC` — clip the reference audio to this many seconds before it drives the duration estimate (upstream parity: 12 s). Default `12`; set `0` to disable the clip.
- `CRISPASR_F5_REF_TRIM_SILENCE` — strip leading/trailing silence and collapse internal silences >~1 s in the reference audio (upstream parity). Default on; set `0` to disable.
- `CRISPASR_F5_TEXT_LEN_BYTES`

### FastConformer (shared encoder)

- `CRISPASR_FC_ATTN_CONT`
- `CRISPASR_FC_BUCKET`
- `CRISPASR_FC_FUSED_QKV`
- `CRISPASR_FC_GPU_MANUAL_ATTN`
- `CRISPASR_FC_MAX_LAYERS`
- `CRISPASR_FC_MEM_DEBUG`
- `CRISPASR_FC_NO_FLASH`
- `CRISPASR_FC_PROFILE`
- `CRISPASR_FC_PROF_FP`
- `CRISPASR_FC_PROF_FP_COLS`
- `CRISPASR_FC_PW_Q8`
- `CRISPASR_FC_TILED_ATTN`
- `CRISPASR_FC_TILED_BLOCK`
- `CRISPASR_FC_WINDOWED_ATTN`
- `CRISPASR_FC_WINDOW_BLOCK`

### FastPitch

- `CRISPASR_FASTPITCH_BENCH`
- `CRISPASR_FASTPITCH_DUMP_DIR`
- `CRISPASR_FASTPITCH_FORCE_TOKENS`

### Ffmpeg

- `CRISPASR_FFMPEG_LOG`

### FireRed ASR / VAD

- `CRISPASR_FIRERED_BEAM_F32`
- `CRISPASR_FIRERED_BENCH`
- `CRISPASR_FIRERED_ENC_CPU`
- `CRISPASR_FIRERED_GGML_ATTN`
- `CRISPASR_FIRERED_LID_BENCH`
- `CRISPASR_FIRERED_MATVEC_CACHE`
- `CRISPASR_FIRERED_NO_REPEAT_BREAK`
- `CRISPASR_FIRERED_VAD_BENCH`
- `CRISPASR_FIRERED_VAD_DEBUG`
- `CRISPASR_FIRERED_VAD_FORCE_SCALAR`
- `CRISPASR_FIRERED_VAD_SERIAL`

### FunASR / SenseVoice

- `CRISPASR_FUNASR_BENCH`
- `CRISPASR_FUNASR_DUMP_STAGES`
- `CRISPASR_FUNASR_EMBED_FAST`
- `CRISPASR_FUNASR_LLM_CPU`
- `CRISPASR_FUNASR_LLM_LAYERS`
- `CRISPASR_FUNASR_NAN_CHECK`
- `CRISPASR_FUNASR_NO_FA`
- `CRISPASR_FUNASR_ENC_CACHE` — `0` disables the exact-T_lfr encoder graph
  cache (repeat-length calls skip the 7.5–23.9 ms graph rebuild). Default on.
- `CRISPASR_FUNASR_STEP_BUCKET` — width of the cached decode-graph Lk buckets
  (default 16, the measured optimum; `>= kv_max_ctx` reproduces the old
  fixed-Lk design, which is a ~69% decode regression — A/B arm only).
- `CRISPASR_FUNASR_STEP_CACHE` — `0` disables the bucketed per-step decode
  graph cache (bit-identical either way). Default on.

### Gemma-4 E2B

- `CRISPASR_GEMMA4_AUTO_CHUNK`
- `CRISPASR_GEMMA4_E2B_BENCH`
- `CRISPASR_GEMMA4_E2B_EMBED_FAST`
- `CRISPASR_NO_REL_POS` — drop the relative-position (matrix BD) term from the
  Gemma-4 E2B encoder attention scores, leaving only the content term. Parity
  bisection aid; not prefixed per-backend for historical reasons.

### GLM-ASR

- `CRISPASR_GLM_ASR_BENCH`
- `CRISPASR_GLM_ASR_DEBUG`
- `CRISPASR_GLM_ASR_EMBED_FAST`
- `CRISPASR_GLM_ASR_LEGACY_PROMPT`
- `CRISPASR_GLM_ASR_SINGLE_WINDOW`

### Granite speech / NLE

- `CRISPASR_GRANITE_BENCH`
- `CRISPASR_GRANITE_DEC_GALLOCR`
- `CRISPASR_GRANITE_DEC_PROFILE`
- `CRISPASR_GRANITE_DISABLE_ENCODER_GRAPH`
- `CRISPASR_GRANITE_ENC_F16`
- `CRISPASR_GRANITE_FORCE_SCALAR`
- `CRISPASR_GRANITE_NLE_BENCH`
- `CRISPASR_GRANITE_NLE_EDIT_DUMP`
- `CRISPASR_GRANITE_QUANT_ALL`

### HiFT vocoder

- `CRISPASR_HIFT_FULL_IDFT`

### HTDemucs (source separation)

- `CRISPASR_HTDEMUCS_BLAS` — route the CrossTransformer matmuls through
  `cblas_sgemm` (default **ON** where Accelerate is available). `=0` selects the
  scalar path. The transformer is ~86% of an unoptimised forward pass, so this is
  the dominant knob (measured 44x on the transformer, 4.6x overall).
- `CRISPASR_HTDEMUCS_FASTCONV` — batched im2col + one GEMM for the CPU convs
  (default **ON**). `=0` selects the original per-time-frame scalar path.
  Measured `enc.conv2d` 10.0 s -> 0.17 s and `enc.rewrite` 12.2 s -> 0.30 s.
- `CRISPASR_HTDEMUCS_WCACHE` — cache F32 copies of weight tensors by pointer
  (default **ON**). `=0` re-reads and re-converts on every access, which the
  DConv stacks do ~6k times per encoder layer.
- `CRISPASR_HTDEMUCS_GGML` — run the ggml graph path instead of CPU/BLAS.
  Since #414 the default is **AUTO**: ON exactly when a real GPU backend is
  present and permitted (where the fused graph measured ~20x faster than
  BLAS — RTF 0.37 vs 7.4 on an RTX 3090 Ti), OFF on CPU-only hosts (where
  graphs measured slower than BLAS). `=1`/`=0` force either way.
- `CRISPASR_HTDEMUCS_GPU` — GPU permission (CUDA > Metal > Vulkan). Default
  AUTO follows the caller's use_gpu (CLI default on); an explicit `=0`/`=1`
  beats the caller in both directions — so `=0` genuinely opts out even
  though the CLI defaults `use_gpu=true` (#414 review catch). On GPU-less
  hosts everything resolves to the BLAS path regardless.
- `CRISPASR_HTDEMUCS_NO_BCAST_CAST` — disable the issue-#398 fix that casts
  non-F32 affine/bias weights to F32 in-graph before broadcast add/mul sites
  (bisection aid). With `=1` the pre-fix graph is rebuilt, which on CUDA
  aborts in `binbcast.cu` (`nb10 % sizeof(src1_t)`) because the F16 GGUF
  stores the DConv GroupNorm affines (`*.dconv.layers.N.4.weight`) as F16.
- `CRISPASR_HTDEMUCS_PROFILE` — print a per-phase wall-time breakdown of one
  forward pass (stft / enc / transformer / dec / istft).
- `CRISPASR_HTDEMUCS_DEBUG` — verbose per-layer shape and NaN diagnostics.
- `CRISPASR_HTDEMUCS_SKIP_TIME` — skip the time branch (bisection aid).
- `CRISPASR_HTDEMUCS_FUSED` — single fused graph (encoder+transformer+decoder
  on-device, no per-layer host↔device roundtrips). Default **AUTO**: ON with
  the GPU graph path (#414), OFF otherwise. `=1` alone implies the graph path
  it needs; `=0` on GPU keeps the per-layer-graph bisection arm.
  The full decision table is unit-locked in tests/test-htdemucs-gates.cpp.
- `CRISPASR_HTDEMUCS_MEMSTATS` — log each weight-cache admission and the running
  cache total in MB.
- `CRISPASR_HTDEMUCS_NO_SEGMENT` — process the whole track in one pass instead of
  the 25%-overlap segment schedule (A/B against the old behaviour).

All three optimisation gates are output-equivalent: the per-stage diff reports
45/45 stages passing with them ON or OFF.

### Higgs STT

- `CRISPASR_HIGGS_DEBUG`
- `CRISPASR_HIGGS_STT_BENCH`
- `CRISPASR_HIGGS_STT_EMBED_FAST`
- `CRISPASR_HIGGS_STT_FUSED_QKV`

### IndexTTS

- `CRISPASR_INDEXTTS_AA_BACKEND`
- `CRISPASR_INDEXTTS_AA_SCALAR`
- `CRISPASR_INDEXTTS_AUDIO24K_FILE`
- `CRISPASR_INDEXTTS_BEAM_SIZE`
- `CRISPASR_INDEXTTS_BENCH`
- `CRISPASR_INDEXTTS_COND_FILE`
- `CRISPASR_INDEXTTS_DEBUG`
- `CRISPASR_INDEXTTS_KV_DEVICE_COPY`
- `CRISPASR_INDEXTTS_LATENT_FILE`
- `CRISPASR_INDEXTTS_MEL_CODES_FILE`
- `CRISPASR_INDEXTTS_MEL_FILE`
- `CRISPASR_INDEXTTS_SPK_NORM`
- `CRISPASR_INDEXTTS_TEXT_NORMALIZER`
- `CRISPASR_INDEXTTS_VOCODER_AA`
- `CRISPASR_INDEXTTS_VOCODER_RAW`
- `CRISPASR_INDEXTTS_VOC_BENCH`
- `CRISPASR_INDEXTTS_VOC_FORCE_GPU`

### Irodori TTS

- `CRISPASR_IRODORI_CAPTION`
- `CRISPASR_IRODORI_CAPTION_TOKEN_IDS`
- `CRISPASR_IRODORI_CFG_CAPTION`
- `CRISPASR_IRODORI_CFG_INTERVAL`
- `CRISPASR_IRODORI_CFG_INTERVAL_DEBUG`
- `CRISPASR_IRODORI_CFG_SPEAKER`
- `CRISPASR_IRODORI_CFG_TEXT`
- `CRISPASR_IRODORI_CODEC_CPU`
- `CRISPASR_IRODORI_CODEC_GPU`
- `CRISPASR_IRODORI_CPU`
- `CRISPASR_IRODORI_DEBUG`
- `CRISPASR_IRODORI_DECODE_CHUNK`
- `CRISPASR_IRODORI_DECODE_CTX`
- `CRISPASR_IRODORI_DIT_TIMING`
- `CRISPASR_IRODORI_DUMP_LATENT`
- `CRISPASR_IRODORI_DUMP_TEXT_STATE`
- `CRISPASR_IRODORI_DUMP_TOKENS`
- `CRISPASR_IRODORI_DUMP_V_PRED0`
- `CRISPASR_IRODORI_ENC_DUMP`
- `CRISPASR_IRODORI_ENC_PRENORM`
- `CRISPASR_IRODORI_FASTCONV`
- `CRISPASR_IRODORI_LAYERS`
- `CRISPASR_IRODORI_ODE_STEPS`
- `CRISPASR_IRODORI_PERSIST_GRAPH`
- `CRISPASR_IRODORI_REF_NOISE`
- `CRISPASR_IRODORI_TOKEN_IDS`
- `CRISPASR_IRODORI_T_LATENT`

### Kokoro

- `CRISPASR_KOKORO_BENCH`
- `CRISPASR_KOKORO_DEBUG`
- `CRISPASR_KOKORO_DEBUG_INTERMEDIATES`
- `CRISPASR_KOKORO_DUMP_STAGES`
- `CRISPASR_KOKORO_FASTCONV`
- `CRISPASR_KOKORO_FASTCONV_DEBUG`
- `CRISPASR_KOKORO_G2P`
- `CRISPASR_KOKORO_GEN_FORCE_METAL` / `CRISPASR_KOKORO_GEN_GPU` — either one puts
  the generator (vocoder) stage on Metal.
- `CRISPASR_KOKORO_SEED`
- `CRISPASR_KOKORO_USE_GPU`
- `CRISPASR_KOKORO_VOICE_GGUF`

### KugelAudio

- `CRISPASR_KUGELAUDIO_CPU_ONLY`
- `CRISPASR_KUGELAUDIO_DEBUG`

### Kyutai STT

- `CRISPASR_KYUTAI_RVQ_FAST`
- `CRISPASR_KYUTAI_STT_BENCH`

### LFM2-Audio

- `CRISPASR_LFM2_AUDIO_BENCH`
- `CRISPASR_LFM2_AUDIO_CPU`
- `CRISPASR_LFM2_SNAP_LAYERS`

### M2M-100 translate

- `CRISPASR_M2M100_BENCH`
- `CRISPASR_M2M100_GPU`

### MarbleNet VAD

- `CRISPASR_MARBLENET_VAD_BENCH`
- `CRISPASR_MARBLENET_VAD_SERIAL`

### Mel-Band RoFormer (source separation)

- `CRISPASR_MBR_PROFILE` — print a per-stage wall-time breakdown of one forward
  pass (stft+pack / band_split / run_time / run_freq / mask_est / synthesize).
- `CRISPASR_MELBAND_GGML` — run the ggml graph path instead of the legacy CPU
  path. Since the Change-176 graph port the default is **AUTO**: ON exactly
  when a real GPU backend is present and permitted (the fused single graph
  measured ~112x faster than the per-layer graphs — RTF ~0.09 vs ~10 on an
  RTX 3090 Ti), OFF on CPU-only hosts. `=1`/`=0` force either way.
- `CRISPASR_MELBAND_GPU` — GPU permission (CUDA > Metal > Vulkan). Default
  AUTO follows the caller's use_gpu (CLI default on); an explicit `=0`/`=1`
  beats the caller in both directions — so `=0` genuinely opts out even
  though the CLI defaults `use_gpu=true` (#414 review semantics). On GPU-less
  hosts everything resolves to the CPU path regardless.
- `CRISPASR_MELBAND_FUSED` — single fused graph: band-split + the full
  time/freq transformer stack + mask estimator on-device in one graph, no
  per-layer host↔device roundtrips (the measured-fastest path on GPU).
  Default **AUTO**: ON with the GPU graph path, OFF otherwise. `=1` alone
  implies the graph path it needs; `=0` on GPU keeps the per-layer-graph
  bisection arm. The full decision table is unit-locked in
  tests/test-mel-band-gates.cpp.
- `CRISPASR_MELBAND_SEG_S` — override the Demucs-style segment length in
  seconds (`params.segment_seconds`; <=0 → the checkpoint's trained
  `chunk_size` from GGUF metadata, 8 s Kim fallback — do NOT expect 10 s: the
  earlier hardcoded 10 s ran RoPE 25% past the trained window, review #422).
  The attention matrix is O(T²·bands·heads), so long inputs are split into
  segments with 25% overlap and a triangular weight (bounds VRAM; the
  unsegmented whole-buffer path OOMs on any clip beyond ~10 s).
- `CRISPASR_MELBAND_NO_SEGMENT` — process the whole track in one pass instead
  of the segmented overlap-add schedule (A/B against the old behaviour).

### MeloTTS

- `CRISPASR_MELOTTS_BENCH`
- `CRISPASR_MELOTTS_BERT`
- `CRISPASR_MELOTTS_FORCE_SCALAR`
- `CRISPASR_MELOTTS_WEIGHT_CACHE`

### Mimi codec

- `CRISPASR_MIMI_NONCAUSAL`

### MiMo-ASR

- `CRISPASR_MIMO_ASR_BENCH`
- `CRISPASR_MIMO_ASR_DIAG`
- `CRISPASR_MIMO_ASR_DUMP_STAGES`
- `CRISPASR_MIMO_ASR_GPU`
- `CRISPASR_MIMO_FORCE_CPU`
- `CRISPASR_MIMO_SMOKE_DUMP`
- `CRISPASR_MIMO_SMOKE_GPU`
- `CRISPASR_MIMO_TOKENIZER_GPU`
- `CRISPASR_MIMO_TOK_CPU`
- `CRISPASR_MIMO_TOK_CPU_RVQ`
- `CRISPASR_MIMO_TOK_VERIFY_RVQ`

### mini-omni2

- `CRISPASR_MINI_OMNI2_BENCH`

### Moonshine

- `CRISPASR_MOONSHINE_ALL_GPU`
- `CRISPASR_MOONSHINE_BENCH`
- `CRISPASR_MOONSHINE_ENC_ATTN`
- `CRISPASR_MOONSHINE_NO_REPEAT_BREAK`
- `CRISPASR_MOONSHINE_STREAMING_BENCH`
- `CRISPASR_MOONSHINE_STREAMING_GPU`
- `CRISPASR_MOONSHINE_STREAM_BENCH`

### MOSS family

- `CRISPASR_MOSS_AUDIO_BENCH`
- `CRISPASR_MOSS_AUDIO_EMBED_FAST`
- `CRISPASR_MOSS_AUDIO_ENC_FLASH`
- `CRISPASR_MOSS_AUDIO_ENC_MANUAL`
- `CRISPASR_MOSS_AUDIO_FORCE_CPU`
- `CRISPASR_MOSS_AUDIO_MEL_FILE`
- `CRISPASR_MOSS_DIARIZE_BENCH`
- `CRISPASR_MOSS_DIARIZE_DEBUG`
- `CRISPASR_MOSS_DIARIZE_DUMP_CONV`
- `CRISPASR_MOSS_DIARIZE_ENC_FLASH`
- `CRISPASR_MOSS_DIARIZE_ENC_MANUAL`
- `CRISPASR_MOSS_DIARIZE_FORCE_CPU`
- `CRISPASR_MOSS_DIARIZE_NO_LOOPFIX`
- `CRISPASR_MOSS_TRANSCRIBE_BENCH`
- `CRISPASR_MOSS_TRANSCRIBE_ENC_DUMP`
- `CRISPASR_MOSS_TRANSCRIBE_ENC_FLASH`
- `CRISPASR_MOSS_TRANSCRIBE_ENC_MANUAL`
- `CRISPASR_MOSS_TRANSCRIBE_FORCE_CPU`
- `CRISPASR_MOSS_TRANSCRIBE_L0_DUMP`
- `CRISPASR_MOSS_TRANSCRIBE_MEL_DUMP`
- `CRISPASR_MOSS_TRANSCRIBE_NO_LOOPFIX`
- `CRISPASR_MOSS_TTS_BENCH`
- `CRISPASR_MOSS_TTS_LOCAL_DEBUG`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_FA_PATH`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_HIDDEN`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_PROMPT_IDS`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_STOP`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER`
- `CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH`
- `CRISPASR_MOSS_TTS_LOCAL_FORCE_FRAMES`
- `CRISPASR_MOSS_TTS_LOCAL_GREEDY_AUDIO`
- `CRISPASR_MOSS_TTS_LOCAL_GREEDY_TEXT`
- `CRISPASR_MOSS_TTS_LOCAL_INJECT_LAYER`
- `CRISPASR_MOSS_TTS_LOCAL_INJECT_PATH`
- `CRISPASR_MOSS_TTS_LOCAL_MAX_FRAMES`
- `CRISPASR_MOSS_TTS_LOCAL_NO_GPU`

### MP3 codec

- `CRISPASR_MP3_ENCODER`

### Diarization — foxnose (#324)

- `CRISPASR_DIARIZE_COUNT` — speaker-count estimator: `bic` (default, the
  upstream GMM/BIC + silhouette sweep) or `eigengap`. Eigengap is better on
  well-separated synthetic data and cheaper, but under-counts on real speech
  (11.4 % vs 5.3 % DER on VoxConverse) — see `docs/foxnose-diarize/PLAN.md`
- `CRISPASR_DIARIZE_BIC_WINDOW` — score silhouette only in a `[k-2, k+3]` window
  around the BIC anchor instead of the full `[min, max]` range. The full range is
  the default: the BIC anchor is unreliable in both directions (measured errors
  of +5 / -3 / -3 on 4/5/6 well-separated blobs) and when it over-counts the
  window is stranded above the truth and cannot climb back to it
- `CRISPASR_WESPEAKER_BENCH` — per-stage embedder timings (fbank / resnet /
  resnet_windows). Counting invocations of these is also how you check WHICH
  embedding path actually ran
- `CRISPASR_WESPEAKER_DEBUG` — embedder diagnostics
- `CRISPASR_DIARIZE_DEBUG` — chosen speaker count, the reason it was chosen,
  and the per-k silhouette curve behind it. Worth reading before trusting a
  count: on a borderline file the decision can rest on a <1 % score gap
- `CRISPASR_DIARIZE_EMBED_WORKERS` — windows embedded concurrently (default:
  `-t`). Each worker gets its own context sharing one copy of the weights
- `CRISPASR_SPEAKER_EMBED_WORKERS` — the same worker count for the standalone
  `crispasr-diarize` CLI's speaker-embedding stage
- `CRISPASR_SPEAKER_EMBED_THREADS` — ggml threads per embedder context
  (default: `-t`). Honoured by the pluggable embedders and by wespeaker
- `CRISPASR_DIARIZE_SPAN_EMBED=1` — run ONE network pass per span of windows
  instead of one per window. 1.78x less diarization CPU for +0.30 mean DER on
  the VoxConverse dev shard; off by default because accuracy is the better
  default for a diarizer. See `docs/cli.md#diarization`
- `CRISPASR_DIARIZE_SPAN_WINDOWS` — windows per span (default 32). Measured NOT
  to affect the accuracy cost — identical from N=2 to N=32 — so there is
  nothing to tune here; larger is simply faster
- `CRISPASR_WESPEAKER_CONV` — conv lowering for the WeSpeaker embedder:
  `im2col` (default) lowers each conv to explicit IM2COL + MUL_MAT nodes so
  the GEMMs reach the Accelerate BLAS backend on CPU and the simdgroup
  mul_mm kernels on Metal; `direct` restores GGML_OP_CONV_2D. Measured on
  esrit.wav (215 s, `-t 8`): diarization delta 9.8 s -> 5.9 s (~1.6x),
  embeddings cosine 1.0 vs direct, DER identical per file (7.32 % shard mean)
- `CRISPASR_WESPEAKER_GPU=1` — run the WeSpeaker embedder on the GPU backend
  (single context; the CPU worker pool is stood down because workers borrow
  weights that now live in a GPU buffer). With the im2col default and batched
  windows Metal reaches parity with the 8-worker CPU schedule on an M-series
  (8.5 s vs 8.7 s wall on esrit.wav) but does not beat it, so CPU stays the
  default; the switch exists for machines where the GPU/CPU balance differs
- `CRISPASR_DIARIZE_BATCH_EMBED` — batch independent 1.2 s windows into one
  graph along ne[3] (arithmetic-identical to per-window; cosine 1.0). Default:
  ON under `CRISPASR_WESPEAKER_GPU=1` (collapses ~350 Metal dispatches into
  ~11 and is what got Metal from a 2x loss to parity), OFF on CPU (measured
  12.1 -> 14.9 s wall on esrit.wav: ggml's CPU conv loops ne[3], so fusing
  buys no GEMM shape and the 32-window chunks starve the worker schedule).
  `1`/`0` forces either way
- `CRISPASR_WESPEAKER_BATCH` — max windows per batched graph (default 32 on
  GPU, 16 on CPU; cap 32). Only meaningful where the batch path is active

### GigaAM-v3

- `CRISPASR_GIGAAM_BENCH` — per-stage timings (mel / encoder / decode)
- `CRISPASR_GIGAAM_DEBUG` — encoder output min/max/mean
- `CRISPASR_GIGAAM_FLASH` — `ggml_flash_attn_ext` in the encoder (opt-in; the
  manual QK^T path is what the per-stage diff was validated on)
- `CRISPASR_GIGAAM_FORCE_SCALAR` — scalar LSTM/joint loops instead of cblas
- `CRISPASR_GIGAAM_QUANT_ALL` — let `crispasr-quantize` quantize the heads and
  the pre-encode convs too (default keeps them at source precision)

### Nemotron

- `CRISPASR_NEMOTRON_BENCH`
- `CRISPASR_NEMOTRON_CONTEXT_PRESET`
- `CRISPASR_NEMOTRON_DEBUG`
- `CRISPASR_NEMOTRON_DECODE_TIMING`
- `CRISPASR_NEMOTRON_FORCE_SCALAR`
- `CRISPASR_NEMOTRON_GGML_DECODE`
- `CRISPASR_NEMOTRON_MAES`
- `CRISPASR_NEMOTRON_NO_WINDOW_MASK`
- `CRISPASR_NEMOTRON_STREAMING`
- `CRISPASR_NEMOTRON_STREAM_DEBUG`

### OmniASR

- `CRISPASR_OMNIASR_BENCH`
- `CRISPASR_OMNIASR_DEBUG`
- `CRISPASR_OMNIASR_DUMP_DIR`
- `CRISPASR_OMNIASR_KEEP_F16_HEAD`
- `CRISPASR_OMNIASR_KEEP_F16_TAIL`
- `CRISPASR_OMNIASR_QUANT_ALL`

### OmniVoice

- `CRISPASR_OMNIVOICE_ACENC_BISECT`
- `CRISPASR_OMNIVOICE_AUTO_LANG` — **default on.** When no language was requested
  (`-l` / `-tl` / the server's `"language"` / `set_target_language`), guess one
  from the text being spoken and use it if it maps to an id the model knows.
  An explicitly requested language always wins; a low-confidence guess resolves
  to nothing and behaves exactly as before. `=0` restores the old
  always-language-agnostic behaviour. Exists because SubtitleEdit's language
  menu is not yet wired to its request payload (#13273).
- `CRISPASR_OMNIVOICE_BENCH`
- `CRISPASR_OMNIVOICE_CFG_INTERVAL`
- `CRISPASR_OMNIVOICE_CHUNK`
- `CRISPASR_OMNIVOICE_CLASS_TEMP`
- `CRISPASR_OMNIVOICE_CODEC_FASTCONV`
- `CRISPASR_OMNIVOICE_CODEC_GPU` — codec placement override (`1` = GPU, `0` = CPU). Unset defaults to GPU on
  CUDA and CPU on Metal/CPU.
- `CRISPASR_OMNIVOICE_CPU`
- `CRISPASR_OMNIVOICE_DEBUG`
- `CRISPASR_OMNIVOICE_DEBUG_CODES`
- `CRISPASR_OMNIVOICE_DEBUG_SUM`
- `CRISPASR_OMNIVOICE_DUMP_CODES`
- `CRISPASR_OMNIVOICE_ENCODE_DIFF`
- `CRISPASR_OMNIVOICE_FRAMES_PER_CHAR`
- `CRISPASR_OMNIVOICE_FUSED_STEP`
- `CRISPASR_OMNIVOICE_GUIDANCE`
- `CRISPASR_OMNIVOICE_HUBERT_REF`
- `CRISPASR_OMNIVOICE_NUM_STEPS`
- `CRISPASR_OMNIVOICE_PERSISTENT_GRAPH`
- `CRISPASR_OMNIVOICE_POS_TEMP`
- `CRISPASR_OMNIVOICE_REF_RATE_CHECK`
- `CRISPASR_OMNIVOICE_TOKENIZER_GGUF`
- `CRISPASR_OMNIVOICE_UNIFIED_CFG`
- `CRISPASR_OMNIVOICE_UPSTREAM_WEIGHTS`
- `CRISPASR_OMNIVOICE_VOICE_CACHE`

### OpenVoice2

- `CRISPASR_OPENVOICE2_BENCH`
- `CRISPASR_OV2_DUMP_DIR`
- `CRISPASR_OV2_FORCE_SCALAR`
- `CRISPASR_OV2_NO_NORMALIZE`
- `CRISPASR_OV2_TAU`

### Opus codec

- `CRISPASR_OPUS_DEBUG`
- `CRISPASR_OPUS_DECODER`
- `CRISPASR_OPUS_ENCODER`

### Orpheus

- `CRISPASR_ORPHEUS_BENCH`
- `CRISPASR_ORPHEUS_BUCKET`
- `CRISPASR_ORPHEUS_DEBUG`
- `CRISPASR_ORPHEUS_DIFF_GPU`
- `CRISPASR_ORPHEUS_DIFF_MAXGEN`
- `CRISPASR_ORPHEUS_PROMPT_IDS`
- `CRISPASR_ORPHEUS_SNAC_CODE`
- `CRISPASR_ORPHEUS_SNAC_GPU`
- `CRISPASR_ORPHEUS_SNAC_T_SUPER`

### OuteTTS

- `CRISPASR_OUTETTS_BENCH`

### Paraformer

- `CRISPASR_PARAFORMER_BENCH`
- `CRISPASR_PARAFORMER_GPU`

### Parakeet

- `CRISPASR_PARAKEET_ATT_CONTEXT`
- `CRISPASR_PARAKEET_BENCH`
- `CRISPASR_PARAKEET_CHUNK_OVERLAP`
- `CRISPASR_PARAKEET_CHUNK_SECONDS`
- `CRISPASR_PARAKEET_DEBUG`
- `CRISPASR_PARAKEET_DECODE_TIMING`
- `CRISPASR_PARAKEET_ENC_CACHE`
- `CRISPASR_PARAKEET_ENC_PROBE`
- `CRISPASR_PARAKEET_FORCE_SCALAR`
- `CRISPASR_PARAKEET_GGML_DECODE`
- `CRISPASR_PARAKEET_INTERNAL_CHUNKING`
- `CRISPASR_PARAKEET_LONGFORM`
- `CRISPASR_PARAKEET_LONGFORM_WINDOW`
- `CRISPASR_PARAKEET_MAES`
- `CRISPASR_PARAKEET_MEM_COEFF`
- `CRISPASR_PARAKEET_MEM_POLICY`
- `CRISPASR_PARAKEET_PIPELINE`
- `CRISPASR_PARAKEET_QUANT_ALL`
- `CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM`
- `CRISPASR_PARAKEET_STREAM_CHUNK`
- `CRISPASR_PARAKEET_STREAM_OVERLAP`
- `CRISPASR_PARAKEET_STREAM_THRESHOLD`
- `CRISPASR_PARAKEET_VAD_SLICE_CAP`
- `CRISPASR_PARAKEET_VRAM_BUDGET_MB`

### Parler-TTS

- `CRISPASR_PARLER_BUCKET`
- `CRISPASR_PARLER_DEBUG`
- `CRISPASR_PARLER_DESC_IDS`
- `CRISPASR_PARLER_DIFF_MAXGEN`
- `CRISPASR_PARLER_DUMP_ENC`
- `CRISPASR_PARLER_PROMPT_IDS`
- `CRISPASR_PARLER_TTS_BENCH`

### Piano transcription

- `CRISPASR_PIANO_SERIAL` — `1` restores the single-threaded conv/GRU/linear
  compute (the default parallel path is bit-identical; #305, ~1.7x).

### Piper

- `CRISPASR_PIPER_FORCE_SCALAR`
- `CRISPASR_PIPER_TTS_BENCH`
- `CRISPASR_PIPER_WEIGHT_CACHE`

### Pocket-TTS

- `CRISPASR_POCKET_DUMP_DIR`
- `CRISPASR_POCKET_FORCE_LATENTS`
- `CRISPASR_POCKET_FORCE_NOISE`
- `CRISPASR_POCKET_MANUAL_BACKBONE`
- `CRISPASR_POCKET_MANUAL_MIMI`
- `CRISPASR_POCKET_MAX_FRAMES`
- `CRISPASR_POCKET_MIMI_DUMP`
- `CRISPASR_POCKET_MIMI_SCALAR`
- `CRISPASR_POCKET_TTS_BENCH`
- `CRISPASR_POCKET_VOICE_CACHE`
- `CRISPASR_POCKET_VULKAN_MIMI_MAX_FRAMES`

### Pyannote segmentation

- `CRISPASR_PYANNOTE_LEGACY`
- `CRISPASR_PYANNOTE_SEG_BENCH`
- `CRISPASR_PYANNOTE_SEG_DUMP`
- `CRISPASR_PYANNOTE_CHUNK_S` — audio per chunk of parallel inference, in
  seconds (default 60; `0` restores the pre-#326 single scan over the whole
  file). Chunking is decided by audio LENGTH and never by thread count, so
  posteriors do not change with `-t`
- `CRISPASR_PYANNOTE_CHUNK_CONTEXT_S` — real audio spliced either side of a
  chunk and then trimmed (default 5), which absorbs the convolutions' zero
  padding and the LSTM's zero initial state

### Qwen3-ASR

- `CRISPASR_QWEN3ASR_QUANT_AUDIO`

### Qwen3-ASR / Qwen3-TTS

- `CRISPASR_QWEN3_ASR_BENCH`
- `CRISPASR_QWEN3_ASR_EMBED_FAST`
- `CRISPASR_QWEN3_ASR_FUSED_QKV`
- `CRISPASR_QWEN3_SYSPROMPT_LANG`
- `CRISPASR_QWEN3_TTS_BENCH`
- `CRISPASR_QWEN3_TTS_CODEC_ALLOW_FULL`
- `CRISPASR_QWEN3_TTS_CODEC_CHUNK`
- `CRISPASR_QWEN3_TTS_CODEC_CPU`
- `CRISPASR_QWEN3_TTS_CODEC_CTX`
- `CRISPASR_QWEN3_TTS_CODEC_FASTCONV` — codec conv fast path. **Default ON**; `0` opts out.
- `CRISPASR_QWEN3_TTS_CODEC_FORCE_METAL`
- `CRISPASR_QWEN3_TTS_CODEC_GGUF`
- `CRISPASR_QWEN3_TTS_CODEC_GPU`
- `CRISPASR_QWEN3_TTS_HIP_CODEC_NATIVE` — bypass the #337 ROCm codec-encoder
  correctness fallback and run it natively on HIP. Diagnostic/A/B only until
  the triggering Daphne spans pass on a real AMD GPU.
- `CRISPASR_QWEN3_TTS_HIP_CP_NATIVE` — bypass the #337 CPU fallback for the
  ROCm 0.6B-F16 code predictor. Diagnostic/A/B only; its native gfx1100 path
  was observed to emit all-NaN logits.
- `CRISPASR_QWEN3_TTS_CODEC_TRACE`
- `CRISPASR_QWEN3_TTS_CP_BACKEND`
- `CRISPASR_QWEN3_TTS_CP_DIRECT`
- `CRISPASR_QWEN3_TTS_CP_MTP_NOFUSE`
- `CRISPASR_QWEN3_TTS_CP_STEP0_CACHE`
- `CRISPASR_QWEN3_TTS_DEBUG`
- `CRISPASR_QWEN3_TTS_DUMP_DIR`
- `CRISPASR_QWEN3_TTS_EMBD_CHECK`
- `CRISPASR_QWEN3_TTS_FUSED_QKV`
- `CRISPASR_QWEN3_TTS_LK_BUCKET`
- `CRISPASR_QWEN3_TTS_NO_EMBD_CACHE`
- `CRISPASR_QWEN3_TTS_O15` / `_O15_SKIP_REALLOC`
- `CRISPASR_QWEN3_TTS_PROF`
- `CRISPASR_QWEN3_TTS_SEED`
- `CRISPASR_QWEN3_TTS_TALKER_SCHED`
- `CRISPASR_QWEN3_TTS_VULKAN_NATIVE`
- `CRISPASR_QWEN3_TTS_DUMP_LOGITS=<dir>` — write the raw per-frame talker
  logits (f32, before the repetition penalty and the suppress mask) plus a
  top-5 line to stderr. The instrument for a cross-backend diff: tokens
  alone cannot tell a miscompute from amplified rounding (#337).
- `CRISPASR_QWEN3_TTS_REPLAY_CODES=<file>` — 16 whitespace-separated codec ids
  per frame; the decode uses them instead of sampling. Teacher forcing, and
  the ONLY way to compare two backends step by step: pin the whole frame or
  the 15 residual codebooks (which must be sampled) diverge and the diff
  measures trajectory, not arithmetic (#337).
- `CRISPASR_QWEN3_TTS_REPLAY_TOKENS=<file>` — the weaker form: codebook-0 ids
  only. Useful for forcing a trajectory, NOT sufficient for a logits diff.
- `CRISPASR_QWEN3_TTS_GREEDY` — force the talker's codebook-0 sampler to argmax
  (top_k=1). The frame sequence then depends only on the logits, so two
  backends agree if and only if their logits agree — this is the lever for
  telling a GPU miscompute apart from a sampling difference, and without it a
  CPU-vs-GPU token comparison proves nothing (#337).
- `CRISPASR_QWEN3_TTS_MAX_FRAMES`
- `CRISPASR_QWEN3_TTS_SKIP_REF_DECODE`

### SenseVoice

- `CRISPASR_SENSEVOICE_BENCH`
- `CRISPASR_SENSEVOICE_NO_FA`

### Sidon

- `CRISPASR_SIDON_FASTCONV` — DAC convolution mode (`off`, `k1-f16`, `k1-f32`, or `full`). Unset defaults to
  `k1-f16` on CUDA and `off` on Vulkan/CPU.
- `CRISPASR_SIDON_RPE` — relative-position-bias formulation: `bucket-direct`, `bucket`, or `expand` (legacy
  `[head_dim, T, T]` expansion; keeps the Vulkan `mul_mat` batching branch). All three are algebraically
  equivalent. Unset defaults to **AUTO**, resolved per graph build because the choice depends on the input
  length: `expand` on a GPU backend while its extra transient footprint (`4·T²·(head_dim+1−heads)`, i.e.
  `196·T²` bytes for the shipped model — 58 MiB at `T=557`, 1.64 GiB at the 3000-frame cap) fits
  `CRISPASR_SIDON_RPE_BUDGET_MB`, and `bucket-direct` otherwise. AUTO exists because `expand` measured **2.7×
  faster in the predictor** on the #416 reporter's GTX 1660 SUPER (213.80 ms vs 575.25 ms at `T=557`, same
  file and binary). AUTO stays off on CPU: that speedup is one device's measurement and CPU behaviour is
  left exactly as it was. An explicit value is honoured as given and never auto-overridden — it is the #416
  bisection handle. The decision table is unit-locked in `tests/test-sidon-rpe-gates.cpp`.
- `CRISPASR_SIDON_RPE_BUDGET_MB` — AUTO's budget in MiB for `expand`'s extra transient footprint (default
  `256`, which admits `expand` up to `T≈1170`, ~23 s of audio). `0` disables AUTO, pinning `bucket-direct`.
- `CRISPASR_SIDON_DECODER_CHUNK_FRAMES` — maximum DAC core size in feature frames (default `512`). `0` decodes
  the whole utterance in one graph (~4.5 GiB at `T≈2825` vs ~0.79 GiB chunked). Chunked output is bit-exact
  against the whole-utterance decode.
- `CRISPASR_SIDON_LOOKAHEAD` — set to `0` to disable the input padding (one leading predictor frame plus 1.5 s
  of right-side lookahead). Padding is on by default; without it the last ~12 ms of every clip is a full-scale
  transient.
- `CRISPASR_SIDON_MAX_FRAMES` — predictor input cap in feature frames (default `3000`, ~58.5 s after the
  lookahead). Guards the `O(T^2)` attention.
- `CRISPASR_SIDON_DEBUG` — print per-stage scheduler workspace sizes (per backend) after graph allocation.
- `CRISPASR_SIDON_DUMP_HANDOFF` — directory/path for the predictor→DAC handoff tensor dump.

### Sherpa

- `CRISPASR_SHERPA_LID_BIN`
- `CRISPASR_SHERPA_LID_TIMEOUT_SEC` / `CRISPASR_SHERPA_TIMEOUT_SEC` — wall-clock
  timeout for the external sherpa LID / diarize helper, scaled by audio length.

### Silero LID

- `CRISPASR_SILERO_FORCE_SCALAR`
- `CRISPASR_SILERO_LID_BENCH`
- `CRISPASR_SILERO_LID_DEBUG`
- `CRISPASR_SILERO_LID_DUMP`
- `CRISPASR_SILERO_LID_LEGACY`
- `CRISPASR_SILERO_LID_MAX_S`
- `CRISPASR_SILERO_LID_MIN_LOGIT` — evidence gate (#409): discard a silero LID answer whose top-1 RAW logit is below this floor (default `-2.0`; `-999` disables) and fall back to whisper-tiny LID. The raw-logit magnitude separates in-domain from out-of-domain audio where the softmax probability does not.
- `CRISPASR_SILERO_LID_TRACE`
- `CRISPASR_SILERO_LID_TRACE_OFF`
- `CRISPASR_SILERO_LID_TRUNC`
- `CRISPASR_SILERO_LID_VULKAN`

### SpeechT5

- `CRISPASR_SPEECHT5_DUMP_DIR`
- `CRISPASR_SPEECHT5_FASTCONV`
- `CRISPASR_SPEECHT5_FASTCONV_DEBUG`
- `CRISPASR_SPEECHT5_TTS_BENCH`

### T5 translate

- `CRISPASR_T5_DIFF`
- `CRISPASR_T5_GPU`
- `CRISPASR_T5_TRANSLATE_BENCH`

### TabCNN (guitar tablature)

- `CRISPASR_TABCNN_DEBUG`
- `CRISPASR_TABCNN_NO_GPU`

### TaDa TTS

- `CRISPASR_TADA_ACOUSTIC_CFG`
- `CRISPASR_TADA_ALLOW_VULKAN`
- `CRISPASR_TADA_BATCH_PREFILL`
- `CRISPASR_TADA_BENCH`
- `CRISPASR_TADA_BUCKET_MIN`
- `CRISPASR_TADA_CFG_INTERVAL`
- `CRISPASR_TADA_CFG_INTERVAL_DEBUG`
- `CRISPASR_TADA_CODEC_BENCH`
- `CRISPASR_TADA_CODEC_DUMP`
- `CRISPASR_TADA_CODEC_GGUF`
- `CRISPASR_TADA_CODEC_VULKAN_NATIVE`
- `CRISPASR_TADA_CTC_ASR`
- `CRISPASR_TADA_DIFF_TEXT`
- `CRISPASR_TADA_DO_SAMPLE`
- `CRISPASR_TADA_DUMP_ACOUSTIC_FEATURES`
- `CRISPASR_TADA_DUMP_FEATURES`
- `CRISPASR_TADA_DUMP_FM_STEPS`
- `CRISPASR_TADA_DUMP_TIME_BEFORE`
- `CRISPASR_TADA_ENCODER_DEBUG`
- `CRISPASR_TADA_EXTRA_STEPS`
- `CRISPASR_TADA_FM_B2`
- `CRISPASR_TADA_KEEP_F16_HEAD`
- `CRISPASR_TADA_KEEP_F16_TAIL`
- `CRISPASR_TADA_MAX_EXPANDED_FRAMES`
- `CRISPASR_TADA_NOISE_TEMP`
- `CRISPASR_TADA_NO_BUCKET`
- `CRISPASR_TADA_NUM_CANDIDATES`
- `CRISPASR_TADA_NUM_FM_STEPS`
- `CRISPASR_TADA_PROMPT_CACHE`
- `CRISPASR_TADA_PROMPT_TEXT`
- `CRISPASR_TADA_QUANT_ALL`
- `CRISPASR_TADA_REPETITION_PENALTY`
- `CRISPASR_TADA_SCORER`
- `CRISPASR_TADA_TALKER_TIMING`
- `CRISPASR_TADA_TEMPERATURE`
- `CRISPASR_TADA_TOP_K`
- `CRISPASR_TADA_TOP_P`
- `CRISPASR_TADA_VULKAN_NATIVE`
- `CRISPASR_TADA_WAV_CLONE` — `1` enables on-the-fly TaDa voice cloning from a
  reference `.wav` + transcript through the session C-ABI (#201). Off by default:
  without it a `.wav` voice reference is still rejected with `-2`, preserving the
  historical behaviour until the decoded-output roundtrip has validated the path.

### TitaNet speaker

- `CRISPASR_TITANET_BENCH`
- `CRISPASR_TITANET_DUMP`
- `CRISPASR_TITANET_DUMP_MEL`
- `CRISPASR_TITANET_FORCE_SCALAR`
- `CRISPASR_TITANET_GGML`
- `CRISPASR_TITANET_GPU`
- `CRISPASR_TITANET_LEGACY`
- `CRISPASR_TITANET_REF_MEL`

Three compute paths exist and all are kept working; the default is the fastest
one measured per platform.

| path | selected by | measured, M1, 2 s segment |
| --- | --- | --- |
| legacy (Accelerate / hand-rolled) | default where `HAVE_ACCELERATE` | **71.7 ms** |
| ggml graph, CPU | default elsewhere; `CRISPASR_TITANET_GGML=1` | 277.3 ms |
| ggml graph, GPU | `CRISPASR_TITANET_GGML=1 CRISPASR_TITANET_GPU=1` | 31.9 ms *(but see below)* |

All three agree to cosine **1.000000** with each other and **0.999996** against
NVIDIA's `nemo_en_titanet_large.onnx` export fed the same mel — so the choice is
purely about speed.

⚠ **`CRISPASR_TITANET_GPU=1` is opt-in because it loses on real workloads
despite winning the micro-benchmark.** Diarization embeds one segment per call
at *variable* lengths, so every call reshapes the graph and the GPU allocator
re-reserves; and `CRISPASR_SPEAKER_EMBED_WORKERS` runs several embedders at once,
which contend for the one GPU. End-to-end on a 600 s clip, 47 segments:

```
workers=4, legacy   9994 ms    <- default, fastest
workers=1, legacy  12673 ms
workers=1, GPU     15275 ms
workers=4, GPU     49866 ms
```

Keep it for evaluating a discrete GPU (where the balance may differ) or for
`CRISPASR_SPEAKER_EMBED_WORKERS=1` on a machine with weak CPU cores. Bucketing
segment lengths so the graph shape stops changing is the work that would make
this path win generally.

`CRISPASR_TITANET_DUMP_MEL=<path>` writes the computed mel as `[T][n_mels]`
float32 — the counterpart to `CRISPASR_TITANET_REF_MEL`. Feeding that dump to an
upstream ONNX export separates the front-end from the network, which a single
end-to-end cosine cannot do.

### VibeVoice

- `CRISPASR_VIBEVOICE_ASR_PROMPT`
- `CRISPASR_VIBEVOICE_ASR_SAMPLE`
- `CRISPASR_VIBEVOICE_ATTN_PREC`
- `CRISPASR_VIBEVOICE_BENCH`
- `CRISPASR_VIBEVOICE_BITNET_ACT_QUANT`
- `CRISPASR_VIBEVOICE_DEBUG`
- `CRISPASR_VIBEVOICE_DUMP_DIR`
- `CRISPASR_VIBEVOICE_ENC_BACKEND` — `{auto|cpu|gpu}` backend for the σ-VAE
  tokenizer ENCODERS (the ASR direction). `auto` (default) diverts them to CPU
  on Vulkan devices with a 65535 `maxComputeWorkGroupCount` (Intel Arc/Iris/
  UHD, llvmpipe), where the conv dispatches over long audio overflow and abort
  (issue #418 — the ASR twin of the decoder's #52 fallback,
  `CRISPASR_VIBEVOICE_VAE_BACKEND`). `gpu` forces the active backend (the
  repro arm); `cpu` forces the fallback anywhere.
- `CRISPASR_VIBEVOICE_ENCODER_CHUNK_SECONDS`
- `CRISPASR_VIBEVOICE_ENCODER_CONTEXT_SECONDS`
- `CRISPASR_VIBEVOICE_GELU_TANH`
- `CRISPASR_VIBEVOICE_GPU`
- `CRISPASR_VIBEVOICE_LM_BUCKETS`
- `CRISPASR_VIBEVOICE_NO_INPUT_NORM`
- `CRISPASR_VIBEVOICE_NO_LM_BUCKETS`
- `CRISPASR_VIBEVOICE_PRED_SCHED`
- `CRISPASR_VIBEVOICE_QUANT_ALL`
- `CRISPASR_VIBEVOICE_RAW_TRANSCRIPT` — `1` keeps the pre-#300 single segment
  holding the model's raw JSON blob instead of splitting it into one segment per
  utterance with the speaker in the structured field.
- `CRISPASR_VIBEVOICE_REF_FEATURES`
- `CRISPASR_VIBEVOICE_REUSE_PRED_GRAPH`
- `CRISPASR_VIBEVOICE_TTS_CFG_SCALE`
- `CRISPASR_VIBEVOICE_TTS_DUMP`
- `CRISPASR_VIBEVOICE_TTS_DUMP_DECODER`
- `CRISPASR_VIBEVOICE_TTS_DUMP_PERFRAME`
- `CRISPASR_VIBEVOICE_TTS_FLASH_ATTN`
- `CRISPASR_VIBEVOICE_TTS_LATENTS`
- `CRISPASR_VIBEVOICE_TTS_NOISE`
- `CRISPASR_VIBEVOICE_TTS_SEED`
- `CRISPASR_VIBEVOICE_TTS_TRACE`
- `CRISPASR_VIBEVOICE_TTS_TRACE_FRAME`
- `CRISPASR_VIBEVOICE_VAE_BACKEND`
- `CRISPASR_VIBEVOICE_VOICE_AUDIO`

### VoxCPM2

- `CRISPASR_VOXCPM2_BENCH`
- `CRISPASR_VOXCPM2_BUCKET_CUDA`
- `CRISPASR_VOXCPM2_CFG_INTERVAL`
- `CRISPASR_VOXCPM2_CFG_INTERVAL_DEBUG`
- `CRISPASR_VOXCPM2_CFG_VALUE`
- `CRISPASR_VOXCPM2_CPU_ONLY`
- `CRISPASR_VOXCPM2_FA_CPU`
- `CRISPASR_VOXCPM2_FORCE_SCALAR`
- `CRISPASR_VOXCPM2_INFERENCE_STEPS`
- `CRISPASR_VOXCPM2_MAX_LEN`
- `CRISPASR_VOXCPM2_NAN_CHECK`
- `CRISPASR_VOXCPM2_NO_BUCKET`
- `CRISPASR_VOXCPM2_USE_GRAPH` — persistent-graph decode. **Default ON**; `0` opts out.
- `CRISPASR_VOXCPM2_USE_REF`
- `CRISPASR_VOXCPM2_VAE_ENC_DIFF`
- `CRISPASR_VOXCPM2_VAE_TRACE`
- `CRISPASR_VOXCPM2_VAE_MAX_SAMPLES` - maximum 16 kHz input samples accepted by one `voxcpm2-vae` upscaling call
  (default `960000`, or 60 seconds). Split longer audio, or raise this only when enough RAM/VRAM is available.

### Voxtral / Voxtral-TTS

- `CRISPASR_VOXTRAL_BENCH`
- `CRISPASR_VOXTRAL_FUSED_QKV`
- `CRISPASR_VOXTRAL_TTS_CODEC_FROM_FILE`
- `CRISPASR_VOXTRAL_TTS_DEBUG`
- `CRISPASR_VOXTRAL_TTS_DIFF_DUMP`
- `CRISPASR_VOXTRAL_TTS_FM_STEPS`
- `CRISPASR_VOXTRAL_TTS_SEMANTIC_CB`
- `CRISPASR_VOXTRAL_TTS_TEXT`
- `CRISPASR_VOXTRAL_TTS_TIMING`
- `CRISPASR_VOXTRAL_TTS_VOICE`

### Voxtral-4B

- `CRISPASR_VOXTRAL4B_BENCH`
- `CRISPASR_VOXTRAL4B_FUSED_QKV`
- `CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER`
- `CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS`
- `CRISPASR_VOXTRAL4B_STREAM_DEBUG`
- `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD`
- `CRISPASR_VOXTRAL4B_STREAM_DIFF`
- `CRISPASR_VOXTRAL4B_STREAM_LIVE`
- `CRISPASR_VOXTRAL4B_STREAM_TIMING`

### VAD (encoder/decoder Silero-style)

- `CRISPASR_VAD_ENCDEC_CONV_CAST`
- `CRISPASR_VAD_ENCDEC_CPU`
- `CRISPASR_VAD_ENCDEC_DEBUG`
- `CRISPASR_VAD_ENCDEC_PERSIST`
- `CRISPASR_VAD_ENCDEC_SERIAL_MEL`

### VAD (WebRTC)

- `CRISPASR_WEBRTC_VAD_MODE` — WebRTC VAD aggressiveness, `0`–`3` (default `1`).
  Only consulted when the caller did not pass an explicit mode.

### Wav2Vec2

- `CRISPASR_WAV2VEC2_BENCH`
- `CRISPASR_WAV2VEC2_DUMP_DIR`
- `CRISPASR_WAV2VEC2_VERBOSE`

### Whisper (Tiron speaker attribution)

- `CRISPASR_WHISPER_TIRON` — `0` forces the stock whisper decode for A/B. The
  Tiron constrained-decoding grammar is auto-on whenever the model's vocab has
  speaker tokens; plain greedy loses ~5 cpWER.
- `CRISPASR_WHISPER_TIRON_DEBUG`
- `CRISPASR_WHISPER_TIRON_MAX_SPEAKERS`
- `CRISPASR_WHISPER_TIRON_NOSPEECH` — `0` disallows an initial `<|nospeech|>`.

### WavTokenizer

- `CRISPASR_WAVTOK_BENCH`
- `CRISPASR_WAVTOK_DUMP_DIR`
- `CRISPASR_WAVTOK_FIXED_CODES`

### Zonos

- `CRISPASR_ZONOS_CPP_DUMP_DIR`
- `CRISPASR_ZONOS_DECODE_CHUNK`
- `CRISPASR_ZONOS_DECODE_CTX`
- `CRISPASR_ZONOS_DIFF_N_STEPS`
- `CRISPASR_ZONOS_FASTCONV`
- `CRISPASR_ZONOS_SPEAKER_EMB_PATH`
- `CRISPASR_ZONOS_TTS_BENCH`
- `CRISPASR_ZONOS_TTS_TEXT`
- `CRISPASR_ZONOS_VULKAN_NATIVE`
