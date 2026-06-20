# Testing

CrispASR has two tiers of tests: **unit tests** (no models, fast) and
**integration / live tests** (need GGUF models on disk).

## Unit tests

679 unit tests run unconditionally in ~20 seconds with no model files:

```bash
ctest --test-dir build -L unit --timeout 30
```

These cover: audio chunking, mel preprocessing, CTC/beam decode,
sentence splitting, WAV metadata, stream finalization, registry lookup,
watermark embed/detect, cache helpers, GPT-2 BPE tokenizer,
BERT WordPiece tokenizer, bench env-var gating, per-backend param
defaults and null-guard coverage (43 backends), and more.

## Integration tests

~25 integration tests need real GGUF models. They are gated by env vars
and **SKIP cleanly** when the vars are unset (Catch2 `SKIP()` → exit
code 4, mapped to ctest "Skipped" via `SKIP_RETURN_CODE 4`).

### Quick start

```bash
# Point at your local model cache:
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models

# Source all env vars at once:
source tests/env-live-tests.sh

# Run only previously-failed tests:
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300

# Run all live tests:
ctest --test-dir build -L live --output-on-failure --timeout 300
```

### Environment variables

`tests/env-live-tests.sh` sets every env var the live tests expect.
Override `CRISPASR_MODELS_DIR` to point at your model directory; all
other vars derive from it unless individually overridden.

| Variable | Used by | Notes |
|---|---|---|
| `CRISPASR_MODELS_DIR` | All auto-download + search | Also checked by `crispasr_cache.cpp` well-known dirs |
| `CRISPASR_MODEL_WHISPER` | Beam search, VAD tests | Default: `~/.cache/crispasr/ggml-tiny.bin` |
| `CRISPASR_MODEL_GLM_ASR` | Beam search (GLM-ASR) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_QWEN3_ASR` | Beam search (Qwen3-ASR) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_CANARY` | Beam search (Canary) | Large model, may timeout on CPU |
| `CRISPASR_MODEL_COHERE` | Beam search (Cohere) | Large model, may timeout on CPU |
| `PARAFORMER_MODEL` | Paraformer live tests | F16 GGUF |
| `PARAFORMER_MODEL_Q4K` | Paraformer Q4_K parity | Q4_K GGUF |
| `PARAFORMER_AUDIO_ZH` | Paraformer Chinese test | 16kHz mono WAV |
| `CRISPASR_TEST_DIARIZE_MODEL` | Diarization (pyannote) | pyannote-seg-3.0 GGUF |
| `CRISPASR_TEST_TITANET_MODEL` | Diarization (embedder) | titanet-large GGUF |
| `CRISPASR_TEST_DIARIZE_WAV` | Diarization | Multi-speaker 16kHz mono WAV |
| `CRISPASR_CHAT_TEST_MODEL` | Chat LLM smoke test | Needs chat template (not harrier) |
| `CRISPASR_MODEL_NEMOTRON` | Nemotron live tests | Q4_K GGUF (~458 MB) |
| `CRISPASR_MODEL_NEMOTRON_F16` | Nemotron F16/Q4_K parity | F16 GGUF (~1.3 GB) |
| `CRISPASR_MODEL_LFM2` | LFM2-Audio live tests | Q5_K GGUF (~1.6 GB). **Not Q4_K** — produces 0 tokens. |
| `CRISPASR_MODEL_DIA` | Dia TTS live tests | Q4_K GGUF (~892 MB) |
| `CRISPASR_MODEL_OUTETTS` | OuteTTS live tests | Q4_K GGUF (~600 MB) |
| `CRISPASR_MODEL_WAVTOK` | WavTokenizer (OuteTTS codec) | F16 GGUF (~100 MB) |

### Test groups

| Tests | Group | Model | Timeout |
|---|---|---|---|
| #100-103 | Paraformer | paraformer-zh-f16.gguf (~422 MB) | 30s |
| #110-112 | Nemotron | nemotron Q4_K + F16 (~1.7 GB total) | 300s |
| #218-220 | Beam: whisper | ggml-tiny.bin (~75 MB) | 120s |
| #221-228 | Beam: other backends | 2-5 GB models | 300s+ (CPU) |
| #77-78 | Diarize (pyannote + TitaNet) | ~50 MB total | 120s |
| #409 | Chat (LLM) | Any chat GGUF | 120s |
| #456-458 | CLI integration | Auto-download (whisper base) | 300s |
| #460-461 | VAD (full + thresholds) | ggml-tiny.bin + silero | 120s |
| #462 | Backend regression | Auto-download (many backends) | 600s |
| #463 | Benchmark-quick | parakeet-tdt-0.6b-v3 | 300s |
| #464 | Progress output | Auto-download (whisper + parakeet) | 300s |

### Auto-download and model cache

Tests that use `-m auto --auto-download` (CLI, backends, benchmark,
progress) resolve models via the registry (`crispasr_model_registry.cpp`)
and the cache system (`crispasr_cache.cpp`). The cache probes these
locations in order:

1. `--cache-dir` CLI override (or `cache_dir_override` in C API)
2. `$CRISPASR_MODELS_DIR` env var
3. `/mnt/storage/gguf-models` (dev machine convention)
4. `/Volumes/backups/ai/crispasr-models` (macOS dev convention)
5. `~/.cache/crispasr` (platform default)
6. `~/.cache/crispasr-models` (legacy)
7. `~/.cache/huggingface/hub` (HF download cache)

If none of the probed paths has the file, it downloads from HuggingFace.

### Writing new integration tests

- Use Catch2 `SKIP()` when env vars are unset — this returns exit code 4
- In CMakeLists.txt, add `SKIP_RETURN_CODE 4` and
  `WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"` to the `PROPERTIES` of
  `catch_discover_tests`
- Use single-word labels (CMake's `catch_discover_tests` splits
  semicolons in `PROPERTIES` values)
- Add the env var to `tests/env-live-tests.sh`
