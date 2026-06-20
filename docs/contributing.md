# Contributing — adding a new backend

Adding a new ASR model to CrispASR is a focused exercise in five
files. The worked examples to copy from are the existing
`crispasr_backend_*.cpp` adapters.

> **‼️ `clang-format` MUST be v18 — never use v22 (Homebrew's default
> `clang-format` keg, or Xcode's bundled clang-format, both currently
> ship v22).** CI pins `clang-format-18`
> (`.github/workflows/lint.yml`); v22 silently re-wraps lines and
> fails the lint job with ~80+ violations across the project's C++
> files. Anyone formatting with v22 will produce a commit that breaks
> CI even though the code "looks formatted" locally.
>
> **Use `./tools/format.sh`** — it locates clang-format-18 via
> Homebrew's `llvm@18` keg or apt's `clang-format-18` and refuses to
> run any other version. `./tools/format.sh` checks;
> `./tools/format.sh --fix` rewrites in place. The script's scope
> mirrors `lint.yml` exactly so local check ≡ CI check.
>
> **Install** clang-format-18:
> - macOS: `brew install llvm@18` (binary lands at
>   `/opt/homebrew/opt/llvm@18/bin/clang-format`)
> - Ubuntu / Debian: `sudo apt install clang-format-18`
> - Cross-platform via pip: `pip install 'clang-format==18.1.8'`
>   (puts `clang-format` v18 on `PATH`; works as long as your pip
>   user-bin precedes Homebrew's bin)
>
> **Do NOT** put `/opt/homebrew/bin/clang-format` (currently v22) on
> `PATH` for this project; it will silently mis-format. The wrapper
> script defends against this by hardcoded version-check.

## 1. Land the model's C API in `src/yourmodel.{h,cpp}`

Following the established convention:

```c
struct yourmodel_context * yourmodel_init_from_file(const char * path, yourmodel_context_params p);
void                       yourmodel_free(struct yourmodel_context *);
char *                     yourmodel_transcribe(struct yourmodel_context *, const float * samples, int n);
```

Use `src/core/mel`, `src/core/ffn`, `src/core/attention`, and
`src/core/gguf_loader` wherever they fit — they cover ~80 % of the
boilerplate.

**Add bench instrumentation.** Every runtime gets per-stage RAII timing
gated by `YOURMODEL_BENCH=1`:

```cpp
static bool yourmodel_bench_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("YOURMODEL_BENCH");
                  v = (e && *e && *e != '0') ? 1 : 0; }
    return v != 0;
}
struct yourmodel_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit yourmodel_bench_stage(const char* n)
        : name(n), t0(std::chrono::steady_clock::now()) {}
    ~yourmodel_bench_stage() {
        if (!yourmodel_bench_enabled()) return;
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  yourmodel_bench: %-22s %.2f ms\n", name, ms);
    }
};
```

Place `yourmodel_bench_stage _b("stage_name");` at each pipeline stage
(mel, encoder, decoder, vocoder, etc.). Zero overhead when the env var
is unset (~1 ns cached getenv).

## 2. Write the backend adapter

Create `examples/cli/crispasr_backend_yourmodel.cpp`:

```cpp
#include "crispasr_backend.h"
#include "whisper_params.h"
#include "yourmodel.h"

namespace {
class YourmodelBackend : public CrispasrBackend {
public:
    const char * name() const override { return "yourmodel"; }
    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | /* ... */;
    }
    bool init(const whisper_params & p) override { /* yourmodel_init_from_file(...) */ }
    std::vector<crispasr_segment> transcribe(
        const float * samples, int n, int64_t t_off,
        const whisper_params & p) override { /* call yourmodel_transcribe and return segments */ }
    void shutdown() override { /* yourmodel_free(...) */ }
private:
    yourmodel_context * ctx_ = nullptr;
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_yourmodel_backend() {
    return std::make_unique<YourmodelBackend>();
}
```

## 3. Register with the factory

In `examples/cli/crispasr_backend.cpp`:

```cpp
std::unique_ptr<CrispasrBackend> crispasr_make_yourmodel_backend();
// ...
if (name == "yourmodel") return crispasr_make_yourmodel_backend();
// ...
std::vector<std::string> crispasr_list_backends() {
    return { ..., "yourmodel" };
}
```

Add the architecture string to `crispasr_detect_backend_from_gguf()`
so `general.architecture` auto-detection works.

## 4. Wire into CMake

In `examples/cli/CMakeLists.txt`:

```cmake
add_executable(${TARGET}
    # ...
    crispasr_backend_yourmodel.cpp
)

target_link_libraries(${TARGET} PRIVATE
    # ...
    yourmodel_lib
)
```

## 5. Optional: add to the model registry

If your model has a canonical Q4_K HuggingFace release, add it to
`src/crispasr_model_registry.cpp` so `-m auto` / `-m <name> --auto-download`
works (one `k_registry[]` row: `{"name", "file.gguf", "https://…", "~size",
nullptr, nullptr}`).

For TTS backends that need a codec companion (e.g. SNAC, DAC), add the
companion file + URL in the same registry row:
```c
{"yourmodel", "yourmodel.gguf", "https://…/yourmodel.gguf", "~2 GB",
 "snac-24khz.gguf",                           // companion_file
 "https://…/snac-24khz.gguf",                 // companion_url
 "~80 MB"},                                   // companion_size
```

## 5b. TTS backend wiring — `CAP_TTS` and `--codec-model`

TTS backends need extra wiring beyond ASR:

1. **`CAP_TTS` capability flag**: set in `capabilities()` so the CLI
   TTS path (`--tts "text"`) routes through your backend's
   `synthesize()` method. Without this, the CLI says "does not support
   TTS".

2. **`synthesize()` override**: return a `std::vector<float>` of mono
   PCM at the backend's native sample rate (24 kHz for SNAC/Mimi-based
   backends, 44.1 kHz for DAC/Zonos). Default returns empty.

3. **`tts_sample_rate()` override**: return the output sample rate if
   it's not 24000 (default). The CLI uses this for WAV header + any
   downstream resampling.

4. **Codec companion loading**: if your backend needs a separate codec
   GGUF (SNAC, DAC, Mimi), handle `params.tts_codec_model` in `init()`:
   ```cpp
   #include "crispasr_model_mgr_cli.h"
   #include "crispasr_model_registry.h"
   // ... in init():
   std::string codec_path = p.tts_codec_model;
   if (!codec_path.empty() && codec_path != "auto")
       codec_path = crispasr_resolve_model_cli(codec_path, ...);
   if (codec_path.empty())
       codec_path = discover_snac(p.model); // look next to model
   if (codec_path.empty()) {
       CrispasrRegistryEntry entry;
       if (crispasr_registry_lookup(p.backend, entry, ...) && ...)
           codec_path = crispasr_resolve_model_cli(entry.companion_filename, ...);
   }
   ```
   See `crispasr_backend_orpheus.cpp` or `crispasr_backend_mini_omni2.cpp`
   for worked examples.

## 6. Expose through the C ABI, bindings, and docs

§2–§5 make `--backend X` work on the CLI. To reach every other consumer
(Python, Go, Dart, server) and `-m <file>` auto-detection, wire the points
below. **`chatterbox` is the canonical template** — `grep -n chatterbox`
(or `CA_HAVE_CHATTERBOX`) in each file shows the exact pattern to copy.

### CLI auto-detection — `examples/cli/crispasr_backend.cpp`
So `-m model.gguf` *without* `--backend` routes correctly, add the name to
**both** passes of `crispasr_detect_backend_from_gguf()`:
- Pass 1 (filename heuristic): `if (contains_ci("yourmodel")) return "yourmodel";`
- Pass 2 (GGUF `general.architecture`): `else if (a == "<arch>") result = "yourmodel";`
  — `<arch>` is whatever your converter passes to `GGUFWriter(arch=...)`.

The `--list-backends` capability row is read live from the backend's
`capabilities()`, so it needs no separate edit.

### C ABI — `src/crispasr_c_api.cpp` (this is what the bindings/server call)
Python/Go/Dart/server use the **session C ABI**, not the CLI factory.
Nine edit points, each mirroring the `CA_HAVE_CHATTERBOX` blocks:
1. **include + flag:** `#if __has_include("yourmodel.h")` → `#include` →
   `#define CA_HAVE_YOURMODEL 1` → `#endif`.
2. **session struct field:** `#ifdef CA_HAVE_YOURMODEL  yourmodel_context* yourmodel_ctx = nullptr;  #endif`.
3. **`crispasr_session_open_explicit()`** dispatch: when `s->backend ==
   "yourmodel"`, build params + `yourmodel_init_from_file(model_path, p)`.
4. **`crispasr_session_synthesize()`** (TTS) or transcribe dispatch:
   `if (s->yourmodel_ctx) return yourmodel_synthesize(s->yourmodel_ctx, text, out_n_samples);`.
4b. **`crispasr_session_speech_to_speech()`** (S2S) — if the backend has
   `CAP_S2S`. Dispatch to `yourmodel_speech_to_speech(ctx, in, n, lang, &text, &n_out)`.
   The server exposes this via `POST /v1/audio/speech-to-speech`.
5. **`crispasr_session_free()`:** free the ctx.
6. **`crispasr_session_set_temperature()`** — if sampling-capable.
7. **`crispasr_session_set_tts_seed()`** — if seedable.
8. **`crispasr_session_available_backends()`:** append `,yourmodel`. The
   Python binding rejects any backend missing from this list, so this is
   not optional.
9. **`set_ask` wiring** — if the backend is an instruct-tuned audio-LLM
   (has a user/system prompt template): add a `yourmodel_set_ask(ctx,
   prompt)` setter to the runtime, and forward `s->ask` in the transcribe
   dispatch. This lets `set_ask()` override the default transcription
   instruction. Currently wired for: granite, voxtral, qwen3-asr,
   glm-asr, gemma4-e2b, mimo-asr.
9b. **`params.language` wiring** — if the backend is an audio-LLM, also
   inject a language hint into the prompt when `params.language` is set
   and non-auto. Pattern: `else if (!params.language.empty() && params.language
   != "auto") { sys_instruction = "Transcribe the speech in " + lang_name(params.language) + "."; }`.
   For **English-only** models (moonshine-streaming, kyutai-stt), emit a
   `fprintf(stderr, ...)` warning instead of silently ignoring the flag.
   Currently wired for: granite (v3 + v4 templates), qwen3-asr, glm-asr,
   moss-audio, mimo-asr; warned for moonshine-streaming, kyutai-stt.
10. **`crispasr_session_set_speaker_id()`** — if the backend is a
    multi-speaker TTS model with integer-indexed speakers (e.g. melotts,
    piper, fastpitch). Add a dispatch block that bounds-checks against
    `yourmodel_num_speakers()` and calls `yourmodel_set_speaker_id()`.
    Also wire `crispasr_session_n_speakers()` to return the count.
    For **name-based** speaker selection (orpheus-style), wire
    `crispasr_session_set_speaker_name()` instead.

### CMake — link into the C-ABI library, `src/CMakeLists.txt`
§4 links the backend into the CLI binary. The C ABI needs it in
`libcrispasr` too. In the "C-ABI wrappers" block (grep
`target_link_libraries(crispasr-lib PUBLIC`):
```cmake
if (TARGET yourmodel_lib)
    target_link_libraries(crispasr-lib PUBLIC yourmodel_lib)
endif()
```

### Bindings — docstrings only for a new *backend* (dispatch is automatic once the C ABI is wired)
- `python/crispasr/_binding.py` — add the name to the TTS-backend lists in
  the `synthesize` comment + docstring.
- `bindings/go/crispasr_session.go` — add to the header/type comments.
  Go links `-lcrispasr` (not per-backend), so **no LDFLAGS change**.
- `flutter/crispasr/lib/src/crispasr.dart` — add to the synthesize
  docstring. Dart uses `DynamicLibrary.lookupFunction` with symbol-presence
  checks, so new C-ABI functions are discovered automatically.

### Bindings — adding a new *session setter* (`crispasr_session_set_*`)
A new setter is **not** auto-discovered; every wrapper exposes it explicitly
and they are kept at full parity. Add the new method to **all six** wrappers
(mirror the nearest existing setter in each — argtypes/restype, error-on-rc≠0):
- `python/crispasr/_binding.py` — ctypes method on `Session`.
- `bindings/go/crispasr_session.go` — the cgo-preamble `int crispasr_session_set_X(...)`
  declaration **and** the `Set X` method.
- **Rust (repo root, not `bindings/`)**: `crispasr-sys/src/lib.rs` extern decl
  **and** `crispasr/src/lib.rs` safe `pub fn`.
- `flutter/crispasr/lib/src/crispasr.dart` — `lookupFunction` + method.
- `bindings/java/.../CrispasrSession.java` — JNA `Lib` interface decl + method.
- `bindings/ruby/ext/ruby_crispasr_session.c` — `extern` decl, `rb_session_set_X`,
  and a `rb_define_singleton_method` registration.
- The HTTP server (`examples/cli/crispasr_server.cpp`) exposes the equivalent as
  a per-request `form_*` field on the transcription endpoints (or a startup flag
  for resident post-processors).
- `bindings/javascript/emscripten.cpp` — WASM/JS (built with emcc).
The canonical surface is `include/crispasr_session.h`; `docs/bindings.md` has the
per-wrapper setter table.

### Docs
- `README.md` — model-table row (TTS or ASR section).
- `docs/tts.md` — backend table row (TTS backends).
- `docs/architecture.md` — a `### yourmodel` section under "Per-backend
  architecture details"; README/tts.md link `docs/architecture.md#yourmodel`.

### Build targets (don't be fooled by stale binaries)
- `crispasr` → the **library** (libcrispasr / `.dylib`).
- `crispasr-cli` → the **CLI binary** (`OUTPUT_NAME crispasr`).
- `crispasr-diff` → the diff harness.

After C-ABI edits, build **`crispasr`** (the dylib) and re-test the Python
`Session` — building only `crispasr-cli` may leave the dylib stale, and the
binding then loads an old backend list. Verify with:
```python
import crispasr
assert "yourmodel" in crispasr.Session.available_backends()
crispasr.Session("model.gguf", backend="yourmodel")  # opens?
```

> ⚠️ **Commit from a separate `git worktree`, or `git pull --rebase`
> immediately before committing.** A `git add -A` / `commit -a` from a
> parallel session over a stale tree silently reverts others' work through
> the shared `.git/index` — this clobbered the entire §135 CSM landing
> (commit `100b9ee5`). `git config pull.rebase true` is set in this repo.

## Running integration / live tests

Unit tests (429 of them) need no models and pass unconditionally. The
~25 integration ("live") tests need real GGUF models on disk and are
env-var-gated — they SKIP cleanly when the env vars are unset.

**Quick start:**
```bash
# Point at your local model cache (auto-download also probes this):
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models

# Source all env vars at once:
source tests/env-live-tests.sh

# Run only the previously-failed tests:
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300
```

`tests/env-live-tests.sh` sets every env var the live tests expect.
Override `CRISPASR_MODELS_DIR` to point at your local model directory;
all other vars derive from it unless individually overridden.

**Key env vars** (see `env-live-tests.sh` for the full list):

| Variable | Used by |
|---|---|
| `CRISPASR_MODELS_DIR` | Well-known search dir for all model lookups |
| `CRISPASR_MODEL_WHISPER` | Beam search + VAD tests |
| `PARAFORMER_MODEL` | Paraformer live tests |
| `CRISPASR_TEST_DIARIZE_MODEL` | Diarization live tests |
| `CRISPASR_CHAT_TEST_MODEL` | Chat (LLM) smoke test |

Tests that use `SKIP()` return exit code 4 (Catch2 convention). The
CMakeLists.txt sets `SKIP_RETURN_CODE 4` so ctest reports them as
"Skipped" rather than "Failed".

## Common pitfalls

### Mel spectrogram

- **FFT size must match upstream exactly.** Whisper uses `torch.stft` with
  `n_fft=400` — a 400-point DFT, NOT zero-padded to 512. Zero-padding
  changes frequency bin spacing (`k*sr/400` vs `k*sr/512`) and corrupts the
  mel projection. Use `core_mel` with a matching FFT callback (a naive
  O(N*N_freqs) DFT is fast enough for N=400).
- **Hann window variant**: `torch.hann_window(N)` (periodic) differs from
  `np.hanning(N+1)[:-1]` (symmetric) by ~2.4e-7 per sample. Enough to
  cause cos_min≈0.95 on the mel comparison. Store the correct variant in
  the GGUF.
- **Frame count**: `torch.stft` produces N+1 frames and drops the last one
  (`stft[..., :-1]`). `core_mel::compute` may produce 1 extra frame.
  Truncate: `T_mel = n_samples / hop_length`.
- **Filterbank layout**: `core_mel::FbLayout::MelsFreqs` = `fb[m*n_freqs+k]`
  (numpy `(n_mels, n_freqs)` row-major). HF `WhisperFeatureExtractor` uses
  `FreqsMels` (transposed). Check which one your upstream model uses.

### Multi-stream audio LLMs

Models that interleave text + audio codec streams (mini-omni2-style):
- N input streams embedded and averaged — audio features replace pad
  positions in audio streams only (NOT the text stream)
- Special task tokens at stream end select the mode (ASR/TTS/S2S)
- For decode-step performance, batch all N token embeddings in one
  `ggml_get_rows` call instead of N separate graph builds

### Reference dump memory management (8 GB RAM)

PyTorch model loading OOMs easily on this machine. Patterns:
- Load encoder, capture activations, `del model; gc.collect()` before
  loading the LLM
- `del state_dict; gc.collect()` after `model.load_state_dict()`
- Use `--stages mel_spectrogram,encoder_output` to skip LLM loading
  entirely when only testing the audio pipeline

## Watermarking tests

All TTS output is automatically watermarked. When changing TTS output
paths, ensure these test suites still pass:

```bash
# Unit tests (no model needed)
build/bin/test_server_wav_writer    # WAV LIST/INFO metadata
build/bin/test_watermark            # Spread-spectrum embed/detect
build/bin/test_tts_provenance       # ID3v2, C2PA, consent, edge cases
build/bin/test_audioseal "[unit]"   # AudioSeal API surface

# Live tests (need AudioSeal GGUF)
python3 models/convert-audioseal-to-gguf.py -o audioseal.gguf
AUDIOSEAL_GGUF=audioseal.gguf build/bin/test_audioseal "[live]"

# Cosine parity with PyTorch (need reference .npy files)
AUDIOSEAL_GGUF=audioseal.gguf build/bin/test_audioseal_cosine
```

## Regression-test your backend

For ASR backends, the transcript is the regression target:

```bash
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > before.txt
# ... make changes ...
cmake --build build --target crispasr-lib
./build/bin/crispasr --backend yourmodel -m model.gguf -f samples/jfk.wav -np > after.txt
diff before.txt after.txt && echo BIT-IDENTICAL
```

For TTS backends the output is audio (not text), and diffusion
samplers are stochastic, so a `diff` of two runs won't compare. The
regression target is "audio cosine similarity vs the official
model's output, with the same Gaussian noise pinned in both runs":

```bash
# 1. Run the official PyTorch model with hooks that capture the
#    per-frame init noise plus the conditions / latents per frame
HF_HOME=/path/to/hf-cache python tools/run_official_vibevoice.py \
    --text "Hello, how are you today?" \
    --voice voices_pt/en-Emma_woman.pt \
    --output-wav /tmp/ref.wav \
    --output-dir /tmp/ref_dump

# 2. Run crispasr with the same noise pinned and per-frame dumps
VIBEVOICE_TTS_NOISE=/tmp/ref_dump/noise.bin \
VIBEVOICE_TTS_DUMP=/tmp/cpp_dump VIBEVOICE_TTS_DUMP_PERFRAME=1 \
./build/bin/crispasr --tts "Hello, how are you today?" \
    -m vibevoice-realtime-0.5b-tts-f16.gguf \
    --voice vibevoice-voice-emma.gguf \
    --tts-output /tmp/cpp.wav -ng

# 3. Audio cos at xcorr peak (accounts for any leading-silence trim)
python -c "import sys; sys.path.insert(0, 'tools'); \
    from _audio_diff import cos_report; \
    print(cos_report('/tmp/ref.wav', '/tmp/cpp.wav'))"
# OFFICIAL: 182400 samples = 7.60s  rms=0.0653
# AFTER_FIX: 171459 samples = 7.14s  rms=0.0672
# cos at zero shift  = 0.0027
# cos at xcorr peak  = 0.9991  lag=7741 samples = 322.5 ms
```

`cos at xcorr peak ≥ 0.999` is "essentially bit-exact modulo F16
quantization". A drop indicates a real divergence — pair the audio
diff with the per-frame stage diff (next section) to localise.

## Debug a new backend against PyTorch ground truth

Bit-identical regression against the previous C++ version proves the
change was neutral, but it doesn't tell you the C++ forward pass is
correct in the first place. For that, use the ground-truth tools:

```bash
# 1. Capture PyTorch reference activations at every named stage
python tools/dump_reference.py --backend voxtral \
    --model-dir /path/to/hf/voxtral-mini-3b-2507 \
    --audio samples/jfk.wav \
    --output /tmp/voxtral-ref.gguf

# 2. Compare your C++ forward pass against the reference, stage by stage
./build/bin/crispasr-diff voxtral \
    voxtral-mini-3b-2507-q4_k.gguf \
    /tmp/voxtral-ref.gguf \
    samples/jfk.wav
#
# [PASS] mel_spectrogram    shape=[128,3000]  cos_min=0.99998  max_abs=3e-5
# [PASS] projector_output   shape=[375,3072]  cos_min=0.99985  max_abs=4e-4
# summary: 2 pass, 0 fail, 0 skip (cos threshold 0.999)
```

**`crispasr-diff` works for TTS backends too**, not only ASR. For TTS,
the 4th argument (`audio.wav`) is ignored — pass any valid WAV (e.g.
`samples/jfk.wav`). Text and other TTS inputs come from env vars
(`<BACKEND>_SYN_TEXT`, `ZONOS_TTS_TEXT`, `CHATTERBOX_SYN_TEXT`, …).
See the **`chatterbox`** dispatch in `crispasr_diff_main.cpp` as the
canonical TTS template; `zonos-tts` follows the same pattern.

```bash
# TTS example — conditioning_prefix stage for Zonos:
ZONOS_TTS_TEXT="Hello world." ZONOS_SPEAKER_EMB_PATH=/path/to/jfk_speaker_emb.bin \
python tools/dump_reference.py --backend zonos-tts \
    --model-dir Zyphra/Zonos-v0.1-transformer \
    --audio samples/jfk.wav \
    --stages conditioning_prefix \
    --output /tmp/zonos-ref.gguf

ZONOS_TTS_TEXT="Hello world." ZONOS_SPEAKER_EMB_PATH=/path/to/jfk_speaker_emb.bin \
./build/bin/crispasr-diff zonos-tts \
    zonos-v0.1-transformer-q4_k.gguf \
    /tmp/zonos-ref.gguf \
    samples/jfk.wav
# [PASS/FAIL] conditioning_prefix  shape=[2048,22,2]  cos_min=…  cos_mean=…
```

The Python dumper uses PyTorch forward hooks to capture intermediate
activations (mel, per-encoder-layer output, projector, LLM block
output, logits, argmax) and writes them to a single **GGUF tensor
archive**. The C++ side loads the archive via
`core_gguf::load_weights` and runs the backend's public stage helpers
(`*_compute_mel`, `*_run_encoder`, etc.) to produce the same tensors,
then the shared `crispasr_diff::Ref` compares them with **cosine
similarity per row**, **max-abs error**, **RMS**, and — for logits —
**top-1 argmax match rate**.

Adding a new backend to the dumper is a ~60-line file in
`tools/reference_backends/<name>.py` that registers PyTorch forward
hooks and returns a dict `{stage_name: ndarray}`. Worked examples:

- `tools/reference_backends/qwen3.py`, `voxtral.py`, `cohere.py`,
  `parakeet.py`, `gemma4.py`, `omniasr_llm.py`, `granite.py` —
  encoder-decoder / Audio-LLM ASR backends. Use the `_hooks.py`
  forward_hook helpers (`capture_modules`, `drop_hooks`, `finalize`).
- `tools/reference_backends/qwen3_tts.py`, `qwen3_tts_codec.py`,
  `qwen3_tts_spk.py`, `qwen3_tts_cenc.py` — TTS prefill / encoder
  backends. Use `capture_modules(..., first_call_only=True)` for
  hooks that fire once per stage (e.g. talker prefill called from
  inside `generate()`).
- `tools/reference_backends/vibevoice.py`, `vibevoice_tts.py` —
  VibeVoice σ-VAE encoder + TTS pipeline.

For **autoregressive / diffusion-sampler diffs** the per-stage
capture above isn't enough — bugs that appear only after several AR
steps don't show up in a frame-0 cos diff. Two extra helpers:

- `tools/reference_backends/_iter_capture.py` — companion to
  `_hooks.py` for "monkey-patch a sampler entry point and append one
  tensor per iteration to a `{stage: [...]}` dict". Used to capture
  `pos_cond / neg_cond / noise / v_cfg_step0 / latent` per frame
  inside `sample_speech_tokens` for vibevoice.
- `tools/_audio_diff.py` — sample-wise audio cos at zero shift, cos
  at the cross-correlation peak (so leading-silence trims and
  causal-padding offsets don't tank the score), spectral band-power
  table, and a one-call `cos_report(a_path, b_path)` for CLI use.

`tools/run_official_vibevoice.py` is the worked example combining
all three: it loads the upstream model, monkey-patches
`sample_speech_tokens` via `_iter_capture.patch_method`, captures
`acoustic_embed` via a standard forward_hook, and writes both
`perframe_<stage>_f<NNN>.bin` files (matching the C++ runtime's
`VIBEVOICE_TTS_DUMP_PERFRAME=1` output) and a `noise.bin` for the
C++ side to replay.
