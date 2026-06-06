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

### Bindings — docstrings only (dispatch is automatic once the C ABI is wired)
- `python/crispasr/_binding.py` — add the name to the TTS-backend lists in
  the `synthesize` comment + docstring.
- `bindings/go/crispasr_session.go` — add to the header/type comments.
  Go links `-lcrispasr` (not per-backend), so **no LDFLAGS change**.
- `flutter/crispasr/lib/src/crispasr.dart` — add to the synthesize
  docstring. Dart uses `DynamicLibrary.lookupFunction` with symbol-presence
  checks, so new C-ABI functions are discovered automatically.

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
