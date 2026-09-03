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

**Forward the generation cap — do NOT hardcode the decode limit.** An
autoregressive/LLM ASR backend has a decode loop bounded by some `max_new`. If
you hardcode it (`const int max_new = 512;`), a user transcribing long audio in
a single pass (`--chunk-seconds 0`) is silently truncated and `--max-new-tokens`
does nothing — this was #292, found in **10 backends at once**. Instead:

- put the cap on the context as a field defaulting to the backend's sensible
  value (`int max_new_tokens = 512;` — keep whatever the old constant was, so
  the default does not regress), and read it for BOTH the decode loop bound and
  the KV-context sizing;
- add a `yourmodel_set_max_new_tokens(ctx, n)` setter (mirror `set_beam_size`),
  declared in the header;
- in the adapter's `transcribe()`, forward it **only when the user set it
  explicitly**:
  `yourmodel_set_max_new_tokens(ctx_, p.max_new_tokens_explicit ? p.max_new_tokens : 0);`
  The `max_new_tokens_explicit` flag exists precisely so the CLI's global 512
  default never SHRINKS a backend whose own default is higher (diarize is 1024).
  Pass `<= 0` to keep the backend default.
- forward it on the session path too (`src/crispasr_c_api.cpp`, next to where the
  backend's other setters are called): `yourmodel_set_max_new_tokens(s->yourmodel_ctx, s->max_new_tokens);`
  — the session default is 0, so no explicit flag is needed there.

The same rule applies to any other decode knob (temperature, beam) — the audit
for #292 confirmed those were already forwarded; max_new_tokens was the one gap.

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

Add the architecture string to the shared table in
`src/core/arch_backend_map.h` so `general.architecture` auto-detection works
on **every** surface at once (CLI, C ABI, and therefore every binding).

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

   **Also register the rate in `crispasr_session_output_sample_rate()`
   (`src/crispasr_c_api.cpp`) — the session ABI cannot reach the CLI
   adapter, so the getter keeps its own per-backend table (#332).** Every
   audio-producing ctx must appear there: a backend at the 24 kHz default
   adds its ctx to the 24 kHz fallthrough list; a non-24 kHz backend adds
   the explicit rate (or its `<name>_sample_rate(ctx)` getter when the
   rate is a model hparam). A ctx missing from the table makes the
   session report "no audio output" (0) for a backend that synthesizes.

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

### Auto-detection — filename pass + the shared architecture table
So `-m model.gguf` *without* `--backend` routes correctly, wire **both** passes:
- Pass 1 (filename heuristic, CLI only) in `examples/cli/crispasr_backend.cpp`:
  `if (contains_ci("yourmodel")) return "yourmodel";`
- Pass 2 (GGUF `general.architecture`) in **`src/core/arch_backend_map.h`**:
  `{"<arch>", "yourmodel"},` — `<arch>` is whatever your converter passes to
  `GGUFWriter(arch=...)`, verbatim, including underscores. List every spelling
  that ships.

That table is the single source of truth for both the CLI's pass 2 and the C
ABI's `crispasr_detect_backend_from_gguf()`. **It used to be two copies, and
issue #335 is what that cost:** they drifted by 113 architecture strings, so
granite-speech (converter writes `granite_speech`, the C-ABI copy only knew
`granite-speech`) opened fine in the CLI — rescued by the filename pass — and
returned a NULL session from Rust/Python/Go/Dart. Only the CLI has a filename
pass, so a missing table entry is invisible on the surface you are most likely
to test on. `tests/test-arch-backend-map.cpp` pins the mapping and drives the
real C-ABI export over a synthesised metadata-only GGUF.

The `--list-backends` capability row is read live from the backend's
`capabilities()`, so it needs no separate edit.

### C ABI — `src/crispasr_c_api.cpp` (this is what the bindings/server call)
Python/Go/Dart/server use the **session C ABI**, not the CLI factory.
Ten edit points, each mirroring the `CA_HAVE_CHATTERBOX` blocks:
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
   glm-asr, gemma4-e2b, mimo-asr, higgs-stt, ark-asr, moss-audio,
   moss-diarize, mini-omni2, lfm2-audio.
9b. **`params.language` wiring** — if the backend is an audio-LLM, also
   inject a language hint into the prompt when `params.language` is set
   and non-auto. Pattern: `else if (!params.language.empty() && params.language
   != "auto") { sys_instruction = "Transcribe the speech in " + lang_name(params.language) + "."; }`.
   For a model that cannot honour the request, emit a `fprintf(stderr, ...)`
   warning instead of silently ignoring the flag — moonshine-streaming warns
   "English-only model; language=… ignored", and kyutai-stt asks the context
   (`kyutai_stt_supports_language`) and names what the loaded GGUF actually
   supports (#366: the blanket "English-only" warning was wrong for the default
   `stt-1b-en_fr`). Both warnings live in the CLI adapters.
   Currently wired for: granite (v3 + v4 templates), qwen3-asr, glm-asr,
   moss-audio, mimo-asr, higgs-stt, ark-asr.
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
- ⚠ `bindings/go/whisper.go` — **REGENERATE THE cgo LDFLAGS**. This step used
  to be documented here as unnecessary ("Go links `-lcrispasr`, not
  per-backend"), which is wrong and cost two red-CI incidents (basic-pitch,
  then mt3): the `#cgo linux/darwin LDFLAGS` lines enumerate EVERY static
  library by name, so a new `add_library(<backend> STATIC …)` in
  `src/CMakeLists.txt` must be added there or the Go bindings fail to link
  with `undefined reference to <backend>_*`. Do not hand-edit the list — run

  ```bash
  python tools/sync_go_cgo_ldflags.py           # regenerates from a cmake graph
  python tools/sync_go_cgo_ldflags.py --check   # what CI asserts
  ```

  `tools/check-backend-wiring.py` runs the `--check` form; a failure prints
  `RESULT: FAIL (Go cgo LDFLAGS drift)` and names this script.
- `flutter/crispasr/lib/src/crispasr.dart` — add to the synthesize
  docstring. Dart uses `DynamicLibrary.lookupFunction` with symbol-presence
  checks, so new C-ABI functions are discovered automatically.

### Bindings — adding a new *session setter* (`crispasr_session_set_*`) or *method*
This applies to any explicit session entry point — the `crispasr_session_set_*`
setters **and** the data-returning session methods such as
`crispasr_session_synthesize`, `crispasr_session_synthesize_raw`, and
`crispasr_session_speech_to_speech`. None of these are auto-discovered (only a new
*backend* is — see the section above); every wrapper exposes them explicitly and
they are kept at **full parity**. When you add one, add it to **all seven** wrappers
(plus the WASM/JS binding and the server, listed below — mirror the nearest existing
setter/method in each: argtypes/restype, error-on-rc≠0,
and for PCM-returning methods copy `synthesize`: return the malloc'd `float*` as an
owned buffer then free via `crispasr_pcm_free`; free any `out_text` via
`crispasr_session_translate_text_free`):
- `python/crispasr/_binding.py` — ctypes method on `Session`.
- `bindings/go/crispasr_session.go` — the cgo-preamble `int crispasr_session_set_X(...)`
  declaration **and** the `Set X` method.
- **Rust (repo root, not `bindings/`)**: `crispasr-sys/src/lib.rs` extern decl
  **and** `crispasr/src/lib.rs` safe `pub fn`.
- `flutter/crispasr/lib/src/crispasr.dart` — `lookupFunction` + method.
- `bindings/java/.../CrispasrSession.java` — JNA `Lib` interface decl + method.
- `bindings/ruby/ext/ruby_crispasr_session.c` — `extern` decl, `rb_session_set_X`,
  and a `rb_define_singleton_method` registration.
- `bindings/csharp/CrispASR/NativeMethods.cs` — `[DllImport]` P/Invoke, **and**
  the public method on `Session` in `Session.cs`. Callback delegates need
  `[UnmanagedFunctionPointer(CallingConvention.Cdecl)]` and a static field to
  survive GC; `const char*` params are `[MarshalAs(UnmanagedType.LPUTF8Str)]`.
- The HTTP server (`examples/cli/crispasr_server.cpp`) exposes the equivalent as
  a per-request `form_*` field on the transcription endpoints (or a startup flag
  for resident post-processors).
- `bindings/javascript/emscripten.cpp` — WASM/JS (built with emcc).
The canonical surface is `include/crispasr_session.h`; `docs/bindings.md` has the
per-wrapper setter table.

### Append-only ABI structs — update EVERY hand-written mirror in the same commit

Some ABI entry points take struct pointers whose layout is defined only in
`src/crispasr_c_api.cpp` (`crispasr_diarize_seg_abi`, `crispasr_diarize_opts_abi`)
or in `include/crispasr_session.h` (`crispasr_vad_abi_opts`). Bindings that
cannot include a C header lay these structs out **by hand**, byte for byte:
the Go cgo preamble, `crispasr-sys/src/lib.rs` (`#[repr(C)]` mirrors), and the
flutter binding's offset-written buffers in `crispasr.dart`.

The C side reads **every field unconditionally**, so a mirror that is one
append behind makes the C side read past the caller's allocation — undefined
behaviour on every call, not just calls that use the new fields. This is not
hypothetical: #324 appended the FoxNose fields to `crispasr_diarize_opts_abi`
and updated only the Go mirror; #332 found the Rust and Dart mirrors 24 bytes
short, with a garbage pointer handed to `std::string` on the C side.

Rules when you touch one of these structs:

1. **Append at the END; never reorder or insert.** The structs are
   append-only by contract — old mirrors must stay prefix-compatible
   while they're being caught up (they aren't safe, but they're findable).
2. **Update every mirror in the same commit.** The authoritative mirror
   list lives in the comment next to the struct in `crispasr_c_api.cpp`;
   extend that list when a new binding grows a mirror.
3. **Keep the layout guards in sync**: the `static_assert`s next to the
   struct in `crispasr_c_api.cpp` pin the canonical sizes/offsets, and each
   mirror carries its own guard — `crispasr-sys`'s `diarize_abi_layout`
   test, flutter's `DiarizeMethod` index-parity smoke test, and
   `tests/test-session-abi-nulls.cpp` (which calls the ABI through a
   fourth hand-written mirror, so a silent layout change fails a unit
   test even if a binding is missed).
4. **Enum values that cross the ABI are append-only too** — the Dart
   `DiarizeMethod`/`LidMethod` enums dispatch on `.index`, and the Rust
   enums on `as i32`; a reorder or mid-enum insertion silently routes
   calls to the wrong method. The index-parity tests pin this.

**C# is CI-tested** (`.github/workflows/bindings-csharp.yml`) — it compiles the
binding against the ABI and runs `CrispASR.Tests` on **ubuntu and windows**, each
against a real moonshine GGUF and `samples/jfk.wav`, with
`CRISPASR_CS_REQUIRE_LIVE=1` so a skipped live test is a failure. Do not let it
drift; it was unbuilt for a long time and shipped a units bug (#291) precisely
because nothing compiled the wrapper against the header.

Three C#-specific rules the second round of #291 established:
- **The managed assembly is `CrispASR.Net.dll`, never `CrispASR.dll`** — the
  native library is `crispasr.dll` and Windows file names are case-insensitive,
  so the two would be one file. `NativeLibraryResolverTests` pins this on every
  platform, including the Linux ones where the collision cannot occur.
- **Every time value the binding exposes is SECONDS** (`Segment`, `Word`,
  `AlignedWord`, `VadSpan`, `StreamingUpdate`, the music types). Convert at the
  boundary via `Session.Seconds()`, which also preserves the C ABI's `-1`
  "no timing" sentinel instead of scaling it to `-0.01`.
- **A live test may not skip silently in CI.** Route every guard through
  `Live` (`bindings/csharp/CrispASR.Tests/Live.cs`); a bare
  `if (!CanLoadLibrary()) return;` makes the suite green on a machine with no
  native library at all, which is how "all xunit tests pass" came to mean
  nothing.

`bindings/csharp/README.md` is the consumer-facing guide: which release archive
to download, where to put the native library, and the `CRISPASR_LIBRARY_PATH`
escape hatch.

### Docs
- `README.md` — model-table row (TTS or ASR section).
- `docs/tts.md` — backend table row (TTS backends).
- `docs/architecture.md` — a `### yourmodel` section under "Per-backend
  architecture details"; README/tts.md link `docs/architecture.md#yourmodel`.

### Build targets (don't be fooled by stale binaries)
- `crispasr-lib` → the **library** (`OUTPUT_NAME crispasr` → libcrispasr / `.dylib`).
- `crispasr-cli` → the **CLI binary** (`OUTPUT_NAME crispasr`).
- `crispasr-diff` → the diff harness.

(There is no bare `crispasr` target — `add_library(whisper ALIAS crispasr-lib)`
is the only alias.)

After C-ABI edits, build **`crispasr-lib`** (the dylib) and re-test the Python
`Session` — building only `crispasr-cli` may leave the dylib stale, and the
binding then loads an old backend list. Verify with:
```python
import crispasr
assert "yourmodel" in crispasr.Session.available_backends()
crispasr.Session("model.gguf", backend="yourmodel")  # opens?
```

**Audit the wiring automatically.** After adding a backend, run the audit — it
checks every canonical backend (those with a dedicated CLI adapter) against the
required surface (factory / c_api dispatch + `available_backends` list /
feature-matrix / cli.md beam list) and reports advisory coverage gaps (test,
reference dumper, README, registry, streaming). Aliases and standalone reference
dumpers are handled, not false-flagged:

```bash
python tools/check-backend-wiring.py --crispasr ./build/bin/crispasr   # exit 1 on a required gap
```

> ⚠️ **Commit from a separate `git worktree`, or `git pull --rebase`
> immediately before committing.** A `git add -A` / `commit -a` from a
> parallel session over a stale tree silently reverts others' work through
> the shared `.git/index` — this clobbered the entire §135 CSM landing
> (commit `100b9ee5`). `git config pull.rebase true` is set in this repo.

## 7. Task-shaped backends (`--separate`, `--pitch`, `--chords`)

Everything above assumes the backend produces TEXT and therefore fits
`transcribe()`. Some do not: source separation returns stems, pitch returns F0
frames, chord recognition returns a chord timeline. None of those can be
expressed as `crispasr_segment`s, so they get their OWN task surface instead of
being forced through the transcribe contract. Precedents:
`htdemucs`/`mel-band-roformer` (`--separate`), `crepe` (`--pitch`), `btc-chords`
(`--chords`).

The trap: because such a backend never appears in a transcribe path, it is easy
to ship it working end-to-end while it remains invisible to the CLI's backend
registry. `btc-chords` did exactly that — runtime, `--chords` dispatcher,
session C ABI and wasm bindings all shipped and verified, while `btc` appeared
NOWHERE in `examples/cli/crispasr_backend.cpp`. `--list-backends` did not know
it existed, and `docs/feature-matrix.md` is generated from
`--list-backends-json`, so any hand-written row for it would have been silently
dropped on the next regeneration.

Wire ALL of the following:

1. **Task dispatcher** — `examples/cli/crispasr_<task>_cli.{h,cpp}`, called from
   `crispasr_run_backend()` **and** from an early route in `cli.cpp`, before any
   transcribe backend is constructed. Both: a dispatch in `crispasr_run.cpp`
   alone still falls through to whisper and dies on "invalid model data".
2. **Capability bit** — a new `CAP_<TASK>` in `examples/cli/crispasr_backend.h`,
   added to BOTH capability-name tables in `crispasr_backend.cpp` (the text one
   and the JSON one) so it shows up in `--list-backends` and the matrix.
3. **Redirect shim** — `examples/cli/crispasr_backend_<name>.cpp` implementing
   `CrispasrBackend` whose `init()` prints "run it with `--<task>`" and returns
   false, with `capabilities()` returning the task bit. This is what puts the
   backend in `--list-backends` and gives `--backend X` (without the task flag)
   a clear error instead of a confusing one. Copy
   `crispasr_backend_crepe.cpp` or `crispasr_backend_btc.cpp`.
4. **Factory + roster** — the `if (name == ...)` alias line and the roster list
   entry in `crispasr_backend.cpp`, plus the shim's forward declaration, plus
   the file in `examples/cli/CMakeLists.txt`.
5. **Both detect passes** — the filename heuristic in `crispasr_backend.cpp`
   (CLI only) and the GGUF `general.architecture` entry in the shared table
   `src/core/arch_backend_map.h` (CLI *and* every binding, since both the CLI's
   pass 2 and `crispasr_detect_backend_from_gguf()` read it). The table was
   itself two divergent copies until #335; crepe and htdemucs had already been
   caught getting a null session in every binding while working in the CLI, and
   granite-speech was that same bug again — unifying the table is what stops it
   recurring. Only the CLI has a filename pass, so **testing
   auto-detection through the CLI alone proves nothing about the bindings.**
6. **Session C ABI** — task-specific entry points, NOT `transcribe()`. Follow
   `crispasr_session_pitch*` / `crispasr_session_chords*`: a `run` call
   returning a count, an `n_*` accessor, a FLAT float view for the bulk read,
   and any string lookup separately. Flat views must be all-float even when a
   field is logically an integer — a mixed int/float struct read through a float
   view misreads the int lanes.
7. **Language-wrapper binding** — UNLIKE a plain transcribe/synthesize backend
   (which the wrappers pick up automatically via the generic dispatch), a task
   surface adds NEW functions that each wrapper must bind explicitly, or the
   backend is C-only. This is where C# sat neglected: `tab`/`beats`/`chords`/
   `piano`/`pitch`/`separate`/`convert` were in the C ABI but bound in no
   managed wrapper. When you add a task surface, bind its run call + getters in
   every wrapper that exposes typed methods — `bindings/csharp` (`SessionMusic.cs`
   is the precedent: one P/Invoke per native function, a `readonly struct` result
   type, one `Marshal.Copy` of the flat view), plus python/go/flutter/etc. as
   applicable. **Normalise time to seconds** in the wrapper even when the flat
   view is milliseconds (chords/piano/pitch are ms; beats is already seconds) —
   an inconsistent unit across methods is the #291 bug.
8. **Regenerate the matrix** — `python tools/gen-feature-matrix.py`. Do not
   hand-edit `docs/feature-matrix.md`; it is generated and says so at the top.

Then run the audit, which now checks the reverse direction too (advertised by
the C ABI but unreachable from the CLI):

```bash
python tools/check-backend-wiring.py --crispasr ./build/bin/crispasr
```

## Running integration / live tests

Unit tests (1063 of them) need no models and pass unconditionally. The
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
| `CRISPASR_MODEL_ALIGNER` | CTC aligner live tests (canary-ctc-aligner) |
| `CRISPASR_CHAT_TEST_MODEL` | Chat (LLM) smoke test |

Tests that use `SKIP()` return exit code 4 (Catch2 convention). The
CMakeLists.txt sets `SKIP_RETURN_CODE 4` so ctest reports them as
"Skipped" rather than "Failed".

## Common pitfalls

### Does your model already punctuate? Declare it (`CAP_PUNCTUATION_NATIVE`)

CrispASR auto-enables FireRedPunc for any backend that advertises neither
`CAP_PUNCTUATION_NATIVE` nor `CAP_PUNCTUATION_TOGGLE`
(`crispasr_punctuation_policy.h`). Run that pass over text a model already
punctuated and you get `your country..` — and, before the fix below,
`ANd so` as well. Every LLM-decoder ASR backend emits punctuated, sentence-cased
text, so **it must declare `CAP_PUNCTUATION_NATIVE`**; CTC and other
unpunctuated backends must NOT (they need the pass).

**Do not decide this by reading the model card, and do not use
`--no-punctuation` to check.** That flag *strips* punctuation after the fact
(`crispasr_run.cpp`), so a model that punctuates itself looks unpunctuated
under it — which is exactly how a whole audit can reach the wrong conclusion.
Print the real thing instead:

```bash
FIREREDPUNC_DEBUG=1 crispasr -m <model> -f samples/jfk.wav --backend <name> 2>&1 | grep PUNCDBG
# [PUNCDBG] in=<And so, my fellow Americans, … your country.>   ← the model's OWN output
# [PUNCDBG] out=<And so, my fellow Americans, … your country..> ← the pass double-punctuating
```

If `in=` already carries commas/stops and sentence case, add the cap. The
2026-07-27 audit found `moonshine-streaming` and `mimo-asr` needed it, while
`canary-qwen` (cased but unpunctuated) and `firered-asr` (ALL CAPS,
unpunctuated) correctly rely on the pass — so this is per-backend evidence, not
a blanket flag.

### The modular libraries have TWO copies — patch both

`crisp_punc/`, `crisp_lid/` and `crisp_truecase/` each exist twice: the sibling
directory (preferred, and what normally links) and a fallback copy under `src/`
that `src/CMakeLists.txt` builds only when the sibling directory is missing from
a checkout. A fix applied to one copy silently does nothing in the normal build.

This is not hypothetical: #308's capitalisation fix landed in
`src/fireredpunc.cpp` while `crisp_punc/src/fireredpunc.cpp` — the copy that
actually links — kept the bug for months. Every symptom pointed at the file that
was already correct, and instrumenting that file produced no output at all,
which is the tell. `tests/test-copies-in-sync.cpp` now fails when the two
diverge (it was extended from the punc pair to all 14 duplicated files); keep it
green rather than deleting the assertion.

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

### GPU / CUDA correctness (it works on Metal ≠ it works on CUDA)

Apple-silicon Metal uses unified memory, so the CPU backend can read GPU
buffers. That hides two whole classes of bug that abort on CUDA — always
sanity-check a GPU-enabled backend on a real discrete GPU (Kaggle T4/P100),
not just M1.

- **Compute a `sched`-allocated graph with the `sched`, never a raw backend.**
  If you allocate with `ggml_backend_sched_alloc_graph(sched, gf)`, you MUST run
  it with `ggml_backend_sched_graph_compute(sched, gf)`. Calling
  `ggml_backend_graph_compute(backend_cpu, gf)` instead dereferences
  GPU-resident weights on the CPU backend — a silent no-op on Metal, an illegal
  access on CUDA. (FastPitch shipped this on its decoder + vocoder graphs while
  the encoder graph was correct — §204.)
- **`core_convt::convt1d_decomp` is channel-major; `convt1d_decomp_tf` is
  time-first.** The decomposed ConvTranspose1d's first op is
  `mul_mat(w_perm[IC, K*OC], x)`, which needs `x->ne0 = IC`. `core_hifigan` (and
  any pipeline built from `ggml_conv_1d`) is **time-major** `(T, C)` — `ne0 = T`,
  not `IC` — so it must use the `_tf` variant. The wrong one aborts on
  `ggml_can_mul_mat` on *all* backends. When a `mul_mat`/conv asserts inside a
  shared vocoder, suspect the input's contiguous axis, not the weights (§204).
- **`ggml_backend_sched` + a weight-less first op corrupts the GPU graph.** If a
  subgraph's leading op is weight-less (e.g. RMSNorm/LayerNorm before any matmul),
  the scheduler (op_offload=false) places that op *and the leaf input* on the CPU
  backend, then inserts a CPU→GPU copy of the activation to feed the next op — and
  on the current ggml revision that cross-backend copy is **miscomputed**, so the
  whole subgraph is garbage on GPU while CPU stays bit-correct. (Graphs whose
  first op uses a weight keep the input on the GPU and are fine — that's why
  encoders/adapters worked but the LFM2 backbone didn't, §206.) **Diagnose** with
  `GGML_SCHED_DEBUG=2` — look for a leaf input on `[CPU]` and `#<backend>#...#0`
  copy nodes. Gotcha: a standalone op test "on Metal" via the sched may silently
  run on CPU for *both* the CPU and Metal runs, trivially matching and masking the
  bug — confirm each op's real backend. **Fix:** when a whole subgraph is supported
  on the active backend, compute it *directly* with `ggml_gallocr` +
  `ggml_backend_graph_compute(ctx->backend, gf)` instead of the scheduler — a
  single-backend graph never inserts the broken copy. (Pinning the input with
  `set_tensor_backend` or `op_offload=true` did **not** help; only avoiding the
  split did.) Fixed lfm2-audio (§206) and kugelaudio (§209), which share this bug.
- **Mixing gallocr and sched when one graph has a Metal-unsupported op.** If most
  graphs need the gallocr fix above but ONE uses an op the active backend can't run
  (kugelaudio's VAE decoder uses `ggml_pad` — the causal-conv left-pad — which Metal
  rejects), gallocr has no fallback and aborts (`unsupported op 'PAD'`). Keep that
  one graph on `ggml_backend_sched` (which falls back to CPU for the unsupported op;
  on CUDA `PAD` is supported, so it runs fully on GPU) and run the rest on gallocr.
  This is safe as long as the sched-routed graph's *first* op uses a weight (input
  lands on GPU) — only the leaf-input + weight-less-first-op combo triggers the
  broken cross-backend copy; mid-graph CPU splits are fine. (A `ggml_concat`-of-a-
  scaled-view "manual pad" to avoid `GGML_OP_PAD` fails when pad > input length —
  you can't build zeros wider than the source view.)
- **Host-syncing ops crash a CUDA-graph-captured graph.** `ggml` can capture and
  replay an LLM decode step as a single `cudaGraphLaunch` (CUDA-graph capture, on
  by default via `GGML_CUDA_GRAPHS_DEFAULT`, arch-gated to sm_80+; disable with
  `GGML_CUDA_DISABLE_GRAPHS=1`), but only if *every* op in the graph is
  capture-safe. `get_rows_cuda_kquant` copies the index tensor D2H and calls
  `cudaStreamSynchronize` to read it on the host before launching its per-row
  dequant kernels — a host sync is illegal inside capture, so a graph containing
  GET_ROWS on a k-quant source crashes. F16/F32/legacy-quant GET_ROWS use the
  capture-safe `k_get_rows` path and are unaffected. (This bit
  `granite-speech-4.1-2b-q4_k.gguf`'s quantized token embedding under capture —
  §210.) **Fix:** deny CUDA-graph capture for graphs with k-quant GET_ROWS
  (`ggml_cuda.cu`), and keep the k-quant path on the legacy per-call rebuild. The
  same rule applies to any future op that host-syncs (e.g. reads a shape/extent on
  the host before dispatching) — it cannot live in a captured graph; route it
  through the non-captured path instead. **Related:** even on a topology-stable
  captured graph you must still `ggml_backend_sched_reset` + `sched_alloc_graph`
  per step — ggml's capture bookkeeping rejects the documented reuse-shortcut
  (`ggml-backend.h:285`) with `CUDA error: invalid argument` mid-decode.

## Watermarking tests

All TTS output is automatically watermarked (on by default). The mark can be
turned off with the `--no-watermark` CLI flag or the `CRISPASR_NO_WATERMARK`
env var — equivalent opt-outs, both emit a one-time stderr warning
(`watermarking disabled. AI usage marking responsibility rests with the
operator.`). The warning text is deliberately jurisdiction-neutral and does not
name a specific statute; see `docs/issue-260/PLAN.md` for the regulatory
background (incl. EU AI Act Art. 50). When changing TTS output paths, ensure
these test suites still pass:

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
