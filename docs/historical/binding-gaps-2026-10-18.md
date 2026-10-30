> This is the initial source audit, not current completion status. Python/C#
> checkpoints are in HISTORY.md; Java/Ruby now have compiled-probe repairs for
> the core rows below. Dart/Go/Rust/WASM integration, auxiliary inventories and
> hosted/model acceptance are tracked in PLAN.md. Do not count this table's
> historical omissions as still absent without inspecting the current code.

# Non-Python wrapper audit — 2026-10-18

Source snapshot: `c12707299c5dcbca7bafe60a073688250238cdd9` (the following
main commit changes documentation only). This is a source audit of callable
APIs and result readers, not a runtime certification or an exhaustive finished
coverage manifest. Python's new header census does not establish parity for
these wrappers. Bare declarations do not establish a usable public method.

## Python census boundary

The previous Python guard covers `include/crispasr_session.h`, not every public
header. `include/crispasr.h` additionally declares eight functions without Python
bindings: `crispasr_audio_load`, `crispasr_audio_load_at_rate`,
`crispasr_audio_load_stereo`, `crispasr_pcm_to_wav`, `crispasr_c2pa_sign`,
`crispasr_c2pa_free`, `crispasr_session_input_channels` and
`crispasr_session_output_channels`. These remain open. The legacy by-value
parallel entry point is already handled by the compiled adapter and is not an
additional omission. Extend the inventory across public headers and document
real equivalents explicitly before claiming complete Python C-ABI coverage.

## Confirmed gaps

| Wrapper | Existing relevant functionality | Confirmed gaps |
| --- | --- | --- |
| Dart / Flutter | Session word alternatives and speaker labels; piano/programs and beats | Shared-session token/raw-byte results; chords, tablature and ContentVec conversion; PCM voice references, sentence-streaming TTS, distinct TTS CFG-scale setter and TADA companion overrides |
| Go | ASR/session controls and auxiliary APIs; separate legacy Parakeet token reader | Shared-session tokens; word-alternative getters are declared but not consumed; piano/chords/beats/tab/conversion operations; the TTS additions above |
| Rust | Safe `set_voice_samples`; ASR, alignment and auxiliary APIs; separate legacy Parakeet token access | Shared-session tokens; safe session words omit alternatives; session segments omit native speaker labels; music/conversion and remaining TTS additions |
| Java | ASR/auxiliary APIs, speaker labels; separate legacy Parakeet tokens | Shared-session tokens; `Word` omits alternatives despite JNA declarations; music/conversion, PCM voice references and remaining TTS additions |
| Ruby | ASR/auxiliary APIs and legacy Parakeet token access; individual result alternative probability accessor | Shared-session tokens; usable alternative-text/structured readout needs wiring; music/conversion, PCM voice references and remaining TTS additions |
| C# | Broad music/conversion surface in `SessionMusic.cs`; structured word alternatives | Shared-session tokens; new conversion replay-dimension getters and noise-length validation; PCM voice references and remaining TTS additions |
| WASM/JS | Session ASR and setters; piano/programs and chords; legacy Parakeet tokens | Shared-session tokens; word-alternative functions are declarations without a result readout; beats/tab/conversion and remaining TTS additions; backend/build availability must be checked separately |

The shared `crispasr_session_result_n_tokens` / `result_token_*` API is absent
from all seven audited non-Python wrapper implementations. Legacy
`crispasr_parakeet_result_token_*` calls are a different result API and do not
close this gap for the backend-agnostic session, chunked, VAD or logits paths.

Two additional correctness findings need behavior tests and fixes:

- Dart's shared session reader and Rust's safe session reader divide negative
  word timestamps by 100 unconditionally. A native `-1` therefore becomes
  `-0.01`, contrary to the documented missing-time convention. Preserve the
  binding's documented missing value and add an explicit availability predicate
  without breaking existing callers. Exercise every reader, offset/remap and
  partial/streaming result path rather than fixing only the default reader.
- C# `SessionMusic.Convert` accepts optional replay-noise arrays and passes them
  to native code without checking their lengths. Native conversion copies
  `inter_channels * frames` and `frames * upsample_product` floats; a short
  caller array can cause an out-of-bounds read. The new native dimension getters
  make validation possible. Add checked products and pre-FFI rejection of short
  buffers, using the actual loaded model dimensions; do not hard-code them.

## Source entry points

- Dart: `flutter/crispasr/lib/src/crispasr.dart`, `SessionSegment`, shared result
  reader, `pianoNotesWithPrograms` and `beats`.
- Go: `bindings/go/crispasr_session.go`, cgo declarations and public result
  construction; declarations for alternatives currently have no call sites.
- Rust: `crispasr-sys/src/lib.rs` plus `crispasr/src/lib.rs`, `SessionWord`,
  `SessionSegment`, `parse_result` and duplicated VAD result construction.
- Java: `bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java`,
  JNA `Lib`, `Word`/`Segment` and result construction.
- Ruby: `bindings/ruby/ext/ruby_crispasr_session.c`, native functions plus Ruby
  method registration; both are necessary for a usable wrapper API.
- C#: `bindings/csharp/CrispASR/NativeMethods.cs`, `Session.cs`, `SessionMusic.cs`
  and the public result types. A P/Invoke declaration alone is insufficient.
- WASM: `bindings/javascript/emscripten.cpp`, native declarations, result object
  assembly and `emscripten::function` registrations. Build-gated backends require
  explicit capability reporting; a registered wrapper cannot create a backend.

## Acceptance and repair order

Active tasks live in PLAN.md. First fix C# replay-buffer validation, then shared
result metadata and missing-time semantics (Dart first, followed by the other
wrappers). Next complete Dart operations, Go/Rust, then Java/Ruby/C#/WASM.
Every family needs an explicit native-symbol → callable-public-method/result
mapping, with documented equivalent, retired or unavailable platform cases.
Do not present a count of symbol mentions as a coverage percentage.

Reuse the model-free native fixtures to test actual language FFI calls:
retained metadata/raw bytes, independent copies after native free/mutation,
missing times, empty results, errors, disposal and callback/thread lifetime.
Run the language's compiler/analyzer and test suite locally where practical;
use its GitHub CI toolchain for unavailable platforms. Test native exported
functions against the same pinned library. Keep real-model output and GPU
acceptance separate; the Python fixture success is not evidence for another
language's layout, marshaling or buffer ownership.
