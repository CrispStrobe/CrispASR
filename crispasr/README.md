# crispasr

Safe Rust wrapper for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports 17 ASR backends including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

`crispasr` and `crispasr-sys` are on [crates.io](https://crates.io/crates/crispasr),
but the FFI crate still needs a native `libcrispasr` to link against. Pick one
of two modes:

**A — build from source (git dependency).** `build.rs` cmakes `libcrispasr` for
you (needs `cmake` + a C++ toolchain; the git checkout supplies the sources):

```toml
[dependencies]
crispasr = { git = "https://github.com/CrispStrobe/CrispASR" }
```

**B — link a pre-built library (crates.io dependency).** Build/install
`libcrispasr` once, then point `CRISPASR_LIB_DIR` at it — this is required
because the crates.io package does **not** vendor the C/C++ sources, so
`build.rs` cannot cmake it:

```toml
[dependencies]
crispasr = "0.8"
```

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR && cmake -B build && cmake --build build -j && sudo cmake --install build
# then, in your project:  export CRISPASR_LIB_DIR=/usr/local/lib
```

## Quick start

```rust
use crispasr::Session;

let model = Session::open("ggml-base.en.bin")?;
let pcm: Vec<f32> = /* decoded mono 16 kHz audio */ vec![0.0; 16000];
for seg in model.transcribe(&pcm)? {
    println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
}
```

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, the model registry, and the CLI.

## License

MIT — see [LICENSE](LICENSE).

`Session::open_with_options` accepts `SessionOptions` with native defaults.
`TruecaseModel` and `PCSModel` own their contexts; mutable `process` returns
`Result<String, String>` and Drop releases the model. `Session::score_texts`
returns copied candidate scores and token counts. `drain_streamed_segments`
returns copied metadata; progress/count/reset helpers are also available.
Raw token polling is unsafe because native storage is borrowed and selected
globally: callers must serialize drain/inference/close until copying finishes.

Word `confidence` preserves native -1 when unavailable. VAD and owned polling
share the ordinary result reader, including count checks and RAII cleanup.


`align_words_ultra(base_path, head_path, text, pcm, n_threads)` provides
experimental Qwen Ultra local-head alignment with owned `UltraAlignedWord` values
and `f64` seconds, preserving fractional milliseconds. Supply both converted
GGUFs and 1–1,920,000 mono 16 kHz samples; empty/NUL-containing strings and native
failures return errors. The head runs on CPU. See the
[local-head guide](../docs/qwen3-ultra-local-head-port.md) for qualification scope.

## Legacy Whisper compatibility

The deprecated `CrispASR` type remains available. Its opaque-parameter sys calls
now link to the native `*_by_ref` exports, preserving the Rust function names.
Use a matching native library with those exports and the option helpers; missing
exports cannot be treated as silent no-ops. Empty/oversized PCM, invalid strategy
or VAD settings, NUL-containing VAD paths and null native parameters return errors.
VAD and tinydiarize can be enabled together. Native parameter allocations are
freed on success and native failure; the VAD path remains alive through inference.

`CrispASR::new_with_dtw(path, DtwPreset::BaseEn)` enables experimental DTW
alignment before loading. `DtwPreset::TopMost(n)` requires positive `n`; the
native loader checks the selected layers/heads against the model dimensions.
Custom head arrays are not exposed. Use a native library whose DTW helper
disables incompatible flash attention when enabling DTW; older helpers can
leave flash enabled and the loader then disables DTW. Ordinary `new` keeps
native defaults. `token_dtw_timestamp(segment, token)` checks result indices
and returns native centiseconds, preserving `-1` when unavailable.

`transcribe_pcm_parallel(&pcm, &options, processors)` uses the native
progress-tracked pointer API, retaining the same option validation, VAD-path
lifetime and copied segments as sequential decoding. Processors must be positive;
chunk boundaries can reduce accuracy. `detect_language(&pcm, n_threads)` runs
Whisper's mel/encoder language pass and returns an owned `(code, probability)`.
It requires positive threads and validates native errors and bounded UTF-8 output.
This mutates native encoder state; serialize operations on the model.

`token(segment, index)` and `tokens(segment)` return owned `WhisperToken`
values with IDs, raw bytes, native-centisecond timing/DTW, probabilities and
owned alternative candidates. Negative unavailable values are preserved. `text`
uses replacement decoding; concatenate `raw` to reconstruct split UTF-8 exactly.
Snapshots survive later decoding and model drop. `transcribe_pcm_with_alternatives`
accepts the existing `TranscribeOptions` plus an alternative count (native 0–32
clamp); greedy decoding captures candidates, while beam decoding does not.
Existing `TranscribeOptions` and `Segment` struct literals remain unchanged.
Snapshot readers limit segments to one million tokens, alternatives to 32 per
token and text to 1 MiB per token/candidate, rejecting invalid counts or buffers.

`WhisperDecodeControls` adds per-call language/prompt strings, boolean flags,
threads, greedy best-of, temperature, length/token limits and alternatives without
changing `TranscribeOptions`. Construct it with `Default::default()` and assign
fields; it is non-exhaustive so future controls can be added compatibly:

```rust
let mut controls = crispasr::WhisperDecodeControls::default();
controls.language = Some("de".into());
controls.initial_prompt = Some("Grüße".into());
controls.n_threads = Some(4);
controls.token_timestamps = Some(true);
controls.alt_n = Some(5);
// With an already loaded legacy model and mono 16 kHz PCM:
// model.transcribe_pcm_with_controls(&pcm, &options, &controls)?;
// model.transcribe_pcm_parallel_with_controls(&pcm, &options, &controls, 2)?;
```

`None` preserves native defaults; `Some(false)` explicitly clears a flag.
Threads/best-of must be positive, limits nonnegative and temperature finite and
nonnegative. Language must be nonempty; language/prompt strings must be NUL-free
and remain owned until native inference and parameter cleanup finish. Empty
prompts are valid. Alternative counts retain the native 0–32 clamp; native beam
and chunk-merge behavior still applies. All controls share existing VAD/tinydiarize
validation, native failure cleanup and owned segment handling on both routes.

Prefer `Session` for backend dispatch and native exception handling. Repairing
the legacy pointer ABI does not make the deprecated direct Whisper API safe
against C++ exceptions. Model/platform acceptance remains separate from the
compiled ABI operation probes recorded in the repository's PLAN.md.
