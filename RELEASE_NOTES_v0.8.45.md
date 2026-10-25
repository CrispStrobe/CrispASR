# v0.8.45 — draft, not released

**Final qualification pending.** The recurring CSM-1B nightly regression in [run 40178598589](https://github.com/CrispStrobe/CrispASR/actions/runs/40178598589) produced “The grip brown fox.” for the pinned nine-word phrase (WER 0.6667; required maximum 0.15). Restoring the original model's missing rotary scaling passes the controlled AVX512/AVX2 speech checks and the original F32-to-native F16 stage checks. Final default-path, exact published-weight recipe and actual CUDA checks must finish before release preparation is completed.

These notes describe the changes after `v0.8.44` and the CSM correction under qualification on the working branch. Final source and validation receipts must be recorded before publication.

The bounded [CPU-profile experiment](docs/csm-cpu-profile-2026-10-25.json)
reproduces the failure on an AMD EPYC 9V74 with native flags. Two native
syntheses fail both ASR judges; two AVX2 syntheses pass both. Repeats have exact
PCM identity within each profile. This isolates synthesis sensitivity in that
experiment. The subsequent original-model audit identified dropped Llama-3
rotary scaling in both CSM transformer stacks.

## CSM speech correction

- The converter preserves the original model's scaling parameters and separate depth-decoder theta. The runtime applies the original frequency factors to backbone and depth graphs, including generation and C ABI diagnostic paths. Existing published CSM weights use canonical original parameters when the older GGUF lacks scaling metadata; malformed or unsupported new metadata is rejected.
- Pinned original operators, full F32 model hooks and magnitude-sensitive F16 stage checks qualify the corrected computation. The controlled AVX512 native speech path scores WER0 and AVX2 scores WER1/9, across both repeats and both ASR judges at the unchanged0.15 limit. Traced/untraced PCM and actual-synthesis/diagnostic logits agree. Final corrected-default and actual CUDA acceptance remain pending in this draft. See the [CSM evidence](docs/csm-cpu-profile-2026-10-25.json).

## Granite Speech 5.0

- Native YaRN conversion and runtime support add the licensed Granite Speech 5.0 2B model. The approved F16 profile and a separately qualified Q4_K profile are published at the [pinned model revision](https://huggingface.co/cstr/granite-speech-5.0-2b-GGUF/tree/e1df7d2cabb0ddba9863d7fc90cb6e78e8f19a6d). The model registry now selects Q4_K by default and keeps an explicit F16 choice; unsupported precision profiles remain rejected.
- The actual `crispasr-quantize` default recipe produces the qualified Q4_K value/down F16 artifact byte for byte. The accepted model is 4,074,560,192 bytes, SHA256 `63324167e3a85a67539a44712aef1a4624882561a939597cc20e1a31e864674d`. It preserves the reviewed sensitive tensors and needs no importance matrix or calibration audio. The older generic Q4, Q6 and Q8 candidates were not promoted.
- Acceptance covers original F32 stage values and magnitudes, prompt IDs, two exact 27-token English JFK decodes, and repeated renamed-model C ABI Session and CLI paths on hosted CPU and scoped Tesla T4 CUDA controls. Native context is 4096; rotary checks through 16383 do not qualify full 16k inference, other speech, throughput or general GPU accuracy. See [F16 evidence](docs/granite5-2026-10-23.json) and [Q4_K evidence](docs/granite5-q4k-2026-10-24.json).

## Qwen3-TTS voice packs and model safety

- Model-matched 0.6B and 1.7B designed default voice packs replace the incompatible shared default. The selected “clear” source passed pinned reference and speech checks; a raw candidate that failed full transcription was rejected. CPU and actual Tesla T4 campaigns each passed 13 checks with WER 0 for their tested requests. A mismatched Base voice pack is rejected before prefill instead of reaching the crash path.
- The [pinned voice-asset revision](https://huggingface.co/cstr/qwen3-tts-voices-GGUF/tree/9ab96c5ba3022bd2ed46306ded0489a24096f8b5) retains source declarations, an Apache voice-asset notice, a separate MIT recipe notice and provenance. The voice identity and upstream training rights have not been independently verified; no likeness, listening or broad hardware claim follows. See [method](docs/qwen3-designed-voices.md) and [receipt](docs/qwen3-designed-voices-2026-10-23.json).
- CLI and C ABI routing reject unsupported or foreign Parakeet GGUF schemas before filename, alias or legacy Whisper fallback. Custom-license acceptance text now requires acknowledgment of the user's own obligations; it does not grant additional model rights.

## OmniVoice audio and Apple Metal

- Pinned upstream audio preprocessing and output cleanup are available through the runtime, with selectable legacy paths. The corrected hosted CPU campaign passed nine saved audio/ASR cases at the unchanged WER 0.2 gate (maximum 1/9); saved codec codes and upstream DSP output matched their pinned controls. This is bounded intelligibility and DSP evidence, not proof of universal click removal, speaker similarity, original Python TTS parity or GPU parity. The distinct `omnivoice-terms` gate remains required.
- Eligible F16/F32 Higgs encoding on Apple ARM macOS/Metal defaults to the validated precise hybrid/graph/fused path with CPU recovery. The M1 controls retained exact codes on their tested references and Session paths. This does not promote the path on Linux, CUDA, iOS or quantized companions. Optional resident codec and longer-input experiments have separate controls and limits. See [CPU audit](docs/omnivoice-audio-processing.md) and [Metal scope](docs/omnivoice-metal-defaults-2026-10-23.md).

## Bindings, packaging and validation

- Dart Session results decode incomplete UTF-8 safely. Rust `synthesize_raw` reports the actual last native synthesis error on failure. The contributor fixes retain their original authorship and passed hosted native binding and ABI controls.
- C# managed speaker-roster tests enforce validation without native-test skips. Wheel publication checks validate platform ABI floors and isolate CPU distributions from non-wheel receipts; Python 3.10 remains supported by those checks. The original `v0.8.44` CPU wheels and sdist were recovered and anonymously byte-verified. That recovery is **not** a `v0.8.45` publication.
- Source `008b421e7` passed all 13 CI jobs and its secret scan. Separate source `85a620366` lint and deep-analysis runs and hosted binding/ABI jobs passed. Those checks predate the current CSM investigation and do not clear the later recurring nightly failure. A fresh final-source check is required after its resolution.

Known open dependencies remain the reporter-specific #532 reproducer, #483 GTX 1660/MX150 MMQ evidence, #461 B390 Vulkan quality trace and #456 license review. None is represented here as resolved by this draft.
