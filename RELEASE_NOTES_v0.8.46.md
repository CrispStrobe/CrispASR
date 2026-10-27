# v0.8.46 — Nemotron qualified F16 default and packaging provenance

This correctness release promotes fixes that were qualified after `v0.8.45`.
No model weights, voice packs or quantization recipes were changed to produce it.

## Nemotron streaming ASR default

- The managed `nemotron` download now selects the original-qualified F16 model
  (`cstr/nemotron-3.5-asr-streaming-GGUF` revision
  `bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2`, 1,286,368,480 bytes, SHA-256
  `14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09`).
  Managed quant overrides accept only F16; a missing Q4 filename never silently
  resolves to F16. Explicit existing Q4 files remain usable for diagnostics.
- The former default plain Q4 fails the fixed original-model controls on CPU and
  on a real Tesla T4 (11/26 original-transcript matches, 8/22 prompt matches,
  pre-encode stage gate failure). F16 passes 26/26 and 22/22 on both devices,
  with all stage/magnitude gates, the CLI control, opaque-filename C ABI
  detection and fresh/reused Session controls. This does not establish streaming
  parity, performance, or accuracy across the full language roster.
- Managed downloads retain the full OpenMDW-1.1 agreement and the pinned origin
  notice. See the [policy and evidence](https://github.com/CrispStrobe/CrispASR/blob/v0.8.46/docs/nemotron-default-policy-2026-10-26.md).

## Diagnostics and packaging

- VoxCPM2 `CRISPASR_VOXCPM2_BENCH` output now reports VAE host permutation
  separately from upload (#461 diagnostics). Graph, math and defaults are unchanged;
  no new B390 speed claim is made.
- Python wheel staging records original and staged native-library identities and
  carries the native bundle's license notices into platform wheels. No license is
  replaced and no new redistribution permission is inferred.
- The MUSA Docker recipe selects the SDK 5.2 / architecture 31 compile profile.
  No MUSA image is published by this release; MUSA runtime and hardware
  inference remain unqualified.

## Unchanged and still open

Qwen3-TTS voice packs, Granite5 Q4_K, CSM rotary scaling and OmniVoice
processing are as qualified in `v0.8.45`. #532, #483, #461 and #456 await their
external reproducer, hardware or license input; #557 Linux GPU wheels is planned.
