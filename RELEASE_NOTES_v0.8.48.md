# v0.8.48 — Melo/OpenVoice conditioning and CLI error fixes

## Speech synthesis

- Restore original OpenVoice flow ordering, zero-speaker bias and final decoder
  activation. Fix Melo word/token/phone alignment, final decoder activation,
  stochastic duration flow inventory and spline math.
- Managed Melo V2 downloads now select the qualified F16 BERT companion under
  the filename discovered by both CLI and Session/C ABI. Both model URLs are
  pinned to the qualified revision. This companion is approximately 238 MB;
  existing explicit quantized companion choices remain supported.
- Forward the selected CLI GPU backend to OpenVoice conversion, preserve native
  output rates, handle the neutral spoken disclosure separately, and report
  missing or failed conversion instead of silently returning an unconverted
  voice. Reject invalid speaker IDs and propagate native setter failures through
  the C#, Java and Ruby bindings.
- Retain BERT's Apache-2.0 attribution and full license independently of the
  Melo/OpenVoice MIT notices.

## Recognition and CLI

- Preserve word boundaries when Parakeet emits standalone boundary markers
  (#558). Reporter-recording and device-specific acceptance remain pending.
- Make `--detect-language` perform language detection and exit before resolving
  an ASR model. Accept `--lid-model auto` as the default provider selection and
  preserve normal automatic-language transcription (#560). Full-recording
  language aggregation remains separate work; #560 stays open.
- Diagnose CPU-sensitive data2vec Q4 rounding using paired original controls.
  The regression gate allows one Q4 word error while requiring an exact F16
  transcript and the unchanged stage/magnitude checks. Missing required stages
  now fail validation.

## Validation and limits

The full rebuilt-current Melo/OpenVoice application passes the unchanged
four-case speech/speaker matrix on CPU and real T4 CUDA, including original
component checks, both 27-stage converter references, real Session/C ABI and CLI
calls, output rates, exact reseeded repeats and observed F16 BERT discovery
without overrides. CPU run 40825101583 and CUDA full-source kernel version 5
have authenticated private evidence and verified executable checkpoints. The
failed Q4 BERT controls remain failed; this is a declared F16 default change.

Hosted CLI run 40807940665 accepts the standalone LID action with pinned
Whisper/Silero providers, multiple inputs, failure exits, ASR/HF bypass and
ordinary transcription. Model-free regression controls reproduce the routing
errors before their fixes.

The qualified F32-sensitive converter remains a validation checkpoint, not a
newly published model asset. Converter-model publication is separate from this
software release. Voice preset identity remains unknown; validation does not grant voice identity
or raw-audio redistribution rights. Physical AMD/Intel/ARM GPU wheel execution,
GTX1660/MX150 MMQ behavior, Arc B390 timing and authorized MUSA runtime/licensing
checks remain unverified. No new device claims are made by this release.
