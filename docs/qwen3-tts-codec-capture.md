# Qwen3-TTS reliable codec capture

This feature is opt-in and is disabled unless CRISPASR_QWEN3_TTS_CAPTURE_DIR is set to an absolute local directory.

## Enable

Set the environment variable before starting the CrispASR process:

- Windows PowerShell: $env:CRISPASR_QWEN3_TTS_CAPTURE_DIR = 'D:\Qwen3-Capture'
- Linux/macOS: export CRISPASR_QWEN3_TTS_CAPTURE_DIR=/var/tmp/qwen3-capture

The directory must be absolute. Capture errors are logged but never fail or alter synthesis. Do not point this at a shared or production output directory. The capture directory contains voice-derived codec tokens and text; restrict its filesystem permissions and delete it when no longer needed.

Optional provenance metadata and precomputed hashes can be supplied by the operator:

- CRISPASR_QWEN3_TTS_CAPTURE_MODEL_NAME
- CRISPASR_QWEN3_TTS_CAPTURE_MODEL_REVISION
- CRISPASR_QWEN3_TTS_CAPTURE_MODEL_SHA256
- CRISPASR_QWEN3_TTS_CAPTURE_TOKENIZER_SHA256
- CRISPASR_QWEN3_TTS_CAPTURE_CODEC_SHA256
- CRISPASR_QWEN3_TTS_CAPTURE_REFERENCE_SHA256
- CRISPASR_QWEN3_TTS_CAPTURE_RUNTIME_REVISION

The runtime does not hash model, codec, tokenizer or reference-source files on the synthesis path. If precomputed hashes are absent, the manifest records paths and file sizes, with SHA256 left null.

## HTTP source-text pairing

When capture is enabled on a Qwen3-TTS backend, including Base, CustomVoice, and VoiceDesign, /v1/audio/speech accepts an optional capture_source_text string. This field is metadata only: it is not passed to the model and does not affect synthesis. It lets a caller record the original/raw request text alongside the exact text that was actually sent to Qwen as model_input_text. For multi-sentence requests, request_source_text is request-level metadata and may cover more than one captured sentence segment.

Do not put credentials, secrets, or unrelated personal data in either speech input or capture_source_text. Authorization headers and other request headers are never copied into capture manifests.

The spoken AI-disclaimer prefix is deliberately excluded from per-sample training captures. The HTTP response hash still covers the final response, including any disclaimer, watermark and C2PA container changes.

The existing X-Crispasr-Request-Id response header is the correlation key. Each Qwen synthesis segment stores that request ID. The HTTP manifest for the same ID stores the final response-body SHA256 and, for streaming, each successfully written application chunk's size and SHA256.

## Files

The capture root contains:

- samples/<generation-id>/manifest.json
- samples/<generation-id>/generated_codes.i32le
- samples/<generation-id>/reference_codes.i32le when reference codes are active
- samples/<generation-id>/decoder_input_codes.i32le for whole-clip decode, or decoder_windows/*.i32le for streaming decode
- requests/<request-id>/manifest.json for the final HTTP response body

Code arrays are signed 32-bit little-endian integers, row-major [frames, codebook]. The manifest records the shape and SHA256. The capture stores the complete generated code sequence and the exact code array passed to each codec-decoder invocation. Streaming window records include the frame offset, pre-roll, and discarded sample count so each decoder input can be reconstructed and audited.

The selected voice identifier and any instruction text are recorded. Effective seed, configured and effective temperature, greedy sampling, reference-decode policy, and language ID are included so a captured result has enough inference metadata to be interpreted. The reference source itself (audio WAV or voice-pack file) and the speaker-embedding vector are not copied; their paths and available fingerprints are recorded. The reference-code sequence is stored when present; the speaker embedding is represented by a SHA256 fingerprint. The decoded PCM is not copied; its float32-LE SHA256, sample rate, channel count, sample count, and duration are recorded. The HTTP response manifest records the hash of the final response body after server-side processing, encoding, and optional C2PA signing. For streaming, the body hash covers application payload bytes successfully passed to sink.write; it excludes HTTP chunk framing.

## Completeness and overwrite rules

- Every generation gets a newly created sample directory. Existing files are never overwritten.
- Every HTTP request gets a newly created request directory keyed by the existing request-correlation ID. A duplicate ID is rejected rather than overwritten.
- Files are written to a temporary sibling and atomically linked into their final path without replacing an existing file.
- manifest.json is written last. capture_complete is true only when generation reached EOS, generated codes, at least one decoder input, every decoder result, and decoded PCM metadata were all captured successfully.
- Failed/abandoned captures produce an incomplete manifest or no manifest; they must not be used as training data.
- A client disconnect produces an HTTP manifest with complete: false and the hash of the payload successfully written before disconnect.
- Capture is separate from CRISPASR_QWEN3_TTS_DUMP_DIR; the diagnostic dump remains unchanged.

## No inference changes

Capture does not change prompts, model weights, sampling, codec inputs, decoder outputs, audio resampling, watermarks, or response content. When the environment variable is unset, no capture directories or files are created. No model is loaded by the unit tests; tests use synthetic integer codes and PCM arrays only.
