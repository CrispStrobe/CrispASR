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

The selected voice identifier and any instruction text are recorded. Effective seed, configured and effective temperature, greedy sampling, reference-decode policy, and language ID are included so a captured result has enough inference metadata to be interpreted. By default the reference source (audio WAV or voice-pack file) and speaker-embedding vector are not copied; paths and available fingerprints are recorded. The reference-code sequence is stored when present; the speaker embedding is represented by a SHA256 fingerprint unless the Base-only private export below is separately enabled. The decoded PCM is not copied; its float32-LE SHA256, sample rate, channel count, sample count, and duration are recorded. The HTTP response manifest records the hash of the final response body after server-side processing, encoding, and optional C2PA signing. For streaming, the body hash covers application payload bytes successfully passed to sink.write; it excludes HTTP chunk framing.

## Private Base speaker-embedding export (separate opt-in)

The regular Capture records the SHA256 of the **actual** speaker-embedding pointer
received from the already-running Qwen3-TTS context. The C++ getter need not
be exported from a DLL or called from Python. To preserve the **exact**
2048-element float32 vector for the experimental Base-ICL Trainer, set
`CRISPASR_QWEN3_TTS_CAPTURE_SPEAKER_EMBEDDING=1` **before starting the isolated
Capture process**, in addition to the required capture directory. Any other
value, or an unset variable, disables vector export. The 2048-value contract
targets the 1.7B Base model. The 0.6B Base model has a 1024-value speaker
embedding: requesting export for that model marks the capture incomplete and
writes no embedding file. With export disabled, its existing hash-only capture
is unchanged. CustomVoice and VoiceDesign behavior is unchanged.

On an opted-in 1.7B Base request with a valid embedding, Capture writes
`samples/<generation-id>/speaker_embedding.f32le` (8192 bytes, 2048 little-endian
float32 values). It serializes the already-owned runtime pointer **once**;
there is no second speaker inference, and nothing changes the synthesis graph
or exported DLL symbols. The `reference` manifest gains
`speaker_embedding_file`, `speaker_embedding_shape` and
`speaker_embedding_size_bytes`; the pre-existing
`speaker_embedding_sha256` must match the exact file bytes. When export is
disabled, the file and shape are null and size is zero.

Writing uses the existing atomic, no-overwrite hard-link writer. When export
is explicitly requested but the pointer is missing, the shape is not [2048],
a value is non-finite, the SHA differs, or the write fails, the sample is
marked `capture_complete=false`. **Synthesis still proceeds**, but that
capture must not enter a training corpus. No source/reference WAV, private
embeddings or generated training examples belong in Git or hosted CI artifacts.

The speaker embedding can identify a voice. Export only with the relevant
rights and consent, into a locally protected private directory with restrictive
ACLs, and securely remove it when no longer required. This opt-in **does not**
constitute user consent or authorize any GPU experiment, model reload, or
training. Previously prepared DLLs still need an independently reviewed rebuild
for this feature; setting the variable on an old binary will not enable it.

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

The manually dispatched `Qwen capture variant live audit` workflow exercises
capture-enabled CustomVoice and VoiceDesign HTTP requests with pinned models.
It checks buffered and streaming manifests and decoder arrays, transcribes each
variant's audio, and exercises a streaming failure and client disconnect. The
model-free `tests/test_qwen3_capture_live_audit.py` checks that its manifest
validator rejects corrupted or misattributed captures.

On 2026-11-06, [hosted audit 42236556427](https://github.com/CrispStrobe/CrispASR/actions/runs/42236556427)
passed both variants with buffered and streaming PCM, transcription, an
injected stream failure and an actual socket disconnect. The disconnect was
recorded as incomplete with `response_capture_abandoned`; the HTTP library
may release the provider before a subsequent write detects the disconnect.
This run uses CPU Q8 models and does not establish GPU or all-model coverage.
