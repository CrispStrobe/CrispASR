# German Moonshine deployment

`models/german-moonshine-sources.json` records the 14 requested upstream
repositories, pinned revisions, inherited licenses, and release eligibility.
The permissive repositories represent three distinct checkpoints: official
Streaming Tiny and Small, and dattazigzag's original Tiny. Source checkpoints
and their exports share weights; separate downloads are not separate models.
Fidoriel weights and their derivatives remain non-commercial, including ONNX
repositories whose cards advertise Apache licensing. They are excluded from
normal release offerings. Existing non-commercial GGUF development gating is
unchanged.

## Build

Install the pinned CPU ONNX Runtime SDK with `scripts/fetch_onnxruntime.sh`,
then configure CMake with `-DCRISPASR_ONNXRUNTIME_ROOT=/path/to/sdk`. Without
that option the existing GGUF backends still build. CrisperWeaver's macOS
build script supplies and bundles the SDK, including license notices.
Windows users can supply an official SDK root manually. The current SDK
fetch helper supports Apple silicon, Linux x64 and Linux arm64.

## Models

Registry aliases `moonshine-streaming-{tiny,small}-de-onnx` use the complete
int8 five-graph bundle; append `-f32` for full precision. `moonshine-onnx`
defaults to German Streaming Small int8. All companion paths are isolated
by model. `moonshine-tiny-de-phreak87-onnx` uses the MIT dattazigzag checkpoint
in the older two-graph format. `moonshine-tiny-de-dattazigzag` is its GGUF
alternative. CrisperWeaver offers the same six deployment choices.

Use `--backend moonshine-onnx -m /path/to/encoder_int8.onnx -l de` for batch
recognition. For incremental PCM, pipe mono 16 kHz s16le audio into
`--stream --stream-json --stream-step 640`. Passing `encoder.onnx` selects
full precision. These German-only checkpoints do not need a language prompt.
Specifying German on a multilingual Parakeet checkpoint does not provide
this same constraint.

The five-graph streaming path carries frontend state, processes canonical
40 ms frontend packets, and retains stable encoder output using the encoder's
bounded attention dependency window. Partial decoding refreshes cross-KV,
prefills the older draft prefix, and permits its trailing eight tokens to
revise. Final flush decodes again from BOS. Drafts therefore are provisional.
The older two-graph export uses complete utterance recognition, not native
streaming; its merged cached decoder branch is not trusted. Decoder prefix
refresh is still work per update, so measure latency on the target device.

The streaming position limit is 4096 encoder frames (about 82 seconds);
CrisperWeaver splits utterances at 60 seconds. ONNX uses CPU inference.
Silence and noisy audio can still yield hallucinations, even with a German
checkpoint. German-only does not guarantee correct recognition.

## Verification

`test-moonshine-tokenizer` covers binary/JSON byte fallback for German umlauts
and rejects malformed vocabularies. `tools/test_moonshine_session.py LIB MODEL
WAV` replays a 16 kHz mono PCM16 WAV through the C API, checking two microphone
packet sizes, final-text parity, repeated flush and feed-after-flush handling.
CrisperWeaver's `fixed_german_live_test.dart` accepts `CRISPASR_TEST_LIVE_WAV`
and optional `CRISPASR_TEST_LIVE_EXPECT` for private real-audio replay. It
checks the actual worker, VAD, native streaming mode, final units and errors.
No private recordings are committed to either repository.
