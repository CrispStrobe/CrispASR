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
revise. Final flush reruns the batch path from retained original PCM: whole-utterance
frontend, encoder and BOS decoding. Windowed frontend rounding on x86 can change
int8 output even after full encoder recomputation. Replaying the same batch path
therefore preserves final/batch text; partial updates retain bounded encoder
work. PCM history costs 64 kB per second at 16 kHz float32 and is cleared at
flush. Drafts remain provisional.
The older two-graph export uses complete utterance recognition, not native
streaming; its merged cached decoder branch is not trusted. Decoder prefix
refresh is still work per update, so measure latency on the target device.

The streaming position limit is 4096 encoder frames (about 82 seconds);
CrisperWeaver splits utterances at 60 seconds. ONNX uses CPU inference.
Silence and noisy audio can still yield hallucinations, even with a German
checkpoint. German-only does not guarantee correct recognition.

Set `MOONSHINE_ONNX_BENCH=1` to report per-graph CPU wall time for the
frontend, encoder, adapter, cross-KV and decoder. Timing is disabled by default.

## Verification

`test-moonshine-tokenizer` covers binary/JSON byte fallback for German umlauts
and rejects malformed vocabularies. `tools/test_moonshine_session.py LIB MODEL
WAV` replays a 16 kHz mono PCM16 WAV through the C API, checking two microphone
packet sizes, final-text parity, repeated flush and feed-after-flush handling.
CrisperWeaver's `fixed_german_live_test.dart` accepts `CRISPASR_TEST_LIVE_WAV`
and optional `CRISPASR_TEST_LIVE_EXPECT` for private real-audio replay. It
checks the actual worker, VAD, native streaming mode, final units and errors.
No private recordings are committed to either repository.

For the five-graph deployments, Python ONNX Runtime can dump graph-boundary
references for the native diff harness:

```sh
python tools/dump_reference.py --backend moonshine-onnx \
  --model-dir /path/to/encoder_int8.onnx --audio german.wav --output ref.gguf
build/bin/crispasr-diff moonshine-onnx /path/to/encoder_int8.onnx ref.gguf german.wav
```

Install `onnxruntime`, `tokenizers`, `numpy` and `gguf` in the reference Python
environment. The diff requires the optional ONNX SDK build. It captures live
frontend state, encoder, adapter, cross-KV and first decoder outputs from the
actual batch path. Input PCM and tensor shapes must match exactly; each stage
must pass cosine and magnitude checks, and final text must match Python's
standard tokenizer. A missing stage fails. This verifies native wrapper
execution against an independent Python driver using the same deployed graphs;
it does not establish exporter parity against the original PyTorch checkpoint.
The older two-graph export and GGUF checkpoint use separate acceptance checks.

The `Moonshine ONNX and native stream acceptance` workflow runs these checks
on relevant pull requests and main changes. It downloads each isolated German
bundle from an empty cache, checks real CLI/C ABI speech output, compares all
four five-graph variants against the Python driver, and replays Nemotron/Qwen
streams with different PCM packet sizes. This optional SDK configuration is
therefore tested even though ordinary builds do not enable ONNX Runtime.
