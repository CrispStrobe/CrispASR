# Go bindings for Whisper

This package provides Go bindings for crispasr. They have been tested on:

  * Darwin (OS X) 12.6 on x64_64
  * Debian Linux on arm64
  * Fedora Linux on x86_64

The "low level" bindings are in the `bindings/go` directory and there is a more
Go-style package in the `bindings/go/pkg/whisper` directory. The most simple usage
is as follows:

```go
import (
	"github.com/CrispStrobe/CrispASR/bindings/go/pkg/whisper"
)

func main() {
	var modelpath string // Path to the model
	var samples []float32 // Samples to process

	// Load the model
	model, err := whisper.New(modelpath)
	if err != nil {
		panic(err)
	}
	defer model.Close()

	// Process samples
	context, err := model.NewContext()
	if err != nil {
		panic(err)
	}
	if err := context.Process(samples, nil, nil, nil); err != nil {
		return err
	}

	// Print out the results
	for {
		segment, err := context.NextSegment()
		if err != nil {
			break
		}
		fmt.Printf("[%6s->%6s] %s\n", segment.Start, segment.End, segment.Text)
	}
}
```

## Chat / LLM

The same package also binds the `crispasr_chat_*` C ABI (text in, text out) as
`whisper.ChatSession`: one-shot and streaming generation, prompt-token
counting, and an abort predicate for cancellation.

```go
sess, err := whisper.ChatOpen("gemma-3-1b-it-Q4_K_M.gguf", nil)
if err != nil {
	panic(err)
}
defer sess.Close()

params := whisper.DefaultChatGenerateParams()
params.SetMaxTokens(128)
messages := []whisper.ChatMessage{{Role: "user", Content: "Say hello."}}

// Cancellation: the predicate returns true to CONTINUE and false to abort, the
// same way round as the C callback and as EncoderBeginCallback on the ASR side.
// The resulting error wraps whisper.ErrChatAborted, so a cancel is
// distinguishable from a fault.
sess.SetAbortCallback(func() bool { return !cancelled.Load() })

err = sess.GenerateStream(messages, &params, func(chunk string) {
	fmt.Print(chunk)
})
if errors.Is(err, whisper.ErrChatAborted) {
	fmt.Println("\ncancelled")
}
```

`ChatOpenParams` and `ChatGenerateParams` carry the C structs the ABI's own
defaults functions fill and are read and written through accessors, the same
way `Params` works on the ASR side. Setting one option therefore leaves every
other one at the ABI default — `SetMaxTokens` alone does not turn the
temperature into 0 and the decode greedy — and an untouched value is the
defaults, not Go's zeroes.

Pass the whole conversation on every call: the session compares the templated
prompt against the tokens it already holds and decodes only what is new.
`sess.CountTokens(messages)` reports how long that prompt is, to compare
against `sess.NCtx()`.

`whisper.ChatAIDisclosureText()` returns the canonical "you are talking to an
AI" wording. Show it visibly at or before the first turn of anything
conversational — see `include/crispasr_chat.h` for why that is a duty and not
an option.

The chat tests need a GGUF chat model and are gated on
`CRISPASR_CHAT_TEST_MODEL`, the same env var the C++ chat suite uses; without
it they skip.

## Building & Testing

In order to build, you need to have the Go compiler installed. You can get it from [here](https://golang.org/dl/). Run the tests with:

```bash
git clone https://github.com/CrispStrobe/CrispASR.git
cd crispasr/bindings/go
make test
```

This will compile a static `libwhisper.a` in a `build` folder, download a model file, then run the tests. To build the examples:

```bash
make examples
```

To build using cuda support add `GGML_CUDA=1`:

```bash
GGML_CUDA=1 make examples
```

The examples are placed in the `build` directory. Once built, you can download all the models with the following command:

```bash
./build/go-model-download -out models
```

And you can then test a model against samples with the following command:

```bash
./build/go-whisper -model models/ggml-tiny.en.bin samples/jfk.wav
```

## Using the bindings

To use the bindings in your own software,

  1. Import `github.com/CrispStrobe/CrispASR/bindings/go/pkg/whisper` (or `github.com/CrispStrobe/CrispASR/bindings/go` into your package;
  2. Compile `libwhisper.a` (you can use `make whisper` in the `bindings/go` directory);
  3. Link your go binary against whisper by setting the environment variables `C_INCLUDE_PATH` and `LIBRARY_PATH`
     to point to the `whisper.h` file directory and `libwhisper.a` file directory respectively.

Look at the `Makefile` in the `bindings/go` directory for an example.

The API Documentation:

  * https://pkg.go.dev/github.com/CrispStrobe/CrispASR/bindings/go
  * https://pkg.go.dev/github.com/CrispStrobe/CrispASR/bindings/go/pkg/whisper

Getting help:

  * Follow the discussion for the go bindings [here](https://github.com/CrispStrobe/CrispASR/discussions/312)

## License

The license for the Go bindings is the same as the license for the rest of the crispasr project, which is the MIT License. See the `LICENSE` file for more details.

`SessionOpenWithOptions(path, backend, options)` accepts `SessionOptions`;
start from `DefaultSessionOptions()` for native GPU/flash defaults. Use
`BackendCapabilities` / `ListBackendCapabilities` for verbs. `TruecaseInit`
and `PCSInit` return owned processors with `Process` and idempotent `Close`.
`DrainStreamedSegments` returns owned metadata or nil when empty. Serialize
all global polling with inference/close of the most recently streaming session.

Word probability `P` preserves native -1 when unavailable; it is not replaced
with a fabricated 1.0. Invalid native counts/dimensions return errors.


## Legacy Whisper parameter ownership

`Params` remains copyable: `copy := params` preserves independent scalar and
prompt/VAD-path values. `SetInitialPrompt` and `SetVADModelPath` retain Go
strings. Sequential and parallel inference create temporary C strings and free
them on return, including native failure. No parameter `Close` is required.
NUL-containing strings return an error at inference; nil contexts, empty PCM
and invalid parallel processor counts are rejected before native calls.
Serialize operations on a native context and do not mutate one `Params` value
concurrently; independent value copies may be configured separately.

`Whisper_init_with_dtw(path, AHEADS_BASE_EN, 0)` enables experimental DTW
before model loading. The higher-level package exposes `NewWithDTW` with the
same arguments. `AHEADS_N_TOP_MOST` requires a positive int32 top-layer count;
model presets require zero. Native loading checks the preset against model
dimensions. Unsupported/custom presets, invalid paths and null allocations
return errors. Ordinary `Whisper_init` keeps its nil-on-error signature and
native defaults; both constructors use pointer parameters and matching frees.
Use a native library whose DTW helper disables incompatible flash attention.
`Whisper_full_get_token_dtw_t` checks indices and returns native centiseconds,
preserving `-1` when unavailable; copied `TokenData` also exposes `TDtw()`.
`Whisper_token_snapshot(segment, token)` and `Whisper_token_snapshots(segment)`
return owned IDs, raw bytes, native-centisecond timing/DTW, probability and
alternatives. `Text` is a Go string containing the raw bytes, including incomplete
UTF-8; concatenate `Raw` values for exact reconstruction. Negative unavailable
metadata is preserved, and snapshots survive model close. Readers reject invalid
indices/counts and malformed native alternative buffers; limits are one million
tokens per segment, 32 alternatives per token and 1 MiB per token/candidate text.

The higher-level package exposes `TokenSnapshots(ctx, segment)` through optional
`TokenSnapshotProvider` support. It returns the same owned snapshots, including
DTW and alternatives. Existing `Context` implementations and `Token` struct
literals are unchanged; unsupported providers and closed models return errors.

Callback registry access is synchronized across contexts. The registry lock is
released before invoking a callback, allowing registration changes during
callback delivery. Low-level raw native contexts still require caller
serialization of inference/close/re-entry; registry synchronization alone does
not enforce their ownership.

The higher-level `pkg/whisper` Model now enforces a nonblocking native borrow
shared by all its Contexts. `Close`, decode, result/token copying, language
selection and new-context creation return `ErrModelBusy` if another operation
owns the model, including calls from callbacks and foreign threads. Methods
without error returns (native token/language queries and timing controls) panic
with that sentinel on busy access and `ErrInternalAppError` after close. Handle
lifecycle through the error-returning operations; do not call native queries
from callbacks. `String` remains safe and reports a busy model. Close is
idempotent after inference ends; a rejected close leaves the model usable.

A hosted real tiny.en/JFK CPU run independently verifies callback/concurrent
rejection, cancellation recovery, parallel decode and owned outputs. See
[the audit](../../docs/historical/go-model-lifecycle-audit-2026-10-19.json) for exact source
and limits; this does not certify raw contexts, other devices or model modes.

Go-owned option setters use a separate mutex and can update the next decode's
snapshot while inference runs; `SetLanguage` requires the native borrow. Contexts
share the latest native result set: read it before another decode, or retain the
owned Segment/token snapshots. This does not provide independent decoder states
for multiple Contexts. Low-level `Whisper_lang_id` now releases its temporary
C string, and high-level `Languages` includes the final supported language ID.

`Whisper_full_parallel` uses the native pointer-parameter parallel wrapper,
including CrispASR's global progress lifecycle. The additive
`Whisper_full_parallel_with_progress` also accepts a per-call progress callback;
encoder, segment and progress closures are removed on success and native failure.
Existing `Whisper_full_parallel` callers keep their signature. Use a matching
native library exporting `crispasr_transcribe_parallel_by_ref`.

Both low-level `Params` and the higher-level `Context` expose `SetBestOf`,
`SetDetectLanguage`, `SetSuppressBlank`, `SetTdrz`, and `SetAltN`. Candidate
counts clamp to 0–32; nonpositive best-of values leave the setting unchanged.
Custom implementations/mocks of the higher-level `Context` interface need
these five new methods. `Params` is a Go options type, not a native ABI struct;
pass it through the binding methods.


The higher-level package also supports parallel processing without extending
its existing `Context` interface:

```go
err := whisper.ProcessParallel(context, samples, 2, nil, onSegment, onProgress)
```

This uses optional `ParallelProcessor`; custom contexts lacking that capability
return an error. One processor selects ordinary sequential decoding. Counts
must be positive int32 values; empty PCM, nil/closed contexts and native failures
return errors. Progress frequency and transcript boundaries depend on the native
decoder. Callback-specific single-segment mode applies only to that call, and
successful calls reset `NextSegment` iteration. The built-in Model rejects
overlapping native calls and callback close/re-entry with `ErrModelBusy`;
custom Context implementations define their own ownership policy. Operation/race
probes establish transport, callbacks and cleanup, not model quality or speed.
