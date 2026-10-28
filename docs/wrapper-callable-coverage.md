# Callable wrapper audit

This ledger records reviewed public entry points and their operation probes.
It is incremental: unmapped functions remain pending in PLAN.md. A native
symbol declaration, regex match or generated stub is not a callable API.
The census scope is the 297 unique `crispasr_*` functions in the two shared
headers, plus the separate 14 chat exports. It does not inventory every
upstream `whisper_*` declaration in `include/crispasr.h`.

## TitaNet

Native functions: `crispasr_titanet_init`, `crispasr_titanet_embed`,
`crispasr_titanet_cosine_sim`, `crispasr_titanet_free`.
Embedding succeeds with exactly **192 floats written**, not zero or one.
Similarity is the native dot product for normalized model embeddings.

| Wrapper | Public entry points | Operation proof |
| --- | --- | --- |
| Python | `TitaNet(...)`, `.embed`, `.cosine_sim`, `.close` | `test_python_legacy_pointer.py` |
| C# | `TitaNet.Open`, `.Embed`, `.CosineSim`, `.Dispose` | `CSharpTitaNetContract` |
| Java | `CrispasrAuxiliary.TitaNet`, `.embed`, `.cosineSimilarity`, `.close` | `JavaAuxWrapperContract` |
| Go | `TitaNetInit`, `TitaNetContext.Embed`, `TitaNetCosineSim`, `.Close` | `TestProbeTitaNetFloatCountAndClosedContext` |
| Rust | `TitaNet::new`, `.embed`, `::cosine_sim`, `Drop` | `crispasr/tests/probe.rs` |
| Dart | `CrispasrTitaNet`, `.embed`, `.cosineSim`, `.close` | `ffi_behavior_test.dart` |
| Ruby | `Session.titanet_init/embed/cosine_sim/free` | `ruby-contract-program.rb` |
| WASM | `TitaNet`, `.embed`, `titanetCosineSimilarity`, `.close`/`.delete` | `WasmWrapperContract` |

The shared compiled fixture checks success, zero/short/oversized native counts
and resource lifetime. That proves transport, not real speaker accuracy.

## Closed-roster speaker database

Native functions: `crispasr_speaker_db_open`, `_count`, `_match`, `_enroll2`,
`_free` (all with the `crispasr_speaker_db` prefix). The legacy `_load` and
`_enroll` exports are retired refusing stubs; they are not implementation
alternatives. Open and enrollment preserve the native roster/attestation
contract. A negative match score means **no match**, including at a negative
threshold. Enrollment through `_enroll2` writes a profile to disk; reopen an
existing database to refresh its retained roster.

| Wrapper | Public entry points | Operation proof |
| --- | --- | --- |
| Python | `SpeakerDB(...)`, `.count`, `.match`, `.enroll`, `.close` | `test_python_legacy_pointer.py` |
| C# | `SpeakerDb.Open`, `.Count`, `.Match`, `SpeakerDb.Enroll`, `.Dispose` | `CSharpSpeakerDbContract` |
| Java | `CrispasrAuxiliary.SpeakerDb`, `.count`, `.match`, `SpeakerDb.enroll`, `.close` | `JavaAuxWrapperContract` |
| Go | `SpeakerDBOpen`, `.Count`, `.Match`, `.Enroll`, `.Close` | `TestProbeSpeakerDatabaseRosterScoresEnrollmentAndLifetime` |
| Rust | `SpeakerDB::open`, `.count`, `.match_embedding`, `.enroll`, `Drop` | `probe_speaker_database_roster_scores_enrollment_and_lifetime` |
| Dart | `CrispasrSpeakerDB`, `.count`, `.match`, `.enroll`, `.close` | `ffi_behavior_test.dart` |
| Ruby | `Session.speaker_db_open/count/match/enroll/free` | `ruby-contract-program.rb` |
| WASM | `SpeakerDB`, `.count`, `.match`, `.enroll`, `.close`/`.delete` | `WasmWrapperContract` |

Dart optionally uses the newer handle-enrollment export when present; its
existing return-code handling remains compatible. Rust's old `load` method
now explicitly directs callers to `open`, rather than invoking a retired
native function. The operation fixture checks actual UTF-8 result copies,
missing consent/roster, native failure codes, no-match semantics and lifetime.

## Generic speaker primitives: reviewed callable adapters

Native `crispasr_speaker_embedder_*_abi` has make/free/dim/name/embed operations;
embedding succeeds with **1**. `crispasr_speaker_cluster_abi` returns the number
of clusters, with negative failure. Pyannote cache compute/free/apply uses
zero-success apply and native centisecond segment/offset fields.

| Wrapper | Public operations | Operation proof |
| --- | --- | --- |
| Ruby | `CrispASR::SpeakerEmbedder.new`, `.dimension`, `.name`, `.embed`, `.close`; `CrispASR.speaker_cluster(rows, threshold, maximum)`; `CrispASR::PyannoteCache.new`, `.apply(segments, offset_cs)`, `.close` | `RubyWrapperContract`, native shared fixture; TypedData GC and recursive coercion/GC checks |
| WASM | `SpeakerEmbedder`, `.dimension`, `.name`, `.embed`, `.close`/`.delete`; `speakerCluster(rows, threshold, maximum)`; `PyannoteCache`, `.apply(segments, offsetCs)`, `.close`/`.delete` | `WasmWrapperContract`, same shared fixture compiled into WASM |
| Go | `NewSpeakerEmbedder`, `.Dim`, `.Name`, `.Embed`, `.Close`; `AgglomerativeCluster`; `NewPyannoteCache`, `.Apply`, `.Close` | `TestProbeGenericSpeakerOwnershipAndReturnCodes`, `TestProbeGenericClusterBoundsAndNativeOutput`, `TestProbeGenericPyannoteOwnedResource`, `TestProbeGenericSpeakerRejectsNulAndOversizedOptions`, shared actual-header/POD oracle |
| Rust | `SpeakerEmbedder::new`, `.dim`, `.name`, `.embed`, `Drop`; `agglomerative_cluster`; `PyannoteCache::compute`, `.apply`, `Drop` | `probe_generic_speaker_returns_clusters_and_cache`, same shared compiled oracle |
| Python | `SpeakerEmbedder`, `.dim`, `.name`, `.embed`, `.close`; `agglomerative_cluster`; `PyannoteCache`, `.apply`, `.close` | `test_python_speaker_primitives.py`: seven actual-header fixture cases, coercion/closed guards and exact success/count/label contracts |
| Dart | `CrispasrSpeakerEmbedder`, `.dim`, `.name`, `.embed`, `.close`; `crispasrAgglomerativeCluster`; `CrispasrPyannoteCache`, `.apply`, `.close` | Strict analyzer and40 FFI tests, including three shared-oracle generic cases; allocator arenas release temporary buffers on exceptions |
| C# | `SpeakerEmbedder`, `.Dimension`, `.Name`, `.Embed`, `.Dispose`, `.Cluster`; `PyannoteCache`, `.Apply`, `.Dispose` | Actual .NET auxiliary probe with malformed counts/labels, exact success, unassigned output, NUL paths and failure cleanup |
| Java | `CrispasrAuxiliary.SpeakerEmbedder`, `.dim`, `.name`, `.embed`, `.close`; `.cluster`; `.PyannoteCache`, `.apply`, `.close` | Real JNA auxiliary probe, same native faults as C# plus existing ownership/closed guards |


Go's oversized cluster dimension previously overflowed the size product and
panicked on an empty buffer. Both Go and Rust accepted an embedding return code
of2 instead of requiring the native1-success contract. The new probes observed
these failures before repair. Counts/conversions are checked before allocation
or native entry, and returned cluster counts/labels are validated. Native
zero-count/all-unassigned output remains distinct from malformed labels. Go
cache offsets are centiseconds; Rust cache offsets/segments are seconds. Each
method returns owned data and closes/drops its model/cache once. Go's generic
handles require caller serialization across operations and Close; the separate
high-level Whisper model guard does not cover them. These are operation probes,
not real speaker-quality or universal concurrency acceptance.

Clustering returns labels plus `n_clusters` (Ruby) / `nClusters` (WASM).
Ruby/WASM cache apply returns owned `{t0,t1,speaker}` records and leaves input
records unchanged. Their explicit close is idempotent; subsequent operations
raise. Go/Rust apply updates caller segment speakers only after native success.
Python/Dart accept only exact1-success and validate count/label results before
returning owned data. C#/Java already required1; their malformed count/label
and NUL-path gaps are repaired. Dart bounds division precedes allocation, and
its cache packing rejects nonfinite/out-of-range timings with arena cleanup.
These generic handles remain caller-serialized. Actual speaker-model quality,
thread contention and further platforms remain separate acceptance tasks.

## Go/Rust options, text and polling follow-up

| Native family | Go public operation | Rust public operation | Operation proof |
| --- | --- | --- | --- |
| Session open with params | `SessionOpenWithOptions`, `DefaultSessionOptions` | `Session::open_with_options`, `SessionOptions::default` | Literal runtime 48-byte struct, custom/default values and zero reserved fields |
| Backend capabilities | `BackendCapabilities`, `ListBackendCapabilities` | Existing `Session::backend_caps` / `list_backends_with_caps` | New Go growth/unknown-backend probe; existing Rust methods still need their ledger proof row |
| Truecase / PCS init/process/free text/free | `TruecaseInit`, `PCSInit`, promoted `.Process`/`.Close` | `TruecaseModel::open`, `PCSModel::open`, `.process`, Drop | Prefix-specific allocation/free oracle, empty text and failures |
| Candidate scores | Existing `Session.ScoreTexts` | `Session::score_texts` | Actual candidate/option transport, owned scores/counts and positive native error |
| Owned streamed segment result | `DrainStreamedSegments` | `drain_streamed_segments` | Shared metadata reader, copied raw bytes/alternatives/speaker and result free |
| Progress/count/reset polling | Existing `GetProgress`, `ResetProgress`, `GetStreamed*Count`, `ResetStreamed*` | `get_progress`, `reset_progress`, `get_streamed_*_count`, `reset_streamed_*` | Serialized global polling fixture |
| Borrowed token polling | `DrainStreamedTokens` with documented caller serialization | **unsafe** `drain_streamed_token_bytes` / `drain_streamed_tokens` | Protected-page raw fragments/empty tokens and malformed counts |

Global polling selects the most recently streaming session. An owned segment
result can be copied after the call; the borrowed token buffer cannot be used
concurrently with another drain/inference/close. Rust makes that requirement
an explicit unsafe contract. These checks do not prove concurrent model behavior.

## Ruby session lifecycle and v2 loader

`Session.open`, `.open_explicit` and `.open_with_options` return opaque integer
tickets; `Session.close` closes child `Session::Stream` handles first. Stream
open/feed/text/flush/live-decode/close paths validate ownership and reject use
after parent close. The new loader accepts all five v2 configuration controls,
with runtime ABI version/reserved fields generated internally.

The 155 guarded Session/Stream/Chat operations preserve arities and aliases and
hold per-context borrows across coercion/callback/thread switches. This count
is a guard synchronization fact, not a native export coverage percentage.
Direct context, DB and text handles also use opaque tickets. Tests force native
address reuse and still reject old closed tickets; native callbacks continue to
use actual native pointers internally.

`RubyWrapperContract` validates this through concrete stream and loader probes,
including native close ordering and callback userdata. Remaining mic and legacy
Whisper ownership paths remain separate tasks in PLAN.md.

## Additional reviewed paths

- C# `Session.TranscribeVad(pcm, rate, model, language, options)` and Java
  `CrispasrSession.transcribeVad(pcm, rate, model, language, options)` expose
  all six VAD ABI fields. The three-argument calls retain their original
  default/native route. Probes compile the literal native option layout and
  verify each field, language, input checks and callback-error cleanup.
- Java `translateText`, `enhanceAudioRnnoise` and
  `TextProcessor(model, TextProcessorKind.PUNCTUATION)` are public operations.
  Allocated UTF-8 output remains a pointer until copied and freed. Empty
  output and native failure are distinct; RNNoise unavailability raises.
- Go `DrainStreamedTokens` reads exactly the native count of NUL-terminated
  strings. It preserves raw fragments and empty tokens. A protected-page
  fixture reproduced the previous guessed-byte-length read crashing before
  the repair. Polling still selects the most recently streaming native
  session; callers must serialize polling with its inference and close.
- Dart `diarizeSegments` uses target-width FFI structs and an arena freed on
  every exit. Its probe verifies all option fields, centisecond conversion,
  turn-buffer retry, invalid native counts and stereo input bounds. Java's
  corresponding pointer offsets now use `Native.POINTER_SIZE`; a 32-bit JVM
  execution remains a platform validation task.

Real-model output, browser scheduling, full WASM size and native build tests
remain separate acceptance gates. See PLAN.md for the remaining audit rows.


## WASM load options and registry/cache

`asrOpenWithOptions`/`ttsOpenWithOptions` call
`crispasr_session_open_with_params` using all five v2 controls and zero reserved
padding. The probe compiles the literal runtime POD. Invalid options preserve
the live singleton; a native load failure after valid replacement leaves it closed.
Single-thread modules enforce one thread. Threaded/browser scheduling remains
an independent pending gate.

`registryLookup`, `registryLookupByFilename`, `registryBackends`,
`registryDefaultBundle`, `cacheDirectory` and `cacheEnsureFile` cover the seven
registry/cache ABI exports. Bundle info and artifact enumeration are combined
into one owned result. The compiled Node contract checks string transport,
unknown versus empty bundles, all three artifact kinds, license metadata,
cache arguments, native errors, buffer growth, retry ceilings and unterminated
native output. This is a model-free transport test, not browser download proof.


## WASM generic diarization and language-aware VAD

`diarizeSegments` and `diarizeSegmentsWithTurns` cover the two generic diarization
ABI exports. The compiled fixture uses literal runtime segment/turn/options
structs on wasm32. It checks field transport, stereo channels, absolute times,
owned output, empty turns, negative/oversized counts, native failures and bounded
buffer growth. Model methods retain their distinct path fields.

`asrTranscribeVad` covers `crispasr_session_transcribe_vad_lang`, a superset of
the no-language export. The compiled fixture verifies all six literal VAD
option fields, language, native-default null options, invalid-input rejection,
result ownership and metadata. These contracts do not establish model accuracy
or threaded browser scheduling.


WASM's `sessionTranscribeWithLogits` uses a shared native-to-owned transcription
helper. The operation probe checks negative segment/word/alternative/token
counts, negative/overflowing logit dimensions, missing buffers, copied grids,
native-result freeing and logit-flag reset on C++ and JS encoding errors.


## Ruby microphone ownership (separate microphone header)

`Whisper::CrispASR::Mic.open/start/stop/close/default_device_name` wrap the five
microphone exports declared in `src/crispasr_mic.h`, outside the shared 311-export
census. Handles are opaque tickets. The compiled native-thread fixture checks
PCM copies, callback-close, repeated/concurrent close, foreign/retired tickets,
start/stop/open failures, queue overflow, GC, callback exceptions, pump kill and
an injected copy exception after dequeue. A test-only atomic allocation counter
checks queued/pending chunks return to zero. This does not prove audio-device
access on a target platform.


## Go legacy Whisper parameters

Low-level `Params` and higher-level `Context` expose `SetBestOf`,
`SetDetectLanguage`, `SetSuppressBlank`, `SetTdrz`, and `SetAltN`; the last
matches the native 0–32 clamp. These are public scalar-field equivalents of
the corresponding five `crispasr_params_set_*` exports. `SetInitialPrompt` and
`SetVADModelPath` retain Go values; sequential/parallel inference allocate and
free temporary native strings instead of leaking allocations in setters.

`TestProbeLegacyParamsCopiesStringsControlsAndFailures` exercises actual
public-header struct transport, independent copies, repeated replacement,
GC inside callbacks, parallel/sequential calls and failures. Invalid PCM,
processor counts and NUL strings never reach the native oracle.
`Whisper_full_parallel` and the additive `Whisper_full_parallel_with_progress`
route through `crispasr_transcribe_parallel_by_ref`; the latter delivers per-call
progress. The oracle counts the pointer export independently and exercises
encoder/segment/progress callbacks, GC and callback retirement after success
and native failure. This proves routing, not actual model progress frequency.
`TestProbeLegacyEmptyPCMReturnsError` reproduced the old index-out-of-range
panic and now passes. The contract also compiles the higher-level package;
that compilation does not run its model-dependent tests.

`Whisper_init_with_dtw` and high-level `NewWithDTW` now expose built-in DTW
presets/top-layer selection. Ordinary `Whisper_init` also uses native pointer
parameters with matching cleanup. `Whisper_full_get_token_dtw_t` validates
indices/counts; `TokenData.TDtw()` exposes the copied field. Both preserve native
centiseconds and -1. `TestProbeLegacyDTWConstruction` checks all presets,
constructor cleanup/errors, incompatible flash attention, invalid indexed access
and DTW sentinels; `TestProbeModelDTWConstructor` exercises the higher-level model
constructor and close. Those are operation probes, not model-dependent tests.
`TestProbeLegacyCallbacksAcrossContexts` exercises eight contexts through native
callbacks, including registration changes within delivery. The strict hosted
Go/Rust contract enables Go's race detector. Custom heads and remaining legacy context/resource controls are still pending.
The higher-level `ProcessParallel(ctx, pcm, processors, encoder, segment, progress)`
helper and optional `ParallelProcessor` capability now expose the native parallel
pointer/progress route without changing the existing Context interface. One
processor shares the ordinary Process path; per-call single-segment policy does
not mutate future options. `parallel_probe_test.go` exercises dispatch counters,
callbacks/token metadata, iteration reset, failure/abort/recovery, invalid counts,
empty PCM and nil/closed/unsupported contexts. The strict Go/Rust race-enabled
contract runs these operation probes in addition to package compilation.
The built-in high-level Model now has the lifecycle guard/probes described
below; raw low-level ownership and actual Go-model/device acceptance remain open.

The owned-token and compatible higher-level DTW paths are described below.


WASM token polling copies all NUL-separated strings into owned C++ storage
before constructing JS arrays. A regression replaces the JS typed-array
constructor with one that poisons the borrowed native buffer: later fragments
were corrupted before the fix and now survive. Negative counts and positive/null
pairs throw; zero-count drains return empty arrays. This covers servicer-side
JS re-entry; concurrent compute/session-close ownership remains pending.


## WASM threaded dispatch and async v2 loading

The shared dispatcher now serves async ASR, model loading and TTS, keeping emval
ownership on the caller's servicer. A compiled PROXY_TO_PTHREAD fixture verifies
work/encoder errors, callback exceptions, rejected/throwing queue submission,
readiness, overlap rejection, callback chaining and atomic progress polling.
Payload and emval counts return to their baselines. A JS finally retires delivery
even when JS skips C++ destructors; no return proxy is needed.

`asrOpenWithOptionsAsync` and `ttsOpenWithOptionsAsync` pass all v2 fields and
reserved padding against the literal runtime POD, and load on compute. The
fixture exercises the public async TTS function with allocated/empty/malformed
PCM, native errors and a throwing JS typed-array constructor. Native PCM is
freed before encoding. The same assertion program runs in Node and headless
Chromium with actual pthreads and COOP/COEP. These are component contracts,
not real-model transcript/roundtrip or full-module acceptance.


## WASM retained session streams

Native `crispasr_session_stream_kind/open` and stream close/feed/flush/text/
live-decode operations have public `asrStreamKind/Open/Close/Feed/Flush/GetText/SetLiveDecode`
adapters. The legacy Whisper-context `crispasr_stream_open` is a distinct pending
ledger row. Sync stream methods serve single-thread modules; threaded modules
provide the corresponding six `*Async` operations plus `asrCloseAsync`.

`WasmStreamContract` and `WasmDispatchContract` compile these actual adapters
against the public stream declarations and a native ownership/thread oracle.
The probes cover exact options and PCM, native failure, text growth, copied
UTF-8/time/counter results, ticket retirement and parent replacement/close.
Only one child is supported per session, matching the native ABI contract;
batch ASR inference refuses while that child exists. Real-model acceptance remains pending in PLAN.md.


## WASM retained ASR callbacks

`crispasr_session_set_segment_callback`, `_set_token_callback` and
`_set_progress_callback` map to `asrSetSegmentCallback`, `asrSetTokenCallback`
and `asrSetProgressCallback`. `asrDispatchEvents` drains copied events, including
during compute. Callback userdata carries a nonreused generation ticket;
JS functions remain on the servicer. Segment times retain native centiseconds,
token bytes preserve incomplete/empty UTF-8, and progress preserves sample counts.
Registration/clear requires idle compute and no retained stream. Null restores
the native default polling hooks. Parent close clears all callback generations.

Actual `WasmStreamContract` and `WasmDispatchContract` fixtures store the typed
public-header callbacks and emit borrowed buffers which are immediately mutated.
They also exercise foreign-thread emission, stale userdata, replacement/close,
callback and encoder errors, backlog limits and popped-event cleanup. This does
not prove each backend emits every hook; real-model event acceptance is in PLAN.


## Legacy Rust and Java Whisper operations

Rust `CrispASR::new` and `transcribe_pcm[_with_strategy/_with_options]` retain
source-compatible sys function names that link to native pointer-parameter
exports. The deprecated path remains deprecated; the symbol repair does not
change its native exception-safety limitations. `new_with_dtw` configures built-in
heads or positive top layers before native initialization; the native helper
selects compatible attention. `token_dtw_timestamp` validates result indices
and returns native centiseconds/-1. The literal-header Go/Rust
fixture exercises initialization, copied segments, VAD plus tinydiarize, native
failure/null parameters, input rejection and parameter/context cleanup.

Rust `transcribe_pcm_parallel` shares sequential option validation and copied
results, using `crispasr_transcribe_parallel_by_ref`. `detect_language` calls
`crispasr_detect_language` on PCM and returns an owned code/probability pair.
`probe_legacy_rust_parallel_and_audio_language_detection` verifies processors,
threads, samples, VAD plus tinydiarize, copied metadata, empty/error results,
parameter cleanup and malformed language output. Neither method establishes
Whisper model accuracy or makes the deprecated direct API exception-safe.

Java `WhisperCpp.getContextDefaultParams/getFullDefaultParams` return independent
Java-owned copies. `initContext` forwards GPU/flash/device and DTW flags,
preset/top count, custom heads and memory size through native pointer parameters.
Custom heads are snapshotted for model lifetime. `fullTranscribe` and
`fullTranscribeWithTime` use the pointer inference export; close/reinitialization
retire old ownership and reject overlap/re-entry, including foreign callback
threads. `JavaLegacyWrapperContract` compiles the actual legacy classes against
`build_legacy_wrapper_probe.py`, rather than counting raw JNA declarations.
The legacy full-params layout requires a 64-bit JVM; broader token/parallel
features and actual-model/platform acceptance remain in PLAN.md.


## Owned legacy Whisper tokens and alternatives

The family includes `whisper_full_n_tokens`, token text/ID/data, the scalar
`crispasr_token_t0/t1/p/dtw_t` helpers, alternative count/ID/probability/text and
`crispasr_params_set_alt_n`. Go's token-data field reads are bulk equivalents of
the scalar metadata helpers; Rust calls the scalar helpers without mirroring a
struct returned by value.

| Wrapper | Callable operation | Probe |
|---|---|---|
| Rust | `CrispASR::token/tokens`, owned `WhisperToken` and candidates; `transcribe_pcm_with_alternatives` enables capture using existing options | `probe_legacy_rust_owned_token_snapshots_and_alternative_capture` |
| Go | `Whisper_token_snapshot(s)` returns owned `WhisperTokenSnapshot` values; existing `Params.SetAltN` enables capture | `TestProbeLegacyOwnedTokenSnapshots` |
| Go higher-level | `TokenSnapshots(ctx, segment)`, optional `TokenSnapshotProvider`; `Context.SetAltN` enables capture | `TestProbeModelTokenSnapshotsWithoutChangingContextInterface` |

Raw token and alternative bytes are copied, including incomplete UTF-8, and
metadata preserves native centiseconds and unavailable negative values. Rust
also provides replacement-decoded display text; Go strings keep the same bytes
as `Raw`. No borrowed native string or result pointer escapes. Snapshots remain
valid after native buffers are mutated and the model is closed. Existing Go
`Context` implementations/`Token` literals and Rust `TranscribeOptions`/`Segment`
literals keep their shapes. Rust's original alternative-capture convenience
method remains sequential; the controls API below also forwards capture during
parallel decode. Native worker/merge and model acceptance remain in PLAN.md.

Probes exercise long alternative buffer retries, token/alternative count limits,
negative/oversized lengths, missing terminators, inconsistent embedded NULs,
null/empty native text, invalid indices, unavailable sentinels and partial UTF-8
reconstruction. The limits are one million tokens per segment, 32 alternatives
and 1 MiB text per token/candidate. These compiled public-header fixtures do not
establish real model accuracy/candidate quality. Raw low-level Go/Rust legacy
contexts retain their lifetime contracts; the built-in high-level Go Model now
guards copying/close as described below.


## Rust legacy Whisper scalar controls

`WhisperDecodeControls`, `transcribe_pcm_with_controls` and
`transcribe_pcm_parallel_with_controls` expose all **19 non-VAD scalar setter
routes**, including language/prompt strings, eleven flags, threads, maximum
length/tokens, best-of, temperature and alternative count. The existing
`TranscribeOptions` supplies the six VAD/tinydiarize setter routes, covering the
current **25 `crispasr_params_set_*` exports**. This does not cover every field
of the upstream raw `whisper_full_params` struct or the separate callback APIs.

`None` leaves native defaults intact; explicit false/zero values apply where
valid. Positive threads/best-of, nonnegative limits, finite/nonnegative temperature,
nonempty language and NUL-free strings are checked before native parameter
allocation. Owned prompt/language C strings outlive inference and parameter free.
Alternative counts use native clamping. Existing options and result shapes remain
unchanged; the new controls type is non-exhaustive and constructed with Default.

The Go/Rust contract compiles the actual runtime's 25 C++ setters against the
public headers, replacing mirrored C setter bodies. Field oracles check every
control on sequential/parallel and greedy/beam routes, explicit clears, unset
native defaults, VAD plus tinydiarize, UTF-8 strings, native failures and cleanup.
The oracle also churns heap blocks before reading borrowed string fields; an
injected early owner drop reproduced a mismatch and the restored lifetime passes.
`probe_legacy_rust_decode_controls_reject_invalid_before_parameter_allocation`
also checks that rejected inputs never request native parameters. A source-family
tripwire flags new setters without public Rust routes; that check supplements
operation probes and does not establish behavioral coverage by itself. Actual
model behavior, worker errors and parallel timestamp merging remain in PLAN.md.


## Native legacy parallel runtime regression scope

`tests/test_whisper_parallel_runtime.py` compiles the literal runtime loop and
segment layout against real public parameter/token headers, controlled worker
states and real threads. Twenty-two UBSan cases cover chunk token/DTW shifts,
unavailable sentinels, alternative association, wide/fractional boundary math,
worker/main errors, worker exceptions, null state allocation, silence cleanup,
parent VAD mapping retention and sequential/tiny-input routing. It is a runtime
glue gate, separate from the language transport fixtures. Seven of these cases
add exact bounded-interval coverage, fractional/clipped/large durations and
negative/tiny-limit routing. Callback semantics and actual-model/device
acceptance remain in PLAN.md. The hosted-only
`tools/ci-heavy/whisper_wrapper_validate.py` prepares actual tiny.en pointer/Python
metadata acceptance; its artifacts must be reviewed before claiming completion.

`tests/test_whisper_vad_timeline.py` supplies thirteen UBSan cases using the literal VAD
stitcher/interpolator, token getters, scalar adapters, sequential/parallel routes
and real state-decoder entry before a controlled inference body. Token/DTW
getters map available times to original audio on a returned copy and preserve
negative sentinels; repeated reads do not change stored tokens. Plain decode
clears old VAD maps, and failed VAD clears old results. Controlled speech
intervals verify actual stitching and 1/2/3-worker getter consistency, not actual
VAD/model alignment quality or device execution. Those remain in PLAN.md.

### High-level Go legacy lifecycle (2026-10-19)

The built-in `pkg/whisper` Model owns a nonblocking native borrow shared across
its Contexts: decode, close, result/token copy, native setters/queries and context
construction are mutually exclusive. Error-returning methods return
`ErrModelBusy`; native queries without error returns panic on busy/closed access.
Go-owned parameter setters and decode snapshots use a separate mutex, so callback
updates can affect the next call. `String` handles busy/closed models safely.
Compiled public-method probes cover callback close, same/shared-context overlap,
result/cursor reads, recovery/abort, native pthread re-entry, parameter races and
owned result retention after close. Low-level raw Go Context pointers, independent
native result states and other-language lifecycle remain separate tasks.

## C# explicit setter failures

`Session.SetPuncModel`, `SetVoice` and `SetSpeakerId` require native zero-success.
Native `-2` means model load failure, invalid voice-reference settings, or an
out-of-range speaker ID respectively. These errors now throw
`InvalidOperationException` with the native return code. Previously the shared
helper silently accepted `-2` as an unsupported tuning control. Optional tuning
controls retain that compatibility behavior.

The actual public C# methods run against a compiled public-header fixture in
`CSharpWrapperContract`, covering `0`, `-1`, `-2`, `-3` and positive failure
codes, plus the unsupported tuning case. The new fixture failed before the
repair (`setter swallowed rc=-2`); afterward its current/older-library runs
pass 101/35 assertions. This proves wrapper error transport, not model quality.
Native punctuation replacement unloads the old model before attempting the new
load; the exception exposes that failure so callers can recover.

Java `CrispasrSession.setSpeakerId` and Ruby `Session.set_speaker_id` had the
same out-of-range `-2` omission. Both now reject it. Their real public adapters
run against a shared two-speaker C fixture, accepting 0/1 and rejecting negative,
upper-bound and maximum-integer IDs. Both new probes fail before the repair;
afterward Java passes 95 current/9 older-ABI assertions and Ruby passes 343
assertions, with all 155 Ruby context guards still checked. Python, Go, Rust
and Dart already reject this error; WASM exposes the return code to its caller.
