# CrispASR — C# / .NET binding

`net8.0` binding over the unified CrispASR Session C ABI: ASR, TTS, streaming,
VAD, language detection, alignment, and the music/task backends.

There is no NuGet package yet — build from source and reference the project:

```xml
<ProjectReference Include="path/to/CrispASR/bindings/csharp/CrispASR/CrispASR.csproj" />
```

The managed assembly builds as **`CrispASR.Net.dll`**; the namespace and every
type name are plain `CrispASR`, so `using CrispASR;` is all your code needs.

---

## 1. Get the native library

The binding calls into `libcrispasr` — a separate native library that is **not**
built by `dotnet build`. Download it from
[Releases](https://github.com/CrispStrobe/CrispASR/releases), taking the
`libcrispasr-*` archive for your platform (the `crispasr-*` archives are the CLI
application, not the library):

| Archive | Use it when |
|---|---|
| `libcrispasr-windows-x86_64.tar.gz` | **Windows default.** CPU, needs AVX2 (any CPU from ~2014 on). |
| `libcrispasr-windows-x86_64-cpu-legacy.tar.gz` | Windows on a pre-AVX2 CPU, or the default build crashes on startup. Portable-CPU build; slower. |
| `libcrispasr-windows-x86_64-cuda.tar.gz` | NVIDIA GPU. Self-contained — bundles the CUDA 12.8 runtime DLLs. |
| `libcrispasr-windows-x86_64-cuda-non-cuda.tar.gz` | Same CUDA build **without** the runtime DLLs, for when you already have `cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll` (published separately once per release). Saves ~600 MB per upgrade. |
| `libcrispasr-windows-x86_64-vulkan.tar.gz` | Any Vulkan GPU (AMD/Intel/NVIDIA), no CUDA install. |
| `libcrispasr-linux-x86_64.tar.gz` / `-avx512` / `-cuda` / `-hip` / `-vulkan` / `libcrispasr-linux-arm64.tar.gz` | Linux. |
| `libcrispasr-macos-arm64.tar.gz` | Apple Silicon (Metal). |

Everything you need is in the archive's `bin/` directory: the library itself plus
its `ggml*` siblings. Keep them together — the library loads them from its own
directory.

## 2. Tell the binding where it is

Any one of these works; the resolver takes the first hit
(`CrispASR.NativeLibraryResolver` implements the search):

1. **`CRISPASR_LIBRARY_PATH`** — the library file, or the directory holding it.
   The simplest thing to try when something is not loading.
2. **`runtimes/<rid>/native/`** under your application directory (the NuGet layout).
3. **`native/`, `bin/`, or `lib/`** under your application directory — copy the
   archive's `bin/` folder in as-is and it is found.
4. **Next to your application**, flat.

```csharp
// If you would rather not use an environment variable:
Environment.SetEnvironmentVariable("CRISPASR_LIBRARY_PATH", @"C:\crispasr\bin");
Console.WriteLine(CrispASR.NativeLibraryResolver.Load());   // prints what it loaded
```

> **Windows note (issue #291).** Through v0.8.29 the managed assembly was
> `CrispASR.dll` and the native library is `crispasr.dll` — the same file name on
> a case-insensitive filesystem, so copying the native library next to the
> managed one *overwrote* it, and bare-name P/Invoke probing found the managed
> assembly instead of the library. The assembly is `CrispASR.Net.dll` now, the
> two coexist, and the resolver additionally refuses to hand a managed assembly
> to the native loader.

When the library still cannot be found, the `DllNotFoundException` lists every
path that was probed — read it before guessing.

## 3. Transcribe a file

```csharp
using CrispASR;

using var session = Session.Open("moonshine-base-de-fidoriel-q4_k.gguf");
foreach (var seg in session.TranscribeFile("speech.wav"))
    Console.WriteLine($"[{seg.T0:F2}-{seg.T1:F2}] {seg.Text}");
```

`TranscribeFile` decodes with the same decoder the CLI uses — WAV, MP3, FLAC,
OGG, Opus, AAC, M4A, WebM, AMR, any sample rate, any channel count. If you
already have samples, `Audio.Load(path)` gives you the 16 kHz mono `float[]`
that every ASR entry point takes:

```csharp
float[] pcm = Audio.Load("speech.mp3");        // 16 kHz mono float32
Segment[] segs = session.Transcribe(pcm);      // long audio is auto-chunked
```

The .NET base class library has no audio decoder, so a hand-rolled WAV reader
used to be the usual reason a model that worked under `crispasr.exe` produced
nothing here. Use `Audio.Load` and the two paths decode identically.

**Model files with companions.** Some backends need a sidecar next to the GGUF —
moonshine loads `tokenizer.bin` from the model's own directory. The CLI
downloads both; a manual download must fetch both too, into the same folder.

## 4. Reuse the session

A `Session` holds the loaded model until `Dispose`. Open once, transcribe many:

```csharp
using var asr = Session.Open("model.gguf");
foreach (var clip in clips)
    Handle(asr.Transcribe(clip));    // no reload
```

A `Session` is **not** thread-safe — one per thread, or serialize the calls. The
exception is standalone `Session.VadSegments`, which loads and frees its VAD
model on every call; if you also transcribe, use the session-integrated VAD path
so one session covers both.

## 5. Quieten the logs

```csharp
Logging.SetMinLevel(LogLevel.Warn);                        // drop info/debug
Logging.Silence();                                         // mute everything
Logging.SetCallback((level, msg) => myLogger.Log(level, msg));
```

This binds `whisper_log_set`, which the native side also wires to `ggml_log_set`,
so it covers model load, backend init and decode. A few backends still
`fprintf` directly and are governed by per-session verbosity instead.

## 6. Time units

**Every time value this binding exposes is in seconds** —
`Segment.T0/T1`, `Word.T0/T1`, `AlignedWord.T0/T1`, `VadSpan.T0/T1`,
`StreamingUpdate.T0/T1`, and the `…Seconds` members on the music types. The C
ABI reports centiseconds; the conversion happens once, at the boundary.

A backend that produces no timing for a unit reports **`-1`** rather than a
scaled sentinel — moonshine, for instance, emits token text and probabilities
but no word times. Check for `-1` before treating a value as a timestamp.

*(Through v0.8.29 `Segment`, `Word` and `AlignedWord` were `long` centiseconds
while the rest of the binding was seconds — issue #291. They are `double`
seconds now; a build that used to print `[0-1100]` for an 11-second clip now
prints `[0.00-11.00]`.)*

## 7. Tests

```bash
cd bindings/csharp
dotnet test CrispASR.Tests/CrispASR.Tests.csproj
```

Tests that need the native library or a model skip themselves by default. Set
**`CRISPASR_CS_REQUIRE_LIVE=1`** to turn those skips into failures — CI does, so
a suite that loads no library can no longer report green. To run the real-model
end-to-end tests, also set:

```bash
export CRISPASR_MODEL_ASR=/path/to/moonshine-tiny-q4_k.gguf   # tokenizer.bin beside it
export CRISPASR_AUDIO_SAMPLE=/path/to/CrispASR/samples/jfk.wav
export CRISPASR_ASR_BACKEND=moonshine
export CRISPASR_LIBRARY_PATH=/path/to/libcrispasr/bin
export CRISPASR_CS_REQUIRE_LIVE=1
```

Optional extras, each gating its own tests: `CRISPASR_MODEL_WHISPER`,
`CRISPASR_VAD_MODEL`, `CRISPASR_MODEL_TABCNN`, `CRISPASR_MODEL_TTS`. A variable
that is set but points at a missing file always fails — that is a
misconfiguration, not a skip.

## Additional native operations

`ModelRegistry` exposes registry entries, exact default bundles and cache
operations. An unknown lookup/bundle returns null; native download errors
propagate. `SpeakerEmbedder`, `PyannoteCache` and `Diarization.Assign` expose
speaker extraction, clustering and segment/turn labeling. All C# times are
seconds. `SessionOptions` configures v2 session creation; `Session.VadSlices`
and `Session.DetectTextLanguage` provide the corresponding auxiliary tasks.

For Whisper-specific legacy controls, `WhisperContext` uses opaque native
parameters and pointer adapters:

```csharp
using var whisper = WhisperContext.Open("ggml-base.bin", dtw: true);
var segments = whisper.Transcribe(pcm, new WhisperDecodeOptions {
    Language = "de", InitialPrompt = "Grüße", TokenTimestamps = true,
    Processors = 2
});
```

Tokens include owned raw UTF-8 pieces, alternatives and DTW times. Concatenate
raw pieces before decoding; a missing time is -1. Library versions without the
new parallel pointer export cannot provide `Processors > 1`. Do one operation
at a time per context. Dispose streams before their owner, or dispose the owner
to close all of its child streams.

`ParakeetContext` also exposes the direct legacy decoder, copying words and raw
tokens before freeing the native result. `Transcribe(pcm, offsetSeconds: 1.23)`
converts the offset to native centiseconds. It has no language parameter; use
`Session` when selecting language hints.

`TitaNet.Embed` returns an owned 192-float vector. Native success reports the
number of floats written, not a zero status. The binding rejects other counts
and calls after disposal. Similarity helpers expect embeddings already
normalized to unit L2 length by TitaNet.

`TranscribeVad(pcm, sampleRate, model, language, new VadOptions { ... })`
accepts per-call language and all six native VAD controls. Threshold defaults
to 0.5; speech/silence/padding durations are milliseconds; chunk length is
seconds. The original three-argument overload retains native defaults.
`SpeakerDb.Match` preserves a negative native score as no match, and DB
operations reject disposed handles. `SpeakerDb.Enroll` writes to disk;
reopen an existing DB to refresh the retained roster.

## Experimental Ultra local-head alignment

`Session.AlignWordsUltra` accepts the converted base aligner and separate local-head
GGUF, a transcript, and mono 16 kHz PCM (up to 120 seconds):

```csharp
AlignedWord[] words = Session.AlignWordsUltra(
    "qwen3-forced-aligner-ultra-f16.gguf", "ultra-local-head-f32.gguf",
    "Hello, world!", pcm16kMono, nThreads: 4);
Console.WriteLine($"{words[0].Text}: {words[0].T0:F6}–{words[0].T1:F6} seconds");
```

Times retain fractional milliseconds as double-precision seconds. Character spans
are empty. A failed native alignment throws `InvalidOperationException`; invalid
arguments throw `ArgumentException`. The loaded native library must export the
experimental Ultra C ABI. See the [local-head guide](../../docs/qwen3-ultra-local-head-port.md)
for conversion and the limited model qualification; broader language and GPU
acceptance remain open.
