# Getting started

A first walkthrough for someone who has never run CrispASR. Download one file,
make it speak, transcribe the result back, then run it as a server. No repo
clone, no Python, no model hunting.

The rest of the documentation is a reference catalogue — 143 backends. Ignore it
until this page works.

- [1. Download the binary](#1-download-the-binary)
- [2. Check it runs](#2-check-it-runs)
- [3. Make it speak](#3-make-it-speak)
- [4. Transcribe it back](#4-transcribe-it-back)
- [5. Run it as a server](#5-run-it-as-a-server)
- [When something goes wrong](#when-something-goes-wrong)
- [Where to go next](#where-to-go-next)

You need about 10 minutes, ~600 MB of disk for the two models, and an internet
connection for the first run of each command (models are cached afterwards).

---

## 1. Download the binary

One file from [**Releases**](https://github.com/CrispStrobe/CrispASR/releases/latest).
Unpack it and work from the folder you unpacked into — the binary is
`crispasr.exe` on Windows, `crispasr` elsewhere.

| Your machine | Take this asset |
|---|---|
| **Windows, no NVIDIA GPU** | `crispasr-windows-x86_64-cpu.zip` |
| **Windows, CPU older than 2013** | `crispasr-windows-x86_64-cpu-legacy.zip` |
| **Windows + NVIDIA GPU** | `crispasr-windows-x86_64-cuda.zip` (or `-cuda13.zip` on Turing/GTX 16xx or newer) |
| **macOS, Apple Silicon** | `crispasr-macos-arm64.tar.gz` — Metal GPU + Accelerate (`crispasr-macos.tar.gz` is the arm64 alias) |
| **macOS, Intel** | `crispasr-macos-x86_64.tar.gz` — CPU + Accelerate; no Metal |
| **Linux, no GPU** | `crispasr-linux-x86_64.tar.gz` |
| **Linux + NVIDIA GPU** | `crispasr-linux-x86_64-cuda.tar.gz` |

Three things that trip people up here:

- **The default Windows and Linux CPU builds require AVX2 + FMA** (Intel 2013+ / AMD 2015+). On an
  older CPU the AVX2 build dies on an illegal instruction. Take the
  `-cpu-legacy` asset instead — same features, slower per core.
- **The CUDA packages are self-contained.** You do *not* need to install the
  CUDA Toolkit, and you do not need to set `CUDA_PATH`. Only the NVIDIA
  **driver** matters.
- **No GPU is not a problem.** The plain CPU builds are the normal case, and the
  CUDA builds fall back to CPU when no NVIDIA driver is present (since v0.8.30).
  The `-hip` and `-vulkan` builds are the exception: they *require* their driver
  and do not fall back.

Full asset matrix, including the split CUDA-DLL downloads:
[install.md](install.md#windows-cpu-which-zip-380).

## 2. Check it runs

```bash
crispasr --version
```

```console
=== build info ===
  version       : 0.8.31
  git sha       : 91e8afad
  ...
  ggml backends : cpu
```

The `ggml backends` line tells you what this build can use — `cpu`, `cpu,cuda`,
`cpu,metal`. If you got a version banner, the download was correct and you are
past the hardest part.

> On Windows, write `.\crispasr.exe` instead of `crispasr` unless you have put
> the folder on your `PATH`. Every command below is a **single line** — do not
> add line continuations you copied from elsewhere. PowerShell continues lines
> with a backtick `` ` ``, `cmd.exe` with `^`, and mixing the two is a common
> cause of "the command did nothing".

## 3. Make it speak

`-m auto` downloads the model on first use and reuses it afterwards, so there is
nothing to find or install. It lands in `~/.cache/crispasr/`
(`%USERPROFILE%\.cache\crispasr` on Windows).

```bash
crispasr --backend kokoro -m auto --tts "The quick brown fox jumps over the lazy dog." --tts-output hello.wav
```

First run downloads ~135 MB, then:

```console
crispasr: resolving kokoro-82m-q8_0.gguf (~135 MB) via -m auto
kokoro: loaded 459 tensors from '.../kokoro-82m-q8_0.gguf'
kokoro: phonemes: 'ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.'
crispasr: TTS output written to 'hello.wav' (78000 samples @ 24000 Hz, 3.25 sec)
```

Play `hello.wav`. That is the TTS half working.

**If your player refuses the file or stalls on it**, the C2PA provenance
manifest is the likely reason — some players (VLC at the time of writing) do not
understand it. Disable it, which requires you to take on the marking duty
explicitly:

```bash
crispasr --backend kokoro -m auto --tts "Hello there." --tts-output hello2.wav --no-c2pa --accept-marking-responsibility
```

```console
crispasr: note: C2PA signing disabled (--no-c2pa); 'hello2.wav' written without a manifest (audio watermark still applied)
```

CrispASR also warns that Kokoro's preset voice is not declared synthetic or
real. Add `--speaker-identity synthetic` to silence that for a preset voice.
Why any of this exists: [eu-ai-act.md](eu-ai-act.md).

## 4. Transcribe it back

```bash
crispasr --backend parakeet -m auto -f hello.wav -l en
```

First run downloads ~467 MB, then:

```console
crispasr: audio: 52000 samples (3.2 s) @ 16000 Hz, 4 threads
crispasr: transcribed 3.2s audio in 3.30s (1.0x realtime)
The quick brown fox jumps over the lazy dog.
```

`-l en` skips language auto-detection, which would otherwise fetch a small extra
model. Both halves now work — swap in your own `.wav` and you are running.

Subtitles instead of plain text — `-of` names the output file without its
extension:

```bash
crispasr --backend parakeet -m auto -f hello.wav -l en -osrt -of hello
```

```console
$ cat hello.srt
1
00:00:00,000 --> 00:00:03,120
The quick brown fox jumps over the lazy dog.
```

## 5. Run it as a server

Loading a model takes seconds; a server pays that once and keeps it in memory.

```bash
crispasr --server --backend parakeet -m auto --port 8199
```

```console
crispasr-server: listening on 127.0.0.1:8199
  POST /inference                  — upload audio (native JSON)
  POST /v1/audio/transcriptions    — OpenAI-compatible API
  GET  /health                     — server status
```

From a second terminal:

```bash
curl -s http://localhost:8199/health
# {"status": "ok", "backend": "parakeet"}

curl -s -F "file=@hello.wav" http://localhost:8199/v1/audio/transcriptions -F "response_format=text"
# The quick brown fox jumps over the lazy dog.
```

`--host 0.0.0.0` accepts remote connections; API keys, hot model swapping and
concurrency are in [server.md](server.md).

## When something goes wrong

Add `-v` to any command first — it prints the build banner, the GPU devices
found, and the resolved model path. Then check these three, which cover most
first-run failures:

**It printed the banner and then stopped.** No error, no output file. That is a
crash, not a refusal, and the exit code names it in one step:

```powershell
$LASTEXITCODE          # PowerShell, immediately after the failing run
```

```bash
echo $?                # Linux / macOS
```

`-1073741795` (`0xC000001D`) is an **illegal instruction**: you are on the wrong
build for your CPU — take the `-cpu-legacy` asset from step 1. Any other code,
and the table in
[troubleshooting.md](troubleshooting.md#it-printed-the-banner-then-nothing-happened)
tells you what it means and what to send in a bug report.

**`model '...' not found locally`.** CrispASR never downloads without being
asked. Either let it resolve one for you, or point at a file you already have:

```bash
crispasr --backend parakeet -m auto -f audio.wav                       # resolve + download
crispasr --backend parakeet -m name.gguf --auto-download -f audio.wav
crispasr --backend parakeet -m /full/path/to/name.gguf -f audio.wav
```

If your models live in one directory, set `CRISPASR_MODELS_DIR` to it and a bare
filename is found there. `--dry-run-resolve` prints every path a command *would*
open, and whether each is on disk, without loading anything — the fastest way to
tell a missing file from a broken one. More:
[troubleshooting.md](troubleshooting.md#model--not-found-locally).

**Is it the GPU?** One flag answers it. `--no-gpu` forces everything onto the
CPU:

```bash
crispasr --backend kokoro -m auto --tts "test" --tts-output out.wav --no-gpu
```

Works with `--no-gpu` and fails without it → a GPU-backend problem. Fails both
ways → not the GPU. Details:
[troubleshooting.md](troubleshooting.md#is-it-the-gpu).

## Where to go next

| You want to… | Go to |
|---|---|
| See every backend this binary has | `crispasr --list-backends` |
| Pick a better ASR model | [README — supported backends](../README.md#supported-backends) |
| Word timestamps, VAD, output formats | [cli.md](cli.md) |
| Other TTS engines and voices | [tts.md](tts.md) |
| Clone a voice from a recording | [tts.md](tts.md) — requires `--i-have-rights`; read [eu-ai-act.md](eu-ai-act.md) first |
| Live mic / streaming | [streaming.md](streaming.md) |
| HTTP API in depth | [server.md](server.md) |
| Build it from source | [install.md](install.md) |
| Something is broken | [troubleshooting.md](troubleshooting.md) |
