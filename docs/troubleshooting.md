# Troubleshooting

First-run problems, in the order people actually hit them. If your symptom
isn't here, open an issue with the [checklist at the bottom](#what-to-put-in-a-bug-report).

## Table of contents

- [Get more output first](#get-more-output-first)
- [It printed the banner, then nothing happened](#it-printed-the-banner-then-nothing-happened)
- [Is it the GPU?](#is-it-the-gpu)
- [`model '...' not found locally`](#model--not-found-locally)
- [Windows: which download, and the CUDA DLLs](#windows-which-download-and-the-cuda-dlls)
- [A backend refuses to start](#a-backend-refuses-to-start)
- [What to put in a bug report](#what-to-put-in-a-bug-report)

---

## Get more output first

Before anything else, make the tool tell you more. Cheapest lever first.

**`-v` — verbose.** Prints the build banner, the GPU devices it found, the
resolved model path, and per-stage progress. Put it on every run you intend to
report. For library / bindings use, where there is no CLI flag to pass, set
`CRISPASR_VERBOSE=1` instead.

**`--dry-run-resolve` — which files, without opening them.** This prints every
path CrispASR *would* use and then exits. Nothing is loaded, so it still works
when the real run crashes:

```console
$ crispasr --backend kokoro -m auto --tts "x" --tts-output y.wav --dry-run-resolve
model:
  requested: auto
  backend:   kokoro
  registry:  kokoro-82m-q8_0.gguf
  url:       https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main/kokoro-82m-q8_0.gguf
  size:      ~135 MB
  status:    cached/local
  path:      /home/you/.cache/crispasr/kokoro-82m-q8_0.gguf
companion:
  ...
```

Read the `status:` and `path:` lines. This separates *finding* a file from
*loading* it — which is the key split when a run dies during startup — and it
also catches the case where a companion file (codec, voice pack) is the one
actually missing. Compare the `size:` against the file on disk: a truncated or
half-downloaded GGUF is a common and completely silent cause of trouble.

**Capture the whole log.** Almost all diagnostic output goes to **stderr**, and
a bare `>` redirect captures only stdout — so "there was no output" is sometimes
a capture problem rather than a program problem. Grab both streams:

```powershell
crispasr ...args... *> crispasr-log.txt          # PowerShell 7+
```

```bat
crispasr ...args... > crispasr-log.txt 2>&1      REM cmd.exe
```

```bash
crispasr ...args... > crispasr-log.txt 2>&1      # Linux / macOS
```

**Deeper levers**, when the above isn't enough:

| Lever | What it adds |
|---|---|
| `-debug` / `--debug-mode` | Extra decoder-level diagnostics (whisper-family backends). |
| `GGML_SCHED_DEBUG=2` | Which backend each graph node was placed on — for "correct on CPU, wrong on GPU". |
| `CRISPASR_<BACKEND>_DEBUG=1` | Per-backend step diagnostics. See [environment-variables.md](environment-variables.md). |

---

## It printed the banner, then nothing happened

**Symptom:** the build-info banner and some `crispasr[verbose]:` lines appear,
then the program stops. No error, no output file, no obvious clue.

This is almost always a **crash**, not a silent refusal. Every failure path in
the CLI prints something before returning, so "no message" means the process
died rather than returned. On Windows in particular the console swallows the
fault and you get no dialog and no text (this is what
[#380](https://github.com/CrispStrobe/CrispASR/issues/380) and
[#397](https://github.com/CrispStrobe/CrispASR/issues/397) both looked like).

**Step 1 — read the exit code.** This is the single most informative thing you
can send us, and it takes one command:

```powershell
# PowerShell, immediately after the failing run
$LASTEXITCODE
```

```bat
REM cmd.exe
echo %ERRORLEVEL%
```

```bash
# Linux / macOS
echo $?
```

| Exit code | Hex | Means |
|---|---|---|
| `-1073741795` | `0xC000001D` | **Illegal instruction** — the binary uses a CPU feature this machine lacks. See below. |
| `-1073741819` | `0xC0000005` | **Access violation** (segfault). A genuine bug — please report it. |
| `-1073740791` | `0xC0000409` | Stack buffer overrun / fail-fast. Please report it. |
| `139` | — | Segfault on Linux/macOS. Please report it. |
| `132` | — | Illegal instruction on Linux/macOS. |
| `11`, `12`, `13`, `14` | — | Not a crash — a normal error return. CrispASR printed a reason; scroll up. |

**Step 2 — if it's an illegal instruction**, the binary executed an instruction
the host cannot run. That can mean either the CPU is older than the artifact's
advertised baseline **or the artifact was packaged incorrectly**. Do not assume
the first and blame the user's CPU without comparing the evidence.

The default Windows CPU zip targets an **AVX2 + FMA** baseline (Intel Haswell
2013+ / AMD Excavator 2015+). If the host genuinely lacks AVX2 or FMA, download
`crispasr-windows-x86_64-cpu-legacy.zip` instead — a generic x86-64/SSE2 build
that runs anywhere from Westmere up, just slower per core. Recent builds check
this at startup and print the exact missing feature
(`CRISPASR_IGNORE_CPU_ISA=1` overrides the check). Full detail:
[install.md § Windows CPU: which zip?](install.md#windows-cpu-which-zip-380).

If the host **does** support the documented baseline, treat the crash as a
release bug. This happened in v0.8.29: the Windows CUDA package accidentally
inherited AVX-512 from its GitHub runner and crashed in `ggml-cpu.dll` on modern
AVX2-only machines ([#374](https://github.com/CrispStrobe/CrispASR/issues/374),
[#397](https://github.com/CrispStrobe/CrispASR/issues/397)). The v0.8.30 Windows
CUDA package is pinned to AVX2. Replace the whole extracted directory when
upgrading so the old DLL is not loaded beside the new executable.

**Step 3 — if it's an access violation**, narrow *where* it dies before
reporting. Add `-v` and note the last line printed:

| Last line you see | Where it died |
|---|---|
| `resolved model = '...'` | Loading the model — the file was found, and opening/parsing it crashed. |
| `backend '...' initialised OK` | After load — during synthesis or transcription. |
| nothing past the banner | Startup, before any model work. Usually the CPU-ISA case above. |

Two follow-ups narrow it further, and neither can crash the way the real run
does: [`--dry-run-resolve`](#get-more-output-first) confirms the files it was
about to open really are on disk and the expected size, and
[`--no-gpu`](#is-it-the-gpu) tells a GPU fault from a model fault.

**On Windows, a minidump pins it exactly.** The repo has a ProcDump recipe:
[windows-illegal-instruction-dumps.md](windows-illegal-instruction-dumps.md).
Use the documented triage dump, not `-ma` — a full dump can contain your audio,
model data, and file paths.

---

## Is it the GPU?

One flag answers this. `--no-gpu` forces the whole pipeline onto the CPU:

```bash
crispasr --backend kokoro -m auto --tts "test" --tts-output out.wav --no-gpu
```

- **Works with `--no-gpu`, fails without it** → a GPU-backend problem. Tell us
  your GPU, driver version, and which zip/tarball you downloaded.
- **Fails both ways** → not the GPU. The model, the file, or the CLI arguments.

Fallback is package-specific. The v0.8.30 Linux CUDA tarball loads CUDA
dynamically and can fall back to CPU; Linux HIP/Vulkan and the Windows GPU
archives require their matching driver/runtime. See the exact
[install matrix](install.md#prebuilt-linux-tarballs--which-one-to-download-355).
`--no-gpu` forces CPU once the process can start, but cannot repair a missing
dependency in a hard-linked package.

---

## `model '...' not found locally`

CrispASR does **not** download models unless you ask it to. When the file
isn't there you get an explicit block naming the file and its size:

```
crispasr: model 'parakeet-tdt-0.6b-v3-q4_k.gguf' not found locally.
  Available for download: parakeet-tdt-0.6b-v3-q4_k.gguf (~467 MB)
  Use --auto-download or -m auto to download automatically.
```

Three ways forward:

```bash
crispasr --backend parakeet -m auto -f audio.wav              # resolve + download
crispasr --backend parakeet -m name.gguf --auto-download -f audio.wav
crispasr --backend parakeet -m /full/path/to/name.gguf -f audio.wav
```

Downloads land in `~/.cache/crispasr/` — on Windows
`%USERPROFILE%\.cache\crispasr` (override with `CRISPASR_CACHE_DIR`).
To see exactly which paths would be used without running anything, use
`--dry-run-resolve`.

**If you do *not* see this block, the file was found.** That matters when
diagnosing a crash: it means the problem is in loading the file, not locating it.

---

## Windows: which download, and the CUDA DLLs

| Zip | Use when |
|---|---|
| `crispasr-windows-x86_64-cpu.zip` | Default. Needs AVX2 + FMA (2013+ Intel / 2015+ AMD). |
| `crispasr-windows-x86_64-cpu-legacy.zip` | Host genuinely lacks AVX2 or FMA. CPU-only. |
| `crispasr-windows-x86_64-cuda.zip` | NVIDIA GPU + AVX2/FMA CPU. Self-contained. |
| `crispasr-windows-x86_64-cuda-non-cuda.zip` | Same baseline, and you already have the three matching runtime DLLs. |
| `crispasr-windows-x86_64-vulkan.zip` | Vulkan GPU + AVX2/FMA CPU (AMD/Intel/NVIDIA). |

The CUDA packages bundle **CUDA 12** runtime DLLs — `cudart64_12.dll`,
`cublas64_12.dll`, `cublasLt64_12.dll`. They are published once per release and
shared by both CUDA zips; see
[install.md § Windows CUDA: split downloads](install.md#windows-cuda-split-downloads-342)
and check the `-runtime-sha256.txt` manifest before reusing DLLs from an older
download.

Two consequences worth knowing:

- A separate CUDA Toolkit install is **not** required — the self-contained zip
  ships what it needs. Having CUDA 13.x installed system-wide neither helps nor
  is used by these binaries.
- If you took the `-non-cuda` zip, the three DLLs must sit next to
  `crispasr.exe`. A CUDA 13 toolkit does not provide them (its runtime is
  `cudart64_13.dll`), so `-non-cuda` + "only CUDA 13 installed" will not work.

On some Windows laptops the Vulkan device `0` is the Intel iGPU and the NVIDIA
GPU is `1`; if Vulkan looks unexpectedly slow, rerun with `-dev 1`.

---

## A backend refuses to start

These are *normal* error returns — CrispASR prints a reason and exits non-zero.

| Message | Fix |
|---|---|
| `backend '...' is not available in this build` (rc 12) | Backend name typo, or a build without it. `--list-backends` shows what this binary has. |
| `failed to initialise backend '...'` (rc 13) | The line above it names the real cause — a missing codec, an unreadable model, a bad voice pack. |
| `backend '...' does not support TTS` (rc 14) | That backend is ASR-only. `--list-backends` has a `tts` column. |
| `voice cloning requires the --i-have-rights flag` (rc 17) | You pointed `--voice` at a recording. Add `--i-have-rights` to attest you have the speaker's consent. See [eu-ai-act.md](eu-ai-act.md). |
| `--voice is a WAV but --ref-text was not set` | Cloning from a recording needs `--ref-text "<exact transcript of that wav>"`. |

---

## What to put in a bug report

Paste all of this — it is usually enough to diagnose without a round trip:

1. **The full command**, verbatim.
2. **The complete output** with `-v` added, captured with
   [both streams redirected](#get-more-output-first) (the build-info banner at
   the top is the important part — it names the version, the backends compiled
   in, and your GPU).
3. **The exit code** (`$LASTEXITCODE` / `echo $?`).
4. **The `--dry-run-resolve` output** for the same command — it shows which
   files were going to be opened and whether each is actually on disk.
5. **Which download** you used — the exact zip/tarball filename, or the
   `cmake` line if you built it yourself.
6. **Whether `--no-gpu` changes anything.**
7. **Model files**: which GGUFs, from which HuggingFace repo, and their sizes
   on disk (a truncated download is a real and common cause).
