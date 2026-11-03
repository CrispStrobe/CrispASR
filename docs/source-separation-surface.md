# Source-separation surface (§248) — shared design

Music/voice **source separation** is a new task category, distinct from ASR
(text out) and TTS (synthetic audio out): it takes one mixed audio input and
returns **N named stems of the user's own audio**. The supported backends are
`htdemucs` (4-stem: `drums`/`bass`/`other`/`vocals`), `mel-band-roformer`
(2-stem: `vocals`/`other` for a 1-stem vocals model; `stem0…stemN` otherwise),
`mini-bs-roformer` (HiDolen V2.5: `bass`/`drums`/`other`/`vocals`),
and `scnet` (SCNet / SCNet-large: `drums`/`bass`/`other`/`vocals`).

This doc is the **single agreed surface** all separation backends route through,
with shared CLI flags and output conventions. Authored by the M1/Metal
session (mel-band-roformer), 2026-07-19, at the maintainer's instruction to
design the shared surface now. The htdemucs session should adopt it.

## Why not the `transcribe()` backend interface

Separation returns **audio, not `crispasr_segment`s**, so it is NOT a
capability layered onto the ASR `transcribe()` path. It gets its own dispatch.
A capability bit (`CAP_SEPARATE`, `1u << 22`, declared by all separation backends)
is only for detection/help text, not for
routing through the transcription loop.

## CLI

```
crispasr --separate -m <model.gguf> -f mix.flac [--stems vocals,drums] [--sep-output-dir DIR]
```

- **`--separate`** enables the task (alias intent: `--task separate`). Routes to
  the separation dispatcher BEFORE the ASR backend is constructed.
- Backend is auto-detected from the GGUF `general.architecture`
  (`htdemucs` / `mel-band-roformer` / `mini-bs-roformer` / `scnet`) after the normal `-m` resolution, so a
  concrete `-m <gguf>` never needs `--backend`. Only the auto-download path
  needs it, to pick which model to fetch (`--backend htdemucs|mel-band-roformer`;
  `mel-band-roformer` is the default resolution key when `--backend` is unset).
- **`--stems LIST`** — comma-separated subset to write (case-insensitive);
  empty / `all` writes every stem. A backend ignores names it doesn't have.
- **Output**: one WAV per selected stem, named
  `<input-stem>_<source>.wav` next to the input, or under `--sep-output-dir`.
  Stereo, 16-bit PCM, at the model's native rate (44100). **No AI-provenance
  INFO chunk** — the audio is the user's, not AI-generated.

SCNet requires a concrete converted GGUF path and does not advertise
auto-download. Both SCNet checkpoints use `general.architecture=scnet`. Its
four-stem API is available through `Session.separate()` in the bindings and
through the HTTP endpoint below. Default inference follows the original
20-second chunks, 50% overlap and one seeded random shift per chunk; `--seed`
selects the repeatable shift sequence (default zero). A direct native
`scnet_params.no_segment` diagnostic bypasses the outer chunk/shift/normalization
wrapper and must not be presented as the default end-to-end inference path.

## Shared code (landed, additive)

`src/core/separation_io.h` (header-only, unit-tested — `tests/test-separation-io.cpp`):

- `struct crispasr_separation_view` — backend-agnostic, **non-owning** result:
  `n_sources, n_channels, n_frames, sample_rate, sources[], source_names[]`.
  Each backend fills this from its own result struct (e.g. `htdemucs_result`,
  which already has exactly these fields) — no backend refactor required.
- `crispasr_stem_output_path(input, source, out_dir, ext="wav")` — deterministic
  naming (extension-strip, dir handling, lowercase source).
- `crispasr_stem_selected(csv, source)` — `--stems` membership.
- `crispasr_stem_to_wav(view, s)` — one stem → interleaved WAV blob via
  `crispasr_make_wav_int16_interleaved` (new multi-channel writer in
  `crispasr_wav_writer.h`, no AI tag).

## Dispatcher (landed)

`examples/cli/crispasr_separate_cli.{h,cpp}` (NEW file — keeps the shared
`cli.cpp` footprint to just flag parsing + one early-dispatch hook, minimizing
collision with the in-flight htdemucs integration):

```
int crispasr_run_separate(const whisper_params& params);
  1. resolve -m, detect arch
  2. read audio -> stereo float @ model rate (reuse audio_resample/wav_reader)
  3. arch == htdemucs         -> htdemucs_init_from_file + htdemucs_separate
     arch == mel-band-roformer-> mel_band_roformer_init_from_file + _separate
     arch == mini-bs-roformer -> mini_bs_roformer_init_from_file + _separate
     arch == scnet            -> scnet_init_from_file + _separate
  4. wrap result in crispasr_separation_view
  5. for each source: if crispasr_stem_selected(--stems) -> write
     crispasr_stem_output_path(...) with crispasr_stem_to_wav(...)
```

The backend `*_separate()` C APIs return the same shape
(`float** sources` + `const char** source_names`), so the wrapper is a few
lines per backend.

## Coordination

- The shared header + WAV writer + tests are **additive** (no existing file
  depends on them) — safe to land while htdemucs is mid-integration.
- The htdemucs session, when it wires its CLI, should call
  `crispasr_run_separate` and adopt this naming, not add a second `--separate`
  path. If it has already started one, we reconcile to THIS spec (one surface).
- `mel_band_roformer_{init_from_file,separate}` (the C API mirroring
  `htdemucs.h`) landed with the MBR C++ backend; the dispatcher's MBR branch is
  live (`examples/cli/crispasr_separate_cli.cpp`), no longer stubbed.

## HTTP server (§381)

```
crispasr --server -m asr.gguf --separate-model htdemucs-f16.gguf --port 8080
```

- **`--separate-model PATH`** — loads a secondary GGUF at startup (auto-detected
  `htdemucs`, `mel-band-roformer`, `mini-bs-roformer` or `scnet`, resolved via `crispasr_resolve_model_cli`).
  Persistent context, serialised by `std::mutex` (concurrent requests queue).
- **`POST /v1/audio/separation`** — multipart `file` (audio upload, any supported
  container) + optional `stems` field (comma list, default all). Returns
  `multipart/mixed` with one `audio/wav` part per selected stem (stereo 16-bit
  PCM, no AI-provenance tag). Each part carries
  `Content-Disposition: attachment; filename="<stem>.wav"`.
- **Errors**: `503` when `--separate-model` not configured (`separation_disabled`),
  `400` for missing/invalid audio or an empty stem selection, `500` on backend
  failure — all in `{"error": {"message", "type"}}` shape.
- The endpoint appears in the startup banner when `--separate-model` is set.

## RESOLVED — one surface (2026-07-19)

Two surfaces briefly coexisted: the `--separate` dispatcher (this spec, which
drives BOTH backends) and an independent `CAP_SEPARATE` transcribe-adapter the
htdemucs session added (`crispasr_backend_htdemucs.cpp` ran `htdemucs_separate`
inside `transcribe()` and returned a synthetic `"[separated: …]"` segment).

**Evaluated both (maintainer: keep the better). The adapter was non-functional:**
its `get_last_result()` had **zero callers**, so the stems it computed were
**never written** — it only printed the fake text segment; and it fed the
pipeline's mono 16 kHz to a model needing stereo 44.1 kHz. `--separate` is the
only functional surface (resamples, both backends, writes correct stems,
validated cos=1.0 + a working CLI producing vocals rms 0.218 vs other 0.005).

**Resolution:** `crispasr_backend_htdemucs.cpp` is now a thin redirect shim —
it keeps htdemucs in `--list-backends` with `CAP_SEPARATE`, and `--backend
htdemucs` without `--separate` prints "run it with --separate" instead of doing
broken separation-through-transcribe. `src/htdemucs.*` (the real backend) is
untouched. One surface: `--separate`.
