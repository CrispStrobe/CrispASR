# CrispASR v0.8.15

Supersedes v0.8.14 (whose Windows binaries were broken by the WebRTC-VAD vendor
— fixed here). Everything since v0.8.13.

## New audio tasks

- **Source separation (`--separate`)** — two native backends:
  - **Mel-Band RoFormer** vocal/instrumental split (MIT weights KimberleyJSN,
    MIT code lucidrains), validated stage-by-stage to cos 1.0. Auto-download via
    `--separate -m auto` (`cstr/mel-band-roformer-vocals-GGUF`).
  - **HTDemucs** hybrid-transformer separation — full parity (pos embeddings,
    per-freq-band dilated-conv residuals, time decoder). Q4_K default on HF.
  - Python `Session.separate()` binding.
- **Speech restoration (`--s2s`)** — **Sidon v0.1** (SaruLab, MIT): w2v-BERT
  predictor + continuous DAC decoder, 16 kHz → 48 kHz (#283, by @KevinAHM).
  Validated against the upstream TorchScript: predictor handoff cos **0.998**,
  ASR round-trip **identical**. Quant ladder + reference intermediates on
  `cstr/Sidon-GGUF` (f16/q8_0/q6_k/q4_k + `sidon-ref.gguf`); `-m auto` fetches
  Q8_0. O(T²)-attention length cap (`CRISPASR_SIDON_MAX_FRAMES`, ~60 s default).
- **MioCodec / MioTTS (§248/§249/§250)** — full native codec decode +
  end-to-end synthesis (weight-norm, SnakeBeta, iSTFT mag/phase), audio parity
  cos 0.999+; GGUFs on HF + registry.

## Fixes & improvements

- **Windows** — the v0.8.14 breakage: `webrtc-vad` now uses `[[noreturn]]` (not
  the MSVC-rejected `__attribute__((noreturn))`) and only defines `WEBRTC_POSIX`
  off-Windows so `spl_init.c` takes the native `CRITICAL_SECTION` branch
  (diagnosed in #279 by @KevinAHM). WASM and Docker-Smoke build fixes too.
- **Resampler** — polyphase downsampling now preserves unity DC gain (#277, by
  @KevinAHM); regression test covers the 48k→44.1k `--separate` path.
- **Library GPU consumers** — C-ABI session/`nemotron_init` now load dynamic
  GGML GPU plugins, so Python/Dart/Rust/Go FFI consumers get real GPU execution
  instead of silent CPU fallback (#282, by @KevinAHM).
- **OmniVoice** — restore reference loudness after decode (#278); default the
  codec to GPU on CUDA, Metal/CPU unchanged (#280); honor open-time and live
  TTS seed controls, now deterministic (#281). All by @KevinAHM.

## Tests

New unit/live coverage: `test-audio-resample` (DC-gain incl. 48k→44.1k),
session dynamic-backend load path, `test-omnivoice-seed` (codes determinism),
sidon length-cap + against-upstream reference harness
(`tools/reference_backends/sidon_ref_dump.py`, `CRISPASR_SIDON_DUMP_HANDOFF`).

With thanks to external contributor **@KevinAHM** for PRs #277–#283.
