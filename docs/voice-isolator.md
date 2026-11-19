# Inworld Voice Isolator (`voice-isolator`, #602)

[Inworld Voice Isolator v1](https://huggingface.co/inworld/voice-isolator)
(Apache-2.0 weights and code, 48 kHz) isolates the foreground voice of noisy
speech. CrispASR runs it natively in ggml as a separation backend with two
stems: `voice`, and `background` = input − voice (upstream's own definition, so
the two stems always sum back to the mono input).

| checkpoint | upstream file | GGUF | behaviour |
|---|---|---|---|
| `voice-isolator-v1` | `denoiser.safetensors` | `voice-isolator-{f16,q8_0,q4_k}.gguf` | offline, two-sided; 8 s windows, 2 s crossfade |
| `voice-isolator-v1-realtime` | `denoiser_realtime.safetensors` | `voice-isolator-realtime-{f16,q8_0}.gguf` | causal; trained look-ahead 0/2/4 frames (default 2 = 20 ms) |

GGUFs: [cstr/inworld-voice-isolator-GGUF](https://huggingface.co/cstr/inworld-voice-isolator-GGUF).

## Usage

```bash
# CLI: writes noisy_voice.wav + noisy_background.wav (mono, 48 kHz, 16-bit)
crispasr --separate -m voice-isolator-q8_0.gguf -f noisy.wav [--stems voice] [--sep-output-dir out/]
crispasr --separate --backend voice-isolator --auto-download -f noisy.wav     # registry: q8_0 default
crispasr --separate --backend voice-isolator-realtime --auto-download -f noisy.wav

# server
crispasr --server -m asr.gguf --separate-model voice-isolator-q8_0.gguf
curl -F file=@noisy.wav -F stems=voice http://localhost:8080/v1/audio/separation

# bindings (Session C ABI): stems come back stereo-interleaved with L = R
python -c "import crispasr; s = crispasr.Session('voice-isolator-q8_0.gguf'); print(s.separate(pcm48k_stereo).keys())"
```

Any input rate/channel count is accepted; the CLI decodes and resamples to
48 kHz mono (channels averaged, as upstream `to_mono`). The C API also offers
`voice_isolator_isolate_mono()` which, like upstream `VoiceIsolator.isolate`,
resamples with torchaudio's `kaiser_best` (`core_torchaudio::kaiser_best_kernel`)
both ways and returns the voice at the input rate — the building block for
cleaning audio before ASR.

## Architecture (from `inworld/voice_isolation/model.py`)

* STFT n_fft 2048, hop 480, periodic Hann, `center=True` reflect padding
  (constant padding when the clip is ≤ 1024 samples).
* 46 bands (`band_edges` of plan 100 Hz→1 kHz … 2 kHz→24 kHz), each band's
  interleaved (re, im) bins → LayerNorm → Linear → 512-wide token, plus a
  learned band embedding (+ look-ahead embedding for the realtime model).
* 7 blocks `T M T M T M T`:
  * T: pre-norm time transformer (split-half RoPE base 10000, *exclusive self
    attention*: each head output minus its projection on its own value) then a
    pre-norm band transformer (plain MHA); exact-erf GELU FFNs. Realtime: causal
    time attention with a 1200-frame window.
  * M: bidirectional Mamba-1 along time (d_state 16, conv 4, expand 2;
    realtime: forward only) then bidirectional Mamba across bands, each with
    `output_proj` + LayerNorm and an outer residual. Runs on ggml
    `ssm_conv`/`ssm_scan` (softplus of Δ inside the scan, as upstream applies it
    before the scan).
* Decoder: constant FiLM (`condition([1, 0])`, folded at load) and per-band
  heads → `[bins, frames, 4]`; the correction `S·2tanh(m) + bounded residual`
  and iSTFT run on the host in F32. Output = input + iSTFT(correction).
  Realtime reads frame t's head output from frame t + look-ahead.

Each block axis runs as its own ggml graph over groups of ≤ 8192 tokens
(sequences are independent, so this is exact); CLI peak RSS is ~0.6 GB.

## Validation (2026-11-19, CPU)

Reference: the unmodified upstream package at `inworld-ai/voice-isolator@c73b3fab`
via `tools/reference_backends/voice_isolator.py` (hooks only). Input: `samples/jfk.wav`
mixed with synthetic music (−3 dB) and pink noise (−10 dB), resampled to 48 kHz
by upstream `audio.resample`. Fixtures: `cstr/crispasr-regression-fixtures`,
`voice-isolator/jfk-noisy-3s/{offline,realtime}-ref.gguf`.

```bash
crispasr-diff voice-isolator voice-isolator-f16.gguf offline-ref.gguf -
```

Stage diff, offline, 7.5 s single chunk (751 frames):

| stage | F32 cos | F16 | Q8_0 | Q4_K |
|---|---|---|---|---|
| stft | 1.000000 (rel 2.8e-6) | | | |
| encode | 0.999999 | 0.999999 | 0.999999 | 0.999999 |
| block0 … block6 (chained) | ≥ 0.99993 | ≥ 0.99993 | ≥ 0.99986 | ≥ 0.9975 |
| block1 … block6 (teacher-forced) | 1.000000 (rel ≤ 5e-7) | | | |
| voice | 0.999997 | 0.999997 | 0.999981 | 0.998834 |
| **voice SI-SDR vs upstream** | **52.4 dB** | **52.4 dB** | **44.3 dB** | **26.3 dB** |

Magnitudes agree (|voice| 43.34 vs 43.36). The chained residual comes from the
encoder's per-band LayerNorm amplifying ~3e-6 STFT rounding in near-silent
narrow bands; every block reproduces the reference to F32 rounding when fed the
reference input. Realtime (3 s, causal, look-ahead 2): F16 voice cos 0.999999,
SI-SDR 56.2 dB vs upstream; Q8_0 46.7 dB.

End to end (`crispasr --separate`, full 11 s → two offline windows), SI-SDR vs the
clean `jfk.wav`: noisy input 2.2 dB → F16 11.6 dB, Q8_0 11.5 dB, Q4_K 11.1 dB
(upstream on the 3 s segment: offline 11.0 dB, realtime 15.3 dB). ASR
(`ggml-tiny.bin`): noisy input "…fellow Americans, **and** not … **and** what you
can do…" → isolated "…fellow American, ask not … ask what you can do for your
country." (parakeet-ctc-0.6b is exact on all of them).

## Known gaps

* No push/pull streaming session: the realtime checkpoint runs as a causal
  render over windows (default 8 s / 2 s), not upstream `streaming.Stream`
  (6 s attention lanes with carried Mamba state, block-wise output).
* Realtime full-clip SI-SDR vs clean degrades after ~4 s on the 11 s fixture
  (1.7 dB overall, ASR still correct) although the 3 s causal forward matches
  upstream at 56 dB; upstream was not run on the later segment (its CPU
  reference needs ~4 GB RSS), so model behaviour vs a runtime issue there is
  unresolved.
* `CRISPASR_VOICE_ISOLATOR_CHUNK_SECONDS` / `_OVERLAP_SECONDS` override the render
  window (debug); `CRISPASR_VOICE_ISOLATOR_VERBOSE=1` prints per-chunk timing.
* GPU backends are selected with `--gpu` but not yet qualified.
