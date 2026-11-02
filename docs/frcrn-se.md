# FRCRN_SE_16K speech enhancement (`frcrn-se`)

`frcrn-se` is a native ggml port of **FRCRN_SE_16K** from
[modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
(code at commit `6b3774dc`, weights
[`alibabasglab/FRCRN_SE_16K`](https://huggingface.co/alibabasglab/FRCRN_SE_16K)
revision `3766e6a6`, both Apache-2.0). It is a single-channel speech
denoiser at 16 kHz and runs through CrispASR's selectable enhancement
pre-pass: the CLI's `--enhance`, the Session ABI's
`crispasr_session_set_enhance`, the one-shot `crispasr_enhance_audio_spec`,
the server's `enhance` field and every wrapper. It is not a transcription
model; opening its GGUF as a session or `--backend frcrn-se` explains that and
exits.

No GGUF is published yet, so there is no registry name — convert locally.

## Convert

```sh
python models/convert-frcrn-se-to-gguf.py \
    --checkpoint FRCRN_SE_16K/last_best_checkpoint.pt \
    --output frcrn-se-16k-f16.gguf --dtype f16      # or --dtype f32
```

`last_best_checkpoint.pt` is the file in the Hugging Face repo above. The
converter loads it with `torch.load(weights_only=True)` and imports no
upstream code; it needs `torch`, `numpy`, `scipy` and `gguf`.

| File | Size | Notes |
|---|---:|---|
| `frcrn-se-16k-f32.gguf` | 58.9 MB | everything F32 |
| `frcrn-se-16k-f16.gguf` | 31.8 MB | conv and FSMN matrices F16; STFT kernels, folded BN, SE layers, biases, FSMN memory taps F32 (recommended) |

| `frcrn-se-16k-q8_0.gguf` | 23.9 MB | `crispasr-quantize frcrn-se-16k-f16.gguf frcrn-se-16k-q8_0.gguf q8_0`: decoder conv kernels and SE gates Q8_0, everything else as in F16 |

**Q8_0 keeps the sensitive tensors at F16.** A plain Q8_0 is wrong inside:
the default name rules quantize the Gated-FSMN `lin`/`proj` matrices, and with
those in Q8_0 the second UNet's stages fall to cosine 0.25 against PyTorch
(the mask to 0.50). The end-to-end waveform still reads 0.9999, because the
broken bins carry little energy, but the network is not computing the right
thing. `crispasr-quantize` therefore uses an allow-list for `frcrn-se`: only
the decoder complex-conv kernels (`u*.dec.*.w_re/w_im[_skip]`) and the SE gate
matrices are quantized; encoder conv kernels, FSMN projections and memory
taps, STFT kernels and folded BatchNorm stay at source precision.

Measured on the CPU, F16 source, against the original PyTorch path:

| Variant | Size | Min stage cosine | End-to-end |
|---|---:|---:|---:|
| default name rules (FSMN + SE quantized) | 30.7 MB | 0.249 (`u2.out`) | 0.99989 |
| FSMN `lin`/`proj` kept F16 | 31.5 MB | 0.99983 | 0.9999999 |
| + all conv kernels Q8_0 | 20.6 MB | 0.99831 | 0.9999976 |
| + decoder kernels kept F16 | 28.2 MB | 0.99908 | 0.9999996 |
| + only u2 encoder kernels kept F16 | 22.2 MB | 0.99803 | 0.9999979 |
| + encoder layers 4–6 kept F16 | 22.1 MB | 0.99783 | 0.9999971 |
| **+ all encoder kernels kept F16 (shipped rule)** | **23.9 MB** | **0.99972** | **0.9999985** |

The shipped rule also passes the windowed paths: 11 s forced windows min stage
0.99971 (end-to-end 0.9999981), 3 s forced windows 0.99977, and the 132 s clip
(176 windows) end-to-end 0.9999977. Denoising quality is unchanged: jfk_noisy
9.05 → 17.27 dB SNR, the same as F16 and PyTorch.

What the converter rewrites (all exact):

- **STFT.** ConvSTFT/ConviSTFT kernels are computed exactly like upstream's
  `init_kernels` (640-sample sqrt periodic Hann, 320 hop, 640-point rfft basis,
  `pinv` for the inverse) and stored as F32 tensors `stft.fwd`, `stft.inv`,
  `stft.window`.
- **Complex convs** keep their two real kernels, flattened to `(C_out, C_in·k_f·k_t)` rows.
- **Complex transposed convs** (stride 2 in frequency, padding 1 in time) become
  stride-1 polyphase convs: output channel `r·C_out + c` is frequency phase `r`,
  time taps are flipped, frequency taps re-indexed; the runtime interleaves the
  phases and crops to `2·(F_in−1)+k_f` rows.
- **Conv bias + eval BatchNorm** fold into a per-(part, channel) affine; complex
  conv bias contributes `b_re − b_im` to the real and `b_re + b_im` to the imaginary part.

## Run

```sh
# write the enhanced audio, no transcription
crispasr --enhance frcrn-se-16k-f16.gguf --enhance-output clean.wav -f noisy.wav

# denoise, then transcribe
crispasr --enhance frcrn-se-16k-f16.gguf -m ggml-base.en.bin -f noisy.wav
```

```python
import crispasr, soundfile as sf
noisy, sr = sf.read("noisy.wav", dtype="float32")           # mono
clean = crispasr.enhance_audio(noisy, "frcrn-se-16k-f16.gguf", sample_rate=sr)
with crispasr.Session("ggml-base.en.bin") as s:
    s.set_enhance("frcrn-se-16k-f16.gguf")                   # sticky pre-pass
    for seg in s.transcribe(noisy, sample_rate=sr):
        print(seg.text)
```

Any input rate works (the enhancer resamples to 16 kHz and back); the output
has the input's length and timeline.

### What one call does (upstream's inference path)

1. `audio_norm` from ClearerVoice's DataReader: scale to −25 dB RMS, then
   again on the above-average-power samples; the factor is undone at the end.
2. `decode_one_audio_frcrn_se_16k` padding, kept bug-for-bug (for example an
   11 s clip is padded with 20000 zeros, not to the next stride).
3. Inputs up to **120 s** run as one pass. The SE layers average over the whole
   clip, so this pass cannot be split without changing the result. Longer
   inputs are decoded in 1 s windows with a 0.75 s stride, keeping the middle
   of each window.

### Memory and environment

The one-pass graph needs about **0.46 MB of compute buffer per 20 ms frame**
plus ~80 MB (311 MiB for 10 s, 363 MiB for 11 s, ~2.9 GB extrapolated to
the 120 s limit) on top of the weights. Large
per-layer work runs in time chunks so the im2col workspace stays bounded.

| Variable | Effect |
|---|---|
| `CRISPASR_FRCRN_SE_ONE_PASS_S` | lower the 120 s one-pass limit to save memory (longer inputs use the windowed path; differs from upstream between the two limits) |
| `CRISPASR_FRCRN_SE_CHUNK_MB` | per-chunk workspace budget (default 96) |
| `CRISPASR_FRCRN_SE_GPU=1` | try the GPU backend (CPU is the default; GPU parity is unverified) |
| `CRISPASR_FRCRN_SE_FORCE_SEGMENT=1` | always use the windowed path (testing) |
| `CRISPASR_FRCRN_SE_DUMP_DIR` | write the first pass's stage tensors as `.npy` (parity) |
| `CRISPASR_FRCRN_SE_PROFILE=1` | per-op wall time to stderr |

## Architecture

DCCRN-style (`frcrn.py`): ConvSTFT → UNet₁ → UNet₂ (fed UNet₁'s raw output) →
mask `tanh(u₂) + tanh(u₁)` → complex multiply with the noisy spectrum →
ConviSTFT. Each UNet (`unet.py`, depth 14, 128 channels) has 7 complex
conv encoders (kernel (5,2)/(2,2), stride (2,1)), complex SE layers on the
skips, frequency-wise FSMN memories (`ComplexUniDeepFsmn_L1`, causal order 20
along frequency, per frame) before encoders 1–6 and after decoders 0–5, a
two-layer time-causal FSMN at the bottleneck, and 7 complex transposed-conv
decoders. Frequency bins per level: 321 → 159 → 78 → 37 → 17 → 7 → 2 → 1.

The ggml graph keeps activations as `[T, F, C, re/im]` (ggml order). The two
parts go through each real kernel as a batch of two (im2col + mul_mat) and are
combined as `re = Wr·xr − Wi·xi`, `im = Wr·xi + Wi·xr`. FSMN depthwise memories
use `ggml_ssm_conv`.

## Parity

Reference: `tools/dump_frcrn_se_reference.py` runs the original PyTorch model
through `decode_one_audio_frcrn_se_16k` and dumps 88 named stages (STFT, every
encoder, SE, FSMN and decoder output of both UNets, mask, masked spectrum,
iSTFT, final output). `tools/frcrn_se_parity.py` runs the native enhancer
through the C ABI and compares. Cosine similarity, CPU, `jfk_noisy.wav`
(JFK + white noise, 9.05 dB SNR):

| Clip | Path | Weights | Min stage cosine (stage) | End-to-end cosine | Agreement SNR |
|---|---|---|---|---:|---:|
| 3 s | one pass (T = 224) | F32 | 0.9999999 (u2.dec1) | 1.0000000 | 100.3 dB |
| 3 s | one pass | F16 | 0.9998533 (u2.se_dec0) | 1.0000000 | 72.5 dB |
| 11 s | one pass (T = 611) | F32 | 0.9999998 (u2.fsmn_dec5) | 1.0000000 | 102.8 dB |
| 11 s | one pass | F16 | 0.9995920 (u2.fsmn_dec5) | 1.0000000 | 71.3 dB |
| 3 s | windowed (forced, 5 × 1 s) | F32 | 1.0000000 | 1.0000000 | 122.5 dB |
| 3 s | windowed (forced) | F16 | 0.9999889 (u2.fsmn_dec4) | 1.0000000 | 74.2 dB |
| 11 s | windowed (forced, 16 × 1 s) | F32 | 1.0000000 | 1.0000000 | 98.0 dB |
| 11 s | windowed (forced) | F16 | 0.9999892 (u2.fsmn_dec4) | 1.0000000 | 71.8 dB |
| 132 s | windowed (> 120 s, 176 × 1 s, not forced) | F32 | – | 1.0000000 | 121.8 dB |
| 132 s | windowed (> 120 s) | F16 | – | 1.0000000 | 72.1 dB |

Windowed rows compare the stages of the first 1 s window; "1.0000000" means
every stage rounds to that. The worst F16 stage is always in the second
UNet's decoder FSMNs, whose activations are heavy-tailed (u2.fsmn_dec5 on the
11 s clip: RMS 1.0, max 680).

Reproduce:

```sh
python -I tools/dump_frcrn_se_reference.py \
    --upstream ClearerVoice-Studio/clearvoice/clearvoice/models \
    --checkpoint FRCRN_SE_16K/last_best_checkpoint.pt \
    --audio noisy.wav --seconds 3 --out ref.npz
python tools/frcrn_se_parity.py --model frcrn-se-16k-f16.gguf --ref ref.npz \
    --lib build/src/libcrispasr.so
```

## Quality

SNR / SI-SDR against clean `samples/jfk.wav` (F16 GGUF, CLI `--enhance-output`):

| Input | Noisy SNR / SI-SDR | Enhanced SNR / SI-SDR | Gain (SI-SDR) |
|---|---:|---:|---:|
| `jfk_noisy.wav` (white, σ = 0.05) | 9.05 / 9.04 dB | 17.27 / 17.19 dB | +8.2 dB |
| white noise, 0 dB | 0.00 / −0.01 dB | 12.83 / 12.61 dB | +12.6 dB |
| white noise, 5 dB | 5.00 / 4.99 dB | 15.24 / 15.11 dB | +10.1 dB |
| white noise, 10 dB | 10.00 / 10.00 dB | 17.76 / 17.69 dB | +7.7 dB |
| pink noise, 0 dB | 0.00 / −0.03 dB | 13.76 / 13.58 dB | +13.6 dB |
| pink noise, 5 dB | 5.00 / 4.98 dB | 16.22 / 16.12 dB | +11.1 dB |
| pink noise, 10 dB | 10.00 / 9.99 dB | 18.62 / 18.57 dB | +8.6 dB |
| speech babble, 0 dB | 0.00 / −0.14 dB | 0.07 / −9.05 dB | −8.9 dB |
| speech babble, 10 dB | 10.00 / 9.96 dB | 1.12 / −3.33 dB | −13.3 dB |

The PyTorch original gives the same 17.27 / 17.19 dB on `jfk_noisy.wav`.
The babble here is four time-reversed, shifted copies of the same speaker: a
single-speaker enhancer keeps (and partly confuses) competing speech, so this
is a separation problem, not a denoising one. For comparison, the built-in
RNNoise reaches 13.83 / 13.70 dB on `jfk_noisy.wav`.

As with every enhancer, check the effect on your ASR model before enabling
it: on `jfk_noisy.wav` whisper tiny.en transcribes the noisy input correctly
and the FRCRN output with small errors ("And so am I, fellow American. Ask,
not.").

## Speed

10 s clip (one pass, T = 499 frames), 2 threads each, x86-64 with AVX-512,
measured back to back on a shared 4-core host under load average ~8, so the
absolute numbers are pessimistic; the ratio is the useful part:

| Engine | Wall time | Compute buffer |
|---|---:|---:|
| PyTorch 2.15 CPU, upstream `decode_one_audio_frcrn_se_16k` | 211.4 s | – |
| native, F32 GGUF | 98.5 s | 311 MiB |
| native, F16 GGUF | 56.8 s | 311 MiB |

About three quarters of native time is the conv GEMMs and their im2col (`CRISPASR_FRCRN_SE_PROFILE=1`).
Windowed decoding (> 120 s) reuses one cached 49-frame graph for every window.

## Tests

- `tests/test-frcrn-se-converter.py` (unit, model-free): STFT kernels and OLA
  round trip, polyphase transposed conv, BN folding, GGUF layout.
- `tests/test_frcrn_se_live.cpp` (`test-frcrn-se`, live): set
  `CRISPASR_MODEL_FRCRN_SE` (and optionally `CRISPASR_MODEL_WHISPER`); checks
  arch detection, session refusal, length at 16/44.1 kHz and sub-window clips,
  real denoising on JFK + noise, and the windowed path.

## License

Code and weights: Apache-2.0 (ClearerVoice-Studio). The converted GGUF records
the upstream commit, Hugging Face revision and the checkpoint's SHA-256.
