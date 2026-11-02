# MossFormer2-SE speech enhancement (48 kHz)

Native ggml port of **MossFormer2_SE_48K** from
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
(Alibaba Tongyi Lab), a 55 M-parameter full-band (48 kHz) speech-enhancement
(denoising) model. It is one of the methods of the selectable enhancement
pre-pass (`--enhance`, `crispasr_session_set_enhance`,
`crispasr_enhance_audio_spec`), see
[cli.md](cli.md#speech-enhancement-denoise----enhance).

| | |
|---|---|
| Upstream code | `modelscope/ClearerVoice-Studio@6b3774dc`, `clearvoice/models/mossformer2_se/` (`TestNet` → `MossFormer_MaskNet`) and `utils/decode.py` `decode_one_audio_mossformer2_se_48k` |
| Weights | ModelScope / Hugging Face `MossFormer2_SE_48K` rev `eff8c979`, `last_best_checkpoint.pt` (sha256 `03692b9f…27c6`) |
| License | Apache-2.0 |
| GGUF architecture | `mossformer2-se` |
| Rate | 48 kHz mono; other rates are resampled in and out by the enhancer |
| Published GGUF | none yet — convert locally (below) |

## Convert

```bash
pip install torch numpy gguf
python models/convert-mossformer2-se-to-gguf.py \
    --checkpoint MossFormer2_SE_48K/last_best_checkpoint.pt \
    --output mossformer2-se-48k-f16.gguf --dtype f16           # f16 | f32
# Q8_0: quantize the F16 file (the arch guard below keeps 52 matrices at F16)
crispasr-quantize mossformer2-se-48k-f16.gguf mossformer2-se-48k-q8_0.gguf q8_0
```

The checkpoint is read with `torch.load(weights_only=True)`. Tensor names match
the [MossFormer2-SS](mossformer2-ss.md) GGUF wherever the modules are shared;
the one structural difference is the FSMN memory (`blk.{i}.fsmn.mem.weight`,
see below). `--dtype f16` stores only `mul_mat` matrices as F16; norms, biases,
PReLU slopes, OffsetScale gamma/beta and every depthwise kernel stay F32.

| File | Size |
|---|---|
| `mossformer2-se-48k-f32.gguf` | 221.1 MB |
| `mossformer2-se-48k-f16.gguf` | 114.0 MB |
| `mossformer2-se-48k-q8_0.gguf` (from F16, with the guard) | 90.4 MB |

`crispasr-quantize` keeps `blk.*.flash.hidden.linear`, `blk.*.fsmn.to_u.linear`
and the mask head `mn.*` at source precision for this architecture. Without
that guard the Q8_0 file is 64.1 MB and still enhances well (end-to-end cosine
0.99997, same SNR), but the residual stream drops to cosine 0.9956 and one FSMN
sub-stage to 0.989 — see [Quantization](#quantization).

## Use

```bash
# write the enhanced audio only (decoded at 48 kHz, full band)
crispasr --enhance mossformer2-se-48k-f16.gguf --enhance-output clean.wav -f noisy.wav
# denoise, then transcribe with any ASR backend
crispasr --enhance mossformer2-se-48k-f16.gguf -m ggml-base.en.bin -f noisy.wav
```

```python
import crispasr
clean = crispasr.enhance_audio(noisy, "mossformer2-se-48k-f16.gguf", sample_rate=48000)
s = crispasr.Session("ggml-base.en.bin")
s.set_enhance("mossformer2-se-48k-f16.gguf")        # pre-pass before every transcribe
```

Passing the GGUF to `-m` (or `crispasr_session_open`) prints how to run it
instead: it is not a transcription model. `--enhance-output` reads the input at
the model's 48 kHz, so a 48 kHz recording keeps its full band; in front of ASR
the 16 kHz input is resampled to 48 kHz and back.

## What runs

Upstream's `decode_one_audio_mossformer2_se_48k` (and `compute_fbank` / `stft`
/ `istft` from `utils/misc.py`), reproduced:

1. The waveform is scaled by 32768 (`MAX_WAV_VALUE`). There is no input RMS
   normalisation for this model (ClearVoice's loader only normalises FRCRN and
   SS inputs).
2. Features: `torchaudio.compliance.kaldi.fbank` — 60 log-mel bins, 40 ms
   frames (1920 samples) every 8 ms (384), symmetric Hamming, pre-emphasis
   0.97, DC removal, 2048-point FFT, power spectrum, mel triangles from 20 Hz to
   Nyquist, `snip_edges` — plus `compute_deltas` (window 5, replicate padding)
   and delta-deltas: `[T, 180]`. Native: `core_kaldi::compute_fbank` with
   `int16_scale` and `WindowType::Hamming`.
3. `MossFormer_MaskNet`: `GroupNorm(1, 180)` → `Conv1d(180, 512, 1)` + scaled
   sinusoidal positions → `Computation_Block` (24 × [FLASH_ShareA_FFConvM →
   Gated_FSMN_Block], LayerNorm, GroupNorm, skip) → PReLU →
   `Conv1d(512, 1024)` of which only speaker 0 (channels 0–511) is used →
   tanh · sigmoid gate → `Conv1d(512, 961)` → ReLU: a magnitude mask `[T, 961]`.
4. `torch.stft(n_fft=1920, hop=384, win=1920, symmetric Hamming,
   center=False)`, multiplied by the mask (real and imaginary parts alike),
   `torch.istft(..., center=False, length=N)` (window-envelope normalised;
   samples past the last full frame are zero, exactly as torch), / 32768.
5. Up to 20 s: one forward. Longer: 4 s windows at a 3 s stride with 0.5 s
   discarded at inner window edges, after upstream's end padding (which, for
   lengths not on the stride grid, appends window + remainder zeros);
   windows whose kept span lies entirely in that padding are skipped, since
   upstream truncates those samples anyway. The output is cut to the input
   length (`ClearVoice.decode`).

The MossFormerM stack is the shared `src/mossformer2_blocks.{h,cpp}` library
(also used by MossFormer2-SS). Apart from docstrings and an unused `mask`
argument, `models/mossformer2_se/mossformer2_block.py` matches the SS file
except for the FSMN block: SE uses `Gated_FSMN_Block` with `UniDeepFsmn`
(`x + p + DepthwiseConv(k = 39)(p)`, `p = project(relu(linear(x)))`), SS the
dilated dense net (`Gated_FSMN_Block_Dilated`). `BlockHParams::fsmn_dilated`
selects between them; layer norms, conv modules, FLASH attention, rotary
embedding and the depthwise-conv kernels (CPU custom op, GPU-portable
`ssm_conv` split) are unchanged.

Deviations from upstream, all deliberate:

- **Kaldi dither.** Upstream fbank adds `1.0 · N(0, 1)` (int16 scale) to every
  frame sample from torch's unseeded global RNG, so two upstream runs never
  agree bit for bit. The port keeps the dither but draws it from a fixed-seed
  per-frame stream (`core_kaldi::FbankParams::dither`), so native runs are
  reproducible. Turning it off is *not* harmless: digital silence then gives
  `log(FLT_EPSILON)` features the model never saw, and on the 33 s clip below
  the last window (which contains end padding) fell from 11.5 to 6.8 dB SNR.
  `CRISPASR_MF2SE_DITHER=<amplitude>` overrides it (0 disables). Parity
  references and `crispasr-diff` use dither 0.
- Inputs shorter than one 1920-sample frame are zero-extended and trimmed
  (upstream raises). Digital silence returns digital silence.

Environment: `CRISPASR_MOSSFORMER2_SE_GPU=1` places the weights on the
preferred GPU backend with the portable depthwise path (validated here only on
CPU, via `CRISPASR_MF2_PORTABLE_DWCONV=1`: bit-for-bit the same cosines);
`CRISPASR_MF2SE_BENCH=1` prints stage wall times.

## Parity

Reference: the original PyTorch model and decode wrapper,
`tools/reference_backends/mossformer2_se.py` (torch 2.15 CPU, torchaudio
2.11.1's pure-Python `compliance.kaldi` / `compute_deltas`,
`rotary-embedding-torch` 0.8.x, pinned upstream sources, dither 0). Native:

```bash
python tools/reference_backends/mossformer2_se.py --upstream ClearerVoice-Studio \
    --checkpoint MossFormer2_SE_48K/last_best_checkpoint.pt --audio noisy48k.wav --gguf ref.gguf
crispasr-diff mossformer2-se mossformer2-se-48k-f16.gguf ref.gguf noisy48k.wav
```

Every stage runs from the native front end (no reference tensor is fed in).
Clip: `jfk_noisy.wav` resampled to 48 kHz, 11 s, one forward (T = 1371 frames);
cosine over the whole tensor:

| Stage | F32 | F16 | Q8_0 |
|---|---|---|---|
| Kaldi fbank `[T, 60]` | 1.00000000 | 1.00000000 | 1.00000000 |
| features + deltas `[T, 180]` | 1.00000000 | 1.00000000 | 1.00000000 |
| GroupNorm | 1.00000000 | 1.00000000 | 1.00000000 |
| encoder + positions | 1.00000000 | 0.99999999 | 0.99999999 |
| layer 0 FLASH `to_hidden` | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 FLASH `to_qk` | 1.00000000 | 0.99999999 | 0.99997672 |
| layer 0 FLASH output | 1.00000000 | 0.99999998 | 0.99998814 |
| layer 0 FSMN conv1 + PReLU | 1.00000000 | 0.99999999 | 0.99998694 |
| layer 0 `UniDeepFsmn` | 1.00000000 | 0.99999994 | 0.99988961 |
| layer 0 `Gated_FSMN` | 1.00000000 | 1.00000000 | 0.99999166 |
| layer 0 output | 1.00000000 | 0.99999997 | 0.99996332 |
| layer 5 output | 1.00000000 | 0.99999997 | 0.99995696 |
| layer 11 output | 1.00000000 | 0.99999996 | 0.99995689 |
| layer 17 output | 1.00000000 | 0.99999997 | 0.99995916 |
| layer 23 output | 1.00000000 | 0.99999997 | 0.99995673 |
| MossFormerM (final LayerNorm) | 1.00000000 | 1.00000000 | 0.99999232 |
| Computation_Block | 1.00000000 | 0.99999994 | 0.99997290 |
| mask `[T, 961]` | 1.00000000 | 0.99999995 | 0.99999340 |
| **end-to-end waveform** | **1.00000000** | **1.00000000** | **0.99999962** |

(`1.00000000` means ≥ 0.999999995; every F32 stage is ≥ 0.999999998, F32
end-to-end max |diff| 1.2e-6. A 3 s clip gives the same table to within
5e-6.) End-to-end on a 33 s clip through the **segmented** path (11 windows):
F32 1.00000000, F16 0.99999999, Q8_0 0.99998608.

With dither on (the runtime default) against upstream with its own random
dither: cosine 0.9999993 (11 s) and 0.9999998 (33 s); upstream with dither
against upstream without it: 0.999995 and 0.9968.

### Quantization

ggml's Q8_0 matmul rounds the activations to 8 bits per 32-block as well as the
weights. In this model the FLASH output `att_u · v · σ(att_v · u)` amplifies
that rounding: with every matrix at Q8_0, `to_hidden` is still at cosine
0.9999993 but the layer output falls to 0.9991, the residual stream to 0.9956
by layer 5 and `UniDeepFsmn` to 0.989. Weight-only Q8_0 of every stack matrix
in PyTorch keeps all stages ≥ 0.99999, so the weights are not the problem.
Keeping `flash.hidden.linear` + `fsmn.to_u.linear` (48 matrices) and the
4-matrix mask head at F16 gives the Q8_0 column above (every stage ≥ 0.99988)
for +26 MB; the mask head alone moves end-to-end from 0.99997 to 0.9999996.

## Quality

48 kHz clips built from `samples/jfk.wav` (resampled ×3, i.e. speech up to
8 kHz) plus full-band noise, scored against that clean signal (CLI,
`--enhance-output`):

| Clip | Input SNR / SI-SDR | PyTorch | native F32 | native F16 | native Q8_0 |
|---|---|---|---|---|---|
| white noise, 5 dB | 5.00 / 5.00 | 17.08 / 17.19 | 17.08 / 17.19 | 17.08 / 17.19 | 17.08 / 17.19 |
| pink (1/f) noise, 0 dB | 0.00 / 0.16 | 12.45 / 12.47 | 12.45 / 12.47 | 12.45 / 12.47 | 12.45 / 12.47 |
| `jfk_noisy.wav` ×3 resampled, 11 s | 9.23 / 9.22 | 16.69 / 16.79 | 16.70 / 16.80 | 16.70 / 16.80 | 16.71 / 16.81 |
| same ×3 concatenated, 33 s (segmented) | 9.23 / 9.22 | 16.39 / 16.46 | 16.39 / 16.46 | 16.39 / 16.46 | 16.39 / 16.46 |

(PyTorch with its default dither; without it the 33 s clip scores 15.37 /
15.44 dB.) On the 0 dB pink clip `tiny.en` transcribed the enhanced audio
*worse* than the noisy input (16 kHz → 48 kHz → 16 kHz round trip) — as with
every denoiser, measure WER on your own audio before enabling it in front of ASR.

## Speed and memory

Shared 4-core Xeon @ 2.1 GHz (AVX-512), load average ~3.5–4 from another job;
2 threads each, wall time of the whole decode (native: CLI including model
load and WAV I/O):

| Clip | PyTorch 2.15 | native F32 | native F16 | native Q8_0 |
|---|---|---|---|---|
| 11 s (one forward, T = 1371) | 3.1–3.6 s | 3.2 s | 7.7 s | 5.7 s |
| 33 s (segmented, 11 × 4 s windows) | 12.0 s | 11.3 s | 30.7 s | 22.1 s |

Native F32 stage split on the 11 s clip: features 79 ms, network 2.73 s,
STFT/mask/iSTFT 163 ms. As for MossFormer2-SS, F32 is the fastest file on CPU
(ggml's F16 and Q8_0 matmuls convert the activations per call); use F16 / Q8_0
for size or GPU. Peak RSS: 223 MB (F16, 11 s), 323 MB (F32, 11 s); segmented
decodes reuse one 4 s graph (188 MB with F16).

## Tests

- `tests/test-mossformer2-se-converter.py` (unit): name map, squeezes,
  precision policy, shared rotary buffer, rejection of foreign (e.g. dilated
  SS-style) tensors, GGUF round trip.
- `test-mossformer2-se "[unit]"`: the enhancer table advertises
  `mossformer2-se`; a weightless GGUF is rejected through the C ABI;
  `compute_deltas` against hand-computed torchaudio values; feature frame
  count and silence features; a unit mask through STFT → iSTFT reconstructs the
  input (incl. torch's zero tail); upstream segmentation padding.
- `MOSSFORMER2_SE_MODEL=mossformer2-se-48k-q8_0.gguf test-mossformer2-se "[live]"`:
  22 s at 16 kHz through the C ABI (segmented at 48 kHz) gains > 5 dB SNR
  (measured 9.07 → 16.42 dB); 48 kHz input, a 500-sample input and digital
  silence keep the same-length contract.
- `crispasr-diff mossformer2-se` (manual): the tables above.
