# MossFormer2-SR speech super-resolution (48 kHz)

Native ggml port of **MossFormer2_SR_48K** from
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
(Alibaba Tongyi Lab), a 110 M-parameter speech super-resolution (bandwidth
extension) model: band-limited speech — 8 or 16 kHz audio, or anything whose
content stops below 24 kHz — in, 48 kHz full-band speech out. It is not a
denoiser, but it is an audio → audio transform at 48 kHz, so it runs through
the same selectable enhancement switch (`--enhance`,
`crispasr_enhance_audio_spec`, `crispasr_session_set_enhance`), see
[cli.md](cli.md#speech-enhancement-denoise----enhance).

| | |
|---|---|
| Upstream code | `modelscope/ClearerVoice-Studio@6b3774dc`, `clearvoice/models/mossformer2_sr/` (`Mossformer` → `MossFormer_MaskNet`, `Generator`), `utils/decode.py` `decode_one_audio_mossformer2_sr_48k`, `utils/bandwidth_sub.py`, `dataloader/meldataset.py` `mel_spectrogram` |
| Weights | ModelScope / Hugging Face `MossFormer2_SR_48K` rev `39eb1f25`: `last_best_checkpoint_m.pt` (MaskNet, sha256 `6cbadb2b…ab36`) and `last_best_checkpoint_g.pt` (generator, sha256 `0bdd13c2…adda`) |
| License | Apache-2.0 |
| GGUF architecture | `mossformer2-sr` |
| Rate | 48 kHz mono in and out; other rates are resampled in (and back, except for `--enhance-output`) |
| Published GGUF | none yet — convert locally (below) |

## Convert

```bash
pip install torch numpy gguf
python models/convert-mossformer2-sr-to-gguf.py \
    --checkpoint-dir MossFormer2_SR_48K \
    --output mossformer2-sr-48k-f16.gguf --dtype f16           # f16 | f32
# Q8_0: quantize the F16 file (the arch guard quantizes the generator only)
crispasr-quantize mossformer2-sr-48k-f16.gguf mossformer2-sr-48k-q8_0.gguf q8_0
```

Both checkpoints are read with `torch.load(weights_only=True)`; the directory's
`last_best_checkpoint` names them (mask net first, generator second). The
MaskNet tensors use the [MossFormer2-SE](mossformer2-se.md) names (that
converter's map is imported), the generator is stored under `gen.*` with its
weight norm folded (`torch._weight_norm`, what upstream's `remove_weight_norm()`
computes) and every convolution as a 2-D matmul matrix: Conv1d
`[Cout, Cin, K]` → `[K·Cout, Cin]` (one `[Cout, Cin]` block per tap),
ConvTranspose1d `[Cin, Cout, K]` → `[Cout·K, Cin]` (the layout
`ggml_col2im_1d` consumes). `--dtype f16` stores only `mul_mat` matrices as
F16; norms, biases, PReLU slopes, Snake alphas, OffsetScale gamma/beta and the
depthwise kernels stay F32.

| File | Size |
|---|---|
| `mossformer2-sr-48k-f32.gguf` | 438.6 MB |
| `mossformer2-sr-48k-f16.gguf` | 222.9 MB |
| `mossformer2-sr-48k-q8_0.gguf` (from F16, generator only) | 171.7 MB |

`crispasr-quantize` quantizes only the generator (`gen.*`) for this
architecture and keeps the MaskNet at source precision — see
[Quantization](#quantization).

## Use

```bash
# 8 / 16 kHz speech -> 48 kHz file (the input is decoded at 48 kHz)
crispasr --enhance mossformer2-sr-48k-f16.gguf --enhance-output speech48k.wav -f call8k.wav
```

```python
import crispasr
wide = crispasr.enhance_audio(narrow_48k, "mossformer2-sr-48k-f16.gguf", sample_rate=48000)
```

`--enhance-output` decodes the input at the model's native 48 kHz and writes
the result at 48 kHz — no change to the enhancer was needed for that, the verb
already works at the native rate. `crispasr_enhance_audio_spec` (and every
binding's `enhance_audio`) keeps its same-rate contract: pass audio resampled
to 48 kHz to get 48 kHz back; at any other rate the result is resampled back to
that rate, which throws the new band away again. As an ASR pre-pass
(`--enhance` with `-m`, `set_enhance`) the 16 kHz recogniser input is
resampled to 48 kHz, extended and resampled back: harmless but pointless,
since the band the recogniser sees is the input's own band, which
`bandwidth_sub` keeps verbatim. Passing the GGUF to `-m` prints how to run it.

## What runs

Upstream's `decode_one_audio_mossformer2_sr_48k`, reproduced (ClearVoice loads
the input with `librosa.resample` to 48 kHz and does not normalise it for this
model):

1. **Log-mel** (`get_mel` → `mel_spectrogram`): reflect padding of
   (n_fft − hop) / 2 = 384 samples, `torch.stft(n_fft = 1024, hop = 256,
   periodic Hann, center = False)`, magnitude `sqrt(re² + im² + 1e-9)`, 80
   Slaney mel bins over **0–8 kHz** (`librosa.filters.mel`, here
   `core_mel::build_slaney_fb`), `log(clamp(x, 1e-5))`: `[T, 80]`,
   T = n / 256. Only 0–8 kHz is ever analysed — the model expects the content
   of a 16 kHz recording.
2. **MaskNet** (`MossFormer_MaskNet(80, 512, 80)`, one speaker):
   `GroupNorm(1, 80)` → `Conv1d(80, 512, 1)` + scaled sinusoidal positions →
   `Computation_Block` (24 × [FLASH_ShareA_FFConvM → Gated_FSMN_Block],
   LayerNorm, GroupNorm, skip) → PReLU → `Conv1d(512, 512)` → tanh · sigmoid
   gate → `Conv1d(512, 80)` → ReLU. The stack is the shared
   `src/mossformer2_blocks.{h,cpp}` with `fsmn_dilated = false`
   (`UniDeepFsmn`, as for SE); the module code is byte-identical to SE's apart
   from docstrings and the `num_spks = 1` head.
3. **Generator** (HiFi-GAN V1 layout, 1024 initial channels, Snake
   activations `x + sin²(αx) / (α + 1e-9)` instead of LeakyReLU):
   `Conv1d(80, 1024, 7)` → 4 × [Snake → `ConvTranspose1d` (×8, ×8, ×2, ×2;
   kernels 16, 16, 4, 4) → mean of 3 `ResBlock1` (kernels 3, 7, 11; dilations
   1, 3, 5; Snake before each conv)] → Snake → `Conv1d(64, 1, 7)` → tanh:
   256 T samples. Native: each Conv1d is a sum of per-tap matmuls over shifted
   views of the zero-padded input (no im2col buffer), each ConvTranspose1d a
   matmul + `ggml_col2im_1d` with the symmetric crop, Snake a fused CPU op
   (`core_act::snake_alpha` on other backends). The generator runs in chunks of
   320 frames with an 18-frame margin on each side (its receptive field is
   ~16 frames), which gives exactly the one-pass result while keeping memory
   bounded (`CRISPASR_MF2SR_GEN_CHUNK=<frames>`, 0 = one pass).
4. **`bandwidth_sub`**: `scipy.signal.stft` (256-point Hann) of the input, the
   cut-off `f_high` = first bin where the cumulative energy reaches 99.96 %;
   `lowpass(input) + highpass(generated)` with 4th-order Butterworth filters at
   `f_high`, zero-phase (`filtfilt`, odd padding), then a 100 ms linear
   cross-fade from the plain input. Native: `mf2sr::butter4` / `filtfilt` /
   `detect_bandwidth` in double precision, matching scipy 1.18 to 1e-12 on
   the coefficients (unit test). So the input's own band reaches the output
   unchanged; the model only adds the band above `f_high`.
5. Up to 20 s: one forward. Longer: 4 s windows at a 3 s stride with 0.5 s
   discarded at inner window edges after upstream's end padding (the window
   plan is `mf2se::segment_plan`, shared with SE); `bandwidth_sub` then runs
   over the padded whole, and the output is cut to the input length.

Deviations, for robustness only:

- Upstream's one-pass branch returns 256 · ⌊n / 256⌋ samples; the runtime
  zero-fills the < 256-sample tail so the output has exactly n samples (the
  parity reference is zero-padded the same way).
- Inputs shorter than one 1024-sample frame are zero-extended and trimmed
  (upstream fails). Inputs shorter than the 100 ms cross-fade use a truncated
  ramp (upstream fails).
- A cut-off of 0 Hz or of the Nyquist frequency — digital silence, or an input
  that is already full band — makes `scipy.signal.butter` raise upstream; the
  runtime returns the input unchanged.

Environment: `CRISPASR_MOSSFORMER2_SR_GPU=1` places the weights on the
preferred GPU backend (portable depthwise path and the unfused Snake; not
validated on a GPU here), `CRISPASR_MF2SR_BENCH=1` prints stage wall times,
`CRISPASR_MF2SR_GEN_CHUNK` sets the generator chunk.

## Parity

Reference: the original PyTorch models and decode wrapper,
`tools/reference_backends/mossformer2_sr.py` (torch 2.15 CPU, librosa 1.0,
scipy 1.18, `rotary-embedding-torch` 0.8.x, pinned upstream sources;
generator weight norm removed as upstream does). Native:

```bash
python tools/reference_backends/mossformer2_sr.py --upstream ClearerVoice-Studio \
    --checkpoint-dir MossFormer2_SR_48K --audio speech48k.wav --gguf ref.gguf [--gen-stages]
crispasr-diff mossformer2-sr mossformer2-sr-48k-f16.gguf ref.gguf speech48k.wav
```

Every stage runs from the native front end (no reference tensor is fed in).
Clip: `samples/jfk.wav` (16 kHz) → `librosa.resample` to 48 kHz, 11 s, one
forward (T = 2062 frames); cosine over the whole tensor:

| Stage | F32 | F16 | Q8_0 |
|---|---|---|---|
| log-mel `[T, 80]` | 1.00000000 | 1.00000000 | 1.00000000 |
| GroupNorm | 1.00000000 | 1.00000000 | 1.00000000 |
| encoder + positions | 1.00000000 | 0.99999999 | 0.99999999 |
| layer 0 FLASH `to_hidden` | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 FLASH `to_qk` | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 FLASH output | 1.00000000 | 0.99999999 | 0.99999999 |
| layer 0 FSMN conv1 + PReLU | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 `UniDeepFsmn` | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 `Gated_FSMN` | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 0 output | 1.00000000 | 0.99999999 | 0.99999999 |
| layer 5 output | 1.00000000 | 0.99999999 | 0.99999999 |
| layer 11 output | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 17 output | 1.00000000 | 1.00000000 | 1.00000000 |
| layer 23 output | 1.00000000 | 1.00000000 | 1.00000000 |
| MossFormerM (final LayerNorm) | 1.00000000 | 1.00000000 | 1.00000000 |
| Computation_Block | 1.00000000 | 0.99999999 | 0.99999999 |
| MaskNet output `[T, 80]` | 1.00000000 | 1.00000000 | 1.00000000 |
| generator output `gen_wave` | 0.99999978 | 0.99999840 | 0.99849211 |
| **end-to-end waveform** | **1.00000000** | **1.00000000** | **0.99999186** |

(`1.00000000` means ≥ 0.999999995.) `gen_wave` is the generator output
before `bandwidth_sub`; `e2e_wave` the final output.

Generator stages, 3 s clip (`--gen-stages`):

| Stage | F32 | F16 | Q8_0 |
|---|---|---|---|
| MaskNet output `[T, 80]` | 1.00000000 | 1.00000000 | 1.00000000 |
| `conv_pre` `[T, 1024]` | 1.00000000 | 0.99999999 | 0.99999999 |
| stage 0 ConvTranspose ×8 `[8T, 512]` | 1.00000000 | 1.00000000 | 0.99999753 |
| stage 0 resblocks | 1.00000000 | 0.99999999 | 0.99999081 |
| stage 1 ConvTranspose ×8 `[64T, 256]` | 1.00000000 | 0.99999999 | 0.99998609 |
| stage 1 resblocks | 1.00000000 | 0.99999998 | 0.99998753 |
| stage 2 ConvTranspose ×2 `[128T, 128]` | 1.00000000 | 0.99999998 | 0.99996924 |
| stage 2 resblocks | 1.00000000 | 0.99999998 | 0.99998308 |
| stage 3 ConvTranspose ×2 `[256T, 64]` | 1.00000000 | 0.99999994 | 0.99993533 |
| stage 3 resblocks | 0.99999998 | 0.99999936 | 0.99951849 |
| generator output `gen_wave` | 0.99999995 | 0.99999871 | 0.99900180 |
| **end-to-end waveform** | **1.00000000** | **1.00000000** | **0.99999328** |

Other clips, end-to-end and generator output:

| Clip | Stage | F32 | F16 | Q8_0 |
|---|---|---|---|---|
| VCTK p234 001–003, 16 kHz → 48 kHz, 13.2 s (one forward) | `gen_wave` | 1.00000000 | 0.99999884 | 0.99910967 |
| same | `e2e_wave` | 1.00000000 | 0.99999999 | 0.99988680 |
| VCTK p234 001–009, 16 kHz → 48 kHz, 43.5 s (segmented, 14 windows) | `e2e_wave` | 1.00000000 | 1.00000000 | 0.99994523 |

### Quantization

The generator turns small errors in its input into large ones in its
noise-like high band, so quantization is judged on `gen_wave` (the generated
band) rather than on the end-to-end waveform, which `bandwidth_sub` fills
mostly with the input's own band (and therefore sits at ≥ 0.9999 whatever is
quantized). Measured on the 3 s clip, `gen_wave` cosine:

| Q8_0 variant (from F16) | Size | MaskNet output | `gen_wave` |
|---|---|---|---|
| F16 (reference point) | 222.9 MB | 0.99999999 | 0.99999871 |
| MaskNet with the [MossFormer2-SE guard](mossformer2-se.md#quantization), generator Q8_0 | 148.1 MB | 0.9999986 | 0.9977 |
| same, generator F16 | 199.3 MB | 0.9999986 | 0.9986 |
| **MaskNet F16, generator Q8_0 (the arch guard)** | **171.7 MB** | 0.99999999 | **0.9990** |
| MaskNet F16, generator stage 0 F16, stages 1–3 Q8_0 | 210.6 MB | 0.99999999 | 0.9997 |

The MaskNet output at the SE guard's precision (cosine 0.9999986) already
costs the generated waveform 0.9986: the generator amplifies its input error
~1000×. Hence the arch guard in `crispasr-quantize` quantizes `gen.*` only.
Even so, Q8_0 `gen_wave` ranges 0.9985–0.9991 over the clips above (stage 3 of
the generator, the 64-channel resblocks, is where it drops below 0.9999); the
F16 file is the one to use when the generated band must match the original
closely, and F32 is the fastest file on CPU anyway (below).

## Quality

QUALITY_TEXT

## Speed and memory

SPEED_TEXT

## Tests

- `tests/test-mossformer2-sr.cpp` `[unit]`: the enhancer advertises
  `mossformer2-sr`; a weightless GGUF is rejected cleanly; `butter4`,
  `lfilter_zi` and `filtfilt` match scipy to 1e-10; `detect_bandwidth` finds a
  tone's band; silence and the degenerate cut-offs return the input; the
  log-mel frame count, silence floor and spectral peak.
- `[live]` (`MOSSFORMER2_SR_MODEL=<gguf>`, SKIP without it): 4 s of
  `samples/jfk.wav` gains > 100× energy above 9 kHz while the band below 6 kHz
  stays the input (cos > 0.999); a 64-frame chunked generator equals the
  one-pass one; sub-frame input, silence and 16 kHz callers keep the
  same-length contract.
- `tests/test-mossformer2-sr-converter.py`: the generator name map, the
  weight-norm fold against torch's own `remove_weight_norm`, the per-tap and
  transposed-conv matrix layouts against `torch.nn.functional.conv1d` /
  `conv_transpose1d`, the precision policy and a GGUF round trip.
- `tests/test-arch-backend-map.cpp`: `mossformer2-sr` in the shared arch table.
