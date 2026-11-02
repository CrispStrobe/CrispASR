# MossFormerGAN-SE speech enhancement

Native ggml port of **MossFormerGAN_SE_16K** from
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
(Alibaba Tongyi Lab), a 3.1 M-parameter 16 kHz speech-enhancement (denoising)
generator. It is one of the methods of the selectable enhancement pre-pass
(`--enhance`, `crispasr_session_set_enhance`, `crispasr_enhance_audio_spec`),
see [cli.md](cli.md#speech-enhancement-denoise----enhance).

| | |
|---|---|
| Upstream code | `modelscope/ClearerVoice-Studio@6b3774dc`, `clearvoice/models/mossformer_gan_se/generator.py` (`SyncANet`) |
| Weights | Hugging Face `alibabasglab/MossFormerGAN_SE_16K` rev `dcb1b20e`, `last_best_checkpoint.pt` (generator only) |
| License | Apache-2.0 (code and model card); no use restriction on the card |
| GGUF architecture | `mossformergan-se` (converter layout 3) |
| Rate | 16 kHz mono; other rates are resampled in and out by the enhancer |
| Published GGUF | none yet — convert locally (below) |

## Convert

The converter reads the checkpoint with `torch.load(weights_only=True)`; the
discriminator checkpoint (`last_best_checkpoint.disc.pt`) is training-only and
not needed.

```bash
pip install torch numpy gguf
python models/convert-mossformergan-se-to-gguf.py \
    --checkpoint MossFormerGAN_SE_16K/last_best_checkpoint.pt \
    --output mossformergan-se-16k-f16.gguf --outtype f16      # f32 | f16 | q8_0
```

`--outtype` applies to matmul weights only (Linear, per-tap conv matrices,
ConvTranspose taps, attention 1x1 convs). Norms, PReLU slopes, biases,
depthwise kernels and the 3-channel input conv always stay F32.

| File | Size |
|---|---|
| `mossformergan-se-16k-f32.gguf` | 12.6 MB |
| `mossformergan-se-16k-f16.gguf` | 7.5 MB |
| `mossformergan-se-16k-q8_0.gguf` | 5.1 MB |

F16 is the recommended file: it is indistinguishable from F32 on the parity
table below. Q8_0 still clears every stage at cosine > 0.999.

## Use

```bash
# write the enhanced audio only
crispasr --enhance mossformergan-se-16k-f16.gguf --enhance-output clean.wav -f noisy.wav
# denoise, then transcribe with any ASR backend
crispasr --enhance mossformergan-se-16k-f16.gguf -m ggml-base.en.bin -f noisy.wav
```

```python
import crispasr
clean = crispasr.enhance_audio(noisy, "mossformergan-se-16k-f16.gguf", sample_rate=16000)
s = crispasr.Session("ggml-base.en.bin")
s.set_enhance("mossformergan-se-16k-f16.gguf")      # pre-pass before every transcribe
```

The server takes the same path in its `enhance` form field. Passing the GGUF to
`-m` (or `crispasr_session_open`) prints how to run it instead: it is not a
transcription model.

## What runs

The decode wrapper is upstream's `decode_one_audio_mossformergan_se_16k`
(`clearvoice/utils/decode.py`), reproduced exactly:

1. `norm_factor = sqrt(N / sum(x^2))` over the **whole** input; digital
   silence returns zeros (upstream would divide by zero).
2. Inputs longer than 10 s are decoded in 10 s windows with a 7.5 s stride;
   1.25 s is dropped at each inner window edge, the tail window starts
   1.25 s early. Up to 10 s is one forward pass.
3. Each window is padded with its own beginning to a multiple of the 100-sample
   hop, then `torch.stft(n_fft=400, hop=100, win=400, periodic Hamming,
   center=True, reflect)`, magnitude compressed to `|X|^0.3` keeping the phase.
4. `SyncANet`: dense encoder (1x1 conv, 4-layer dilated dense net with FSMN,
   stride-2 conv to 101 bins) → 6 blocks of (intra-frequency MossFormer +
   inter-frame MossFormer, each with FSMN gating and squeeze-excitation, then
   full-band multi-head attention over frames) → mask decoder and complex
   decoder (dilated dense net + sub-pixel upsampling back to 201 bins).
5. `out = mask * |X_c| * e^{j angle(X_c)} + complex`, uncompressed with
   `^(1/0.3)`, `torch.istft` (window-envelope normalised), divided by
   `norm_factor`, cropped to the input length.

Native graph notes (`src/mossformergan_se.cpp`):

- Activations are channels-fastest (torch `[C, T, F]` is ggml `[C, F, T]`), so
  Linear / 1x1 conv are plain `mul_mat`, per-position norms are `ggml_norm`,
  and the sub-pixel shuffle is a reshape.
- 2-D convs are per-tap `mul_mat`s over flat shifted views of a zero-padded
  input; the dilated dense net keeps one padded, channel-concatenated input.
  (An im2col for the 256-channel dense layer would be ~2 GB on a 10 s window.)
- Depthwise FSMN / ConvModule kernels run on `ggml_conv_2d_dw_direct`'s
  channels-contiguous path; the MossFormer rotary embedding is `ggml_rope_ext`
  (interleaved pairs, 32 of 128 dims, theta 10000).
- The relu² gated attention has two T×T products per block (the cross-row term
  of the intra pass and the quadratic term of the inter pass). They are
  evaluated in row chunks of at most 256 MiB and accumulated with `ggml_acc`.
- CPU is the validated path. `CRISPASR_MOSSFORMERGAN_SE_GPU=1` places the
  weights on the preferred GPU backend through the scheduler (unvalidated).
  `MOSSFORMERGAN_SE_PROFILE=1` prints per-op wall time.

## Parity

Reference: the original PyTorch generator and decode wrapper
(`tools/dump_mossformergan_se_reference.py`, torch 2.15 CPU,
`rotary-embedding-torch==0.8.3` as pinned upstream). Native:
`build/bin/mossformergan-se-parity model.gguf ref_dir`.

Per-stage cosine, one forward on the first 4 s of `jfk_noisy.wav`
(T = 641 frames; stages are taken from the reference's own model input, the
STFT row from the native front end):

| Stage | F32 | F16 | Q8_0 |
|---|---|---|---|
| input spectrum (native STFT + compression) | 1.00000000 | 1.00000000 | 1.00000000 |
| encoder conv_1 | 1.00000000 | 1.00000000 | 1.00000000 |
| encoder dilated dense | 1.00000000 | 1.00000000 | 0.99999111 |
| encoder out | 1.00000000 | 0.99999999 | 0.99997907 |
| block 0 intra MossFormer | 1.00000000 | 0.99999999 | 0.99996828 |
| block 0 inter MossFormer | 1.00000000 | 0.99999999 | 0.99997289 |
| block 0 attention projection | 1.00000000 | 0.99999999 | 0.99996465 |
| block 0 | 1.00000000 | 0.99999999 | 0.99997671 |
| block 5 | 1.00000000 | 0.99999997 | 0.99995009 |
| mask | 1.00000000 | 0.99999999 | 0.99995663 |
| complex | 1.00000000 | 0.99999984 | 0.99945353 |
| final real/imag | 1.00000000 | 0.99999992 | 0.99972694 |

(`1.00000000` means ≥ 0.999999995; F32 max |diff| ≤ 3e-5 on every stage.)

End-to-end waveform cosine against `decode_one_audio_mossformergan_se_16k`:

| Clip | F32 | F16 | Q8_0 |
|---|---|---|---|
| 4 s (one window) | 1.00000000 (max \|d\| 4.2e-7) | 0.99999999 | 0.99997698 |
| 10 s (one window, T = 1601) | 1.00000000 (max \|d\| 4.2e-7) | 0.99999999 | 0.99997590 |
| 11 s (segmented: 10 s + tail) | 1.00000000 (max \|d\| 4.2e-7) | 0.99999999 | 0.99997419 |

## Quality

`samples/jfk.wav` with added noise, enhanced with the F16 GGUF (CLI), scored
against the clean recording:

| Noise | Input SNR / SI-SDR | Enhanced SNR / SI-SDR |
|---|---|---|
| white, σ = 0.05 | 9.05 / 9.04 dB | 16.95 / 16.87 dB |
| pink (1/f), 5 dB | 5.00 / 5.00 dB | 15.74 / 15.65 dB |
| mains hum + harmonics + white, 0 dB | 0.00 / 0.01 dB | 15.48 / 15.65 dB |
| babble-like (6 shifted, reversed copies of the speech), 5 dB | 5.00 / 4.99 dB | 7.51 / 6.77 dB |

The PyTorch original scores the same on the white-noise clip (16.95 / 16.87 dB).
Babble made of the speaker's own voice is the expected hard case for a
single-channel enhancer.

On the white-noise clip `tiny.en` transcribes the noisy and the enhanced audio
identically ("And so my fellow American, ask not what your country can do for
you. Ask what you can do for your country."), so measure WER on your own audio
before enabling a denoiser in front of ASR.

## Speed and memory

The network is dominated by two T x T relu² products per block (T = frames,
1601 for a 10 s window), so cost grows quadratically up to the 10 s window and
linearly beyond it (segmentation).

Measured on a shared 4-core Xeon @ 2.1 GHz (AVX2) that was heavily loaded by
other jobs (load average 8-10), so absolute numbers are only indicative. To
give both sides the same background load, PyTorch (original decode wrapper,
`torch.set_num_threads(2)`) and the native port (F16, 2 threads) ran
concurrently on the same 10 s clip:

| 10 s clip, 2 threads | Wall time | Peak RSS |
|---|---|---|
| PyTorch 2.15 CPU | 564 s | — |
| native F16 | 585 s | 1.33 GB |

On a 4 s clip (T = 641) at a quieter moment the native F32 decode took
24 s against PyTorch's 39 s (2 threads each, not concurrent). The
T x T score tensors are evaluated in ≤ 160 MiB row chunks, which keeps native
peak memory near 1.3 GB for a full 10 s window; PyTorch materialises the full
`[101, T, T]` score tensors.

## Tests

- `tests/test-mossformergan-se-converter.py` (unit): tensor layouts, precision
  policy, rejection of foreign checkpoints, verbatim lifting of upstream decode
  functions in the reference dumper.
- `test-mossformergan-se "[unit]"`: the enhancer table advertises
  `mossformergan-se`; a weightless GGUF is rejected cleanly through the C ABI.
- `MOSSFORMERGAN_SE_MODEL=mossformergan-se-16k-f16.gguf test-mossformergan-se "[live]"`:
  segmented 11 s decode gains > 5 dB SNR on jfk + white noise; 48 kHz input,
  a 150-sample input and digital silence keep the same-length contract.
- `mossformergan-se-parity` (manual): the tables above.
