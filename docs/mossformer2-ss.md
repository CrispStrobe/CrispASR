# MossFormer2_SS_16K two-speaker speech separation

`mossformer2-ss` is a native ggml port of ClearerVoice-Studio's
**MossFormer2_SS_16K**: it splits a 16-kHz mono recording of two overlapping
talkers into `spk1` and `spk2`. It runs on the shared
[source-separation surface](source-separation-surface.md) (`--separate`, the
Session C ABI and `POST /v1/audio/separation`), not through `transcribe()`.

| | |
|---|---|
| Weights | [`alibabasglab/MossFormer2_SS_16K`](https://huggingface.co/alibabasglab/MossFormer2_SS_16K), revision `407cb030`, `last_best_checkpoint.pt` (sha256 `00a3a48b…a11a`) |
| Code | [`modelscope/ClearerVoice-Studio`](https://github.com/modelscope/ClearerVoice-Studio) @ `6b3774dc` (`clearvoice/models/mossformer2_ss`, `utils/decode.py`) |
| License | Apache-2.0 (model card and code) |
| GGUF architecture / backend | `mossformer2-ss` |
| Parameters | 55.7 M |
| I/O | 16 kHz mono in; two mono stems out, same length as the input |

There is no published GGUF yet, so the backend does not advertise
auto-download: convert the checkpoint locally.

## Convert

```bash
# 1. fetch the checkpoint (any way you like), e.g.
huggingface-cli download alibabasglab/MossFormer2_SS_16K --revision 407cb030 \
    --local-dir MossFormer2_SS_16K
# 2. convert (needs torch, numpy, gguf; the .pt is read with weights_only=True)
python models/convert-mossformer2-ss-to-gguf.py \
    --checkpoint MossFormer2_SS_16K/last_best_checkpoint.pt \
    --output mossformer2-ss-16k-f16.gguf            # --dtype f32 for the F32 file
# 3. optional Q8_0
./build/bin/crispasr-quantize mossformer2-ss-16k-f16.gguf mossformer2-ss-16k-q8_0.gguf q8_0
```

The converter verifies the checkpoint hash (override with
`--allow-checkpoint-mismatch`), converts only the `model` state dict (the
optimizer state in the `.pt` is dropped) and shortens the upstream names to fit
`GGML_MAX_NAME` (`mask_net.mdl.intra_mdl.mossformerM.layers.3.to_hidden.mdl.1.weight`
→ `blk.3.flash.hidden.linear.weight`; the identical per-layer rotary buffer is
stored once as `rotary.freqs`). Precision policy: only mul_mat matrices (linear
layers, k=1 convolutions, encoder/decoder filter banks) are F16; norm affines,
biases, PReLU slopes, the OffsetScale gamma/beta and every depthwise time
kernel stay F32. `crispasr-quantize q8_0` quantizes the 221 linear/pointwise
matrices and leaves the 16-tap encoder/decoder banks in F16.

| File | Size |
|---|---|
| `mossformer2-ss-16k-f32.gguf` | 223.0 MB |
| `mossformer2-ss-16k-f16.gguf` | 116.0 MB |
| `mossformer2-ss-16k-q8_0.gguf` | 65.9 MB |

## Run

```bash
crispasr --separate -m mossformer2-ss-16k-f16.gguf -f meeting.wav
#   -> meeting_spk1.wav, meeting_spk2.wav (16-bit mono, 16 kHz)
crispasr --separate -m mossformer2-ss-16k-f16.gguf -f meeting.wav --stems spk2 --sep-output-dir out/
```

Any input the CLI can decode is resampled to 16 kHz and downmixed to mono.
Through the Session ABI (`Session.separate()` in the bindings) the input is
interleaved stereo at 16 kHz, averaged to mono; each speaker comes back
duplicated into both interleaved channels so the separation ABI stays stereo.
The HTTP endpoint (`--separate-model mossformer2-ss-16k-f16.gguf`) returns
mono WAV parts.

C++ API (`src/mossformer2_ss.h`):

```c
mossformer2_ss_params p = mossformer2_ss_default_params(); // n_threads, use_gpu, gpu_device, no_segment
mossformer2_ss_context* ctx = mossformer2_ss_init_from_file("mossformer2-ss-16k-f16.gguf", p);
mossformer2_ss_result* r = mossformer2_ss_separate(ctx, pcm, n_samples, /*in_channels=*/1);
// r->n_sources == 2, r->n_channels == 1, r->sample_rate == 16000,
// r->sources[0..1] (n_samples each), r->source_names = {"spk1", "spk2"}
mossformer2_ss_result_free(r);
mossformer2_ss_free(ctx);
```

### What `separate()` does (upstream `decode_one_audio_mossformer2_ss_16k`)

The runtime reproduces the ClearerVoice decode function exactly, including its
quirks:

- input RMS is measured on the unpadded input;
- inputs up to 2 s (`one_time_decode_length`) run as one forward, zero-padded
  to the 2-s window;
- longer inputs are padded with upstream's expression (`t < window`:
  to `window`; `t < window + stride`: to `window + stride`; otherwise by
  `t - (t - window) // stride * stride` when not stride-aligned) and decoded in
  2-s windows with a 1.5-s stride, keeping each window's centre and dropping
  0.25 s (`give_up`) at the inner edges;
- each output is rescaled to the input RMS, with the output RMS taken over the
  padded length, and finally truncated to the input length (as
  `ClearVoice.decode()` does).

The only deviation: an all-zero output is left silent instead of becoming NaN
(upstream divides by its zero RMS). ClearVoice's data loader additionally
normalises the input level (`audio_norm`) before decoding and does not undo it
for this model; that only changes the output gain (the network is scale
invariant up to GroupNorm's eps), so the native runtime keeps the caller's
level. `no_segment=true` forces one forward over the whole padded input
(diagnostic).

## Architecture

```
wav [L] -> Conv1d(1, 512, k16, s8) + ReLU                        enc_out  [512, T]
        -> GroupNorm(1, 512) -> Conv1d(512, 512, 1) + ScaledSinu PE      mn_in
        -> 24 x (FLASH_ShareA_FFConvM -> Gated_FSMN_Block_Dilated)  l{i}_flash, l{i}_fsmn
        -> LayerNorm(eps 1e-6) -> GroupNorm(1, 512) + skip           intra_out, mdl_out
        -> PReLU -> Conv1d(512, 1024) -> per speaker: tanh(W x) * sigmoid(W' x)
        -> Conv1d(512, 512) + ReLU                                    masks [512, T, 2]
        -> masks * enc_out -> ConvTranspose1d(512, 1, k16, s8)       dec_spk1/2 [L]
```

FLASH layer: token shift of the first half of the channels; `to_hidden`
(512→2048, splits into v and u) and `to_qk` (512→128) FFConvM modules
(ScaleNorm → Linear → SiLU → x + depthwise conv k=17); OffsetScale into
quad/lin q/k; rotary embedding on the first 32 dims (interleaved pairs, theta
10000); `relu(q·k/256)²` attention inside 256-frame groups plus non-causal
linear attention over the whole sequence (`/n`), shared by v and u; gated
output `(att_u·v)·σ(att_v·u)` → `to_out` (1024→512) → residual. Gated FSMN
block: Conv1d 512→256 + PReLU → LayerNorm → `to_u`/`to_v` (LayerNorm FFConvM)
→ UniDeepFsmn_dilated on u (Linear+ReLU, projection, two-level dilated dense
depthwise net: k=39 dil 1, then k=39 dil 2 over `cat(out1, x)` with two inputs
per group, each InstanceNorm + per-channel PReLU) → `v·(u + fsmn) + x` →
LayerNorm → Conv1d 256→512 → residual.

### Code layout and reuse

- `src/mossformer2_blocks.{h,cpp}` — the MossFormerM stack shared by the
  MossFormer2 family (SS_16K here; SE_48K and SR_48K use the same
  `mossformer2_block.py`, `fsmn.py`, `conv_module.py`): `flash_layer`,
  `gated_fsmn_block`, `mossformer_m`, `computation_block`, `ffconvm`,
  `scale_norm`, `layer_norm`, `global_group_norm`, `prelu`, `dwconv`,
  `sinu_embedding`, plus load-time kernel helpers (`dw_kernels`,
  `tap_kernel`, `dilate_kernel`). Tensor names follow the converter's
  `blk.{i}.flash.*` / `blk.{i}.fsmn.*` scheme.
- `src/mossformer2_ss.{h,cpp}` — the SS encoder, mask head, decoder,
  decode-pipeline segmentation, Session/CLI API and `mossformer2_ss_diff`.

Depthwise time convolutions have two implementations. On the CPU backend a
`ggml_custom_4d` op convolves the `[C, T]` activations directly, vectorised
over channels, with tap-major kernels built at load time (no transposes or
padding copies). Every other backend uses `ggml_ssm_conv` on a transposed,
zero-padded copy, split into the kernel widths all GPU backends implement
(CUDA supports 3/4/5/9/15; 17 = 9+5+3, 39 = 15+15+9, 77 = 15·4+9+5+3), with the
dilation-2 kernel zero-inserted at load time.
`CRISPASR_MF2_PORTABLE_DWCONV=1` forces the portable path on the CPU; both are
covered by `tests/test-mossformer2-ss.cpp` against a naive grouped Conv1d and
both pass the full stage diff. The GPU path has not been run on a GPU yet.

## Parity

Reference: `tools/dump_mossformer2_ss_reference.py` runs the unmodified
upstream PyTorch model (source files hash-pinned) with forward hooks and the
verbatim `decode_one_audio_mossformer2_ss_16k`, writing `.npy` files and/or a
GGUF archive (also reachable as `tools/dump_reference.py --backend
mossformer2-ss`). `crispasr-diff mossformer2-ss MODEL REF.gguf x` replays the
reference input through the native graph and compares every stage present
(global cosine, per-row minimum cosine, norm ratio).

```bash
python tools/dump_mossformer2_ss_reference.py --upstream ClearerVoice-Studio \
    --checkpoint MossFormer2_SS_16K/last_best_checkpoint.pt \
    --audio mix.wav --seconds 3 --gguf ref_3s.gguf [--npy-dir ref_3s/]
./build/bin/crispasr-diff mossformer2-ss mossformer2-ss-16k-f16.gguf ref_3s.gguf x
```

Test input: a two-talker mixture of `samples/jfk.wav` (English, male) and
`samples/paraformer_zh.wav` (Mandarin) at equal RMS. The PyTorch reference
separates it cleanly (3-s clip: `spk1` vs the Mandarin source cos 0.991,
`spk2` vs JFK cos 0.996; cross terms < 0.008).

Global cosine vs PyTorch, 3.0-s clip (48000 samples, T = 5999 frames,
24 groups; single forward for the stages, two-window segmented decode for
`e2e`):

| Stage | F32 | F16 | Q8_0 |
|---|---|---|---|
| `enc_out` | 1.000000000 | 0.999999994 | 0.999999994 |
| `mn_norm` | 1.000000000 | 0.999999947 | 0.999999947 |
| `mn_in` | 1.000000000 | 0.999999974 | 0.999989698 |
| `l0_to_hidden` | 1.000000000 | 0.999999995 | 0.999995592 |
| `l0_to_qk` | 1.000000000 | 0.999999976 | 0.999977805 |
| `l0_flash` | 1.000000000 | 0.999999912 | 0.999958283 |
| `l0_fsmn_conv1` | 1.000000000 | 0.999999944 | 0.999978238 |
| `l0_fsmn_ddn` | 1.000000000 | 0.999999144 | 0.999754075 |
| `l0_fsmn_gated` | 1.000000000 | 0.999999982 | 0.999990015 |
| `l0_fsmn` | 1.000000000 | 0.999999781 | 0.999853209 |
| `l1_flash` / `l1_fsmn` | 1.000000000 / 1.000000000 | 0.999999720 / 0.999999760 | 0.999810474 / 0.999814288 |
| `l2_flash` / `l2_fsmn` | 1.000000000 / 1.000000000 | 0.999999762 / 0.999999739 | 0.999802325 / 0.999798308 |
| `l5_flash` / `l5_fsmn` | 1.000000000 / 1.000000000 | 0.999999595 / 0.999999622 | 0.999654142 / 0.999668736 |
| `l11_flash` / `l11_fsmn` | 1.000000000 / 1.000000000 | 0.999999676 / 0.999999693 | 0.999718797 / 0.999733347 |
| `l17_flash` / `l17_fsmn` | 1.000000000 / 1.000000000 | 0.999999598 / 0.999999618 | 0.999742213 / 0.999755528 |
| `l23_flash` / `l23_fsmn` | 1.000000000 / 1.000000000 | 0.999999709 / 0.999999948 | 0.999700615 / 0.999942178 |
| `intra_out` | 1.000000000 | 0.999999986 | 0.999986141 |
| `mdl_out` | 1.000000000 | 0.999999902 | 0.999917904 |
| `masks` | 1.000000000 | 0.999999873 | 0.999861207 |
| `dec_spk1` / `dec_spk2` | 1.000000000 / 1.000000000 | 0.999999985 / 0.999999990 | 0.999989627 / 0.999992394 |
| `e2e_spk1` / `e2e_spk2` | 1.000000000 / 1.000000000 | 0.999999982 / 0.999999988 | 0.999989039 / 0.999989216 |

F32 values print as 1.000000000 at nine decimals (largest absolute error
1.1e-3 on `l23_*`, whose activations reach ~10²). The 1.94-s clip (19 stages:
encoder, mask-net input, layer-0 sub-stages, layers 0/23, masks, decoder) gives
the same picture: F32 1.000000000 everywhere; the minimum stage cosine is
0.999999302 for F16 and 0.999736195 (`l23_flash`) for Q8_0. The portable `ggml_ssm_conv` depthwise path
(`CRISPASR_MF2_PORTABLE_DWCONV=1`) also passes every stage on that clip.

End-to-end `separate()` vs upstream decode:

| Clip | Path | F32 | F16 | Q8_0 |
|---|---|---|---|---|
| 1.94 s (31000 samples) | one pass (padded to 2 s) | 1.000000000 / 1.000000000 | 0.999999950 / 0.999999987 | 0.999983918 / 0.999986098 |
| 3.0 s | segmented, 2 windows | 1.000000000 / 1.000000000 | 0.999999982 / 0.999999988 | 0.999989039 / 0.999989216 |
| 10.0 s | segmented, 8 windows | 1.000000000 / 1.000000000 | 0.999999981 / 0.999999980 | 0.999980091 / 0.999975897 |

(cosine `spk1` / `spk2`; all norm ratios within 1e-4 of 1.)

## Performance

Measured on a shared 4-core x86 VM (AVX-512, no GPU) while other jobs were
running, so the absolute numbers are noisy (1-min load average in brackets);
ggml's spin-wait thread barriers degrade much more than PyTorch's under
oversubscription, so compare the least-loaded pairs.

| Run | Threads | Time | Load |
|---|---|---|---|
| PyTorch original, one 2-s window `model(x)` | 4 | 7.05 s | 4.3 |
| native F32, one 2-s window (crispasr-diff forward) | 4 | 4.96 s | 3.8 |
| PyTorch original, 10-s clip, `decode_one_audio_mossformer2_ss_16k` (8 windows) | 4 | 97.2 s | 4.3 |
| native F32, 10-s clip, `separate()` (8 windows) | 4 | 138.5 s | ~4 |
| PyTorch original, 10-s clip | 2 | 369.5 s | 9.3 |
| native F32, 10-s clip, CLI `--separate` | 2 | 137.8 s | 6.9 |
| native Q8_0, 10-s clip, CLI | 2 | 263.4 s | 9.3 |
| native F16, 10-s clip, CLI | 2 | 312.2 s | 5.7 |

Per-op profile of one 2-s window (`CRISPASR_MF2SS_PROFILE=1`, F32, 4 threads,
moderate load): mul_mat 2.34 s of ~5.8 s; depthwise custom op 0.60 s;
copies/transposes 0.56 s; sigmoid 0.35 s; the rest are element-wise ops and
norms. On CPU the F32 GGUF is the fastest file: ggml's F16 and Q8_0 matmuls
quantise the activations per call and lose to the F32 GEMM here (F16/F32
mul_mat time ≈ 3.5× in the same contended profile), so use F16/Q8_0 for size
or GPU, F32 for CPU speed. Resident memory of a 10-s CPU run was ~330 MB (F16 file).

Diagnostics: `CRISPASR_MF2SS_BENCH=1` (stage wall times),
`CRISPASR_MF2SS_PROFILE=1` (per-op-type times, node-by-node execution),
`CRISPASR_MF2_PORTABLE_DWCONV=1` (GPU depthwise path on CPU).

## Tests

- `tests/test-mossformer2-ss.cpp` `[unit]` — depthwise conv on both paths vs a
  naive grouped Conv1d (k17, k39, k39 dil 2 with two inputs per group, short
  sequences), GPU kernel-width split, sinusoidal embedding layout, rejection of
  missing/foreign/incomplete GGUFs.
- `tests/test-mossformer2-ss-converter.py` — name map, squeezes, precision
  policy, rotary dedup and GGUF round trip on a synthetic state dict.
- `tests/test-mossformer2-ss.cpp` `[live]` (`CRISPASR_MOSSFORMER2_SS_MODEL`,
  optional `CRISPASR_MOSSFORMER2_SS_REF` for the full diff) — one-pass and
  segmented `separate()`, bit-identical repeat through the cached graph.
- `tests/test-mossformer2-ss-surface.py` (`CRISPASR_MOSSFORMER2_SS_MODEL`) —
  CLI `--separate` WAVs, renamed-model Session auto-detection, stereo ABI
  stems, repeat ownership.
