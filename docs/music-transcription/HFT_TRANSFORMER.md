# hFT-Transformer in ggml

`src/hft_transformer.{h,cpp}` — a port of Toyama et al.'s hierarchical
frequency-time transformer for piano transcription (ISMIR 2023,
`sony/hFT-Transformer`, MIT) to CrispASR's ggml runtime, with a GGUF converter
that quantises it.

**Read the Cost section before choosing this over `onsets-and-frames`.** The
accuracy is real and it is the smallest model that reaches it. The throughput
is not competitive on this hardware, and the measurement says why.

## Why this model, and what was being tested

CrispTuner's benchmark (`CrispStrobe/flutter_tuner`, `bench/REPORT.md` §35–36)
scored six transcribers on MusicNet's test split under
`mir_eval.transcription`. On the three solo-piano pieces:

| transcriber | solo-piano note F1 | size |
| --- | --- | --- |
| Kong / piano-transcription | 71.2% | 154 MB ONNX / 77 MB F16 GGUF |
| **hFT-Transformer** | **70.5%** | 22 MB ONNX |
| Onsets & Frames | 69.1% | 106 MB F32 ONNX |
| Basic Pitch | 57.5% | 110 KB |

hFT matches Kong at one seventh of the size and beats Onsets & Frames by 1.4
points with a fifth of its parameters. §36.3 of that report recommended the
O&F port first and put this one second, **"and measured before it is
believed"**, because the case for it rested on an argument rather than a
number:

> 83.5% of its MatMul is weight GEMM, so int8 kernels apply to the dominant
> cost rather than to a rounding error, and on CPU a good q8 GEMM is typically
> two to three times fp32. That would take 1.36× real time to somewhere near
> 0.5×.

That is the bet this port tested. **It does not pay off on this hardware**, and
the Cost section below is the measurement, the reason, and the condition under
which the reasoning would still hold.

## What is here

| file | what |
| --- | --- |
| `models/convert-hft-transformer-to-gguf.py` | ONNX → GGUF, F32 / F16 / Q8_0 / Q4_0 |
| `src/hft_transformer.{h,cpp}` | the runtime |
| `examples/cli/crispasr_backend_hft_transformer.cpp` | legacy `transcribe()` adapter |
| `examples/cli/crispasr_piano_cli.cpp` | the `--piano` arm (the real surface) |
| `tests/test_hft_transformer_live.cpp` | contract + invariants (`[hft-transformer]`) |
| `tests/hft_parity_dump.cpp` | dumps mel + heads for the ONNX diff |
| `tools/hft_parity.py` | numeric agreement vs native onnxruntime |
| `tools/hft_musicnet_f1.py` | note-level F1 on MusicNet's test split |

```
python models/convert-hft-transformer-to-gguf.py \
    --input /path/hft_transformer.pruned.onnx \
    --output hft-transformer-q8_0.gguf --quant q8_0 \
    --check-mel --verify-fusion

crispasr --piano -m hft-transformer-q8_0.gguf -f piano.wav --piano-format midi
```

**Use the pruned graph.** The unpruned export keeps all fifteen forward
outputs, two of which nobody decodes — `enc_vector` at `[1, 128, 4, 88, 256]`
is 11.5 M floats by itself — and could not be loaded and run at all on the box
these numbers came from. `bench/tool/prune_hft.py` in `CrispStrobe/flutter_tuner`
cuts the graph to `onset_B`, `offset_B`, `mpe_B`, `velocity_B` (1,621 → 1,517
nodes). The converter only ever reads initializers, so the weights are the
same either way; the pruning matters for the ONNX arm of the harnesses.

## Architecture

5.48 M parameters, three stages, all of them transformer:

```
log-mel [T, 256 bins]
  │   32 margin frames of log(1e-8) each side; tail padded to a multiple of 128
  │
  ├─ ENCODER, one sequence per answered frame (batch 128 × 256 freq tokens)
  │     per (frame, bin): the 65-frame window → Conv2d(1,4,(1,5)) → [4·61=244]
  │                       → Linear(244, 256)        ... FUSED, see below
  │     × √256, + pos_embedding_freq[0:256]
  │     3 × { x = LN(x + SelfAttn(x)) ; x = LN(x + FF(x)) }
  │
  ├─ DECODER-FREQ, 88 pitch queries cross-attending to those 256 tokens
  │     x = pos_embedding_freq[0:88]                 (no √256 on the query)
  │     layer_zero : LN(x + CrossAttn(x, enc)) ; LN(x + FF(x))
  │     2 ×        : LN(x + SelfAttn(x)) ; LN(x + CrossAttn(x, enc)) ; LN(x + FF(x))
  │     → [128 frames, 88 pitches, 256] → transpose → [88, 128, 256]
  │
  └─ DECODER-TIME, 128 time tokens (batch 88 pitches)
        x = x·√256 + pos_embedding_time[0:128]
        3 × { x = LN(x + SelfAttn(x)) ; x = LN(x + FF(x)) }
        onset / offset / mpe = Linear(256, 1)   velocity = Linear(256, 128)
```

Attention is 4 heads × 64 with scale 1/8 everywhere. Every head emits
**logits**; the reference `infer.py` thresholds them at 0.5 as though they were
probabilities (really a sigmoid threshold of 0.62), so this runtime applies the
sigmoid and thresholds at 0.5, and `onset_threshold` here means what it says.

### The four things that are wrong-but-runnable

**1. Each layer has ONE LayerNorm, applied two or three times.** The checkpoint
carries a single `layer_norm.weight`/`.bias` per module and the graph reuses it
after every residual add. A port that allocated a set of gains per application
would run, and would be wrong by however much the applications differ. The
converter reads the module, not the graph node, so there is nothing to get
backwards.

**2. The convolution and the token embedding are fused, exactly.** Both are
linear in the 65-tap window and there is no nonlinearity between them, so they
collapse into a single `Linear(65, 256)`:

```
K[m, d] = Σ_c Σ_{j : 0≤j≤60, 0≤m-j≤4}  W_tok[c·61 + j, d] · W_conv[c, m-j]
b[d]    = b_tok[d] + Σ_c b_conv[c] · Σ_j W_tok[c·61 + j, d]
```

16,640 weights where the graph carries 62,464, and two `im2col` passes removed
from the runtime. `--verify-fusion` checks the collapsed form against an
explicit conv + 244×256 matmul on random input and asserts the residual is at
f32 rounding level (**7.1e-07 relative**). The GGUF records
`hft.front_end = "fused-conv-tok-embedding"` and the runtime refuses a file
that declares anything else, so a future unfused converter cannot silently feed
a 244-wide embedding to a front end that has no conv.

**3. The front end has three non-default settings and they are shipped, not
rebuilt.** 256 mel bins, n_fft 2048, hop 256, f_min 0, f_max 8000, at 16 kHz,
**power 2.0**, **constant (zero)** centre padding, then `log(mel + 1e-8)` — an
*add*, not a clamp — with a **periodic** Hann window, the **HTK** mel scale and
**slaney** filter normalisation. `core_mel` can express all of it, but neither
`build_htk_fb` nor `build_slaney_fb` is this combination, so the converter
computes the filterbank and the window, checks them against librosa
(`--check-mel`: **1.5e-08** and **3.0e-08**) and writes them into the GGUF as
`hft.mel_fb` / `hft.window`. The runtime refuses a GGUF without them rather
than guessing.

Note also that hFT does **not** drop a sample before the STFT the way Onsets &
Frames does, so `T = n / hop + 1` here against O&F's `(n-1) / hop + 1`.

**4. `mode_velocity='ignore_zero'` is not plumbing — it is the model's only
working filter.** The reference decoder drops any note whose velocity head
reads zero at the onset frame. §35.4 of the flutter_tuner report swept the
onset threshold from 0.2 to 0.7 and found hFT **completely flat at 52.2% F1**,
which is not saturation (0.3% of the onset head's values sit in [0.2, 0.5)) but
the gate removing exactly the candidates a lower threshold admits. Turning the
gate off and sweeping instead gives 45.1 / 47.0 / 49.8 / 52.1 — so the gate
reaches a *higher* F1 than the best thresholded arm without it, while answering
more often. It is exposed here as `ignore_zero_velocity` (default true) so it
can be measured, not so it can be tuned.

## How it is computed

Unlike `src/onsets_and_frames.cpp` there is no hand-rolled recurrence: the
whole model is ggml graphs. Two graphs per 192-frame window:

* **encoder + frequency decoder**, in chunks of `frame_chunk` frames (default
  32). Nothing in either stage mixes across frames — the encoder reads one
  65-frame window per answered frame and the frequency decoder's queries are
  the same 88 positional embeddings every time — so a chunk is **bit-identical**
  to the unchunked result, and `tests/test_hft_transformer_live.cpp` asserts
  that across a factor of eight in chunk size. What the chunk bounds is the
  attention score tensor: `[256, 256, 4, chunk]` is 16.8 MiB at 32 frames and
  would be 134 MiB for a whole window.
* **time decoder + the four heads**, once per window, on `[256, 128, 88]`.

Both allocators are created once and kept for the life of the context. §36.4 of
the flutter_tuner report named "a fresh ggml allocator per convolution chunk"
as one of the two unfixed causes of the Onsets & Frames arm's throughput gap;
this arm runs four encoder chunks and a time decoder for every 2.048 s of
audio, so it was not worth repeating.

Quantisation covers every 2-D matrix consumed by `ggml_mul_mat` whose
contraction axis is a multiple of 32: 63 tensors — all the attention
projections, all the feed-forward matrices and the velocity head. The
positional embeddings are *added*, not multiplied, so they stay F32; the fused
front end's contraction axis is 65 and stays F32; the three scalar heads are
`[1, 256]` and stay F32 because quantising 256 floats saves nothing and they
sit directly under a sigmoid.

## Numbers

### Sizes

| file | size | what is quantised |
| --- | --- | --- |
| `hft_transformer.pruned.onnx` (the source) | 21.8 MiB | — |
| `hft-transformer-f32.gguf` | 21.8 MiB | nothing |
| `hft-transformer-q8_0.gguf` | **7.0 MiB** | 63 of 157 tensors |
| `hft-transformer-q4_0.gguf` | **4.5 MiB** | the same 63 |

<!-- NUMBERS: parity, F1 and cost tables are filled in below by the measurement
     run; see the commit that adds them. -->
