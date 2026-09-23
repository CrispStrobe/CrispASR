# X-ASR (GilgameshWind/X-ASR-zh-en) — #436, second half

icefall **streaming Zipformer2 transducer** (zh + en, punctuation and casing,
Apache-2.0). Upstream ships sherpa-onnx exports for four chunk sizes (160 /
480 / 960 / 1920 ms) plus one training checkpoint (`streaming_exp/pretrained.pt`).
Target: `--backend xasr`, bit-for-bit behaviour of sherpa-onnx
`OnlineRecognizer` greedy search.

## Blueprint facts (read from icefall `zipformer/{zipformer,scaling,subsampling,decoder}.py`,
## `export-onnx-streaming.py`, and sherpa-onnx `csrc/`)

### Features (sherpa `FeatureExtractorConfig` defaults)
- Kaldi fbank, 80 bins, 25/10 ms, Povey window, preemph 0.97, remove DC,
  low 20 Hz, **high −400 → Nyquist−400 = 7600 Hz**, dither 0,
  **snip_edges = false**, samples in [−1, 1] (`normalize_samples = true`, no ×32768).
- Kaldi builds its triangles in the mel domain, while `core_kaldi` historically
  builds them in Hz. X-ASR uses the opt-in `mel_domain_triangles` (measured below).

### Chunk pump (sherpa `OnlineRecognizerTransducerImpl`)
- Encoder input window `T = decode_chunk_len + 13` frames (13 = 7 + 2·3:
  Conv2dSubsampling loses 7, ConvNeXt's 3-frame look-ahead ×2), shifted by
  `decode_chunk_len` (= 2·chunk_size at 100 Hz).
- `IsReady`: `processed + T < frames_ready` (strict); the tail that does not fill
  a window is never decoded, so callers append silence (the sherpa examples pad 0.66 s).

### Encoder_embed (Conv2dSubsampling, streaming)
- conv(1→8, k3, pad (0,1)) → SwooshR → conv(8→32, k3, s2) → SwooshR →
  conv(32→128, k3, s(1,2)) → SwooshR → ConvNeXt → linear(128·19 → d0) → BiasNorm.
  Frames out = (T−7)//2 − 3 = chunk_size.
- ConvNeXt streaming: depthwise 7×7 conv (time pad 0 with a 3-frame cached
  left context, freq pad 3) → 1×1 conv (128→384) → SwooshL → 1×1 conv (384→128),
  plus a residual that takes the first T−3 frames of the input.
- Swoosh-L(x) = log(1+e^(x−4)) − 0.08x − 0.035; Swoosh-R(x) = log(1+e^(x−1)) − 0.08x − 0.313261687.

### Zipformer2 (streaming_forward)
- Stacks i = 0..S−1, each with a downsampling factor ds_i (typically 1,2,4,8,4,2).
  Input goes through `convert_num_channels` to encoder_dim[i] (truncate or
  zero-pad); stacks with ds_i > 1 are DownsampledZipformer2Encoder:
  - SimpleDownsample: weights = softmax(bias[ds]); mean over each group of ds frames.
    In the streaming export there is **no** padding: chunk_size divides ds.
  - SimpleUpsample: repeat each frame ds times, then truncate.
  - out_combiner = Bypass(src_orig, up).
- Final output = `_get_full_dim_output`: the last stack's output, plus the
  higher channels taken from earlier, wider stacks. Then SimpleDownsample(2),
  then (ONNX wrapper) `joiner.encoder_proj`.
- Per stack, left_context_len = left_context_frames // ds_i. Key padding mask
  (length left + chunk at 50 Hz, subsampled `[::ds]`): left positions are masked
  until `processed_lens` covers them. `processed_mask = (processed_lens <= arange(L)).flip(1)`.
  The masked scores get −1000 before the softmax.
- Positional encoding: CompactRelPositionalEncoding(pos_dim, length_factor 1).
  pe over x ∈ [−(T−1), T−1] with T = seq+left: x_c = √d·sign(x)·(log(|x|+√d) − log √d),
  atan(x_c / (d/2π)), cos/sin interleaved at freqs 1..d/2, **last column = 1**.
  The slice has length left + 2·seq − 1.
- Layer (streaming_forward):
  1. attn_weights = RelPosMHAWeights(src): in_proj → q | k | p. Concatenate
     cached_key, then k. Score = q·kᵀ + rel-shift(p·linear_pos(pos_emb)ᵀ).
     rel-shift: out[t, j] = pos_scores[t, j + (seq−1−t)]. Masked, then softmax.
     No 1/√d scale: it is folded into in_proj.
  2. src += FF1(src) (hidden ¾·ff): in_proj → SwooshL → out_proj.
  3. NonlinAttention(src, attn_weights[head 0]): in_proj → s | x | y (hidden ¾·d each);
     x *= tanh(s); concatenate cached_x (head 0 only; `num_heads` = 1 here because
     attn_weights[0:1]) → attn·x → *y → out_proj. src += it.
  4. src += SelfAttention1(src, attn_weights): in_proj (d → H·vd), concatenate cached_val1,
     attn·v, out_proj.
  5. src += Conv1(src): in_proj (d → 2d) → x·sigmoid(s) → ChunkCausalDepthwiseConv1d → SwooshR → out_proj.
     - Streaming depthwise conv: concatenate cache (k//2 frames).
       causal_conv (kernel (k+1)/2, bias) over it, plus chunkwise_conv (kernel k, zero pad k//2, bias)
       over the chunk only, times the scale 1 + left_edge + right_edge, both taken
       from chunkwise_conv_scale (2, C, k). For chunk < k the scale slices are
       left[:, :chunk] and right[:, −chunk:]; otherwise they are zero-padded to the chunk length.
  6. src += FF2(src) (hidden ff).
  7. src = bypass_mid(src_orig, src) = orig + (src − orig)·bypass_mid.bypass_scale.
  8. src += SelfAttention2(src, attn_weights) (cached_val2).
  9. src += Conv2(src) (cached_conv2).
  10. src += FF3(src) (hidden 5/4·ff).
  11. src = BiasNorm(src): x · mean((x − bias)²)^−½ · exp(log_scale).
  12. src = bypass(src_orig, src).
- Caches per layer: key (left, H·qd), nonlin (1, left, ¾d), val1/val2 (left, H·vd),
  conv1/conv2 (d, k//2); plus the embed ConvNeXt cache (128, 3, 19) and processed_lens.

### Decoder / joiner
- Stateless decoder: Embedding(V, dd) with id < 0 → zero row; Conv1d(dd, dd,
  kernel=context_size, groups=dd/4, no bias, no padding) → ReLU → decoder_proj.
- Joiner: output_linear(tanh(enc_proj + dec_proj)).
- sherpa greedy: tokens start as [−1]·(ctx−1) + [0]. At most **one symbol per
  frame**: argmax, emit if y ∉ {0 (blank), unk}, and re-run the decoder only after an emission.

## Weight source: the sherpa-onnx export, not `pretrained.pt`

`streaming_exp/pretrained.pt` (epoch 4, `model` and `model_avg`) is **not** the
exported model. Of 579 tensors comparable by name, none equals the ONNX
initializer, `decoder.embedding` included (max |Δ| 17.9, and up to 34 elsewhere).
Run on the same fbank chunks, a torch encoder with those weights was unrelated
to the ONNX encoder (cos_min −0.12), and transcribed garbage where sherpa was
right. Both control arms of `tools/kaggle/xasr-ref` fired.

What the export looks like (`tools/kaggle/xasr-onnx-graph`):
- The four chunk exports share every weight bit for bit: named initializers and
  all 344 anonymous MatMul weights.
- Named initializers keep PyTorch names (convs, biases, norms, bypass scales,
  decoder, joiner).
- The Linear weights are `onnx::MatMul_*`, stored transposed, and appear in
  execution order: embed.out; per layer in_proj, linear_pos, ff1 in/out,
  nonlin in/out, attn1 in/out, conv1 in/out, ff2 in/out, attn2 in/out,
  conv2 in/out, ff3 in/out; then encoder_proj. The shapes line up one to one.
- `linear_pos` is not folded.
- The downsample `softmax(bias)` was constant-folded to (ds,1,1) constants,
  in the order 2,4,8,4,2 (stacks 1..5), then 2 (downsample_output). The
  converter stores log-weights, so softmax restores them.
- `chunkwise_conv_scale` never shows up as a folded Mul constant, so it is
  expected to be a full (2, C, K) anonymous initializer, which would make it
  chunk-independent. This is not yet observed: the converter matches it by
  first use + shape and exits if the count or shapes disagree.
- The per-layer `bypass_scale` ("TODO: remove it") is absent, which is fine
  because it is never read.

## Features vs kaldi-native-fbank

`core_kaldi` gained opt-in `snip_edges=false` and `mel_domain_triangles`.
Against knf 1.22.3 on jfk + 1.6 s of silence the frame counts are equal.
Mel-domain triangles give mean |Δ| 5.3e-5, and max 1.9e-4 wherever
log-energy > −5; the few larger values are near-floor bins (log ≈ −15, float32
FFT rounding). The historic Hz-domain triangles would give mean |Δ| 2.6e-3,
max 5.2e-2.

## NOW (2026-09-23)

F16 parity on the first real run (tools/kaggle/xasr-pipeline, fcf080e2): every
stage has worst-frame cos ≥ 0.99999 on jfk and zh at 480 ms and 160 ms. The
tokens and the sherpa-rendered text are identical, and streaming in 370 ms
pieces equals one-shot. Q8_0 (168 MB) is the registry default; Q4_K (93 MB)
keeps zh identical but drops English punctuation.
