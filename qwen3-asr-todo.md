# Qwen3-ASR-0.6B port plan (CrispASR)

Hybrid plan (option A) confirmed 2026-04-08. Reuses our existing whisper
mel + parakeet-style 2D subsampler + new Qwen3 LLM forward. Reference
implementation: github.com/predict-woo/qwen3-asr.cpp (MIT) — used for
architecture findings, **not** vendored.

## Architecture (verified against safetensors shapes)

### Audio tower — 2D-conv subsampler + Whisper-block body
- `conv2d1`: (480, 1, 3, 3)   — Conv2D, treats mel spec as 1×128×T image
- `conv2d2`: (480, 480, 3, 3)
- `conv2d3`: (480, 480, 3, 3)  — three stride-2 freq subsamplings (128→64→32→16)
- `conv_out`: linear (7680 → 896) where 7680 = 480 ch × 16 freq bins
- 18 × encoder block, each:
  - pre-LN `self_attn_layer_norm` (weight + bias)
  - `self_attn`: q/k/v/out_proj (896→896) **with biases** (Whisper-style)
  - 14 heads × 64 head_dim = 896
  - pre-LN `final_layer_norm` (weight + bias)
  - FFN: `fc1` (896→3584) GELU `fc2` (3584→896) **with biases**
- `ln_post` (896, weight+bias) — final norm
- Projector head:
  - `proj1` (896→896) + bias  → GELU
  - `proj2` (896→1024) + bias → output frames matching LLM hidden

**Positional embedding** is sinusoidal, computed at runtime via the Whisper
formula (`SinusoidsPositionEmbedding` class):
```
log_inc = log(10000) / (C/2 - 1)
inv_t   = exp(-log_inc * arange(C/2))
pe      = concat(sin(t*inv_t), cos(t*inv_t)) along channel
```
Shape `(max_pos, 896)` where max_pos comes from `n_window_infer=800`.
NOT stored in safetensors — we compute it on the C++ side identically.

**Chunked input pipeline (verified against reference dump 2026-04-08):**
1. Mel `(128, T)` is split along time into chunks of size `n_window*2 = 100`.
2. Chunks are batched and unsqueezed: `(num_chunks, 1, 128, 100)`.
3. Each chunk runs through `conv2d1/2/3` + GELU independently → `(num_chunks, 480, 16, 13)`.
4. Permute + flatten freq: `(num_chunks, 13, 480*16=7680)` → `conv_out` linear → `(num_chunks, 13, 896)`.
5. **Add positional embedding** sliced to T_chunk_out=13.
6. Mask out padding: flatten valid positions across chunks → `(N_total, 896)` where
   `N_total = sum(aftercnn_lens)` (e.g. 143 for 11s of jfk audio).

**Chunked self-attention via `cu_seqlens`:** the flat `(N_total, 896)` is fed
through 18 encoder layers, each of which applies **windowed attention**: positions
are grouped by `cu_seqlens` (cumulative chunk lengths) and attention only happens
within each window. Window size after CNN is
`window_aftercnn = padded_T_after_cnn * (n_window_infer // (n_window*2))`.
For our case `padded_T_after_cnn=13, n_window_infer=800, n_window=50` → window=104.
**Standard full self-attention will produce wrong output.** This is the
trickiest part of the C++ port.

### Text decoder — stock Qwen3 0.6B (28 layers)
- hidden=1024, 16 heads, **8 KV heads (GQA)**, head_dim=128
- `embed_tokens` (151936, 1024)
- per layer:
  - `input_layernorm` (RMSNorm, weight only)
  - `self_attn`: q_proj (2048, 1024), k_proj (1024, 1024), v_proj (1024, 1024),
    o_proj (1024, 2048), **no biases**
  - `q_norm`, `k_norm` (128,) — Qwen3-specific RMSNorm on per-head Q/K
  - `post_attention_layernorm` (RMSNorm)
  - `mlp`: gate_proj (3072, 1024), up_proj (3072, 1024), down_proj (1024, 3072)
    SwiGLU, no biases
- `model.norm` (RMSNorm)
- `lm_head` (151936, 1024) — stored separately even though config says tied
- RoPE θ=1e6, head_dim=128
- **mrope:** config has `mrope_interleaved` with `mrope_section=[24,20,20]`,
  but reference impl ignores mrope and uses standard 1D RoPE. We do the
  same as a v1 simplification — verify against reference output before
  shipping.

### Audio injection
- ChatML prompt:
  `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
- The N `<|audio_pad|>` (151676) tokens get their token-embedding row
  replaced by the N projector output frames (masked_scatter pattern)
- N = number of frames out of the encoder = `(T_mel - pad) / 8` after
  the 3 stride-2 freq convs (which also subsample T by 8?)
- Special tokens: audio_start=151669, audio_end=151670, audio_pad=151676,
  eos=151645, pad=151643

## Work breakdown

### Done
- [x] Download Qwen3-ASR-0.6B safetensors (1.87 GB) → `/tmp/qwen3-asr-inspect/`
- [x] Reverse-engineer architecture from config + shapes
- [x] Read upstream reference impl (predict-woo/qwen3-asr.cpp), confirmed
      this plan is implementable

### Step 1 — Converter (this session)
- [ ] `models/convert-qwen3-asr-to-gguf.py` in our converter style
      (per-arch namespace `qwen3asr.*`, F16/F32 split via `is_f32_tensor`,
      no llama.cpp gguf-py dependency, no inline quantisation — defer to
      `cohere-quantize`)
- [ ] Tensor name map derived from upstream + verified against actual keys
- [ ] Run on the downloaded weights, produce `qwen3-asr-0.6b.gguf` (~1.5 GB F16)
- [ ] Verify tensor count matches expectation (612 stored → ~610 mapped)

### Step 2 — Encoder ✅ DONE (2026-04-08)
- [x] `src/qwen3_asr.{h,cpp}` — full audio encoder graph builder
- [x] Sinusoidal pos embed precomputed at load time (Whisper formula)
- [x] Conv front-end + 18 Whisper-style pre-LN blocks + ln_post + proj1 + GELU + proj2
- [x] **Diff test on jfk.wav: max abs diff 1.43e-4, mean 7.31e-6, per-row cosine
      sim mean=1.000000 min=0.999999** — F16 numerical precision

#### Notable bugs fixed during Stage 2:
1. **`ggml_permute` semantics** (Stage 1): `permute(t, p0, p1, p2, p3)` means
   "source axis i goes to NEW position p_i", NOT "new axis i comes from source
   axis p_i". I had it backwards. Confirmed by reading ggml.c:3772 (`ne[axis0]
   = a->ne[0]`). Fixed by swapping permute args from (1,2,0,3) to (2,0,1,3)
   for the conv flatten step.
2. **PyTorch conv hooks fire pre-GELU**: `conv1_out.npy` etc. are PRE-gelu
   captures (the hook is on the `nn.Conv2d` module, GELU is applied externally
   via `F.gelu(self.conv2d1(chunk))`). Needed to compare C++ pre-gelu against
   the dumps, not C++ post-gelu.
3. **`cu_seqlens` is GPU-only**: `eager_attention_forward` (used on CPU)
   IGNORES `cu_seqlens` and does standard full self-attention with
   `attention_mask=None`. The windowed attention is only used by
   FlashAttention2 on GPU. **The C++ implementation should NOT apply windowed
   masking on CPU** — it just uses zero mask. Critical: this also means a
   future GPU/flash-attn variant would need explicit windowing, but the
   reference produces full-attention output for our diff target.

### Step 3 — Qwen3 LLM forward ✅ DONE (2026-04-08)
- [x] `qwen3_asr_build_graph_llm()` in `src/qwen3_asr.cpp`
- [x] GQA via `ggml_repeat_4d` (insert singleton, repeat 2×, reshape) — turns
      (head_dim, n_kv=8, T) into (head_dim, n_q=16, T) with each KV head
      duplicated to its corresponding 2 Q heads
- [x] Q-norm/K-norm: `ggml_rms_norm` along head_dim axis × per-head_dim weight
- [x] SwiGLU FFN: `down(silu(gate(x)) * up(x))`
- [x] NEOX-style RoPE θ=1e6 via `ggml_rope_ext` with `GGML_ROPE_TYPE_NEOX`
- [x] Causal mask + softmax with attention scaling
- [x] Token embedding lookup via `ggml_get_rows`
- [x] **Diff test on 9-token text prompt:**
      - max abs logit diff: 3.29e-2
      - mean abs diff: 4.38e-3
      - per-position cosine sim mean=0.999999 min=0.999996
      - **top-1 match: 9/9** (every position picks identical argmax token to PyTorch)
- [x] mrope sidestepped: for text-only input, mrope sections [24,20,20] all
      receive identical position_ids → reduces to standard 1D RoPE. The
      audio-injection path will need re-evaluation if mrope diverges, but for
      pure-text smoke test the simpler RoPE matches reference perfectly.

#### Bugs fixed
- **Attention output width**: each Q has 16 heads × 128 head_dim = 2048,
  NOT d_model=1024. The o_proj is (2048→1024), so the attention output is
  reshaped to (2048, T) before o_proj, then o_proj reduces to (1024, T).
  I had incorrectly used `ggml_reshape_2d(attn, d, T)` (= 1024 width).
  Fixed via `ggml_reshape_2d(attn, hd*n_q, T)`.

### Step 4 — Audio injection + end-to-end ✅ DONE (2026-04-08)
- [x] `qwen3_asr_embed_tokens()` — token-embed lookup helper exposing the
      backend `get_rows` to callers
- [x] `qwen3_asr_run_llm_from_embeds()` — LLM forward starting from a
      precomputed `(d, T)` embeddings buffer instead of token IDs
- [x] Refactored the LLM graph into a shared `qwen3_asr_build_llm_body()`
      called by both the ids-input and embeds-input variants
- [x] `examples/qwen3-asr-test-trace/main.cpp` — end-to-end test that loads
      a Python-dumped trace (input_ids + audio_pad_pos + reference logits +
      reference generated ids), runs encoder → embed → splice → LLM forward
      → greedy decode (no KV cache, full forward each step), and verifies
      the C++ output matches the Python wrapper.
- [x] **End-to-end PASS on jfk.wav:**
      - Next-token logits cosine sim 0.999997, argmax matches (token 11528=" And")
      - Greedy decode generated 30 tokens ending with EOS
      - **All 26 reference tokens reproduced contiguously** at gen offset 3
      - The 3 leading tokens are the language-detection markers that the
        Python wrapper strips before returning text — they ARE in the
        actual model output, just hidden by the wrapper.
      - Decoded transcript: "And so, my fellow Americans, ask not what
        your country can do for you; ask what you can do for your country."
- [ ] BPE tokeniser for the prompt (deferred — for the Stage-5 CLI we can
      either ship a small Python helper that pre-tokenises or implement BPE
      from `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` in C++)

### Step 5 — KV cache, CLI, integration

#### KV cache ✅ DONE (2026-04-08)
- [x] Persistent F32 KV cache `(head_dim=128, max_ctx, n_kv=8, n_layers=28)`
      allocated to backend, lives on `qwen3_asr_context`
- [x] `qwen3_asr_kv_init()` / `qwen3_asr_kv_reset()` lifecycle
- [x] `build_graph_llm_kv()` shared between prefill (n_past=0, n_tokens=T)
      and incremental decode (n_past>0, n_tokens=1):
      - Writes new K/V to cache view at offset `il*nb[3] + n_past*nb[1]`
      - Reads full history `(hd, n_past+T, n_kv)` for attention
      - Causal mask shape `(Lk, T)` constructed CPU-side per call
- [x] `qwen3_asr_run_llm_kv()` C API with auto n_past tracking
- [x] **Trace test PASS with KV cache**: same 26/26 token match
- [x] **Performance on jfk.wav (11s audio)**:
      - prefill 158 tokens: 3.84 s
      - decode 29 tokens: 4.81 s = **165.8 ms/token**
      - end-to-end: ~9 s (was ~120 s without cache → **~25× decode speedup**)
- [x] KV cache: 896 MiB at max_ctx=4096 in F32. F16 cache (next step) would
      cut this in half.

#### CLI ✅ DONE (2026-04-08)
- [x] Mel filterbank baked into GGUF via converter (`audio.mel_filters` =
      WhisperFeatureExtractor.mel_filters in (n_freqs=201, n_mels=128) layout,
      `audio.mel_window` = periodic Hann of length 400)
- [x] `qwen3_asr_compute_mel()` C API: STFT + mel projection + log10 +
      max-8dB clip + (x+4)/4 normalization, matching WhisperFeatureExtractor
- [x] Inline FFT (Cooley-Tukey + DFT leaves) handles non-power-of-2 n_fft=400
- [x] **Mel diff vs Python reference: max abs 2.2e-2, mean 3.2e-6** (F16 noise)
- [x] BPE tokeniser **sidestepped**: the chat-template prompt is fixed except
      for the audio_pad count, so token IDs are hardcoded in `build_prompt_ids()`
- [x] GPT-2 byte-decoder for vocab strings → UTF-8 transcript text
- [x] Tiny WAV reader (16-bit PCM, mono or stereo→mono mixdown, 16 kHz only)
- [x] `examples/qwen3-asr-main/main.cpp` (~280 LOC) ties everything together
- [x] **End-to-end on samples/jfk.wav (4 threads, F32 weights):**
      - mel:     250 ms
      - encoder: 2660 ms
      - prefill: 3032 ms
      - decode:  4391 ms (29 tokens, 151 ms/token)
      - **total: 10.3 s for 11 s of audio**
      - Output: "And so, my fellow Americans, ask not what your country
        can do for you; ask what you can do for your country."
- [x] Last-token-only lm_head optimization (~25% prefill speedup)
- [x] `qwen3_asr_token_text()` C API for vocab string lookup

#### Bugs fixed during Stage 5
1. **lm_head reshape** (during refactor): forgot the `T > 1` guard around the
   view-2d slice. Trivial.
2. **Mel filterbank orientation**: WhisperFeatureExtractor.mel_filters is
   shape `(n_freqs, n_mels)` not `(n_mels, n_freqs)`. My initial C++ access
   was transposed → max diff 1.9 in mel. Fixed.
3. **Hann window**: `np.hanning` is symmetric (divides by N-1), but
   `torch.hann_window(N, periodic=True)` (which librosa/scipy use) divides
   by N. Cosmetic ~1e-7 difference but fixed for cleanliness.

#### Remaining for production
- [ ] F16 KV cache to halve memory + potentially ~2× speedup on the read path
- [ ] Quantize the weights via `cohere-quantize` (Q8_0, Q5_K, Q4_K)
      → big prefill speedup
- [ ] `ggml_flash_attn_ext` for the cached attention path
- [ ] README runtime table entry (this fork's main README)
- [ ] HF upload `cstr/qwen3-asr-0.6b-GGUF` (F16 + Q8_0 + Q4_K)
- [ ] Implement BPE for arbitrary text input (not needed for ASR but enables
      few-shot prompting / language hints in the future)

### Deferred to v2
- ForcedAligner-0.6B port (separate model, 1775 LOC of CTC-style alignment in
  upstream — `nfa-align` already covers the use case for now)
- mrope (only matters if we hit accuracy issues that point at long-context
  positional confusion)
- Streaming inference

## Attribution

Reference implementation by predict-woo (MIT):
https://github.com/predict-woo/qwen3-asr.cpp — used for architecture
discovery and tensor name mapping. No source code vendored. Will be
credited in the README and source headers of the qwen3-asr files.

Model weights: Qwen/Qwen3-ASR-0.6B (Apache-2.0).
