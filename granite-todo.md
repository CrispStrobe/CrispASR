> **STATUS (2026-04-10): COMPLETE.** Working transcription matching HF ground truth.
> All 6 bugs found and fixed. See TODO.md for remaining tasks (quantize, upload, CLI).

# ibm-granite/granite-4.0-1b-speech — port plan

## Architecture (verified against config.json + safetensors index)

Total: 954 tensors, 4.4 GB BF16, three modules.

### 1. Audio encoder — `granite_speech_encoder` (Conformer)
- 16 layers, hidden=1024, 8 heads, dim_head=128
- Conv module: kernel=15, depthwise conv + batch norm
- Input: 160-dim (80 mels × 2 stacked frames), output: 348-dim (CTC head, ignored at inference)
- **Relative position embedding** (`attn.rel_pos_emb`) — similar to Parakeet
- Per-layer tensors (33 per layer, 528 total):
  - attn: pre_norm, to_q, to_kv, to_out, rel_pos_emb
  - conv: batch_norm (5 tensors), depth_conv, up_conv, down_conv, norm
  - ff: pre_norm, ff.0 (Linear), ff.3 (Linear)
  - ff_norm
- **Reuse opportunity**: Very similar to Parakeet's FastConformer encoder.
  Same Conformer pattern (MHSA + Conv + FFN with pre-norms).

### 2. Projector — BLIP-2 Q-Former
- 2 transformer layers, hidden=1024, 16 heads, FFN=4096
- **Learned query tokens**: `projector.query` (fixed-length query sequence)
- Per-layer (27 tensors per layer, 54 total):
  - Self-attention: Q/K/V (with bias) + dense output + LayerNorm
  - Cross-attention: Q/K/V (with bias) + dense output + LayerNorm
  - Intermediate query FFN + output query FFN + LayerNorm
- Final: `projector.linear` (1024→2048, maps to LLM hidden dim)
- `projector.qformer.layernorm` (input norm)
- **New module** — no existing code to reuse. Cross-attention from query
  tokens to encoder output is the core operation.

### 3. Language model — Granite 4.0-1B
- 40 layers, hidden=2048, 16 heads, 4 KV heads (GQA), FFN=4096
- RoPE θ=10000, RMSNorm eps=1e-5, SiLU activation
- **μP (maximal update parameterization) multipliers**:
  - `embedding_multiplier = 12.0` (scales token embeddings)
  - `attention_multiplier = 0.0078125` (scales attention logits = 1/128 = 1/head_dim)
  - `residual_multiplier = 0.22` (scales residual additions)
  - `logits_scaling = 8.0` (scales output logits)
- vocab=100353, max_pos=4096
- `tie_word_embeddings = False` (separate lm_head)
- 362 tensors

### 4. Audio preprocessing
- 80 mel bins (not 128 like Whisper/Voxtral)
- Stacked frames: input_dim=160 = 80 × 2
- downsample_rate=5 (encoder frames → LLM tokens)
- window_size=15 (windowed attention in encoder conv)

## Port approach — Path C (standalone, no llama.cpp dep)

Given our success with Voxtral 4B (standalone 4.4B model, 1 week), the
standalone approach is now proven. The Granite LLM (40 layers, 2048 dim)
is actually smaller than Voxtral's LLM (26 layers, 3072 dim) in terms
of per-token compute, so it should be faster.

### Implementation plan

1. **GGUF converter** (~0.5 day)
   - Map all 954 tensors to GGUF names
   - Bake mel filterbank (80 bins, same Slaney construction)
   - Handle μP multiplier metadata

2. **Audio encoder** (~2 days)
   - Reuse Parakeet Conformer graph patterns
   - Adapt for 160-dim input (stacked frames)
   - Implement rel_pos_emb (similar to Parakeet's)
   - Batch norm folding (already have this from Parakeet/canary_ctc)

3. **Q-Former projector** (~2 days)
   - Learned query tokens (fixed-length, loaded from GGUF)
   - Self-attention among queries
   - Cross-attention from queries to encoder output
   - LayerNorm + FFN per layer
   - Final linear projection (1024→2048)

4. **LLM decoder** (~1.5 days)
   - Copy Voxtral 4B LLM pattern (GQA, RoPE, SwiGLU, RMSNorm)
   - Add μP multipliers (4 scalar multiplications)
   - KV cache with F16
   - Flash attention

5. **CLI + integration** (~1 day)
   - Chat template parsing
   - Audio token injection at placeholder
   - Greedy decode + text output
   - Auto-download support

## Comparison with existing models

| | Granite 1B | Qwen3-ASR | Voxtral 4B |
|---|---|---|---|
| Params | ~1B | 900M | 4.4B |
| Encoder | Conformer 16L | Whisper 18L | Causal 32L |
| Projector | Q-Former (cross-attn) | Conv2D subsample | Linear stack-4 |
| LLM | Granite 40L/2048d | Qwen3 28L/896d | Mistral 26L/3072d |
| Languages | en, fr, de, es, pt, ja | 30 + 22 CN | 13 |
| Expected speed | ~8-12s (Q4_K) | 6.5s | 54s |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
