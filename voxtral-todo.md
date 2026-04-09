# Voxtral-Mini-3B-2507 port plan

`mistralai/Voxtral-Mini-3B-2507` is Mistral's audio-LLM, structurally
similar to Qwen3-ASR but with cleaner reuse of existing code in this fork.

## Architecture (verified against config.json + safetensors index)

### Audio encoder — `audio_tower.*` (literal Whisper-large-v3 encoder)
- 32 layers, hidden=1280, 20 heads, head_dim=64, FFN=5120
- num_mel_bins=128, max_source_positions=1500
- 2 × Conv1D front-end (`conv1`, `conv2` — same as Whisper)
- **Learned absolute position embedding** `embed_positions.weight` (1500, 1280)
- 32 × encoder block, each:
  - `self_attn_layer_norm.{weight,bias}`
  - `self_attn.q_proj.{weight,bias}`
  - `self_attn.k_proj.weight` (NOT a bias — Whisper quirk)
  - `self_attn.v_proj.{weight,bias}`
  - `self_attn.out_proj.{weight,bias}`
  - `final_layer_norm.{weight,bias}`
  - `fc1.{weight,bias}`, `fc2.{weight,bias}` (GELU FFN)
- `layer_norm.{weight,bias}` at the end (post-encoder norm)

This is **byte-for-byte identical** to whisper-large-v3's encoder. The
GGUF tensor names match what `whisper.cpp` already loads. We can
literally call into our existing whisper encoder graph builder.

### Multi-modal projector — `multi_modal_projector.*`
**Verified shapes via HTTP range request on shard 2:**
- `linear_1.weight: [3072, 5120]` = `nn.Linear(in=5120, out=3072)`
- `linear_2.weight: [3072, 3072]` = `nn.Linear(in=3072, out=3072)`

So the projector pipeline is:
1. Audio encoder output `(B, T_audio=1500, 1280)`  for a 30s clip
2. **Stack 4 adjacent frames** along the channel dim:
   `(B, T_audio, 1280) → (B, T_audio/4=375, 5120)`
3. `linear_1` (5120 → 3072)
4. GELU (per `projector_hidden_act: gelu` in config)
5. `linear_2` (3072 → 3072)
6. Output `(B, 375, 3072)` — audio embeddings, ready to splice into the
   LLM input embedding sequence at the audio_token_id placeholder positions.

This is a 4× temporal downsampling, taking the 50 fps Whisper encoder
output to the documented 12.5 fps audio embedding rate.

### LLM — `language_model.*` (vanilla Llama 3 / Mistral)
- 30 layers, hidden=3072, 32 Q heads / **8 KV heads (GQA, ratio 4)**, head_dim=128
- FFN=8192, SwiGLU (`gate_proj`/`up_proj`/`down_proj`)
- RMSNorm, RoPE θ=**1e8** (Mistral default), max_pos=131072 (long context!)
- vocab=131072
- **No biases** anywhere on attention or MLP (`attention_bias: false`, `mlp_bias: false`)
- **No Q-norm/K-norm** — simpler than Qwen3
- audio_token_id=24 (placeholder token ID for audio frame embedding splice)

### Tokenizer — Tekken (Mistral's tiktoken-style BPE)
- File format: `tekken.json` (NOT a sentencepiece .model or HF tokenizer.json)
- 150 000 vocab entries each as `(rank, token_bytes_b64, token_str)` —
  this is a **rank-ordered byte BPE** like tiktoken's cl100k_base
- 1000 special tokens at ranks 0..999 (`<unk>`, `<s>`, `</s>`, `[INST]`,
  `[/INST]`, `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[IMG]`, `[AUDIO]`?,
  ...). Default vocab_size = 131072 (1000 specials + 130072 BPE).
- Pre-tokenizer regex: same as tiktoken cl100k_base, uses Unicode property
  classes (\p{L}, \p{Lu}, \p{Ll}, \p{N}). **Needs unicode-aware regex**
  in the C++ side.
- Encoding: BPE merges are implicit — start with single bytes, greedily
  merge the lowest-rank pair where both halves form an existing vocab entry.

## Reuse plan

Most of this port can reuse existing infrastructure:

| Piece | Source | Effort |
| --- | --- | --- |
| Mel preprocessing (n_fft=400, hop=160, 128 mels) | `qwen3_asr_compute_mel()` | 0 (drop-in, same params) |
| Conv front-end (2 stride-2 Conv1Ds) + sinusoidal pos embed | `whisper.cpp` `whisper_build_graph_encoder` | trivial — already exists |
| Whisper-style encoder block (32 layers) | `whisper.cpp` ditto | trivial — already exists |
| Multi-modal projector (2× Linear) | new, ~30 LOC | trivial |
| Llama-style LLM forward (30 layers, GQA 32/8) | strip Q/K-norm from `qwen3_asr_build_llm_body`, change dims, change RoPE θ | ~100 LOC |
| KV cache (F16) | identical to Qwen3-ASR | reuse |
| Flash-attn on prefill + decode | identical to Qwen3-ASR | reuse |
| GGUF loader (mmap + bind) | adapt `qwen3_asr_load_model` tensor name list | ~150 LOC |
| Audio injection (splice into prompt embeddings) | identical to Qwen3-ASR | ~50 LOC |
| **Tekken tokenizer (NEW)** | rank-based byte BPE + tiktoken regex pre-split | ~300 LOC |
| Converter | adapt `convert-qwen3-asr-to-gguf.py` | ~150 LOC |
| CLI | adapt `qwen3-asr-main` | ~50 LOC |

The audio encoder reuse is the biggest win. We don't need to write
encoder graph code; we just call `whisper_build_graph_encoder` (or
inline the same primitives) on a 32-layer config instead of the
12/24/32-layer Whisper variants the existing whisper.cpp loader knows
about.

## Risks / unknowns

1. ~~Multi-modal projector input shape.~~ ✅ **Confirmed**: stack-4-frames
   (1280×4=5120) → linear_1(5120→3072) → GELU → linear_2(3072→3072).
2. **Tekken tokenizer regex.** The pre-split regex uses Unicode
   property classes (`\p{Lu}` etc.) which `<regex>` in libstdc++
   doesn't support. Options: (a) link against a Unicode-aware regex
   lib like RE2 or onigmo; (b) write a hand-rolled pre-splitter for
   the common case (English/German/whitespace) that approximates the
   regex's behaviour; (c) port tiktoken's C implementation. Option
   (b) is what we did for Qwen3 and it worked fine for ASR-style
   workloads where the prompt is a fixed chat template.
3. **Audio injection chat template.** Voxtral uses the Mistral
   `[INST]...[/INST]` format. Need to check exactly which special
   token signals the start of the audio embedding region — probably
   `[AUDIO]` or `[INST] [audio_token_id]×N [/INST]`. Need to confirm
   from the chat template.
4. **Long-context LLM at 131k positions.** Our existing KV cache is
   sized at max_ctx=4096 by default which is fine for the audio frames
   themselves (375 audio frames per 30s clip = ~30 minutes of audio in
   our cache budget) but the user might want longer.

## Stage breakdown (~4-6 days of focused work)

### Stage V1 — converter + tokenizer + LLM forward (smoke test)
- [ ] `models/convert-voxtral-to-gguf.py` — adapt the qwen3-asr converter
      to handle Voxtral's tensor name patterns. Output ~9.4 GB F16 GGUF.
- [ ] `src/voxtral.{h,cpp}` skeleton with model struct, GGUF loader,
      KV cache lifecycle (copy from `qwen3_asr.{h,cpp}` and rename)
- [ ] LLM body — strip Q/K-norm, change RoPE θ to 1e8, change all dims
- [ ] **Tekken tokenizer** — `src/voxtral_tokenizer.{h,cpp}`,
      ~300 LOC. Load tekken.json from a separate file (or embed at convert
      time as `tokenizer.tekken.{vocab,specials}` blobs in the GGUF).
- [ ] Smoke test: text-only LLM forward on a fixed prompt, diff
      logits against HF reference (similar to qwen3-asr-test-llm).

### Stage V2 — audio encoder
- [ ] Encoder graph: reuse whisper.cpp's encoder block primitives.
      Audio encoder is structurally identical to whisper-large-v3 except
      we load via our own loader (not whisper's GGUF format). Either
      vendor whisper.cpp's encoder primitive functions, or write a thin
      32-layer Whisper encoder graph builder in voxtral.cpp directly
      (~250 LOC, same shape as Qwen3-ASR's audio encoder build).
- [ ] Multi-modal projector — confirm shape after weights download,
      then: reshape `(1500, 1280) → (375, 5120) → linear_1 → GELU → linear_2 → (375, 3072)`
- [ ] Diff against PyTorch reference at proj2 output (similar to Stage 2 of qwen3-asr).

### Stage V3 — audio injection + end-to-end CLI
- [ ] `voxtral_run_llm_kv()` — same KV-cached prefill+decode as Qwen3-ASR
- [ ] Splice audio frames into the prompt embedding sequence at
      `audio_token_id=24` placeholder positions
- [ ] `examples/voxtral-main/main.cpp` — full CLI matching the
      `*-main` style
- [ ] End-to-end test on jfk.wav + the German clips

### Stage V4 — quantization + HF release
- [ ] Quantize via existing cohere-quantize. The 30-layer LLM has
      3072-wide tensors (3072 % 256 = 0, divides cleanly for Q4_K) and
      the encoder has 1280-wide tensors (1280 % 256 = 0 too). So no
      Q4_0 fallback needed. Q4_K should work for everything.
- [ ] Q4_K size estimate: ~2.0 GB (vs F16 9.4 GB)
- [ ] HF upload: `cstr/voxtral-mini-3b-2507-GGUF`
- [ ] README runtime table entry as the 7th runtime

## Next session next step

Verify the actual safetensors tensor shapes once the download finishes,
particularly the `multi_modal_projector.linear_1.weight` shape (which
will reveal whether it's 1280 → 3072 or 5120 → 3072 = "concat 4 frames").
That confirms or refutes assumption (1) above and unblocks the converter.
