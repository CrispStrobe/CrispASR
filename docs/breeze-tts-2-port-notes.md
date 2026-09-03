# Breeze TTS 2 — port notes (issue #412, phase 1)

Companion to `docs/breeze-tts-2-feasibility.md`. That memo is the GO/NO-GO
scoping; this is the build sheet. Everything below is line-cited against
`/mnt/volume1/tmp-overflow/breeze-src/` — `gh/` = the Apache-2.0 inference repo
(`breezeblue-ai/breeze-tts`), bare paths = the HF checkpoint
(`BreezeBlue/Breeze-TTS-2`).

Phase-1 artifacts:

| File | Role |
|---|---|
| `models/convert-breeze-tts-2-to-gguf.py` | HF safetensors → single GGUF, arch `breeze-tts-2` |
| `tools/kaggle/breeze-refdump/` | GPU reference oracle (kernel `chr1s4/crispasr-breeze-refdump`) |
| `tools/reference_backends/breeze_tts_2.py` | local comparator half of the diff harness |

---

## 1. Corrections to the feasibility memo

Two claims in §1.1 of the memo are wrong and would have produced a
plausible-sounding but non-parity text encoder. Fix them before writing C++.

**1. The text encoder is BIDIRECTIONAL, not causal.** The memo reads
`"use_bidirectional_attention": false` and concludes causal. That key is a
misnomer and is never read by the implementation that is actually registered.
`breeze_config.py:19-20` registers the LOCAL `models/t5gemma2_compat.py`
(`AutoConfig.register("t5gemma2_text", T5Gemma2TextConfig)` /
`AutoModel.register(T5Gemma2TextConfig, T5Gemma2TextEncoder)`), and there:

- `t5gemma2_compat.py:429` — `self.is_causal = False`
- `:395`, `:407` — `flash_attn_varlen_func(..., causal=False)` /
  `flash_attn_func(..., causal=False)`
- `:686-731` `_build_additive_attention_mask` — returns `None` for
  `full_attention` layers when there is no padding (i.e. **no mask at all**,
  full bidirectional), and for `sliding_attention` builds a **symmetric**
  window:

  ```
  left_window_size  = (sliding_window + 1) // 2 = 256   # dist =  q-kv in [0, 255]
  right_window_size =  sliding_window // 2 + 1  = 257   # dist in [-256, -1]
  ```

  so query *i* attends to keys `[i-255, i+256]`. The flash path uses the
  equivalent `window_size = (255, 256)` (`:371-373`).

**2. `rms_norm_eps` for the backbone is 1e-6, not the top-level 1e-5.**
The memo flags `rope_theta` and `rope_scaling` as top-level decoys but misses
that `rms_norm_eps` is one too. `breeze_backbone_factory.py:129`
`llm_config = AutoConfig.for_model(**backbone_config)`, then `:178`
`Qwen3RMSNorm(llm_config.hidden_size, eps=llm_config.rms_norm_eps)` →
`backbone_config.rms_norm_eps = 1e-06`. The top-level `1e-05` applies only to
`BreezeRMSNorm` users — i.e. the depth decoder (via `depth_decoder_config`,
which is also 1e-05).

Secondary risk in the memo §6 — **confirmed inert**:
`text_encoder_feature_layer_idx` is *absent* from `config.json` (defaults to
`-1` → `(-1,)`, `breeze.py:1084-1092`) and `text_encoder_layer_projs` is
absent from the checkpoint. `text_encoder_proj_type = "linear"`, so
`self.text_encoder_proj` is a plain `nn.Linear(1152, 2048, bias=False)`
(`breeze.py:1002-1006`) and `_project_segments` takes the
`isinstance(proj, nn.Linear)` fast path (`breeze.py:1488-1490`). No
DimFusion. Do not build it.

---

## 2. Tensor inventory

1115 tensors in 2 shards, 3 466 363 713 params, 6 966 413 058 bytes
(`model.safetensors.index.json` `metadata`).

| HF family | n | params | → GGUF | note |
|---|---:|---:|---|---|
| `text_encoder.embed_tokens.weight` | 1 | 302.0 M | `te.token_embd.weight` | `[262158, 1152]`, scaled by `sqrt(1152)` on lookup |
| `text_encoder.embed_tokens.eoi_embedding` | 1 | 0.0 M | `te.eoi_embd` | `[1152]`, substituted where `id == 256000` |
| `text_encoder.layers.{0..25}.*` | 338 | 697.9 M | `te.blk.N.*` | 13 tensors/layer |
| `text_encoder.norm.weight` | 1 | 0.0 M | `te.output_norm.weight` | |
| `text_encoder_proj.weight` | 1 | 2.4 M | `te_proj.weight` | `[2048, 1152]` |
| `backbone_model.layers.{0..27}.*` | 308 | 1409.4 M | `backbone.blk.N.*` | 11 tensors/layer (incl. q_norm/k_norm) |
| `backbone_model.norm.weight` | 1 | 0.0 M | `backbone.output_norm.weight` | |
| `lm_head.weight` | 1 | 4.2 M | `backbone.codebook0_head.weight` | `[2052, 2048]` — 2051 codes + EOS class |
| `depth_decoder.model.embed_tokens.weight` | 1 | 67.2 M | `backbone.audio_embd.weight` | **tied**, see §2.1 |
| `depth_decoder.model.inputs_embeds_projector.weight` | 1 | 2.1 M | `depth.projection.weight` | `[1024, 2048]` |
| `depth_decoder.model.layers.{0..11}.*` | 108 | 333.5 M | `depth.blk.N.*` | 9 tensors/layer (no q/k_norm) |
| `depth_decoder.model.norm.weight` | 1 | 0.0 M | `depth.output_norm.weight` | |
| `depth_decoder.codebooks_head.weight` | 1 | 31.5 M | `depth.cb_head.{0..14}.weight` | split + transposed, §2.2 |
| **`embed_text_tokens.weight`** | 1 | **537 M** | — **DROP** | `[262158, 2048]`, 1.07 GB |
| **`codec_model.*`** | 350 | **79.3 M** | — **DROP** | Mimi leftover, 0.16 GB |

Dropped: **351 tensors, 616 208 193 params, 1.23 GB bf16** (537 M + 79 M).
The memo's "1.16 GB" is the same two families rounded differently.

Live after the drop: **2 850 155 520 params (2.85 B)** — component totals
1000 M text encoder + 2.4 M proj + 1409 M backbone + 4.2 M lm_head + 434 M
depth decoder, which reconciles exactly with the index metadata
(3 466 363 713 total). F16 GGUF ≈ 5.7 GB; Q4_K ≈ 1.7–1.8 GB.

Per-layer tensor sets (exact, from the index):

```
text_encoder.layers.N.       backbone_model.layers.N.   depth_decoder.model.layers.N.
  pre_self_attn_layernorm      input_layernorm            input_layernorm
  post_self_attn_layernorm     post_attention_layernorm   post_attention_layernorm
  pre_feedforward_layernorm    self_attn.q_proj           self_attn.q_proj
  post_feedforward_layernorm   self_attn.k_proj           self_attn.k_proj
  self_attn.q_proj             self_attn.v_proj           self_attn.v_proj
  self_attn.k_proj             self_attn.o_proj           self_attn.o_proj
  self_attn.v_proj             self_attn.q_norm           mlp.gate_proj
  self_attn.o_proj             self_attn.k_norm           mlp.up_proj
  self_attn.q_norm             mlp.gate_proj              mlp.down_proj
  self_attn.k_norm             mlp.up_proj
  mlp.gate_proj                mlp.down_proj
  mlp.up_proj
  mlp.down_proj
```

Note the asymmetry the converter must respect: **text encoder AND backbone
have q_norm/k_norm; the depth decoder does not.** (`BreezeAttention`,
`breeze.py:289-327`, builds no norms — it is the plain CSM/Llama attention.)

### 2.1 The tied audio embedding

`backbone_model.embed_tokens.embed_audio_tokens.weight` **is not in the
checkpoint.** `breeze.py:913-917` lists it in `_tied_weights_keys` and
`_tie_weights` (`breeze.py:1102-1108`) clones
`depth_decoder.model.embed_tokens` into it when
`config.tie_codebooks_embeddings` (= `true`). Both are `[16*2051, 2048] =
[32816, 2048]`.

The converter emits **one** physical tensor named
`backbone.audio_embd.weight` and sets `breeze.tie_codebooks_embeddings = true`.
The runtime binds both `bb_audio_embd_w` and `dd_token_embd_w` to it. Writing
it twice would cost an extra 134 MB at F16 for nothing.

A converter that blindly looks for the backbone name and finds nothing
produces a GGUF that loads and then emits silence — hence the hard
`sys.exit("FATAL: ...")` guard in the converter.

### 2.2 `codebooks_head` split

`BreezeCodebooksHead.weight` is `[num_codebooks-1, hidden, vocab] =
[15, 1024, 2051]` (`breeze.py:607-609`) and the forward is

```python
F.linear(hidden_states[:, i, :], codebook_weight[i].T)   # breeze.py:620-624
```

i.e. `out = h @ weight[i]` with `weight[i]` of shape `(hidden, vocab)`. The
converter emits 15 separate `depth.cb_head.{i}.weight` tensors, each
**transposed** to `(vocab, hidden) = (2051, 1024)` → GGUF `ne = [1024, 2051]`,
which feeds `ggml_mul_mat(head, cur)` directly (`cur->ne[0] == 1024`).

This deliberately diverges from `csm_tts.cpp`, which keeps the 3-D tensor and
pays a `ggml_cont(ggml_transpose(slice))` on **every** depth step
(`csm_tts.cpp:1454-1462`, again at `:2308-2314`) — 4.2 MB of F16 copy × 15
steps × T frames. Pre-transposing in the converter removes that entirely.
Emitting the slices untransposed is the single easiest way to get a depth
decoder that runs, produces audio, and is wrong.

---

## 3. Constants table (line-cited)

All citations are `gh/models/...` unless the path says otherwise.

### 3.1 Global / audio

| Constant | Value | Source |
|---|---|---|
| `num_codebooks` | 16 | `config.json` |
| `audio_vocab_size` | 2051 | `config.json` |
| `hidden_size` (backbone d) | 2048 | `config.json` |
| `audio_embed_size` | 2048 | `config.json` |
| `text_vocab_size` | 262158 | `config.json` |
| `audio_token_id` (`<\|AUDIO\|>`) | 262144 | `config.json`; tag `templates.py:12` |
| `audio_eos_token_id` (`<\|audio_eos\|>`) | 262145 | `config.json`; tag `templates.py:13` |
| `codebook_pad_token_id` | 2050 | `config.json` |
| `codebook_eos_token_id` | 0 | `config.json` |
| backbone EOS class | 2051 (`= vocab_size`) | `breeze.py:922-923` (`lm_head` is `vocab_size + 1` wide) |
| `bos / eos / pad` | 2 / 1 / 0 | `config.json` |
| reserved codec ids | `[2048, 2051)` masked out of the sampler | `generation_breeze.py:125-131` (`codec_config.codebook_size` = 2048) |
| `tie_codebooks_embeddings` | true | `config.json` |
| `audio_tokens_offsets` | `arange(16) * 2051` | `breeze.py:792-796` |
| `audio_embeds_projector` | **absent** (`audio_embed_size == hidden_size`) | `breeze.py:786-791` |
| speaker tokens `[S0]..[S9]` | 262146..262155 | `config.json` `text_encoder_special_tokens_config.token_ids` |
| `<ins_bos>` / `<ins_eos>` | 262156 / 262157 | same |
| LoRA (r=8) + 12 added tokens | `merged_into_base: true` — nothing to apply | `config.json` |

### 3.2 Text encoder (T5Gemma2)

| Constant | Value | Source |
|---|---|---|
| layers | 26 | `config.json text_encoder_config` |
| `hidden_size` | 1152 | " |
| heads / kv heads | 4 / 1 (MQA) | " |
| `head_dim` | 256 | " |
| `intermediate_size` | 6912 | " |
| activation | `gelu_pytorch_tanh` | " |
| `rms_norm_eps` | 1e-6 | " |
| norm form | `x_normed * (1 + w)` | `t5gemma2_compat.py:125` |
| embed scale | `sqrt(hidden_size)` = 33.941125 | `t5gemma2_compat.py:606` |
| `eoi_token_index` | 256000 → substitute `eoi_embedding` | `t5gemma2_compat.py:579-587` |
| attn scale | `query_pre_attn_scalar ** -0.5` = `256 ** -0.5` | `t5gemma2_compat.py:427` |
| q_norm / k_norm | per-head RMSNorm on Q and K, **before** RoPE | `t5gemma2_compat.py:454-455`, applied `:471-472` |
| causal? | **NO** — `is_causal=False`, `causal=False` | `t5gemma2_compat.py:429`, `:395`, `:407` |
| `sliding_window` | 512, **symmetric**: left 256 / right 257 | `t5gemma2_compat.py:712-720` |
| flash `window_size` | `(255, 256)` | `t5gemma2_compat.py:371-373` |
| `layer_types` | 5×sliding, 1×full, repeated; **full at 5, 11, 17, 23** | `config.json` (26 entries; last two are sliding) |
| RoPE (sliding) | `rope_type="default"`, theta 1e4 | `config.json rope_parameters.sliding_attention` |
| RoPE (full) | `rope_type="linear"`, theta 1e6, **factor 8.0** | `config.json rope_parameters.full_attention` |
| "linear" semantics | `inv_freq /= factor`, `attention_scaling = 1.0` | `t5gemma2_compat.py:177`, `:182` |
| segment encoding | each text segment is its own padded batch row — **no cross-segment attention** | `breeze.py:1416-1424`, `_batched_text_encoder_forward` `breeze.py:1243-1348` |
| bucketing | segments bucketed while `len/min_len <= 2` | `breeze.py:1278-1301` (padding only; does not change math) |

Full-attention layer indices, verbatim from `config.json`:
`[5, 11, 17, 23]`. Layers 24 and 25 are sliding.

### 3.3 Backbone (Qwen3) — from `config["backbone_config"]` ONLY

| Constant | Value | Decoy at top level |
|---|---|---|
| layers | 28 | 28 (same) |
| `hidden_size` | 2048 | 2048 (same) |
| heads / kv heads | 16 / 8 | 16 / 8 (same) |
| `head_dim` | 128 | 128 (same) |
| `intermediate_size` | 6144 | 6144 (same) |
| `rope_theta` | **1 000 000** | ⚠ 500 000 |
| `rope_scaling` | **null** | ⚠ llama3 factor 32, orig 1024 |
| `rms_norm_eps` | **1e-6** | ⚠ 1e-5 |
| `max_position_embeddings` | 40960 | ⚠ 2048 |
| q_norm / k_norm | present (Qwen3) | — |
| `attention_bias` | false | — |

Path: `breeze_backbone_factory.py:124-129` (`AutoConfig.for_model(**backbone_config)`)
→ `:152` `_create_qwen3_layers(llm_config)` → `:163-181` real
`transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer` /
`Qwen3RMSNorm(eps=llm_config.rms_norm_eps)` / `Qwen3RotaryEmbedding(config=llm_config)`.

`BreezeBackboneModelEmbeddings` however is built from the **top-level**
config (`breeze_backbone_factory.py:96`), so `hidden_size` / `num_codebooks` /
`vocab_size` / `audio_embed_size` come from there. Both configs agree on
`hidden_size`, so the only live top-level values are the audio ones.

### 3.4 Depth decoder

| Constant | Value | Source |
|---|---|---|
| layers | 12 | `config.json depth_decoder_config` |
| `hidden_size` | 1024 | " |
| heads / kv heads | 8 / 2 | " |
| `head_dim` | 128 | " |
| `intermediate_size` | 8192 | " |
| `vocab_size` | 2051 | " |
| `num_codebooks` | 16 → 15 decode steps | " ; `breeze.py:604-610` |
| `backbone_hidden_size` | 2048 | " |
| `audio_embed_size` | 2048 | " |
| `max_position_embeddings` | 33 | " |
| `rms_norm_eps` | 1e-5 | " |
| `rope_theta` | 500 000 | " |
| `rope_scaling` | llama3, factor 32.0, low 0.001953125, high 0.0078125, **orig_max_pos 16** | " |
| attn scale | `head_dim ** -0.5` (plain) | `breeze.py:302` |
| causal | yes | `breeze.py:304` |
| q_norm/k_norm | none | `breeze.py:289-327` |
| `backbone_hidden_state_projector` | **None** (2048 == 2048) | `breeze.py:492-496` |
| frame-0 conditioning | backbone `last_hidden_state` written into `inputs_embeds[:, 0]` **unprojected**, then `inputs_embeds_projector` (2048→1024) | `breeze.py:556-561`, `:568` |
| embed offset | `offset = clamp(cache_position - 1, min=0) * vocab_size` | `breeze.py:550-551` |

The `original_max_position_embeddings = 16` inside a llama3 scaling on a
33-position sequence is unusual but is what the checkpoint was trained with.
Reproduce it literally (`rope_freq_factors` in `KvSelfAttnParams`).

### 3.5 Sampling / generation

| Constant | Value | Source |
|---|---|---|
| `temperature` | 0.9 | `generation_config.json` |
| `depth_decoder_temperature` | 0.9 | " |
| `top_k` / `top_p` | 50 / 1.0 | `runtime.py:44-52` |
| `max_new_tokens` | 750 | `generation_config.json` |
| `repetition_penalty` | 1.1 | `infer.py:26` |
| `MAX_SEQ_LEN` | 2048 | `infer.py:25` |
| `MAX_NEW_TOKENS` (CLI) | 1500 | `infer.py:24` |
| default `cfg_scale` | 1.0 | `infer.py:23` |

### 3.6 Codec

Do **not** convert. `runtime.py:94-105` loads `qwen_tts.Qwen3TTSTokenizer`
from the bundled `audio_tokenizer/`, whose `config.json` is
`qwen3_tts_tokenizer_12hz`: `latent_dim 1024`, `decoder_dim 1536`,
`upsample_rates [8,5,4,3]`, 16 quantizers, 24 kHz, 1920× down/upsample —
matching `qwen3_tts.cpp:557-563` exactly. Wire the existing
`cstr/qwen3-tts-tokenizer-12hz-GGUF` in as a registry **companion**, the same
shape as the `qwen3-tts` row at `src/crispasr_model_registry.cpp:726-731`.

### 3.7 Prompt templates (`breeze_infer/templates.py`)

Only two templates exist:

- `tts_instruction` (`:109-114`) — `[S0]<ins_bos>{instruction}<ins_eos>{text}`;
  negative branch = plain `[S0]{text}`.
- `ref_edit_tata` (`:115-121`) — `[S0]{ref_text}` + audio + `[S0]<ins_bos>{ins}<ins_eos>{text}`.

**Voice Clone is not its own template.** It is
`_ref_clone_tata_segments` (`:74-84`):

```
[S0]{ref_text}   |   <|AUDIO|> × T_ref  <|audio_eos|>   |   [S0]{text}
```

reachable either as `ref_edit_tata`'s negative branch (`:95-96`) or as the
`"ref"` branch of `_ref_edit_tata_dual_branches` (`:99-107`). Phase 2 should
expose it directly.

Each `{"type": "text"}` segment is tokenized **with** `add_special_tokens=True`
and then re-rendered, so a Gemma `<bos>` lands at the head of every text
segment (`templates.py:159-161`). `text_ids_len` records one entry per text
segment; `text_ids_mask` is False over the audio placeholders
(`templates.py:186-194`).

---

## 4. Reuse map (symbol level)

### 4.1 `src/csm_tts.cpp` — the parent. ~70 % structural reuse.

| CSM symbol | Reuse for Breeze | Change needed |
|---|---|---|
| `struct csm_hparams` (`:99-141`) | template for `breeze_hparams` | add the whole `te_*` block; split `bb_rms_norm_eps` (1e-6) from `dd_rms_norm_eps` (1e-5) — CSM shares one field and `csm_tts.cpp:2303` even normalises the depth decoder with `hp.bb_rms_norm_eps` |
| `struct llama_layer` (`:149-161`) | backbone + depth layers | **add** `attn_q_norm_w` / `attn_k_norm_w` (Qwen3 backbone + T5Gemma2 encoder need them; depth decoder leaves them null) |
| `struct csm_model` (`:206-241`) | template | drop `seanet_*` / `mimi_*` / `rvq_*` (codec is the companion); add `te_*` + `te_proj_w` |
| `bind_weights` (`:538-684`) | rename map | 3 blocks instead of 2; bind `bb_audio_embd_w` and `dd_token_embd_w` to the same tensor (§2.1) |
| `load_metadata` (`:468-528`) | KV reader | new `breeze.*` keys; read `breeze.te.layer_types` as an array |
| `init_bb_kv_cache` (`:685-714`) / `init_dd_kv_cache` (`:715-748`) | verbatim shape logic | bb: 28L × 8 kv heads × 128; dd: 12L × 2 kv heads × 128, 33 slots |
| `build_backbone_graph` (`:1293-1372`) | **the closest thing to a drop-in** | `KvSelfAttnParams` gains `qk_norm_eps` + q/k norm weights; theta 1e6; no rope_freq_factors |
| `build_depth_graph` (`:1374-1474`) | drop-in | 12 layers instead of 4; `rope_freq_factors` for llama3 scaling; per-codebook head is now a plain `mul_mat` on `depth.cb_head.{i}` (§2.2) |
| `build_audio_frame_embedding` (`:1281-1292`) | **verbatim** | summed-codebook embed with `audio_tokens_offsets`; 16 codebooks instead of 32 |
| `build_text_frame_embedding` (`:1275`) | **delete** | Breeze has no text embedding at the backbone — text comes from the encoder projection |
| AR loop `csm_tts_synthesize_with_reference` (`:1179-…`) | skeleton | prefill is embeds-not-ids; 15 depth steps; EOS is class 2051 in a 2052-wide head |
| `sample_topk` (`:364-411`) | verbatim | add repetition penalty (1.1) — `qwen3_tts.cpp:1852 apply_repetition_penalty` is the ready-made one |
| `csm_tts_run_backbone_dump` (`:2038`), `csm_tts_run_depth_dump` (`:2213`), `csm_tts_run_generate_codes` (`:2346`) | **copy the shape of these three** | they are exactly the entry points `tools/reference_backends/breeze_tts_2.py` diffs against |
| `rvq_dequantize` / `build_mimi_dec_transformer` / `build_seanet_decoder` (`:771`, `:902`, `:991`) | **not reused** | codec is qwen3-tts |

### 4.2 `src/qwen3_tts.cpp` — the codec, free.

| Symbol | Use |
|---|---|
| `qwen3_tts_init_codec_only(path, params)` (`qwen3_tts.h:45`) | load the companion GGUF standalone |
| `qwen3_tts_decode_codes(ctx, codes, n_codes, &n)` (`:256`) | 16×T codes → 24 kHz PCM. Direct target of the `codec_audio` fixture |
| `qwen3_tts_codec_extract_stage(...)` (`:265`) | per-stage codec diff, decoupled from the LLM (validation plan step 2) |
| `qwen3_tts_cenc_extract_stage(...)` (`:87`) | the **encoder** side — this is what produces `ref_codes` from the clone reference wav; `run_cenc_chunked` (`qwen3_tts.cpp:5106`), `cenc_rvq_encode` (`:5328`) |
| `qwen3_tts_synthesize_streaming(...)` (`:293`) | chunked decode, for the streaming path |
| `apply_repetition_penalty` (`qwen3_tts.cpp:1852`) | the 1.1 penalty from `infer.py:26` |
| `qwen3_tts_sum_frame_embed` (`qwen3_tts.cpp:6883`) | second reference implementation of the summed-codebook frame embed |

Qwen3 *layer* shape (q_norm/k_norm + GQA) is already exercised by the qwen3-tts
talker, so `core_attn::kv_self_attn` with `qk_norm_eps` set is a proven path —
see `GqaMode` guidance in `src/core/attention.h:609-613` (qwen3 uses
`GQA_MANUAL_CONT`).

### 4.3 `src/gemma4_e2b.cpp` — the text encoder, ~60 %.

| Symbol | Use |
|---|---|
| `g4e_llm_hparams` `sliding_window` / `rope_theta` / `rope_theta_full` (`:84-98`) | **exactly** the dual-RoPE + hybrid-attention hparam shape Breeze's encoder needs |
| per-layer `layer_type` mask (`:98`, loaded at `:1753`) | `breeze.te.layer_types` maps 1:1 |
| `kvp` / `kvp_full` split (`:1070-1094`) | two `KvSelfAttnParams` — one per rope config — chosen per layer |
| `g4e_gguf_u32/f32` metadata readers (`:1734-1746`) | template for the `breeze.te.*` readers |
| `build_conformer_self_attn` (`:758-922`) | the **non-causal** attention builder; the text encoder is bidirectional (§1) so the KV-cache path is the wrong template — this one is closer |
| Gemma pre/post-norm layer body | `pre_self_attn` / `post_self_attn` / `pre_ffn` / `post_ffn` ordering matches `t5gemma2_compat.py:530-554` exactly |

Differences from Gemma4 the port must add: `1 + w` RMSNorm form,
`embed_scale = sqrt(1152)`, `attn_scale = query_pre_attn_scalar ** -0.5`, the
`eoi_embedding` substitution, `rope_type="linear"` (divide `inv_freq` by 8,
attention_scaling 1.0 — **not** llama3/yarn), and the symmetric sliding window
(Gemma4's is causal-left-only).

### 4.4 Registry

One row in `src/crispasr_model_registry.cpp`, shaped like `qwen3-tts`
(`:726-731`) for the companion plus `raon` (`:1093-1098`) for the license
prose:

```cpp
{"breeze-tts-2", "breeze-tts-2-q4_k.gguf",
 "https://huggingface.co/cstr/breeze-tts-2-GGUF/resolve/main/breeze-tts-2-q4_k.gguf",
 "~1.8 GB",
 "qwen3-tts-tokenizer-12hz.gguf",
 "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
 "~60 MB",
 "other — NON-COMMERCIAL use only (BreezeBlue Research and Non-Commercial "
 "License Agreement v1.1, RESONIA INC; https://huggingface.co/BreezeBlue/Breeze-TTS-2)"},
```

`crispasr_license_requires_acceptance()` already covers `other`, so `-m auto`
will demand `CRISPASR_ACCEPT_LICENSE`. The GGUF also carries
`breeze.license.notice` / `breeze.license.derived_from` /
`breeze.license.summary` so a stray file still states its terms.

### 4.5 One-line registrations still owed (phase 2)

- `tools/dump_reference.py` `REGISTERED_BACKENDS`:
  `"breeze-tts-2": "reference_backends.breeze_tts_2",`
- `src/CMakeLists.txt` PUBLIC-link row for the new `src/breeze_tts_2.cpp`.

---

## 5. The CFG multi-branch design question

### What the model actually does

Three prompt branches (`templates.py:99-107`, `_ref_edit_tata_dual_branches`):

| branch | prompt | ref audio? |
|---|---|---|
| `uncond` | `[S0]{text}` | no |
| `ref` | `[S0]{ref_text}` + audio + `[S0]{text}` | **yes** |
| `ins` | `[S0]<ins_bos>{instruction}<ins_eos>{text}` | no |

Each branch has its **own prompt length, its own KV cache, and its own
`text_ids_mask`** — they are not a batch of one prompt with different
conditioning vectors. `generation_breeze.py:611-656` builds three independent
`model_kwargs` via `_make_branch_kwargs` (`:621`), `:687-706` prefills all three, and
the main path is *skipped entirely* in dual-CFG mode.

Combination, at every backbone step (`generation_breeze.py:822-826`):

```
logits = uncond + cfg_scale_ref * (ref - uncond) + cfg_scale_ins * (ins - uncond)
```

and **again inside the depth decoder**, per codebook step
(`_depth_decoder_generate_with_dual_cfg`, `generation_breeze.py:246-310`; the combine is `:304-307`):
three depth-decoder forwards per codebook, i.e. **45 depth forwards per audio
frame** instead of 15.

Single-CFG (2 branches) uses the same shape with
`logits = uncond + cfg_scale * (cond - uncond)`
(`generation_breeze.py:150`, `:199-201`). Upstream's own fast path caps at
2 (`fast_streaming.py:759`, `_BranchBatch(..., 2, cfg)` — `:75-79`); 3 branches only exist
on the slow `generate()` path.

### Capability matrix

| Capability | branches | needs |
|---|---:|---|
| Voice Clone | 1 | prompt with ref audio, `cfg_scale = 1.0` |
| Voice Clone + guidance | 2 | uncond + ref |
| Voice Design (instruction only) | 2 | uncond + ins |
| Voice Direction (ref + instruction) | 3 | uncond + ref + ins |

A batch-1-only backend ships **Voice Clone only** and silently drops the two
headline features. That must be a stated scope decision, not an accident.

### Option A — widen the graph to `n_branch`

Make `T` in every backbone/depth graph carry `n_branch` independent sequences
and give the KV cache a branch dimension.

- KV cache becomes `ne = (head_dim, max_ctx, n_kv_heads, n_layers * n_branch)`
  or a 5th dim; `core_attn::kv_self_attn` takes `il` as the trailing index, so
  the cheapest encoding is `il = layer * n_branch + branch` — **no change to
  `core_attn`**, only to the cache allocation and the `il` arithmetic.
- Prompts have different lengths, so prefill needs either (a) three separate
  prefill graph runs writing into three cache slices — trivial, prefill is
  once — or (b) a padded batch with a block-diagonal mask.
- Decode is where the win is: one graph, `T = n_branch` tokens, three cache
  slices, one `mul_mat` per weight instead of three. On CPU with F16 weights
  the backbone decode is memory-bandwidth bound on the weights, so 3 branches
  in one pass costs ≈ **1.05–1.2×** a single branch, versus 3×.
- Cost: every `n_past` in the AR loop becomes per-branch (the branches have
  different prefill lengths, so their `n_past` values differ permanently);
  `positions`, the causal mask, and the `kv_indices` scatter all become
  per-branch. Touching `csm_tts.cpp`'s loop structure is the bulk of it.
- Depth decoder: same widening, `T = n_branch` per step, cache 33 slots ×
  n_branch. Its KV cache is reset per frame so this is cheap.
- **Effort: ~3 days** on top of the batch-1 backend. Risk: the per-branch
  `n_past` bookkeeping is exactly the class of bug that produces
  almost-right audio.

### Option B — run branches serially

Keep every graph batch-1; run the backbone step (and each depth step) once per
branch, keeping `n_branch` separate KV caches, and combine the logits in C++.

- Zero changes to the graph builders or `core_attn`. The AR loop gains an
  inner `for (b : branches)`.
- Cost: **3×** the backbone decode and **3×** the depth decode. For a 3-branch
  Voice Direction run that is 45 depth forwards + 3 backbone forwards per
  frame. At 12.5 Hz frame rate, a 10 s utterance is 125 frames → 5 625 depth
  forwards. On the VPS-class CPU that is well past real time; on a GPU build
  it is merely 3× slower.
- Memory: 3× the KV cache. Backbone cache at 2048 ctx is
  `28 × 8 × 128 × 2048 × 2 bytes × 2 (K+V)` ≈ 240 MB per branch → 720 MB for
  three. Non-trivial on an 8 GB box but survivable.
- **Effort: ~0.5 day.**

### Recommendation

**Ship Option B first, behind `CRISPASR_BREEZE_CFG_BRANCHES` (default 1).**
It unlocks all four capabilities immediately at a known cost and, crucially,
gives the diff harness a correct 3-branch reference to compare against.
Option A then becomes a pure performance change validated against a working
Option B — a much safer diff than "new feature + new graph shape at once".
Gate the widened path behind the same env var when it lands (per the repo's
env-gating rule).

---

## 6. Phase-2 riskiest three points

**1. The bidirectional, symmetric-window text encoder.**
The repo has causal *decoder* paths (all sliding windows left-only) and
bidirectional *encoder* paths (whisper/parakeet/gemma4's conformer, all
unwindowed). Breeze's text encoder is the combination neither side has:
bidirectional **and** windowed, symmetrically (§1). The `[i-255, i+256]`
window has to be built
as an explicit additive mask (`t5gemma2_compat.py:712-720`), and the
`full_attention` layers at 5/11/17/23 get **no mask at all** — a
`ggml_flash_attn_ext` call with a causal mask silently produces a plausible
encoder output with wrong tail context, and the failure only shows up as
prosody drift many stages later. Additional traps in the same block: the
`1 + w` norm form, `embed_scale = sqrt(1152)`, `attn_scale =
query_pre_attn_scalar^-0.5`, and `rope_type="linear"` meaning
`inv_freq /= 8.0` with `attention_scaling = 1.0` (not a yarn/llama3 scaler).
Mitigation: diff `te_seg{K}_hidden` per segment against the fixture before
writing a single line of backbone code, and dump `te_seg{K}_layer{J}`
(`BREEZE_DUMP_TE_LAYERS=1`) to bisect the layer.

**2. Prompt assembly / segment boundaries.**
The backbone prefill is *embeddings*, not ids: `inputs_embeds` starts as
zeros, projected text-encoder rows are scattered into the text positions
(`breeze.py:1452-1458`), and the ref-audio codebook embeddings are merged into
the `<|AUDIO|>` positions (`breeze.py:1544+`). Getting this wrong is
invisible until the audio is garbage. Three specific landmines:
(a) segments are encoded **independently** — concatenating them into one
encoder pass changes every hidden state (`breeze.py:1418-1436`);
(b) each text segment is tokenized with `add_special_tokens=True`, so a `<bos>`
appears mid-prompt at every segment head (`templates.py:159-161`);
(c) `text_ids_len` and `text_ids_mask` must agree exactly or the reference
asserts (`breeze.py:1372-1375`) — the C++ has no such assert and will just
misalign. Mitigation: `prompt_input_ids` / `prompt_text_ids_mask` /
`prompt_text_ids_len` / `backbone_inputs_embeds` are all in the fixture set;
diff them before the first backbone forward.

**3. Depth-decoder llama3 RoPE with `original_max_position_embeddings = 16`
on a 33-slot sequence, plus the 15-way head.**
The scaling constants (factor 32, low 0.001953125, high 0.0078125) applied
with an original context of 16 put nearly every frequency in the
"interpolate" regime — a small error in the `rope_freq_factors` computation
changes the codes for codebooks 8-15 while leaving 0-7 plausible, which
sounds like a codec artifact rather than a RoPE bug. Compounding it: the
per-codebook head is indexed by `cache_position - 1` (`breeze.py:617-618`),
so an off-by-one in the head index shifts every codebook by one and still
produces audio. Mitigation: `dd_logits_frame0_cb{1..15}` are dumped
individually with argmax-equality as the acceptance criterion — the harness
localises both bugs to the exact codebook.

Runner-up (worth naming): the backbone `lm_head` is **2052** wide with the
EOS class at index 2051, and ids in `[2048, 2051)` are reserved and must be
masked out of the sampler (`generation_breeze.py:125-131`). Sampling into
2048-2050 yields codes the codec cannot decode.

---

## 7. Validation sequence (phase 2)

1. `te_seg{K}_hidden`, `te_proj_out` — cos ≥ 0.999 per segment.
2. `backbone_inputs_embeds` — cos ≥ 0.999. (Catches prompt assembly.)
3. `backbone_layer{J}_frame0` — bisect any drift; `backbone_logits_frame0`
   argmax must match.
4. `dd_logits_frame0_cb{1..15}` — argmax must match each.
5. `codes` — exact integer equality under greedy (`BREEZE_GREEDY=1`).
6. Codec, independently: feed the fixture's `codes` to
   `qwen3_tts_codec_extract_stage` and diff PCM against `codec_audio`.
7. Mandatory ASR roundtrip on the generated wav (en + zh), plus the reference
   e2e control arm — run `breeze-ref.wav` from the fixture through the same
   ASR first so a roundtrip failure can be attributed.
8. CFG branches diffed separately: uncond-only, then ref, then ins.

Reproduce the fixture with:

```
kaggle kernels push -p tools/kaggle/breeze-refdump      # maintainer only
python tools/reference_backends/breeze_tts_2.py --list
python tools/reference_backends/breeze_tts_2.py --cpp-dump /path/to/cpp/dumps
```
