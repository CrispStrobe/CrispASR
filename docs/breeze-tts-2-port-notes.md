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

---

## Phase-2 state (2026-09-15)

### What exists now

| Artifact | State |
|---|---|
| `src/breeze_tts_2.{h,cpp}` | written, **compile-verified only** — no stage has been measured |
| `examples/cli/crispasr_backend_bt2_tts.cpp` | backend key `bt2-tts`, aliases resolve `breeze-tts-2` |
| registry / arch map / caps table / factory / `crispasr_list_backends()` | wired |
| `tests/test-registry.cpp` | NC gate asserted, with controls |
| `cstr/breeze-tts-2-GGUF` | f16 5.71 GB, q8_0 3.19 GiB, q4_k 2.05 GiB — **pre-norm-fold, being regenerated** |
| reference fixture | **not yet produced** — see below |

Conversion reconciles exactly with §2 of this document: 778 tensors,
2 850.2 M live params, 351 tensors / 633.1 M params dropped.

### The backend key is `bt2-tts`, not `breeze-tts2`

§4 of the Agreement bars the licensor's marks as the **primary name** of a
derivative. The feasibility memo judged `breeze-tts2` acceptable as descriptive
attribution; the owner's instruction for phase 2 was not to lead the key with
it. So the primary key is `bt2-tts` and `breeze-tts-2` / `breeze-tts2` /
`breeze_tts_2` remain as aliases, which keeps the model findable under its
published name while the name CrispASR presents is not theirs. The HF repo
`cstr/breeze-tts-2-GGUF` is unchanged — a repo name IS descriptive attribution.

### Scope decision on CFG: neither Option A nor Option B, yet

§5 recommended shipping Option B (serial branches) behind
`CRISPASR_BREEZE_CFG_BRANCHES`. Phase 2 has **not** done that. What is in the
tree:

* both KV caches are allocated with a trailing dim of `n_layers * n_branch`
  and both graph builders take a `branch` index, so the cache topology and the
  `il = layer * n_branch + branch` arithmetic are already in place and cost
  nothing at `n_branch == 1`;
* `n_branch` is pinned to 1, and the two things Option B still needs are
  per-branch prompt assembly (three different prompts, three different lengths,
  three different `n_past`) and the logits combine at every backbone step plus
  every depth step.

So this build reaches **Voice Clone and plain TTS**, and does not reach **Voice
Design or Voice Direction**. That is enforced rather than documented:
`breeze_tts_2_capabilities()` returns only `1 << BREEZE_CFG_NONE`,
`breeze_tts_2_synthesize_guided()` refuses anything else, and the CLI adapter
refuses `--tts-instruct` at init. The reason to refuse rather than downgrade is
specific to this failure mode: a single-branch run of a Voice Design request
produces fluent, natural speech that ignores the instruction entirely, and no
property of the output reveals it.

### Two decisions worth not re-deriving

**The encoder's "linear" RoPE needs no core change.** `rope_type="linear"` means
`inv_freq /= 8` with `attention_scaling = 1.0`. `core_attn::kv_self_attn`
hardcodes `freq_scale = 1.0f`, so the obvious route is to add a field to
`KvSelfAttnParams`. It is not needed: ggml computes `theta / freq_factors[i]`
per pair, so a **constant** `freq_factors` vector of `8.0` is exactly linear
scaling. Twenty other callers of that function stay untouched.

**Gemma's `1 + w` is folded at conversion time.** `T5Gemma2RMSNorm` is
`x * (1 + w)` and that includes `q_norm`/`k_norm`, which are applied *inside*
`kv_self_attn` as a plain `rms_norm * w`. The converter therefore adds 1.0 to
every `te.*norm*` weight and sets `breeze.te.norm_weights_pre_offset`; the
runtime refuses to load a GGUF that claims the Gemma form without it. Note the
pair of KV keys: `norm_unit_offset` describes the architecture,
`norm_weights_pre_offset` describes the bytes.

### The reference oracle: what it took

The fixture still does not exist, and the reasons are worth recording because
each one cost a run:

1. `kh.provenance` was called before it existed in the harness — the kernel
   clones the harness fresh but carries its own frozen script, so the two
   halves disagreed (gotcha #24). Killed the run in 10 s.
2. The P100 guard exited on every draw. Kaggle was P100-pinned and its torch
   has no sm_60 kernels, so "re-push to redraw" was an instruction to loop
   forever. Now falls back to CPU — the dump needs ~24 frames and 7 GB of host
   RAM, not a GPU. (Re-measured 2026-09-15: the pool handed out a **T4**,
   host RAM 33.7 GB. The P100 pin is not currently holding.)
3. `np.asarray` on a CUDA tensor from the audio tokenizer.
4. `output_hidden_states=True` on a backbone assembled by
   `breeze_backbone_factory` — the wrapper accepts the flag and returns
   `None`. Per-layer states now come from forward hooks. ⚠ **Convention:** the
   hooks fire *after* each layer, so `backbone_layer{J}` is the residual stream
   after layer J — NOT the `hidden_states` convention where index 0 is the
   input embedding. `backbone_inputs_embeds` is that input, dumped separately.
5. `model.generate()` without `audio_tokenizer=`, which drops into the dead
   Mimi branch and reaches for `quantizer.cardinality`.

Everything up to and including `backbone_logits_frame0` has been observed to
dump successfully: prompt L=185 over 2 text segments (27 and 19 tokens),
`te_proj_out [46, 2048]`, `backbone_inputs_embeds [185, 2048]`, all 28
backbone layers.

### Owed before any parity claim

1. `crispasr-diff` arm for `bt2-tts` — deliberately not written yet, so its
   stage shapes can be read off the fixture rather than guessed.
2. Per-stage cosine **and magnitude** against the fixture, in the §7 order.
3. Greedy exact-code equality, then the TTS→ASR roundtrip (en + zh), with the
   reference `breeze-ref.wav` run through the same ASR first as a control.
4. A tokenizer check: the C++ prompt builder uses
   `core_bpe::tokenize_simple` (as `gemma4_e2b.cpp` does) plus explicit
   special-token splitting. `prompt_input_ids` in the fixture is what settles
   whether that reproduces the Gemma tokenizer; `breeze_tts_2_run_prompt_dump`
   exists for exactly that comparison.

### ⚠ Reference-conditioning mismatch to check FIRST

The fixture exists as of 2026-09-15 (62 stages at
`cstr/crispasr-regression-fixtures/breeze-tts-2/`, dumped on a T4). Before
reading anything downstream of it, check this:

`ref_encoded {"sr": 16000, "n_samples": 176000, "ref_frames": 138}` — the
oracle hands `samples/jfk.wav` to the audio tokenizer **at 16 kHz** and lets
the tokenizer resample internally. The C++ adapter instead resamples to 24 kHz
with `core_audio::resample_polyphase` and hands over 24 kHz. Those are two
different resamplers, so `ref_codes` can differ before a single transformer
weight is touched — and a different reference prompt drifts everything after
it, in a way that looks like a model bug.

`ref_codes` is a dumped stage precisely so this is answerable rather than
mysterious: diff it first. If it differs, the fix is to align the oracle and
the runtime on one resampling path, not to chase the divergence downstream.

Frame-0 landmarks for a fast smoke check, from the completed run:
`argmax_cb0 = 404`, and the frame-0 code vector is
`[404, 172, 340, 1357, 644, 528, 1025, 1250, 122, 730, 1219, 1452, 1957, 443, 416, 1187]`.
Generated grid is `codes (24, 16)`; `codec_audio` is 46 080 samples, which is
exactly 24 frames x 1920, so the codec's frame arithmetic checks out.

### Repetition penalty is part of the recipe, not the model

`repetition_penalty` is **absent from `generation_config.json`**. `infer.py:86`
and `breeze_infer/api.py:92` pass `REPETITION_PENALTY = 1.1` at call time, so
it belongs to the shipped *synthesis* recipe, not to the checkpoint's defaults
— which is precisely the §3.5 table's "infer.py:26" footnote, read correctly.

The consequence for parity: the reference oracle calls `generate()` **without**
it, so the fixture's `codes` are penalty-free. A C++ greedy run that applied
1.1 would diverge from the fixture by construction, and the first bisect would
be spent chasing a difference that is ours. The runtime therefore uses 1.0 on
the greedy/diff path and 1.1 on the normal synthesis path, as an explicit
`GenOptions` field rather than something derived from `greedy` — they are two
different questions.

Still open (a validation item, not a known bug): *which* token history HF's
`RepetitionPenaltyLogitsProcessor` sees on the normal path. For this model the
backbone's `input_ids` are audio frames rather than text, so "the history" is
ambiguous; the C++ applies it over the generated codebook-0 tokens. Confirm
against a non-greedy reference run before treating the normal path as faithful.

### Published artifacts, read back rather than asserted

`cstr/breeze-tts-2-GGUF` regenerated 2026-09-15 with the norm fold:
f16 5.32 GiB, q8_0 3.19 GiB, q4_k **2.05 GiB**, 778 tensors / 2 850.2 M live
params, 351 tensors / 633.1 M dropped. The q4_k figure is what the quant
policy predicts once `te.token_embd` is held at source precision; set
`CRISPASR_BREEZE_QUANT_TEXT_EMBD=1` to include it and land near 1.75 GB.

The KV block of the **published** q4_k was fetched with an HTTP range request
and parsed (143 KV entries, 778 tensors). Verified in the shipped file, not in
the converter's intentions:

| key | value | what it rules out |
|---|---|---|
| `breeze.bb.rope_theta` | `1000000.0` | the top-level 500000 decoy |
| `breeze.bb.rms_norm_eps` | `1e-06` | the top-level 1e-5 decoy |
| `breeze.te.causal` | `False` | `use_bidirectional_attention: false` |
| `breeze.te.rope_factor_full` | `8.0` | — |
| `breeze.te.norm_weights_pre_offset` | `True` | a GGUF the runtime would reject |
| `general.license` | `"other"` | a tag that would NOT trip the NC gate |
| `breeze.license.notice` | verbatim, incl. `RESONIA, INC` | §4(b) |

All three config decoys are therefore dodged *in the artifact*, which is the
only place it counts.

### If a stage fails: where to look, in order

The §7 sequence localises by construction — first divergence is the bug — but
the *candidates* differ per stage, and the encoder is where this port has no
precedent to lean on.

| First failing stage | Look at, in this order |
|---|---|
| `ref_codes` | the resampler mismatch above. Nothing else. |
| `prompt_input_ids` | `core_bpe::tokenize_simple` vs Gemma's tokenizer; then the `<bos>` per text segment; then whether `[S0]`/`<ins_*>` were split as single ids by `tokenize_with_specials`. |
| `te_seg*_hidden` | 1. the symmetric window (`[i-255, i+256]`, NOT left-only, and NOT 512 either side); 2. whether full layers 5/11/17/23 got a mask at all (they must get **none**); 3. the `+1` norm fold — applied twice, or not at all; 4. `attn_scale` = `query_pre_attn_scalar^-0.5`; 5. `embed_scale` = sqrt(1152); 6. the dual RoPE (theta 1e4 sliding / 1e6 + constant freq_factors 8.0 full). |
| `te_proj_out` | the projection is one mat-mul; if the hidden matched and this does not, it is an orientation bug. |
| `backbone_inputs_embeds` | the scatter: text rows to text positions, summed ref-audio frames to `<\|AUDIO\|>` positions, and the all-`codebook_eos` frame at `<\|audio_eos\|>`. |
| `backbone_layer{J}` | bisect J. Qwen3 q/k-norm, theta 1e6, eps 1e-6 — if J=0 fails, it is not the backbone, it is the embeds. |
| `dd_logits_frame0_cb{C}` | if low C pass and high C fail, suspect the llama3 `rope_freq_factors` (orig_max_pos 16); if ALL C are shifted by one codebook, it is the `cache_position - 1` head index. |
| `codes` | with every stage above passing, this is sampling: reserved-id masking `[2048, 2051)`, the EOS class at 2051, or the repetition penalty. |

**Escape hatch worth knowing before bisecting the encoder.** The symmetric,
non-causal mask is the one attention shape this repo has not run before.
`CRISPASR_CORE_ATTN_EAGER_F32=1` swaps `ggml_flash_attn_ext` for an explicit
`mul_mat -> soft_max_ext -> mul_mat` with F32 scores. If the encoder is wrong
under flash and right under eager, the bug is in how the mask reaches
flash-attention, not in the weights or the constants — and that A/B costs one
env var instead of a day.

### The harness was tested before the model was

An instrument that cannot report failure makes every number it prints
worthless, so the comparison path was validated on known-answer and degenerate
inputs first, locally, before any Kaggle run scored anything:

* **npy writer** (C++ → numpy): 4 shape/dtype combinations round-tripped with
  exact value equality, not just matching shapes.
* **npy reader** (numpy → C++): run against the REAL fixture files; shapes,
  dtypes and leading values identical to `np.load`.
* **comparator, identical inputs**: 9/9 PASS, cos 1.000000, ratio 1.0000.
* **comparator, injected faults** — the arm that matters:

  | injected | caught by | reported |
  |---|---|---|
  | `te_seg0_hidden` x 2.0, *scale only* | magnitude | **cos=1.000000** and FAIL, `ratio=2.0000` |
  | `backbone_logits` argmax swapped | argmax | cos 0.975, `argmax ref=404 cpp=999 MISMATCH` |
  | one code off by one | int equality | `exact=False first_bad=7` |
  | untouched stages | — | still PASS (no false positives) |

  Exit code 1 on failures, 0 on a clean dump, so the kernel's `compare_rc` is
  a real signal rather than decoration.

The first row is the entire argument for the magnitude columns: a stage wrong
by a uniform factor of two scores a **perfect cosine**. Without `|ref|`/`|cpp|`
a 2x-wrong text encoder would have been reported as a flawless pass, which is
exactly how the htdemucs iSTFT and CQT bugs survived as long as they did.

---

## Phase-2 RESULTS — run 1 (q4_k vs bf16), 2026-09-15

**It synthesises, and the ASR roundtrip reads it back.**

| | |
|---|---|
| WAV | 94 080 samples @ 24 kHz = **3.92 s**, peak 0.53, rms 0.084 (not silence) |
| ASR on our audio | *"The quick brown dot fox jumps over the lazy dog."* |
| target | *"The quick brown fox jumps over the lazy dog."* |
| ASR on the ORACLE's own clip (control) | *"The quick brown fox jumped in."* |
| `--list-backends` | `bt2-tts` present |
| NC gate, `-m auto` without acceptance | **REFUSED** |

The control matters: the oracle's fixture audio is capped at 24 frames
(1.92 s), so its transcript is *supposed* to be truncated. It confirms the ASR
works and that our (uncapped) audio is the more complete of the two. One
inserted word — and that is with a mistokenized prompt and a mis-encoded
reference clip, both measured below. The model is more robust than the port.

### Per-stage, and why 7/55 "passing" understates it

Run 1 diffed the **q4_k** model against a **bf16** reference, which was a
mistake in experimental design, not a result:

```
backbone_layer0   cos=0.999849      backbone_layer10  cos=0.998825
backbone_layer1   cos=0.999672      backbone_layer20  cos=0.998375
backbone_layer5   cos=0.999031      backbone_layer27  cos=0.998024
backbone_logits   cos=0.999230   argmax ref=404 cpp=404  ✓
```

A **monotonic** decay with depth, with magnitude ratios pinned at 0.99-1.01
throughout, is the signature of accumulating quantization noise — not of a
structural bug, which shows up as a step at one layer. Judging the port by
these is judging the quantizer. Re-run on f16 before concluding anything about
the transformer stacks.

`argmax_cb0 = 404` matching the oracle exactly, off oracle-supplied
embeddings, is the strongest single signal that prompt assembly → backbone →
lm_head is right.

### Three real defects, two of them predicted in advance

| Stage | Measured | Cause |
|---|---|---|
| `ref_codes` | 61.8% equal, 138 frames both, first bad at 7 | the resampler mismatch, predicted and recorded before the run. Now removed from the harness by giving both sides 24 kHz, and kept as its own `ref_audio` stage. |
| `prompt_input_ids` | 64.9% equal, our L=208 vs 185 | `core_bpe::tokenize_simple` does not reproduce Gemma's tokenizer. Predicted. **Open.** |
| `dd_codes_frame0_stepwise` | 1/16 equal (only cb0, from the backbone) | the depth decoder diverges from codebook 1 with EXACT inputs. **Open, and the main suspect.** |

### The depth decoder — what has been ruled out

Fed the oracle's `backbone_hidden_frame0` and the oracle's `cb0 = 404`, our
codebook 1 is already wrong. Read line by line against `models/breeze.py`
after the run, these are **eliminated**:

* `BreezeRMSNorm` is plain `weight * normed` (`breeze.py:129-134`) — NOT the
  Gemma `1 + w` form. Our implementation matches.
* `BreezeDecoderLayer` is standard Llama pre-norm (`:405-425`). Matches.
* `BreezeAttention`: `scaling = head_dim ** -0.5`, `is_causal = True`, no
  q/k norm (`:302-305`). Matches.
* The `depth.cb_head.{i}` orientation, re-derived index by index: the
  converter's transpose makes `ggml_mul_mat` compute `h @ weight[i]`, which is
  what `F.linear(h, weight[i].T)` means. Correct.
* Not an off-by-one in the head index either — our cb1 is not the oracle's
  cb2, and no shift of the sequence aligns them.

The discriminating measurement is the **raw** `dd_logits_frame0_cb1` cosine,
which run 1 could not produce because the dump was masked (below). A cosine
near 1 with a different argmax means numeric drift; a cosine near 0 means
something structural.

### A measurement bug worth naming

`dd_logits_frame0_cb*` came back `cos=nan`, `|cpp|=inf`. Not a model failure:
`mask_reserved` writes `-inf` into `[2048, 2051)` and the dump was taken from
the masked buffer, while the reference dumps its logits *before* its own
suppression. The fix is to dump raw and mask a copy for sampling. A stage that
is merely being measured wrong must never be able to report as a catastrophic
failure — it sends the next person to the wrong place.

### The fixture is NOT the clone branch — and the tokenizer target is now known

Two findings from diffing the fixture's `prompt_input_ids` against the real
Gemma tokenizer offline (no Kaggle run needed; `tokenizer.json` is 32 MB).

**1. `meta.json` mislabelled the branch.** It said
`ref_edit_tata (clone branch, cfg_scale=1.0)`. The ids say otherwise: segment 2
is

```
[2, 262146, 262156, 130171, 8207, 532, 14769, 236761, 262157, 818, 3823, ...]
      [S0]  <ins_bos>  ...instruction...        <ins_eos>  The quick ...
```

`prepare_inputs` always builds from `template.build_segments`, which for
`ref_edit_tata` is `_ref_edit_tata_segments` — the **instruction** variant.
`build_negative_segments` (the real clone branch) is only reached when
`guidance_scale != 1.0`. So `guidance_scale=1.0` does keep it single-branch —
the true half of the old claim — but the single branch is the positive,
instruction-carrying one.

Consequence: a C++ prompt built without the instruction **cannot** match those
ids however good its tokenizer is. Run 1's `L=208 vs 185` therefore blamed the
tokenizer for a difference that was partly a missing `<ins_bos>…<ins_eos>`
span. The arm now passes `"Speak clearly and naturally."`, and the fixture
records its own `instruction` so this is not re-derived.

**2. The tokenizer target is exactly reproducible.** With the real tokenizer:

| segment | fixture | `[2] + tokenizers.encode(...)` | match |
|---|---|---|---|
| seg0 (`[S0]` + ref_text) | 27 ids | 27 ids | **exact** |
| seg1 (`[S0]` + syn_text, no instruction) | — | 12 ids | n/a |

seg0 matching exactly confirms the *structure* the port assumes — a `<bos>`
at every text segment head, `[S0]` as the single id 262146 — and confirms that
`Tokenizer.from_file(tokenizer.json)` reproduces the oracle. So the remaining
gap is purely `core_bpe::tokenize_simple`, whose whitespace-split
pre-tokenizer inflated 39 true tokens to ~69. The fix is a real Gemma
pre-tokenizer, and there is now a ground-truth oracle to test it against
offline, one string at a time, with no GPU and no Kaggle run.

### Tokenizer defect: CLOSED, verified offline

`core_bpe::tokenize_simple` was the wrong algorithm, not a near miss. Gemma's
`tokenizer.json` is explicit:

```
normalizer    {"type":"Replace","pattern":{"String":" "},"content":"▁"}
model.type    "BPE",  byte_fallback: true,  514906 merges
```

So: replace every space with U+2581 and BPE-merge across the **whole string**;
word boundaries ride on the ▁ marker. `tokenize_simple` instead whitespace-
splits and pushes every byte through GPT-2's `bytes_to_unicode()` — a different
scheme from the first step onward, which is why 39 true tokens came out as ~69.

Fixed by adding `core_bpe::tokenize_spm_bpe()` (additive — no existing caller
changes) and pointing the Breeze prompt builder at it. `bpe_one()`'s
rank-ordered merge loop was already correct and is reused unchanged.

Verified by compiling the **shipping** header against the real vocab and merges
and tokenizing the oracle's own reference text:

```
fixture seg0 : [2, 262146, 3133, 834, 1041, 12339, 14522, 236764, 2679, 711, ...]
C++ output   : [2, 262146, 3133, 834, 1041, 12339, 14522, 236764, 2679, 711, ...]
EXACT MATCH  : True  (27/27)
```

No build, no GPU, no Kaggle run — the fixture is ground truth and the
tokenizer is a pure function, so it can be held to exact equality on the
workstation. That is the cheapest verification available anywhere in this port
and it should have been done before the first parity run.

One known, benign deviation: Gemma's `byte_fallback` emits `<0xXX>` tokens for
out-of-vocab pieces while `bpe_one` falls back per codepoint. With a
262k-entry vocab the paths coincide for any text the model has embeddings for.

---

## Phase-2 RESULTS — run 2 (f16 vs bf16): THE PORT IS CORRECT

**50/55 stages pass, and the worst cosine across all fifty is 0.999861.**
The five failures are the two known input-side defects, both already fixed but
not yet re-run. Not one model stage fails.

| stage | cos | magnitude ratio | argmax |
|---|---|---|---|
| `te_seg0_hidden` | 0.999963 | 1.0000 | — |
| `te_seg1_hidden` | 0.999970 | 1.0003 | — |
| `te_proj_out` | 0.999933 | 0.9988 | — |
| `backbone_inputs_embeds` | 0.999957 | 0.9992 | — |
| `backbone_layer0…27` | 0.999997 → 0.999976 | 0.998–1.002 | — |
| `backbone_logits_frame0` | 0.999992 | 0.9977 | **404 = 404 ✓** |
| `dd_logits_frame0_cb1…cb15` | 0.999991 → 0.999861 | 0.998–1.001 | **all 15 match** |
| `dd_codes_frame0_stepwise` | — | — | **exact, 16/16** |

That settles every open architectural question at once. The bidirectional
text encoder with its symmetric `[i-255, i+256]` window and unmasked full
layers is right. The dual RoPE — theta 1e4 sliding, theta 1e6 with a constant
`freq_factors` vector of 8.0 for the "linear" full layers — is right, and no
change to `core_attn::kv_self_attn` was needed. Gemma's `1 + w` folded at
conversion time is right. The Qwen3 backbone off the nested config (theta 1e6,
eps 1e-6) is right. The depth decoder's llama3 RoPE with
`original_max_position_embeddings = 16`, and the 15 pre-transposed per-codebook
heads indexed by `cache_position - 1`, are right.

### The depth decoder was never broken — it was the quantizer

Run 1 read 1/16 codes matching at frame 0 and the localisation table pointed
at RoPE scaling and head indices. All of that was wrong: on f16 the SAME code
produces the oracle's entire frame-0 vector exactly —

```
404 172 340 1357 644 528 1025 1250 122 730 1219 1452 1957 443 416 1187
```

— because run 1 diffed **q4_k against bf16**. The tell was already in the
data and was read correctly at the time: a *monotonic* cosine decay with
layer depth at pinned magnitude ratios is accumulating quantization noise, not
a structural bug, which appears as a step at one layer. The lesson is to diff
the reference-precision artifact FIRST and only then ask what quantization
costs; a whole round of architectural suspicion was spent on the quantizer.

**This is also a real finding about the published q4_k**: it does not preserve
frame-0 codes at all (1/16), while still producing intelligible speech. The
quant policy protects the embeddings and the output heads but leaves the depth
decoder's 434 M attention/MLP weights at q4_k, and 15 sequential steps compound
the noise. Worth an A/B before q4_k is called the default.

### The five failures, all input-side, all already fixed

| stage | measured | status |
|---|---|---|
| `ref_codes` | 61.8% | resampler mismatch — harness fixed (both sides 24 kHz); needs the regenerated fixture |
| `prompt_input_ids` / `_mask` / `_len` | 64.9% / 78.4% / 0% | wrong tokenizer + missing instruction — **both fixed**, tokenizer verified 27/27 offline |
| `codes` | 3.9% | downstream of the prompt; generation ran on the mistokenized prompt |

Expected next run: 55/55.

---

## Phase-2 RESULTS — run 3 (f16 + fixed tokenizer + regenerated fixture)

**53/56.** Three failures, one of them a fixture bug introduced by the
previous commit and since fixed.

What the tokenizer fix bought, end to end in the real pipeline:

| stage | run 2 | run 3 |
|---|---|---|
| `prompt_input_ids` | 64.9%, L=208 | **exact**, L=185 |
| `prompt_text_ids_mask` | 78.4% | **exact** |
| `prompt_text_ids_len` | 0% | **exact** |
| `ref_codes` | 61.8% | **98.78%** |
| ASR roundtrip | "The quick brown **dot** fox jumps over the lazy dog." | "**A** quick brown fox jumps over the lazy dog." |

`ref_audio` — the runtime's own 16 → 24 kHz resample against the reference's —
passes at **cos = 1.000000, ratio 1.0001**. `core_audio::resample_polyphase`
matches torchaudio's resampler essentially exactly, which **retires the
resampler as a suspect** for anything downstream. That was worth making its own
stage rather than leaving it inside `ref_codes`.

`ref_codes` at 98.78% (first divergence at flat index 601) is now a clean
codec-encoder-vs-codec-encoder comparison with no resampler in it. ~27 codes of
2208 differ; that residual is the qwen3-tts encoder port, and it is the one
genuinely open numerical item.

### `backbone_inputs_embeds` 0.999957 → 0.937599: a fixture disagreeing with itself

Not the model. The previous commit resampled only the audio the fixture
**dumps**; the prompt is built by `prepare_inputs` →
`encode_prompt_audio(audio_tokenizer, audio_path)`, which re-reads the wav from
disk at its native rate. So `ref_codes` came from 24 kHz while the audio
embeddings inside the fixture's own `backbone_inputs_embeds` still came from
16 kHz.

The proof is worth keeping, because it is a general shape:

```
across a regeneration that changed ref_codes almost completely (35% agreement),
backbone_inputs_embeds came back BYTE-IDENTICAL
```

A prompt that does not move when its own reference codes move is reading a
different audio source. Fixed by writing the resampled clip to disk and
pointing the request at that file, so one audio array feeds the dump, the
prompt and generation alike.

Note how it surfaced: **one stage collapsed while its neighbours held**. A
single end-to-end score would have shown marginally worse audio and nothing
else. That is the argument for per-stage diffing, made by the harness catching
its own author's bug.

### The voice-consent gate stopped the first quant A/B, correctly

Run 1 of the quant A/B pinned the speaker by cloning `samples/jfk.wav`, which
is the better experimental control. All twelve syntheses were refused with
`rc=17` — `crispasr_run.cpp:3518`, voice cloning requires `--i-have-rights`,
attesting *"I have the consent of the speaker whose voice this clones, or it
is my own voice."*

That flag was **not** passed and should not be. `jfk.wav` is a real person, the
attestation is a claim only someone with standing can make, and an automated
benchmark ticking it to obtain its numbers is exactly the box-checking the gate
exists to prevent. The control was right; the experiment was wrong.

The A/B therefore synthesises **unconditioned**, with a fixed seed and length
cap across arms. The speaker is whatever the model produces rather than a
pinned reference — a genuinely weaker control, recorded here rather than
glossed. The metric is still intelligibility of the same four sentences under
three quantizations. If a speaker-pinned version is wanted, a human passes
`--i-have-rights`; that is an escalation, not something to engineer around.

Second lesson from the same run, and a self-inflicted one: the failure stderr
was captured into a dict field that the summary dropped, so twelve identical
refusals reported a bare `rc=17` and the cause had to be recovered by reading
the C++ afterwards. **The reason is now printed at the point of failure.** The
rule it broke — a readout must be able to report failure — is written in these
very notes.

### Fixture self-consistency, proved both directions

The fix (write the resampled clip to disk; point the request at it) was
confirmed by the mirror image of the evidence that exposed the bug:

| regeneration | `ref_codes` | `backbone_inputs_embeds` |
|---|---|---|
| broken (dump resampled, prompt not) | changed almost completely | **byte-identical** |
| fixed (one array feeds both) | **unchanged** (agreement 1.0) | changed (maxabs 0.59) |

First the codes moved and the prompt did not; then the prompt moved to catch up
while the codes stood still. Together those say the prompt now reads the same
audio its own reference codes came from.

---

## Quant A/B — does q4_k deserve to be the registry default? (2026-09-17)

Four sentences, three quantizations, one seed, unconditioned synthesis, scored
by whisper. **Recommendation: keep q4_k. No registry change, no carve-out
change.**

| quant | size | raw WER | **normalised WER** | frame-0 codes |
|---|---|---|---|---|
| q4_k | 2.05 GiB | 0.0814 | **0.0357** | 1/16 |
| q8_0 | 3.19 GiB | 0.0913 | **0.0278** | 16/16 |
| f16 | 5.32 GiB | 0.0278 | **0.0000** | 16/16 |

### The raw numbers rank them wrongly, and it matters

Raw WER puts q4_k *ahead* of q8_0. That ordering is an artifact: whisper writes
"seventeen" as "17" and Americanises "travellers"/"harbour", and q8_0 happened
to collect one more of those spelling artifacts. Those are the ASR's
orthographic conventions, not the model's pronunciation.

Normalising numbers and British/American spellings before scoring flips the
order and separates the arms properly. The per-sentence transcripts, so the
reading can be checked rather than taken:

| | q4_k | q8_0 | f16 |
|---|---|---|---|
| pangram | exact | exact | exact |
| "…seventeen blue umbrellas **in** Manchester…" | exact | **"and" Manchester** | exact |
| "…the weather **had turned**…" | **"returned"** | exact | exact |
| "…close the window…" | exact | exact | exact |

**One real word error each for q4_k and q8_0; none for f16.** The metric now
normalises digits and spelling so a future run cannot be misled the same way.

### What this settles

1. **f16 is materially better** — perfect on all four sentences against one
   error apiece for both quants. Anyone who wants reference quality should take
   the f16, and it is published.
2. **q4_k and q8_0 are indistinguishable.** One real error each; the remaining
   0.008 WER gap is the *length* of one error. At n=4 that is nothing.
3. **Code exactness does not predict audio quality.** This is the sharpest
   result of the three: q8_0 reproduces the oracle's frame-0 codes **exactly,
   16/16**, and still made a real word error, while q4_k matched **1/16** and
   made one too. Every code-level gate this port owns is therefore blind to
   what quantization does perceptually — which is precisely why this A/B had to
   exist and why no code-level metric should be promoted into a quality gate.

So the honest answer is the cheap one: **q4_k loses code exactness and sounds
the same.** Spending +1.14 GiB of every user's download on q8_0 buys exactness
that demonstrably does not reach the audio.

### Where this is weak

* **n = 4 sentences.** Enough to show f16 clear of both quants; nowhere near
  enough to separate q4_k from q8_0, and it is not claimed to. A firm ranking
  of those two would need tens of sentences.
* **Unconditioned speaker.** Cloning would pin the voice and was the better
  control, but it requires `--i-have-rights` — a consent attestation an
  automated benchmark may not make. Each arm therefore has its own voice, which
  is a genuine confound for intelligibility.
* **Single ASR.** whisper's own error floor is inside every number; it is
  shared across arms, so comparisons hold even though absolute values are
  inflated.
