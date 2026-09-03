# Issue #412 — Breeze-TTS-2 scoping

Sources: HF `BreezeBlue/Breeze-TTS-2` (**not gated**, `license: other`) + GH `breezeblue-ai/breeze-tts`
(Apache-2.0 code), cloned to `/mnt/volume1/tmp-overflow/breeze-src/{,gh}`. Line refs are `gh/` paths.

## 1. Architecture
**It is a CSM (Sesame) fork.** `models/breeze.py` is HF's `modeling_csm.py` with `Csm`→`Breeze`:
`BreezeDepthDecoderModel` (breeze.py:464), `BreezeCodebooksHead` (:604),
`BreezeBackboneModelEmbeddings` (:775) are the Csm classes verbatim; `config.json` even retains
CSM's `"backbone_flavor": "llama-1B"` / `"decoder_flavor": "llama-100M"` keys (dead — see §1.2).
Three swaps vs CSM: text encoder added, Qwen3 backbone, Qwen3-TTS codec. Fully autoregressive —
no flow-matching, no diffusion, no masked/parallel decode. 24 kHz mono, codec frame rate 12.5 Hz
(`audio_tokenizer/config.json` `_frame_rate`). Languages **en + zh only** (HF `tags`, README).

### 1.1 Text encoder — `T5Gemma2TextEncoder`, 1.00 B params
`config.text_encoder_config`: 26L, d=1152, 4H/1KVH, head_dim 256, `query_pre_attn_scalar` 256,
`gelu_pytorch_tanh`, vocab 262158, `sliding_window` 512. `layer_types` = 5×sliding then 1×full
(full at 5,11,17,23), with **two RoPE configs** (full → theta 1e6 + linear factor 8.0; sliding →
theta 1e4 default). ⚠ **CORRECTED 2026-09-03 (phase 1):** the config key
`"use_bidirectional_attention": false` is a DECOY — `breeze_config.py:19-20` registers the
local `t5gemma2_compat.py`, where `is_causal=False` (:429) and `causal=False` is passed into
flash-attn (:395, :407): the text encoder is **BIDIRECTIONAL**. Full-attention layers get no
mask at all; sliding layers span a SYMMETRIC `[i-255, i+256]` (:712-720). Also `1+w` norms,
`embed_scale=sqrt(1152)`, `attn_scale=query_pre_attn_scalar^-0.5`, `rope_type="linear"`
(`inv_freq/=8`, `attention_scaling=1.0`).
Per-layer tensors are Gemma-3 shaped: `pre_self_attn_layernorm`, `post_self_attn_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`, `self_attn.{q,k}_norm`. LoRA (r=8) and
the 12 added special tokens are already `merged_into_base: true` → nothing to apply.
Output goes through `text_encoder_proj` (linear 1152→2048, breeze.py:1481) and is **written into the
backbone input embeddings at text positions** (breeze.py:1456-1462: `inputs_embeds` starts as zeros,
`inputs_embeds[text_ids_mask] = text_embeds`). Critically `embed_text_tokens` (262158×2048,
**537 M / 1.07 GB**) is used only on the no-text-encoder fallback (breeze.py:1578) — **dead weight
at inference, droppable from the GGUF**.

### 1.2 Backbone — real Qwen3, 1.41 B params
`breeze_backbone_factory.py:149-175` builds genuine `transformers` `Qwen3DecoderLayer`s from
`config.backbone_config` (28L, d=2048, 16H/8KVH, head_dim 128, ffn 6144, rope_theta **1e6**,
`rope_scaling: null`; q_norm/k_norm present in the index). The top-level `rope_theta` 500000 +
llama3 `rope_scaling` in `config.json` are **decoys** — the factory reads the nested
`backbone_config`. Audio input embedding is CSM's summed-codebook scheme (breeze.py:786-804):
`nn.Embedding(16*2051, 2048)` + `audio_tokens_offsets`, summed over codebooks; `audio_embed_size ==
hidden_size == 2048` so `audio_embeds_projector` is **absent**. `lm_head` 2048→2052 (breeze.py:923).

### 1.3 Depth decoder — 434 M params
12L, d=1024, 8H/2KVH, head_dim 128, ffn 8192, **16 codebooks**, vocab 2051, llama3 rope_scaling
(factor 32, `original_max_position_embeddings` **16**), theta 5e5. Own `embed_tokens` (16*2051 ×
2048) → `inputs_embeds_projector` 2048→1024 (breeze.py:487-489). `codebooks_head.weight` is
`[15, 1024, 2051]` (breeze.py:608), one head per codebook 1..15 — backbone emits codebook 0, depth
decoder runs 15 steps/frame. (CSM: 32 cb / 31 steps / 4L.)

### 1.4 Codec — **already shipped in CrispASR**
`runtime.py:94-105` loads `qwen_tts.Qwen3TTSTokenizer` from the bundled `audio_tokenizer/` dir
(`Qwen3TTSTokenizerV2Model`, `qwen3_tts_tokenizer_12hz`, 16 quantizers, latent_dim 1024,
decoder_dim 1536, upsample_rates `[8,5,4,3]`, 24 kHz, 1920× down/upsample). Decode goes through it,
not Mimi (generation_breeze.py:1278-1336: `codec_model.decode` is the `audio_tokenizer is None`
fallback only). **Verified bit-identical to `Qwen/Qwen3-TTS-Tokenizer-12Hz`**: 496 tensors,
identical name set, shapes, and safetensors `data_offsets`; 170.6 M params, Apache-2.0. CrispASR
already ships the GGUF (`cstr/qwen3-tts-tokenizer-12hz-GGUF`) and a working ggml encoder+decoder in
`src/qwen3_tts.cpp` — `latent_dim=1024`, `decoder_dim=1536`, `upsample_rates{8,5,4,3}` at
qwen3_tts.cpp:557-563 match exactly, plus chunked streaming decode (qwen3_tts.h:284).
The `codec_model.*` Mimi tensors in the main checkpoint are a training leftover — **drop**.
(Phase-1 exact count: 350 tensors / **79.3 M** params, not the 96 M estimated here; with
`embed_text_tokens` 536.9 M the drop totals 351 tensors / 1.23 GB bf16.)

### 1.5 Conditioning / CFG
`breeze_infer/templates.py`: `[S0]`..`[S9]` speaker prefix (:31-37), `<ins_bos>`/`<ins_eos>`
instruction wrapper (:44-50), `<|AUDIO|>`/`<|audio_eos|>` for reference audio (:12-13, :57-80).
Voice clone = ref_text + ref audio codes + target text (:70-80). **Classifier-free guidance with up
to 3 branches** (uncond / ref / ins) — templates.py:100-104, `cfg_scale_ref`/`cfg_scale_ins`,
generation_breeze.py:100-122, fast_streaming.py `_BranchBatch(..., 2, cfg)`. Voice design/direction
depend on it.

### 1.6 Sizes / license
3.48 B params, 6.97 GB bf16 (2 shards) + 682 MB fp32 audio tokenizer. Per-component: backbone
1409 M, text_encoder 1000 M, embed_text_tokens 537 M (droppable), depth_decoder 434 M, codec_model
79.3 M (droppable), lm_head 4.2 M, text_encoder_proj 2.4 M.
**After dropping dead weight: ~2.85 B → Q4_K ≈ 1.7-1.8 GB.**
License: **BreezeBlue Research and Non-Commercial License** (`LICENSE`, 424 lines; code Apache-2.0).
§1.3 explicitly names **quantization** as a Derivative Model → our GGUFs inherit NC. §4 permits
non-commercial redistribution but **requires** a full copy of the agreement, a NOTICE file reading
`"Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License Agreement.
Copyright (c) 2026 RESONIA, INC. All Rights Reserved."`, a prominent "Derived from Breeze TTS 2 …"
line in the model card, and no "Breeze"/"BreezeBlue" as the primary derivative name.
§1.7(b) makes hosting-as-a-service commercial.

## 2. Closest existing runtime
| Component | Closest CrispASR code | Reuse |
|---|---|---|
| Codec (dec+enc, streaming) | `src/qwen3_tts.cpp` (same weights, same GGUF) | **~100%** |
| Backbone AR + summed-codebook embed + depth-decoder loop | `src/csm_tts.cpp` (2478 lines; CSM is the literal parent) | ~70% |
| Text encoder (Gemma sliding/full, dual RoPE, pre/post norms, head_dim 256, sw 512) | `src/gemma4_e2b.cpp` | ~60% |
| Qwen3 layer (q_norm/k_norm) | `src/qwen3_tts.cpp` talker | ~80% |

Verdict: **`csm_tts` + `qwen3_tts` codec + `gemma4_e2b` text encoder**, **~60-65% overall reuse**.
Nothing else in the 61-backend TTS fleet is close — not F5-DiT/cosyvoice3/vibevoice/orpheus shaped.
The codec, historically the hardest part of a new TTS backend, is a **free win**.

## 3. Converter sketch (`convert_breeze_tts2_to_gguf.py`)
Single GGUF, arch `breeze-tts2`; codec stays the existing `qwen3-tts-tokenizer-12hz` GGUF wired in
as a registry companion — do not repack it.
| HF family | GGUF prefix | Notes |
|---|---|---|
| `text_encoder.embed_tokens` / `.layers.N.*` / `.norm` | `te.*` | 26L; store `layer_types` + both rope configs as KV |
| `text_encoder_proj.weight` | `te_proj.weight` | 1152→2048 |
| `backbone_model.embed_tokens.embed_audio_tokens` | `bb.audio_embd` | [16*2051, 2048]; bake `audio_tokens_offsets` in converter |
| `backbone_model.layers.N.*` (incl. `q_norm`/`k_norm`), `.norm` | `bb.*` | Qwen3, theta 1e6, no scaling |
| `lm_head.weight` | `bb.output` | 2052 rows |
| `depth_decoder.model.embed_tokens` / `.inputs_embeds_projector` / `.layers.N.*` / `.norm` | `dd.*` | 12L, llama3 rope scaling |
| `depth_decoder.codebooks_head.weight` | `dd.cb_head` | split `[15,1024,2051]` → 15 2-D tensors |
| `embed_text_tokens`, `codec_model.*` | — | **DROP** (1.16 GB, unused at inference) |

KV: `audio_num_codebooks=16`, `audio_vocab_size=2051`, `codebook_pad=2050`, `audio_token_id=262144`,
`audio_eos=262145`, speaker/instruction ids from `text_encoder_special_tokens_config`, sampling
defaults from `generation_config.json` (temp 0.9, depth temp 0.9, max_new_tokens 750). Tokenizer:
Gemma 262158-vocab from `tokenizer.json`.

## 4. Validation plan
Reference oracle exists and runs — **but GPU-only**: README requires Linux + CUDA, ~7.7 GiB VRAM
eager, and 6.97 GB bf16 will not fit the 8 GB VPS. Deps pinned and public (torch 2.9.1,
transformers 4.57.3, `qwen-tts==0.1.1`). → Ref dumper runs on **Kaggle GPU**.
1. Stage dumps via `attn_implementation="eager"` (breeze.py:55 resolver) + `output_hidden_states`:
   text-encoder final hidden → `text_encoder_proj` out → backbone per-layer h → codebook-0 logits →
   depth-decoder per-codebook logits → 16×T codes. `crispasr-diff` per stage, cos ≥ 0.999.
2. Codec validated independently: feed Python-dumped 16×T codes to the existing `qwen3-tts` path
   (`qwen3_tts_codec_extract_stage`, qwen3_tts.h:265) and diff PCM — decouples codec from LLM bugs.
3. Greedy (temp=0, cfg=1.0) end-to-end token-ID equality vs Python.
4. Mandatory ASR roundtrip on generated audio (en + zh), plus the reference e2e control arm.
5. CFG branches diffed separately: uncond-only, then ref, then ins.

## 5. Effort — **~12 days** (range 10-14)
Converter + GGUF KV/tokenizer 2 · T5Gemma2 text-encoder graph (port `gemma4_e2b.cpp`) 2 ·
Qwen3 backbone + summed-codebook embed (port `csm_tts.cpp`) 1.5 · depth decoder + 15-way
codebooks head 1.5 · prompt templates / speaker+instruction tokens / ref-audio encode wiring 1 ·
CFG multi-branch decode 1.5 · codec integration (reuse `qwen3_tts`) 0.5 ·
Kaggle ref dumper + diff harness + roundtrip 2.

## 6. Recommendation: **GO**
The usual long pole for a new TTS backend — a novel neural codec — is already done and bit-identical
to one we ship, and the LLM half is a fork of a backend we already have. NC is the only structural
blocker and we already handle it.
**Single biggest risk: CFG multi-branch batched decoding.** Voice Design and Voice Direction — the
headline features and the reason the model is interesting — need the backbone run over 2-3 prompt
branches per step (templates.py:100-104, fast_streaming.py `_BranchBatch`). Our TTS ggml graphs are
overwhelmingly batch-1: either widen the graph to n_branch (touching every KV path in the new
backbone) or run branches serially at 2-3× cost, forfeiting the low-latency selling point. Scope it
explicitly — a batch-1-only backend ships Voice Clone but silently drops two of four capabilities.
Secondary risk: `text_encoder_feature_layer_idx` defaults to `(-1,)` and `text_encoder_layer_projs`
is absent from the checkpoint, so the dimfusion path is inert — confirm on the dump before building.

**Registry NC gate required.** Add `license` prose to the `crispasr_model_registry.cpp` entry in the
existing form (cf. `voxtral-tts` :389, `raon-opentts` :1093): `"other — NON-COMMERCIAL use only
(BreezeBlue Research and Non-Commercial License, RESONIA INC; ..."`.
`crispasr_license_requires_acceptance()` already covers `other`, so `-m auto` will demand
`CRISPASR_ACCEPT_LICENSE`. On the HF GGUF repo also ship the LICENSE copy, the verbatim NOTICE
string, and the "Derived from Breeze TTS 2 …" model-card line; §4's naming clause bars "Breeze" as
the *primary product* name — `cstr/breeze-tts-2-GGUF` and backend key `breeze-tts2` are descriptive
attribution and fine.


## Phase-1 addenda (2026-09-03)

Three config decoys now known, all confirmed against the code: the nested `backbone_config`
governs rope (not the top-level keys), backbone `rms_norm_eps` is **1e-6** from that nested
block (not the top-level 1e-5), and `use_bidirectional_attention` is inverted by the compat
module (see above). Converter, port notes and the Kaggle ref-dump kernel are in the tree;
see `docs/breeze-tts-2-port-notes.md`.
