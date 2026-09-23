# §436 — Dolphin (DataoceanAI) ASR family

## NOW — active work

**2026-09-23: F16 parity achieved on the first run** (kernel
`chr1s4/crispasr-436-dolphin-pipeline`, commit 5776f678). Against the upstream
package (dither 0) on paraformer_zh.wav and jfk.wav, every `crispasr-diff` stage
PASSes: fbank 1.000000, subsample ≥0.999999, all 12 E-Branchformer blocks
≥0.999417, encoder output ≥0.999064, CTC log-probs 1.000000 (argmax identical on
every frame), and the decoded text — prompt tokens, predicted <zh><CN>, beam +
rescoring — is IDENTICAL to upstream on both clips. GGUFs on
cstr/dolphin-cn-dialect-small-streaming-GGUF (F16 789 MB, Q8_0 443 MB, Q4_K 258 MB).

Open: Q8_0/Q4_K degrade more than expected in blocks 6-11 (Q8_0 jfk worst-frame
encoder cos 0.34, although CTC argmax still matches every frame and the zh text is
identical at Q8_0 and Q4_K). Next: quant A/B keeping candidate tensors at F16.

Branch `feat/436-dolphin`. Target: `DataoceanAI1/dolphin-cn-dialect-small-streaming`
(asked for in #436), with the runtime written for the whole Dolphin family
(base / small / cn / cn.streaming / cn.prompt share one architecture).
X-ASR (the other half of #436, an icefall Zipformer2 transducer) is a separate
encoder family and follows Dolphin.

## Blueprint (read from DataoceanAI/Dolphin @ 78ea615, not the card)

`dolphin/transcribe.py::transcribe` → `ASRModel.decode(methods=["attention_rescoring"],
beam_size=10)` with the defaults of `decode()`:

| knob | value | where |
| --- | --- | --- |
| features | Kaldi fbank, 80 bins, 25/10 ms, dither only in training | `train.yaml` fbank_conf |
| normalisation | global CMVN from `global_cmvn` (JSON) | cmvn_conf |
| encoder | E-Branchformer, 12 blocks, d=768, 12 heads, cgMLP 3072 (kernel 31), merge conv kernel 31, rel-pos self-attention, conv2d subsampling, **causal**, dynamic chunk trained | encoder_conf |
| inference chunking | `decoding_chunk_size=-1` → full context even for the streaming checkpoint | `decode()` default |
| CTC | vocab 18173, blank 0 | ctc_conf |
| decoder | Transformer, 12 blocks, 12 heads, 3072 FFN | decoder_conf |
| search | CTC prefix beam (10) → attention rescoring, `ctc_weight=0.0`, `reverse_weight=0.0` | `decode()` defaults |
| prompt | `<sos>` + language token + region token (two-level, e.g. `<zh>` `<CN>`) | tokenizer / transcribe |
| hotwords | optional deep biasing (`context_module: cppn`, 2 layers) — out of scope for v1 | hotword.py |

Checkpoint SHA-256 is pinned in `dolphin/model_registry.py`
(small.cn.streaming: `bba8688e…`) — verify before converting (see LEARNINGS:
a CIFS download silently corrupted a checkpoint on 2026-09-23).

## Steps

1. Read `model.py` E-Branchformer / rel-pos attention / cgMLP line by line.
2. Converter (`models/convert-dolphin-to-gguf.py`), CMVN + units baked in.
3. Reference dump backend (`tools/reference_backends/dolphin.py`) — Kaggle CPU.
4. Runtime + diff harness arm; parity per stage; then decoded-output check.
5. Registry, quantizer rules, CLI/C-ABI wiring (12-point checklist).

## Encoder facts the shapes would allow you to get wrong (model.py, read line by line)

1. **"rel_pos" is not relative.** `pos_enc_layer_type: rel_pos` → WeNet's legacy
   `RelPositionalEncoding`: input scaled by √d, `pos_emb = pe[0:T]` (ABSOLUTE
   positions 0..T-1, sin/cos **interleaved** 0::2 / 1::2, max_len 5000).
2. **No rel_shift.** `use_sdpa: true` takes the SDPA branch of
   `RelPositionMultiHeadedAttention`: `bd = (q + pos_bias_v)·(linear_pos(pe))ᵀ`
   (T×T, no shift) is folded into the attention mask, `ac = (q + pos_bias_u)·kᵀ`,
   scores = (ac + bd)/√d_k. FastConformer's rel-pos code (2T-1, shifted) would be
   silently wrong here.
3. **CSGU pads BEFORE its LayerNorm.** Causal cgMLP: `x_g` is left-padded with
   (kernel-1) zeros, THEN LayerNorm, THEN depthwise conv — so the padded frames the
   conv sees are the LayerNorm *bias*, not zeros. Gate activation = identity;
   out = x_r * conv(norm(x_g)); cgMLP = Linear(768→3072) + exact GELU → CSGU →
   Linear(1536→768).
4. **Merge = merge_proj(concat + dwconv(concat)).** Depthwise conv over the 1536
   concat channels, kernel 31, causal (left pad 30, zeros — no norm here), plus
   the un-convolved concat, then Linear(1536→768), added to the residual.
5. **Layer order:** macaron FFN (½·, Swish) → [norm_mha → rel-pos MHA] ‖
   [norm_mlp → cgMLP] → merge → FFN (½·, Swish) → **norm_final per layer**, and
   the encoder's `after_norm` on top of the last layer.
6. **Attention is full-context at inference.** `use_dynamic_chunk` + 
   `decoding_chunk_size=-1` → chunk = whole utterance. Only the two depthwise
   convs are causal.
7. **Subsampling:** Conv2d(1→768, 3, s2)+ReLU, Conv2d(768→768, 3, s2)+ReLU,
   flatten (b, t, c·f) channel-major, Linear(768·19 → 768). Mask keeps [2::2][2::2].

## Features, decoder, search

8. **Features:** `waveform * 32768` → `torchaudio.compliance.kaldi.fbank(**fbank_conf)`
   = 80 bins, 25/10 ms, povey window, snip_edges, **dither 0.1** — upstream's own
   inference is therefore random run to run. The reference dump must pass
   dither=0; the C++ uses 0 (core_kaldi fbank with int16 scale), and the model
   card should say so. Then global CMVN (JSON `global_cmvn`).
9. **Decoder:** pre-norm Transformer, 12 blocks, ReLU FFN, embedding × √d plus
   ABSOLUTE interleaved sinusoids, `after_norm`, untied output Linear.
10. **Prompt:** `[sos, <lang>, <region>, <asr>, <notimestamp>]`. When the caller
    gives no language/region they are predicted greedily by the decoder from
    `[sos]` (`predict_lang_region_timestamp`), one step each.
11. **Search:** WeNet CTC prefix beam (beam 10, per-frame top-k = beam) → attention
    rescoring: score(hyp) = Σ decoder log-p of hyp tokens at offset 4 + log-p(eos);
    ctc_weight 0, reverse_weight 0. Output tokens keep the 4 prefix tokens.
    core_ctc::prefix_beam_search and core_kaldi fbank exist and are reusable.
