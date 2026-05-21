# Paraformer-zh NAR-ASR port — handover

> Self-contained brief for finishing the **Paraformer-zh** port on
> branch `paraformer-port` (pushed to `cohere`). The encoder + CIF
> predictor work; the decoder produces garbage. Estimated 2–4 hours
> to debug the decoder, wire the diff harness, and validate.

## Where things stand

**Branch:** `paraformer-port` (one commit ahead of `cohere/main`).

```
4194142 feat(paraformer): initial Paraformer-zh NAR-ASR port (WIP)
fb8b043 docs(learnings): funasr use_low_frame_rate config mismatch (PLAN #99)
```

**Files added/modified:**

| File | Status |
|------|--------|
| `models/convert-paraformer-to-gguf.py` | Working. Produces 421 MB F16 GGUF. |
| `src/paraformer.h` | Public C API (init/free/transcribe/extract_stage). |
| `src/paraformer.cpp` | ~550 LOC runtime. Encoder + CIF work. Decoder broken. |
| `examples/cli/crispasr_backend_paraformer.cpp` | Thin adapter, `--backend paraformer`. |
| `examples/cli/crispasr_backend.cpp` | Dispatch + list entries added. |
| `examples/cli/CMakeLists.txt` | Source + link entries for paraformer. |
| `src/CMakeLists.txt` | `add_library(paraformer ...)` block. |
| `tools/reference_backends/paraformer.py` | **NOT YET WRITTEN** — needed for diff harness. |

**GGUF already converted** (on the VPS):
```
/mnt/storage/paraformer-zh/paraformer-zh-f16.gguf   421 MB
```

**Upstream model snapshot** (on the VPS):
```
/mnt/storage/paraformer-zh-upstream/
  model.pt  config.yaml  tokens.json  am.mvn  example/asr_example.wav  seg_dict
```

## The model

FunASR **Paraformer-zh** (`funasr/paraformer-zh` on HuggingFace).
~220 M params, non-autoregressive (single-pass decode), primarily
Mandarin Chinese + English. Character-level tokenizer, vocab 8404.
License: FunASR Model License (commercial OK with attribution).

Architecture:

```
Audio (16 kHz mono)
  → Kaldi fbank (80 mel, hamming, 25 ms / 10 ms)
  → LFR: stack 7, stride 6 → (T_lfr, 560)
  → CMVN (am.mvn: AddShift + Rescale, 560-dim)
  → SANMEncoder: 1 entry block (560→512) + 49 main blocks (512→512)
      each: norm1 → fused QKV (1536,in) → FSMN(k=11) + MHA(4 heads) → norm2 → FFN(2048)
  → CifPredictorV2:
      Conv1d(512,512,k=3,pad=1) → ReLU → Linear(512,1) → sigmoid
      → CIF accumulation loop (fire when alpha_accum ≥ 1.0)
      → acoustic_embeds: (N_tokens, 512)
  → ParaformerSANMDecoder: 16 blocks
      each: norm1 → FSMN(k=11, NO QKV) → norm2 → cross-attn(Q=dec, KV=enc) → norm3 → FFN(2048, internal LN)
  → decoders3: 1 post-processing block (norm → FFN with internal LN)
  → after_norm → output_layer(512→8404) → argmax → characters
```

Key architectural points:
- The encoder is **identical** to funasr-nano's SANM encoder and
  reuses `core_sanm::build_block()`.
- The decoder "self-attention" is **FSMN only** (depthwise conv,
  no Q/K/V). The only Q/K/V attention is cross-attention to the
  encoder output.
- The decoder FFN has an **internal LayerNorm** between w_1 and the
  activation: `w_1 → LayerNorm → ReLU → w_2` (no w_2 bias).
- Cross-attention uses a **fused K+V projection** from encoder:
  `linear_k_v` outputs (1024, T_enc) which is split into K (512)
  and V (512).

## What works

1. **Converter** (`models/convert-paraformer-to-gguf.py`): reads
   `model.pt` + `config.yaml` + `tokens.json` + `am.mvn`, writes
   GGUF with 956 tensors. Tested, verified tensor shapes.

2. **Feature extraction**: Kaldi fbank → LFR → CMVN. Identical to
   funasr-nano, using `core_kaldi::compute_fbank` + `core_lfr::stack`.

3. **Encoder**: 50 SANM blocks via `core_sanm::build_block()`. Runs
   correctly, produces (512, T_lfr) encoder output. Verified: the
   encoder graph computes without error.

4. **CIF predictor**: CPU-side Conv1d + sigmoid + accumulation loop.
   On the 13 s Chinese test audio: T_lfr=218 → 110 tokens predicted.
   This matches the expected character count for the reference
   transcript. **Important F16 fix**: CIF weights are F16 in the
   GGUF; the code dequantizes them to F32 before the CPU-side conv
   (see `read_f32()` lambda). Original crash was from reading F16
   data with an F32 byte count.

5. **CLI wiring**: `--backend paraformer` dispatches to the adapter.
   Build + run works end-to-end (no crash). Output is garbage
   characters, not a crash.

## What's broken: the decoder

Running on the Chinese test audio produces:
```
的的他非非的非的他非他非非的非非的非非非的非他的他非非的非非非非的非非非非非非非的非…
```

Expected (from Python reference):
```
正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
```

The decoder runs 16 blocks + post block + output layer → argmax.
The logits come out, argmax picks tokens, characters are emitted.
But the characters are wrong — the decoder hidden states are likely
corrupted.

## Likely decoder bugs (ordered by probability)

### 1. Missing embedding addition to acoustic_embeds

The upstream `ParaformerSANMDecoder.forward` starts with:

```python
x = self.embed(ys_pad)  # lookup: (N, vocab) → (N, 512)
# … but ys_pad during inference is zeros (no teacher forcing)
# WAIT — actually, during inference the model calls:
#   decoder(encoder_out, encoder_out_lens, acoustic_embeds, ys_lengths)
# where acoustic_embeds are the CIF-fired vectors. The decoder's
# forward does:
#   x = self.embed(ys_pad)   ← this is SKIPPED during inference?
#   or: x = acoustic_embeds  ← passed directly?
```

**This needs investigation.** Check `funasr/models/paraformer/model.py`
`Paraformer.calc_predictor()` and `Paraformer.cal_decoder_with_predictor()`
to see exactly what the decoder receives as input. If the embedding
table IS used during inference (e.g., as a learned bias added to the
CIF output), the current C++ code is missing it.

To check from the VPS:
```python
python3 -c "
import inspect
from funasr.models.paraformer.model import Paraformer
print(inspect.getsource(Paraformer.cal_decoder_with_predictor))
"
```

The GGUF has the embedding: `paraformer.dec.embed.w` shape (8404, 512).

### 2. Cross-attention K/V split ordering

The fused `linear_k_v` produces (1024, T_enc). The C++ splits it:
```cpp
K_ = view_2d(kv, D, T_enc, row_bytes, 0);           // first half
V  = view_2d(kv, D, T_enc, row_bytes, D * sizeof(float)); // second half
```

This assumes K is the first 512 dims and V is the second 512.
Upstream does:
```python
linear_k_v = nn.Linear(size, 2*size)
# forward:
kv = self.linear_k_v(encoder_out)
k, v = kv.chunk(2, dim=-1)
```

`chunk(2, dim=-1)` splits along the last dim: first half = K, second
half = V. In ggml the last dim is ne[0], so the split should be along
ne[0]. The current code splits along ne[0] via the byte offset.
**This should be correct**, but verify by dumping the first few K/V
values and comparing with Python.

### 3. FSMN in the decoder: causal vs bidirectional padding

The encoder FSMN uses `pad_w = (K-1)/2` which is symmetric
(bidirectional). The decoder FSMN might need **left-only** (causal)
padding since it operates on the token sequence. Check upstream:

```python
# In MultiHeadedAttentionSANMDecoder or DecoderLayerSANM:
# Is the FSMN conv1d causal? i.e., padding = (K-1, 0) not ((K-1)/2, (K-1)/2)?
```

If the decoder FSMN is causal, the padding in `build_decoder_block()`
is wrong (currently uses `(K-1)/2` symmetric).

### 4. Decoder FFN activation

The C++ does `w_1 → LN → ReLU → w_2`. But upstream might use a
different activation. Check `PositionwiseFeedForwardDecoderSANM` —
it might use `nn.ReLU()` (default) or `Swish` or something else.

### 5. Logits readout transposition

The C++ reads logits as `logits_data[n * V + v]`. But the ggml
tensor `logits` has ne = (V, N). Element at ne-index [v, n] is at
byte offset `(v + n * V) * 4`. So `logits_data[n * V + v]` for
varying v gives a contiguous block of V floats = one token's logits.
**This looks correct** but double-check.

## How to debug

### Quick path: dump decoder input vs Python reference

The fastest way to find the bug is:

1. **Dump the CIF output** (acoustic_embeds) from both C++ and Python
   and compare. If they match, the encoder + CIF are confirmed.

2. **Dump the decoder block 0 output** from both and compare. If
   block 0 output is already wrong, the bug is in the decoder block
   architecture (FSMN / cross-attn / FFN).

3. **Dump the decoder's INPUT** — what Python actually feeds to
   `self.decoders[0]`. This will reveal whether the embedding table
   is used.

### Setting up the diff harness

The crispasr-diff tool compares C++ stage activations against a
Python reference GGUF. To use it for paraformer:

**Step 1: Write `tools/reference_backends/paraformer.py`.**

Use `tools/reference_backends/funasr.py` as the template. The
structure is:

```python
DEFAULT_STAGES = [
    "raw_audio", "mel_features",
    "encoder_output",
    "cif_alphas", "acoustic_embeds",
    "decoder_output",      # post after_norm, pre output_layer
    "generated_text",
] + [f"encoder_layer_{i}" for i in range(50)]

def dump(model_dir, audio_path, stages, ...):
    # 1. Load model via funasr.AutoModel
    # 2. Hook encoder layers + decoder sub-modules
    # 3. Run inference
    # 4. Return dict of stage_name → numpy array
    ...
```

Key hooks needed:
- `encoder.encoders0[0]` through `encoder.encoders[48]` — encoder layer outputs
- `encoder.after_norm` — encoder_output
- Predictor: capture `alphas` and `acoustic_embeds` by monkey-patching
  `Paraformer.cal_decoder_with_predictor` (similar to how funasr.py
  monkey-patches `MultiHeadedAttention.forward`)
- Decoder blocks: hook each `decoder.decoders[i]`
- `decoder.after_norm` — decoder_output
- `generated_text` — via `m.generate()` or `m.inference()`

**Step 2: Add paraformer to `crispasr_diff_main.cpp`.**

Copy the funasr section (around line 3470) and adapt:
```cpp
} else if (backend_name == "paraformer") {
    paraformer_context_params cp = paraformer_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 0;
    paraformer_context* ctx = paraformer_init_from_file(model_path.c_str(), cp);
    ...
    std::vector<std::string> stages;
    stages.push_back("mel_features");
    for (int i = 0; i < 50; i++)
        stages.push_back("encoder_layer_" + std::to_string(i));
    stages.push_back("encoder_output");
    stages.push_back("acoustic_embeds");
    // decoder stages...
    stages.push_back("generated_text");
    ...
}
```

**Step 3: Implement `paraformer_extract_stage()`** in `paraformer.cpp`.
Currently it's a stub returning nullptr. Follow the funasr pattern:
set `ctx->requested_stage`, run the full forward, find the named
tensor in the graph, copy it out.

**Step 4: Generate reference dump + run diff.**

```bash
cd /tmp
python tools/dump_reference.py --backend paraformer \
    --model-dir /mnt/storage/paraformer-zh-upstream \
    --audio /mnt/storage/paraformer-zh-upstream/example/asr_example.wav \
    --output /mnt/storage/paraformer-zh/paraformer-zh-ref.gguf

crispasr-diff paraformer \
    /mnt/storage/paraformer-zh/paraformer-zh-f16.gguf \
    /mnt/storage/paraformer-zh/paraformer-zh-ref.gguf \
    /mnt/storage/paraformer-zh-upstream/example/asr_example.wav
```

## Model downloads (fresh box)

```bash
# Upstream Python model
mkdir -p /mnt/storage/paraformer-zh-upstream
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('funasr/paraformer-zh',
    local_dir='/mnt/storage/paraformer-zh-upstream',
    local_dir_use_symlinks=False)
"

# Pip deps
pip install funasr torch torchaudio

# Convert to GGUF
mkdir -p /mnt/storage/paraformer-zh
python3 models/convert-paraformer-to-gguf.py \
    --input /mnt/storage/paraformer-zh-upstream \
    --output /mnt/storage/paraformer-zh/paraformer-zh-f16.gguf
```

## Build

```bash
# Use a worktree (never checkout branches in the main working tree)
git worktree add /tmp/crispasr-paraformer paraformer-port
cd /tmp/crispasr-paraformer

cmake -B /tmp/paraformer-build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build /tmp/paraformer-build -j2 --target crispasr-cli

# Test
/tmp/paraformer-build/bin/crispasr --backend paraformer \
    -m /mnt/storage/paraformer-zh/paraformer-zh-f16.gguf \
    -f /mnt/storage/paraformer-zh-upstream/example/asr_example.wav \
    --no-prints
```

## Reference transcript for the test audio

Python (`funasr.AutoModel.generate()` on `example/asr_example.wav`):
```
正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
```

## GGUF tensor naming convention

### Encoder
```
paraformer.cmvn_shift          (560,)     F32
paraformer.cmvn_scale          (560,)     F32
paraformer.enc0.blk.0.norm1.w  (560,)     F32   ← entry block, norm1 size = input_size
paraformer.enc0.blk.0.attn.qkv.w (560, 1536) F16
...
paraformer.enc.blk.{0-48}.*    same as funasr SANM naming
paraformer.enc.after_norm.{w,b} (512,)    F32
```

### CIF predictor
```
paraformer.cif.conv.w          (3, 512, 512) F16  ← Conv1d(512,512,3)
paraformer.cif.conv.b          (512,)        F32
paraformer.cif.out.w           (512, 1)      F16  ← Linear(512,1)
paraformer.cif.out.b           (1,)          F32
```

### Decoder
```
paraformer.dec.blk.{0-15}.norm1.{w,b}     (512,)     F32
paraformer.dec.blk.{0-15}.fsmn.w           (512, 11)  F16  ← depthwise conv
paraformer.dec.blk.{0-15}.norm2.{w,b}     (512,)     F32
paraformer.dec.blk.{0-15}.cross.q.{w,b}   (512, 512) F16 / (512,) F32
paraformer.dec.blk.{0-15}.cross.kv.{w,b}  (512, 1024) F16 / (1024,) F32
paraformer.dec.blk.{0-15}.cross.out.{w,b} (512, 512)  F16 / (512,) F32
paraformer.dec.blk.{0-15}.norm3.{w,b}     (512,)     F32
paraformer.dec.blk.{0-15}.ffn.l1.{w,b}    (512, 2048) F16 / (2048,) F32
paraformer.dec.blk.{0-15}.ffn.norm.{w,b}  (2048,)    F32  ← internal LN!
paraformer.dec.blk.{0-15}.ffn.l2.w        (2048, 512) F16  ← no bias
paraformer.dec.post.norm1.{w,b}            (512,)     F32
paraformer.dec.post.ffn.*                  same pattern as above
paraformer.dec.after_norm.{w,b}            (512,)     F32
paraformer.dec.output.{w,b}                (512, 8404) F16 / (8404,) F32
paraformer.dec.embed.w                     (8404, 512) F16  ← may need for decoder input
```

## Existing code to reuse

| Component | Module | Used by |
|-----------|--------|---------|
| Kaldi fbank | `core/kaldi_fbank.h` `core_kaldi::compute_fbank` | funasr, sensevoice, chatterbox |
| LFR stacking | `core/lfr.h` `core_lfr::stack` | funasr, sensevoice |
| SANM block | `core/sanm.h` `core_sanm::build_block` | funasr, sensevoice, **paraformer encoder** |
| GGUF loader | `core/gguf_loader.h` | all backends |
| Greedy decode | `core/greedy_decode.h` | parakeet, canary, etc. |

The decoder is new code — FSMN self-attn + cross-attn + FFN-with-LN.
If Paraformer works well, the decoder block could be extracted into
`core/paraformer_decoder.h` for reuse by future NAR models.

## Validation gate

The fix is correct when:

1. `generated_text` matches the reference byte-for-byte on the
   Chinese test audio (`example/asr_example.wav`).
2. `encoder_output` cos_min ≥ 0.999.
3. `acoustic_embeds` cos_min ≥ 0.99 (CIF is a sequential loop with
   float accumulation — expect some drift).
4. The English JFK sample (`samples/jfk.wav`) produces recognizable
   English output (paraformer-zh handles English too).

## When you finish

- Rebase `paraformer-port` onto `cohere/main`.
- Fast-forward merge into main; delete the branch + worktree.
- Run `tools/test-all-backends.py` to check no regressions.
- Convert Q4_K + Q8_0 quants, upload all three to a new HF repo
  `cstr/paraformer-zh-GGUF`.
- Update PLAN.md, HISTORY.md, README.md (ASR table).
- Add to the model registry for `-m auto` resolution.

## Debug cheat sheet

```python
# Check what the decoder actually receives as input:
python3 -c "
import inspect
from funasr.models.paraformer.model import Paraformer
print(inspect.getsource(Paraformer.cal_decoder_with_predictor))
"

# Run Python reference on test audio:
python3 -c "
from funasr import AutoModel
m = AutoModel(model='/mnt/storage/paraformer-zh-upstream', device='cpu')
res = m.generate(input='/mnt/storage/paraformer-zh-upstream/example/asr_example.wav')
print(res[0]['text'])
"

# Quick C++ test:
/tmp/paraformer-build/bin/crispasr --backend paraformer \
    -m /mnt/storage/paraformer-zh/paraformer-zh-f16.gguf \
    -f /mnt/storage/paraformer-zh-upstream/example/asr_example.wav \
    --no-prints
```

## Files to look at first

| Path | Why |
|------|-----|
| `src/paraformer.cpp` lines 390-440 | CIF predictor (working but CPU-only, could move to ggml graph later) |
| `src/paraformer.cpp` lines 445-480 | `build_decoder_block()` — the broken decoder block |
| `src/paraformer.cpp` lines 490-500 | `build_decoder_post()` — post-processing block |
| `src/paraformer.cpp` lines 550-600 | `paraformer_transcribe_impl()` — full pipeline |
| `funasr/models/paraformer/decoder.py` | Upstream decoder (`ParaformerSANMDecoder.forward`) |
| `funasr/models/paraformer/model.py` | `cal_decoder_with_predictor()` — how decoder is called |
| `funasr/models/paraformer/cif_predictor.py` | `CifPredictorV2.forward` |

— end of handover —
