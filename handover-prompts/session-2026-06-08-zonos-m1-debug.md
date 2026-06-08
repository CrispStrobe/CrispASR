# Handover — 2026-06-08 Zonos TTS debug on M1 MacBook (16 GB)

## Situation

Zonos TTS (§130) runs end-to-end on GPU (P100) without crashes:
AR backbone (26L 2048d GQA) + CFG + 9-codebook delay pattern + DAC
decoder (4-block upsampling conv) all produce output. But the audio
is unintelligible — ASR roundtrips as "Yeah." regardless of input.

**Key symptom:** `cb0 argmax=110` is identical across ALL runs — different
speaker embeddings, different text, different phoneme languages (even
Esperanto vs English). The model's first codebook prediction doesn't
change with conditioning. This points to a bug in how the conditioning
prefix is injected into the backbone, or a fundamental numerical issue.

## What to investigate on M1

The M1 16 GB can run the Q4_K model (~900 MB) on Metal. Unlike the VPS
(CPU-only, 8 GB, too slow for AR with CFG), Metal should complete a
synthesis in 1-2 minutes. The goal: **stage-by-stage diff** between the
Python reference and C++ runtime.

### Step 1: Build

```bash
cd /path/to/CrispASR
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build --target crispasr-cli -j$(sysctl -n hw.ncpu)
```

### Step 2: Download models

```bash
# Q4_K AR model (~900 MB)
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('cstr/zonos-v0.1-transformer-GGUF', 'zonos-v0.1-transformer-q4_k.gguf'))
"
# DAC codec (~104 MB)  
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('cstr/dac-44khz-GGUF', 'dac-44khz-f16.gguf'))
"
# Pre-computed speaker embedding (128-d LDA from JFK)
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('cstr/zonos-v0.1-transformer-GGUF', 'jfk_speaker_emb.bin'))
"
```

### Step 3: Run C++ and capture diagnostics

```bash
export ZONOS_SPEAKER_EMB_PATH=/path/to/jfk_speaker_emb.bin
./build/bin/crispasr \
    --backend zonos-tts \
    -m /path/to/zonos-v0.1-transformer-q4_k.gguf \
    --codec-model /path/to/dac-44khz-f16.gguf \
    --tts "Hello world" \
    --tts-output /tmp/zonos_cpp.wav \
    --seed 42 -v 2>&1 | tee /tmp/zonos_cpp.log
```

Check the log for:
- `lang=en-us` (not `lang=eo` — the old bug)
- `phoneme tokens` count (should be ~12 for "Hello world" with espeak)
- `DIFF cond prefill cb0 argmax=???` — is it 110 on Metal too?
- `AR generated N delayed steps (eos=0/1)` — does it hit EOS?

The C++ also dumps codes to `/mnt/storage/zonos-tts/cpp_codes.txt`
(hardcoded path — change via env if needed).

### Step 4: Run Python reference

```bash
pip install zonos soundfile
python3 tools/reference_backends/zonos_tts_reference.py \
    --text "Hello world" \
    --output /tmp/zonos_ref.wav \
    --dump-dir /tmp/zonos_ref_dump/ \
    --dump-codes /tmp/zonos_ref_codes.txt \
    --seed 42 \
    --language en-us \
    --max-tokens 200
```

This dumps:
- `conditioning_prefix.npy` — the full prefix tensor
- `backbone_layer_NN.npy` — per-layer activations
- `backbone_layer_NN_attn.npy` — attention outputs
- `backbone_layer_NN_mlp.npy` — MLP outputs
- `backbone_norm_f.npy` — final norm
- `output_codes.npy` — generated codes
- `output_pcm.npy` — decoded audio

### Step 5: Compare

```python
import numpy as np

# Compare conditioning prefix
ref_cond = np.load("/tmp/zonos_ref_dump/conditioning_prefix.npy")
print(f"ref cond: shape={ref_cond.shape} mean={ref_cond.mean():.4f} std={ref_cond.std():.4f}")
print(f"ref cond first5: {ref_cond.flatten()[:5]}")

# Compare codes
ref_codes = np.load("/tmp/zonos_ref_dump/output_codes.npy")
print(f"ref codes shape: {ref_codes.shape}")
print(f"ref cb0 first10: {ref_codes[0, 0, :10]}")

# Check if Python ref produces speech
# Play /tmp/zonos_ref.wav — does it sound like "Hello world"?
# ASR roundtrip:
# ./build/bin/crispasr --backend parakeet -m auto -f /tmp/zonos_ref.wav -otxt
```

**Key questions:**
1. Does Python ref produce intelligible "Hello world"?
2. Is the C++ `cb0 argmax` the same as Python's first token?
3. Do the conditioning prefix values match (within Q4_K tolerance)?
4. At which backbone layer do the activations diverge?

## Known bugs (already fixed)

- ~~language_id=25 → Esperanto~~ Fixed: name-based lookup, en-us=24
- ~~DAC ggml_view_2d crash~~ Fixed: use `core_convt::convt1d_crop`
- ~~"auto" language code~~ Fixed: treat as no-op
- ~~Kaggle HF token path~~ Fixed: scan `/kaggle/input/datasets/`

## Suspected root cause

The `cb0 argmax=110` being constant suggests one of:

1. **Conditioning prefix is numerically degenerate** — after project +
   LayerNorm, the prefix collapses to near-zero variance, and the model
   treats it as "no input". The C++ uses eager CPU math for the prefix
   conditioner (not the ggml graph), which may have a precision issue
   with the Fourier feature computation or the project-then-norm order.

2. **KV cache write is broken** — the prefix gets "prefilled" via
   `run_backbone(prefix, T, n_past=0)` but the cache entries aren't
   retained for subsequent single-token steps. Check: does `n_past`
   increment correctly? Does the sched re-use the same KV buffer?

3. **Q4_K precision** — the attention weights at Q4_K may not preserve
   enough precision for the conditioning signal. Test: try F16 GGUF
   (`zonos-v0.1-transformer-f16.gguf`, ~1.8 GB, fits in 16 GB M1).

4. **RoPE type mismatch** — Zonos uses `rotary_emb_interleaved=true`
   (consecutive pairs), mapped to `GGML_ROPE_TYPE_NORMAL`. If the
   upstream model actually uses non-interleaved RoPE, the attention
   pattern would be scrambled.

## Key files

| What | Path |
|------|------|
| Zonos C++ runtime | `src/zonos_tts.cpp` |
| DAC decoder | `src/core/dac_decoder.h` |
| Attention core | `src/core/attention.h` (lines 515-553: fused QKV) |
| FFN core | `src/core/ffn.h` |
| Python reference | `tools/reference_backends/zonos_tts_reference.py` |
| CLI adapter | `examples/cli/crispasr_backend_zonos.cpp` |
| Converter (AR) | `models/convert-zonos-to-gguf.py` |
| Converter (DAC) | `models/convert-dac-to-gguf.py` |

## Kaggle kernel in flight

Kernel v7/v8 (`chr1str/crispasr-zonos-tts-gpu-test`) runs both C++ and
Python reference on P100 with ASR roundtrip of both. v8 also dumps
intermediate activations. Check results:

```bash
PYTHONPATH=/tmp/kaggle_pkg python3 -m kaggle kernels output \
    chr1str/crispasr-zonos-tts-gpu-test -p /tmp/zonos-results
```
