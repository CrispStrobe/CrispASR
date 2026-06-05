# Piper TTS (#128) — Handover Prompt (DONE)

> **Status: COMPLETE.** Committed `a3bb6586`. This handover is rewritten
> post-mortem as a reference for how future model ports should be structured.

## What this is

Native C++ ggml runtime for rhasspy/piper VITS TTS models. End-to-end from
phoneme IDs to 22.05 kHz mono PCM. 14 bugs found and fixed, all verified
via a per-stage diff harness (cos ≥ 0.9996 at every pipeline stage against
ONNX reference).

## Methodology that worked

### 1. Build the diff harness FIRST

Before writing ANY runtime code, build:
- **Python reference dumper** (`tools/reference_backends/piper_tts_diff.py dump`)
  that patches ONNX intermediate node outputs onto the model graph and saves
  every stage to `.npz`
- **C++ stage dumper** (binary `--dump <dir>` flag) that writes matching `.bin`
  files at every pipeline boundary
- **Comparison script** (`piper_tts_diff.py compare`) that loads both,
  handles the (T,C)→(C,T) transpose between C++ row-major and ONNX layout,
  and reports cosine similarity + max absolute error for each stage

The harness lets you answer "where does it first diverge?" in seconds,
not hours. Every fix becomes: run harness → find first red line → fix that
specific stage → re-run harness → verify green.

### 2. Stages that MUST be compared

For a VITS-family model, dump and compare at these boundaries:

```
phoneme_ids                  — must be exact (int64)
enc_layer{0..N}_post_attn    — after each attention+post-norm
enc_layer{0..N}_post_ffn     — after each FFN+post-norm
enc_output                   — raw encoder hidden states (NOT projected mean)
enc_proj                     — projected mean+logvar (384-d for split)
sdp_pre                      — after dp.pre 1x1 conv
sdp_ddsconv                  — after DDSConv (3 layers)
sdp_proj                     — after dp.proj 1x1 conv
sdp_flow{7,5,3}_z            — z after each ConvFlow
sdp_logw                     — after ElementwiseAffine
durations                    — integer durations (must be EXACT)
z_p                          — duration-expanded mean (latent input to flow)
flow{6,4,2,0}_m              — WaveNet m output per coupling block
z_dec                        — flow output (decoder input)
dec_conv_pre                 — HiFi-GAN conv_pre output
dec_stage{0,1,2}_mrf         — MRF output per upsample stage
audio                        — final PCM
```

### 3. Common bug classes found in this port (check these first on any new port)

| # | Bug class | How it manifests in diff | Root cause pattern |
|---|-----------|------------------------|--------------------|
| 1 | **Pre-norm vs post-norm** | enc_layer0 diverges wildly | VITS uses post-norm `x = norm(x + sublayer(x))`, not pre-norm `x = x + sublayer(norm(x))` |
| 2 | **Wrong SDP input** | sdp_pre matches but magnitudes are 10-50x off from expected | SDP receives raw encoder hidden states, NOT the projected mean |
| 3 | **Missing activation** | DDSConv output cos drops to 0.98 | VITS DDSConv applies GELU after BOTH norm1 and norm2; easy to miss the first one |
| 4 | **Wrong dilation pattern** | DDSConv cos drops to 0.986 | DDSConv uses `kernel_size^i` (1,3,9) not `2^i` (1,2,4). Read the ONNX conv attributes! |
| 5 | **Operation order (proj placement)** | SDP conditioning 500x too large | `dp.proj` goes AFTER DDSConv, not in parallel with `dp.pre` |
| 6 | **Missing scaling factor** | Spline params 14x too large → NaN in quadratic formula | ConvFlow widths/heights divided by `sqrt(filter_channels)` before RQS |
| 7 | **GGUF converter weight mapping** | Flow m values 5-15x too small, cos 0.05-0.5 | Anonymous `onnx::Conv_NNNN` weights appear in REVERSE flow order in ONNX graph; converter mapped them forward |
| 8 | **Flip ≠ half-swap** | Flow pre_conv output 10x too small | `torch.flip(x,[1])` reverses ALL channels `[c191..c0]`, not swap halves `[c96..c191, c0..c95]` |
| 9 | **ResBlock structure** | Audio nearly silent (RMS 0.005 vs 0.15) | Each conv in ResBlock has its OWN residual connection, not one skip at the end |
| 10 | **ResBlock dilations** | Audio amplitude 5x too low | Per-kernel dilations (k=3:[1,2], k=5:[2,6], k=7:[3,12]) not hardcoded dilation=1 |
| 11 | **WaveNet dilation_rate** | Flow m values systematically too small | Flow WaveNet uses `dilation_rate=1` (all dilations=1), not `2^i` |
| 12 | **Missing flow transform** | Durations 10x wrong | Piper includes ElementwiseAffine (flows.0) unlike standard VITS which skips it |
| 13 | **GELU variant** | cos 0.986 instead of 1.000 | ONNX uses exact GELU via `erf`, not tanh approximation |
| 14 | **Missing GGUF tensor** | ElementwiseAffine has no effect | `dp.flows.0.logs` constant-folded in ONNX export; converter must recover from `exp(-logs)` |

### 4. How to READ the ONNX model to prevent bugs

For EVERY operation, check the ONNX graph attributes BEFORE writing C++:

```python
import onnx
model = onnx.load('model.onnx')
for node in model.graph.node:
    if 'target_node_name' in node.name:
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in node.attribute}
        print(f'{node.name}: {attrs}')
```

Specifically check:
- **Conv dilations** — never assume; ALWAYS read from ONNX attributes
- **Conv groups** — depthwise vs regular
- **Conv padding** — asymmetric padding exists
- **Slice steps** — negative step = reversal (the Flip!)
- **Operation order** — trace the graph node-by-node, don't assume from paper
- **Anonymous weights** — `onnx::Conv_NNNN` names lose their block assignment;
  verify the mapping by tracing which Conv node references which weight

### 5. Test sequence

1. **Phoneme encoding** — compare IDs against Python `encode_phonemes()`
2. **Encoder** — one layer at a time, cosine at each norm output
3. **SDP** — each DDSConv layer, each ConvFlow, final logw
4. **Duration alignment** — durations must match EXACTLY for downstream to be comparable
5. **Flow** — each coupling block's m output
6. **Decoder** — conv_pre, each MRF stage, final audio
7. **ASR roundtrip** — synthesize → whisper transcribe → verify text

## Files

```
src/piper_tts.{h,cpp}                          — runtime + C ABI
models/convert-piper-to-gguf.py                — ONNX+JSON → GGUF
tests/test-piper-tts.cpp                       — smoke test
tools/reference_backends/piper_tts.py           — ONNX reference
tools/reference_backends/piper_tts_diff.py      — per-stage diff harness
examples/cli/crispasr_backend_piper.cpp         — CLI adapter
```

## Test commands

```bash
# Convert
python3 models/convert-piper-to-gguf.py \
    --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
    --output /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf

# Dump reference
python3 tools/reference_backends/piper_tts_diff.py dump \
    --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
    --ipa "həlˈoʊ wˈɜːld" --out /mnt/storage/piper/ref_stages.npz

# Dump C++ + compare
mkdir -p /mnt/storage/piper/cpp_stages
build/bin/test-piper-tts /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf \
    "həlˈoʊ wˈɜːld" /tmp/out.wav 0.0 0.0 --dump /mnt/storage/piper/cpp_stages
python3 tools/reference_backends/piper_tts_diff.py compare \
    --ref /mnt/storage/piper/ref_stages.npz --cpp /mnt/storage/piper/cpp_stages/

# CLI end-to-end + ASR roundtrip
build/bin/crispasr --backend piper -m <model.gguf> --tts "həlˈoʊ wˈɜːld" --tts-output out.wav
build/bin/crispasr -m models/ggml-base.en.bin -f out.wav
```
