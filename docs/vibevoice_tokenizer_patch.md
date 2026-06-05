# Patching old VibeVoice-ASR GGUFs to add the tokenizer

## Why this exists

The VibeVoice-ASR GGUFs originally uploaded to HuggingFace
(`vibevoice-asr-7b-f16.gguf`, `vibevoice-asr-7b-q4_k.gguf`) are missing
`tokenizer.ggml.tokens` because the early converter only tried to load the
tokenizer from the VibeVoice snapshot itself, which ships no tokenizer files.

Symptom: runtime generates correct token IDs but detokenizes to `""`, and
`-otxt` produces no file.

Fix landed in the repo (commits `a314934` and later):
1. `models/convert-vibevoice-to-gguf.py` now falls back to `Qwen/Qwen2.5-7B`.
2. `src/vibevoice.cpp` uses `lm_head.weight` (not tied to `tok_emb`) and
   appends `<|im_start|>assistant\n` (matches HF `add_generation_prompt=True`).

For anyone already hosting the old GGUFs, we don't need to re-upload 4.5 GB
of quantized weights — tensor data is byte-identical between old and fixed
variants. Only the KV section (~2.6 MB of vocab strings) needs to be added.

`tools/vibevoice_add_tokenizer.py` does exactly that.

## Runbook: patch + reupload from a VPS

Prereqs on the VPS:
- Python 3.10+ with `transformers` and a gguf install (either `pip install
  gguf` or the vendored copy at `ggml/python/gguf` is on the path).
- HuggingFace Hub auth if you're pushing: `huggingface-cli login`.
- Disk: ~10 GB free (old + new GGUF side-by-side while rewriting).
- No GPU, no torch needed. No need to download the safetensors source.

```bash
# 1. Clone or pull the repo
git clone https://github.com/CrispStrobe/CrispASR.git
cd CrispASR
# or: cd CrispASR && git pull

# 2. Install tooling (first time only)
pip install transformers huggingface_hub gguf

# 3. Download the broken GGUF from HF
#    (replace repo name if it changed)
hf download --local-dir . CrispStrobe/vibevoice-asr-gguf \
    vibevoice-asr-7b-q4_k.gguf
# Or use the URL:
# wget https://huggingface.co/CrispStrobe/vibevoice-asr-gguf/resolve/main/vibevoice-asr-7b-q4_k.gguf

# 4. Splice in the Qwen2.5-7B tokenizer
python tools/vibevoice_add_tokenizer.py \
    --input  vibevoice-asr-7b-q4_k.gguf \
    --output vibevoice-asr-7b-q4_k-fixed.gguf
# Expected tail: "Done: ... 4.81 GB, 901 tensors, +151665 vocab strings"

# 5. Verify the fix took
python - <<'PY'
import gguf
r = gguf.GGUFReader("vibevoice-asr-7b-q4_k-fixed.gguf")
print("has tokenizer:", "tokenizer.ggml.tokens" in r.fields)
print("n_vocab_strings:", len(r.fields["tokenizer.ggml.tokens"].data))
print("n_tensors:", len(r.tensors))
PY
# Expect: has tokenizer: True / n_vocab_strings: 151665 / n_tensors: 901

# 6. (Optional) smoke-test decode — only if you have a crispasr build
#    with both decoder fixes (commit 02f1ac8 or later):
#    build/bin/crispasr --backend vibevoice \
#        -m vibevoice-asr-7b-q4_k-fixed.gguf -f samples/jfk.wav \
#        -d 3000 -n 96 -l en -np
#    Expect the JFK JSON transcript.

# 7. Upload the fixed file back to HF
hf upload CrispStrobe/vibevoice-asr-gguf \
    vibevoice-asr-7b-q4_k-fixed.gguf \
    vibevoice-asr-7b-q4_k-fixed.gguf
# Or replace the original name if you prefer — just remember downstream
# users need a crispasr built from 02f1ac8 or later to get usable output.
```

The same procedure works for `vibevoice-asr-7b-f16.gguf` (just swap the
filenames in steps 3-7).

## Gotchas

- The splice refuses to rewrite a GGUF that already has
  `tokenizer.ggml.tokens`. If you want to replace it anyway, delete the
  key manually first.
- Tensor data is not re-quantized. If the source quantization was wrong,
  this tool won't fix it — it only adds metadata.
- The fixed GGUF is useless without a `crispasr` binary built from
  commit `02f1ac8` or later. Both the GGUF and the code side need to be
  fresh.
