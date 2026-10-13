# Original rotary timescale control

The original Float32 RotaryEmbedding generates a nonpersistent timescale buffer
and divides each position by it. The default ggml CPU helper recursively
multiplies a frequency ratio. At position 978, the actual-source tiny coefficient
probe measures maximum error 1.2040138e-4; existing ggml frequency factors using
the original table and base one reduce it to 5.9604645e-8. This is primitive
coefficient evidence, not a neural or speech fix.

The candidate exports the pinned original module's actual Float32 table as
`dia.rope.timescale`, with original source/Torch provenance and a physical
payload audit. Every existing neural coefficient keeps its original export
bytes and order. The extra buffer is 256 bytes for the real head dimension 128;
GGUF header/alignment overhead is measured separately on the real export.

`CRISPASR_DIA_SOURCE_ROPE=1` uses that table through the existing portable
`ggml_rope_ext` operator with frequency base one. All six rotary sites are
covered: encoder Q/K, cached cross K, decoder self Q/K and cross Q. The default
keeps the old RoPE call. Explicit mode rejects missing, wrong-type/shape or
nonfinite/nonpositive tables. No custom GPU kernel or sampler policy is added.

Run the fixed hosted control:

```bash
# Through the registered heavy-cpu workflow with pinned Torch2.6 CPU/DAC/gguf.
python3 tools/ci-heavy/dia_neural_precision_control.py \
  --cross-cache-control --source-rope-control
```

All arms use the same full-original-F32 neural coefficients, critical DAC,
F32 cross caches, canonical reference, short prompt, seed 42, temperature 1.2,
top-p .95, top-k 45 and CFG 3. Only rotary mode changes OFF/ON/OFF. The native
runtime marker records actual mode/table bytes. Each arm requires 40 canonical
stage comparisons plus actual free CLI and C ABI generation/ASR. OFF/repeat
full inputs, logits and both PCM outputs must match exactly; known legacy
speech rejection is retained. ON must meet the original unchanged speech gate.

The earlier late-layer job's log-count failure is retained: the underlying
comparisons return success, but stderr splits a PASS marker. Flushing stdout
before isolated DAC replay fixes the producer; parsing the exact legacy decoder
marker also preserves old interleaved records without dropping failures or
numerical fields. Complete original logs remain unchanged.

This is an unmerged experiment. No new model/default pin, broad speech,
GPU/Q4 acceptance or performance gain follows before the required results.
