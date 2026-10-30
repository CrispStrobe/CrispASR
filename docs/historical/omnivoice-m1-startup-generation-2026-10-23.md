# OmniVoice M1 generation and startup — 2026-10-23

The useful new optimization is `CRISPASR_OMNIVOICE_ATTN_VIEWS=1`: pass strided
Q/K/V layouts directly to attention instead of copying each to a contiguous
buffer. The actual attention operation already accepts those strides. This
removes 84 copies per transformer forward, with unchanged weights, masks,
positions, generation steps and sampling. It remains opt-in pending broader
hardware and full-model precision coverage.

On this M1, with the resident Metal decoder enabled in both arms:

| Counterbalanced warm medians | Current path | Experiment |
|---|---:|---:|
| Generation, attention views | 8.098 s | 7.454 s |
| Complete request, attention views | 8.603 s | 7.986 s |
| Generation, precompiled attention-view repeat | 7.986 s | 7.129 s |
| Complete request, precompiled repeat | 8.490 s | 7.622 s |
| Generation, packed QKV and gate/up | 7.845 s | 7.769 s |
| Generation, unified CFG | 7.653 s | 8.845 s |

Attention views reduce generation latency 7.9% and whole-request latency 7.2%
in the first campaign. The precompiled repeat confirms the gain (10.7% generation,
10.2% complete request); its preflight reports load average 2.72 and 55% system
memory free. Packed projections have no convincing win here, consume about
312 MB of extra Q8 weights and stay off. They concatenate existing serialized
rows without requantizing, retain the original projections and fall back for
incompatible widths/types or failed backend allocation. Unified CFG remains off
on Metal. The precompiled-build unified-CFG repeat also regresses
7.836→9.147 s; it does not change the conclusion.

Every row comes from its own back-to-back campaign, with two arm warmups then
six counterbalanced samples (`0,1,1,0,0,1`). Do not combine different campaigns
into a claimed cumulative speedup. All use English, seed 42, 32 steps, four CPU
threads, 150 target frames, the same pinned Q8 main and F16 tokenizer and the
same sentence: “We should bring enough water and food for the long walk home.”
All 40 native outputs have code and PCM bytes identical to the previously scored
resident baseline (both independent judges WER 0). Output is 96,240 samples at
24 kHz after cleanup; the six-second target is not the final audio duration.

The serialized one-step GPU profiler attributes about 72% of the original
forward to matrix multiplication and 14% to layout copies. With attention views,
contiguous-copy nodes fall 85→1 / 84→0 for conditional/unconditional forwards;
matrix multiplication then accounts for about 81–82%. Its serialized wall times
are not request timings. Matrix kernels and longer inputs are the next useful
targets; fewer generation steps or stale CFG would change the quality recipe.

## Startup and usable builds

Weights were copied unchanged to the internal SSD and both SHA256 values checked
against the [previous pinned receipt](omnivoice-m1-comparison-2026-10-23.json).
Both shader builds use those same file paths. The first internal-source launch
loads in 1.228 s; a precompiled-shader launch loads in 0.824 s. The earlier 144.633 s
external-volume observation is not a controlled cold-cache comparison.

Eight additional separate-process launches, ordered embedded/AOT/AOT/embedded/
embedded/AOT/AOT/embedded, give median model-plus-codec initialization:

| Build | Wall | Process CPU |
|---|---:|---:|
| Embedded Metal source | 0.503 s | 0.519 s |
| Precompiled Metal library | 0.490 s | 0.460 s |

macOS file and shader caches were already warm. No system caches were cleared.
Precompiled library loading itself takes 2–3 ms; cached source-library loading
20–23 ms. This is a small warm-start difference, not a reproducible multi-second
warm-start optimization. Storage placement and keeping a Session alive are the
practical choices here. Precompilation avoids source compilation when its cache
is cold, but this campaign does not establish that cold-cache speedup.

A full Release CLI with precompiled shaders was built and executed with ordinary
watermark/C2PA output and explicit OmniVoice license acceptance. Both pinned ASR
judges recognize its marked WAV perfectly. The built-in detector finds its
watermark (confidence 0.8520) and the WAV contains a C2PA chunk. Build it with a working Apple Metal
compiler, retaining the two metallib files beside the executable:

```sh
cmake -S . -B "$BUILD_AOT" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_METAL_EMBED_LIBRARY=OFF \
  -DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF
cmake --build "$BUILD_AOT" --target ggml-metal-lib crispasr-cli -j3
```

Precompiled resource discovery is validated for this CLI layout. SDK host
executables have different resource layouts; the full Python Session tests use
the regular embedded-source shared library. Packaged build defaults stay as-is.
No ggml revision change, adapted weight publication or model renaming is needed.

## Checks and scope

The actual transformer builder is now tested with F32/F16/Q8_0/Q4_K weights,
asymmetric query/KV heads, separate/unified CFG, and packed projections/views
individually and together. Every output must be finite and nonzero and each logit
must match within 1e-5 absolute error. A deliberate wrong K-view offset makes this
test fail. Allocation failures must publish no packed tensors; byte-preserving
packing, reuse and incompatible-type/width fallbacks have separate checks.
Four selected CTest cases and the actual runtime under ASan/UBSan pass.

The fresh full shared library is exercised through Python Session/C ABI:
unconditioned views, views+packing and views+unified CFG all retain exact baseline
code/PCM bytes. Both uncached synthetic reference voices (`window`, `station`)
also retain exact code/PCM when views are enabled. Their new cross-sentence
outputs and the marked CLI output each score WER 0 under both pinned judges.
These walk-sentence controls do not clear the existing fox-sentence cloning
failures described in the earlier audit.

Enable the useful experimental combination with:

```sh
export CRISPASR_OMNIVOICE_CODEC_SPLIT=1
export CRISPASR_OMNIVOICE_CODEC_RESIDENT=1
export CRISPASR_OMNIVOICE_CODEC_DECOMP=1
export CRISPASR_OMNIVOICE_ATTN_VIEWS=1
```

Keep ordinary license acceptance, voice rights and marking obligations. The
extra `CRISPASR_OMNIVOICE_PACKED_PROJ=1` experiment is available for further A/B,
with its allocation cost charged to the first request. Both new switches default
off. The public native benchmark adds `--paired-attn`, `--paired-proj`,
`--paired-cfg` (eight repeats each) and `--load-only` for reproduction using the
[previous compilation instructions](omnivoice-m1-comparison-2026-10-23.md).

[Machine-readable samples, hashes, transcripts and scope](omnivoice-m1-startup-generation-2026-10-23.json).
