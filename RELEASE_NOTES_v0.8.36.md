# CrispASR v0.8.36

Seven new model arms — two piano transcribers, four ASR backends, one
registry-only fine-tune — and a round of front-end fixes that were each found
because a reference was finally compared against the features the model
actually sees.

The theme carries over from v0.8.34 and v0.8.35: **instruments that could not
fail.** A diff harness that reported PASS for a layer it never compared, a
`use_gpu` field nothing read, a perf job that went green having measured two
arms of forty-eight, a reference dumper whose captured tensor was rewritten in
place after it was captured. Each is fixed here, and several of the numbers
below exist only because one of them was.

---

## New backends

### Piano transcription: Onsets & Frames and hFT-Transformer

Two new arms behind `--piano`, both ported to ggml from their ONNX exports and
both auto-downloading (`-m onsets-and-frames`, `-m hft-transformer`)
(3aee6f11, ea40b1c3). Note-level F1 on all ten MusicNet test pieces,
`mir_eval.transcription`, with the ONNX arm reproducing the published figures
before anything is claimed for the port:

| | size | note F1 | F1 + offsets | solo piano | registry default |
|---|---|---|---|---|---|
| **onsets-and-frames** f32 | 101.9 MiB | 49.6% | 13.8% | 69.0% | |
| onsets-and-frames q8_0 | 30.8 MiB | 49.6% | 13.9% | 69.0% | ✔ |
| onsets-and-frames q4_0 | 18.6 MiB | 49.5% | 13.3% | 68.9% | |
| **hft-transformer** f32 | 21.8 MiB | 52.21% | 18.49% | 70.52% | |
| hft-transformer q8_0 | 7.0 MiB | 52.23% | 18.50% | 70.51% | |
| hft-transformer q4_0 | 4.5 MiB | 52.55% | 18.77% | 70.71% | ✔ |

(5bf99c42, 121abaf1, 0e6193b1)

- **Both f32 ports are F1-identical to their ONNX exports on every piece.**
  O&F post-sigmoid onset max abs 5.3e-07, 100% of decisions identical
  (03dde53c); hFT onset/offset/mpe within 5.8e-06, cosine 1.00000000, velocity
  argmax 100% identical (0e6193b1). Basic Pitch scores 57.5% on the same solo
  piano pieces (03dde53c).
- **The two quantise differently, and the defaults follow.** q4_0 hits O&F's
  frame head, which sets note durations, so it costs F1-with-offsets; in hFT the
  head q4_0 damages most is velocity (cos 0.870), which only feeds the
  `ignore_zero` gate (121abaf1, ea40b1c3).
- **The O&F ONNX export's output names are shifted by one.** The converter
  ignores them and finds the real frame head structurally; believing the names
  costs 9.1% → 5.6% F1-with-offsets and nothing raises (03dde53c).
- **O&F lifecycle work:** one allocator per graph shape instead of one per call
  (−10.3% marginal CPU), and an opt-in ggml-graph BiLSTM
  (`CRISPASR_OAF_GRAPH_LSTM=1`, −23.6% marginal on top). Cumulative
  0.6073 → 0.4163 CPU-s per audio-second (8a3e22a9, 663c6867). The graph
  BiLSTM is not bit-identical to the scalar recurrence, so it stays opt-in even
  though it reproduces 49.6% / 69.0% on MusicNet (67ead34e, 444708db).
- **Apple Silicon CPU:** on GitHub's virtualised 3-vCPU M1, hFT runs at 0.926×
  real time at f32 and 0.563× at q4_0; O&F at ~0.16×. Reproduced on a second
  runner (09c56f8e, f7a9148b). The same run's "quantisation makes O&F 8%
  slower" did **not** reproduce and was retracted; what does reproduce is peak
  RSS 300 → ~220 MiB (f7a9148b).
- **GPU wiring:** both arms now honour `use_gpu` (CUDA > Metal > Vulkan > CPU),
  with `CRISPASR_HFT_NO_GPU` / `CRISPASR_OAF_NO_GPU` to force CPU (3d81244a). A
  device that cannot run `MUL_MAT` for the model's dtypes is now declined with a
  warning instead of aborting (ca74bbda). **Metal throughput is unmeasured:**
  every hosted GitHub macOS image (14, 15, 26, latest) exposes an "Apple
  Paravirtual device" with no simdgroup matmul, so no dense-GEMM ggml model can
  use the GPU there (740bc939, 47755796). The CPU path is verified unchanged:
  O&F 26/26 stages cos 1.0000000, hFT four heads cos 1.00000000 (a141c300,
  47755796).

### Dolphin CN-Dialect (#436)

`-m dolphin` — DataoceanAI's `dolphin-cn-dialect-small-streaming`,
E-Branchformer + Transformer decoder + CTC with CTC prefix beam and attention
rescoring; Mandarin plus Chinese dialects, with language/region predicted by
the decoder when not given (README, 0d18b2a2). F16: every `crispasr-diff`
stage passes and the decoded text is identical to upstream on a zh clip and
jfk (97bc0a38). **Q4_K is the default**, settled by transcripts rather than
encoder cosines (which are chaotic under quantisation, 0.3–0.99): on 15
in-domain clips Q8_0 and Q4_K are each 13/15 identical to F16, the misses being
dropped trailing particles and one character; Q4_K is 42% smaller. English is
out of domain and drifts more at Q4_K (76a648c5). `-bs` sets the beam width,
default 10 (docs/cli.md, 0d18b2a2). The C ABI session accepts a two-level
`source_language` such as `zh-SICHUAN` (0d18b2a2).

### X-ASR zh-en (#436)

`-m xasr` — `GilgameshWind/X-ASR-zh-en`, an icefall streaming Zipformer2
transducer (57a4aa94). Converted from the sherpa-onnx export, not
`pretrained.pt`: the control arms showed `pretrained.pt` is an earlier
checkpoint, 0/579 tensors identical (a4e62b4a). All four exports share every
weight, so one GGUF serves 160 / 480 / 960 / 1920 ms chunks, selected with
`CRISPASR_XASR_CHUNK_MS` (tail silence: `CRISPASR_XASR_TAIL_PAD_MS`)
(a4e62b4a, 70c45a3a). Append-only streaming API and a realtime WebSocket
session; offline transcription runs the same stream (1591b5fe). F16 parity:
worst-frame cos ≥ 0.99999 at every stage, tokens and sherpa-rendered text
identical, and streaming in uneven 370 ms pieces equals one-shot. **Q8_0 is the
default**; Q4_K keeps zh identical but drops English punctuation (5e78cb4f,
docs/xasr/PLAN.md).

### Hojo-ASR-Multi-V1 (#438)

`-m hojo-asr` — Qwen3-Omni audio tower + WeNet Conformer adapter + Qwen3-4B
(README, 8627ef46). No 30 B checkpoint is fetched: all 1013 tensors are in the
one merged safetensors (853bcf51). Q4_K default, 4.4 GB; F16 9.6 GB (de32b50e).
Per PLAN.md, F16 passes 7/7 diff stages and the greedy text is identical to
upstream (ad85e4e8); per the model card, Q4_K matches on French and differs by
one word on German (de32b50e, 19199955).

**Greedy is the default** although the checkpoint's `config.yaml` names beam 4:
beam search here replays each beam's suffix every step, O(B·T²) — about 80,400
forwards versus 200 for a 9 s clip. `-bs 4` still selects it, and the runtime
prints the projected forward count first (9b6f4ffc). The model takes no
language conditioning; an explicit `-l` warns (4eecacd1).

### Orukeet and Confucius4-R2T2 (#445), registry only

- **`-m orukeet`** — `oruk/orukeet`, a parakeet-tdt-0.6b-v3 fine-tune run by
  the Parakeet runtime unchanged. F16 and Q8_0 transcripts identical to NeMo on
  en/de/fr/es; Q4_K (default) same words, one punctuation mark differs on de
  and es. **CC-BY-SA-4.0**, and its final adaptation trained on LibriSpeech
  test-other, so its score there is not held-out (bc04ddaf).
- **`-m confucius4-r2t2`** — NetEase Youdao's streaming Qwen3-ASR-1.7B
  fine-tune with a tied `lm_head`, Q4_K default (4c255bbc). The realtime
  WebSocket session runs R2T2's prefix-rollback algorithm on its `example.py`
  schedule with append-only deltas; `CRISPASR_QWEN3_STREAM=1` enables that
  session for other Qwen3-ASR models (docs/server.md, 4c255bbc). F16 offline and
  streaming final text identical to upstream on en + zh (434fb8cc). **Not open
  source:** NetEase Youdao Model Use License (4c255bbc).

---

## Fixes to existing backends

### Kaldi-fbank front-ends build their triangles in mel

`core_kaldi` built its mel filterbank linear in Hz; Kaldi, kaldi-native-fbank
and `torchaudio.compliance.kaldi` build it linear in mel. Against knf the Hz
form drifts by mean |Δ| 2.6e-3 (max 5.2e-2) in log-mel; mel-domain gives
5.3e-5 (7111bfcc, docs/xasr/PLAN.md). **Mel-domain is now the default** for
sensevoice, funasr, paraformer, wespeaker, CAM++ and dolphin, and firered_asr /
firered_vad's own bank copies are switched the same way (fb64dc12). Verified
after the fact: every reference was rebaked and every stage passes at F16 for
wespeaker, sensevoice, funasr, firered-asr, dolphin and CAM++ (40051218).

Found and fixed on the way (40051218):

- **SenseVoice now applies `am.mvn` CMVN**, as FunASR's loader does despite
  `cmvn_file: null` in the config. Words survived without it; rich tags did not
  (jfk came out `<|ANGRY|>` instead of upstream's `<|EMO_UNKNOWN|>`). The GGUFs
  were re-uploaded with CMVN. **An older SenseVoice GGUF still loads but warns
  once to re-download from `cstr/sensevoice-small-GGUF`** (ad26dac7).
- **Paraformer** now scales by √d_model and adds the sinusoidal PE before the
  first encoder block; `encoder_layer_0` had sat at cos 0.95–0.98 while the text
  matched (242fbb2d). One CIF row still differs on zh, a float32 fire-threshold
  tie 1 ULP from 1.0; the text is identical (40051218).
- **Reference dumpers:** sensevoice/paraformer used a CMVN-less front-end, the
  funasr and hook captures aliased tensors later modified in place, and the
  FireRed dumper used knf's random default dither (ad26dac7, 8a5f9056,
  fb64dc12).

### qwen3-tts on Vulkan: causal-mask width (#337)

The O15 code-predictor's T=2 graph was built with a wider mask than its KV
length. Vulkan's flash-attention shader derives the mask stride from KV, so it
masked the wrong keys and the talker ran to the frame cap: 1188 frames / 95 s
before, 67 frames / 5.4 s after, ASR round-trip WER 0, on an RX 7900 XT
(8c359d80). `kv_self_attn` now asserts the mask is exactly Lk wide on every
backend; an audit of its 34 callers found no other offender (da258942).

### Threaded MKL scaled the mel projection (#453)

With Debian's MKL installed, `find_package(BLAS)` picked `mkl_intel_thread`
next to libgomp, and `cblas_sgemm` returned the upper mel columns **multiplied
by the thread count** — ln(4) of log error at 4 threads, nothing raised
(00822744). `crispasr-core` and `cohere` now prefer OpenBLAS, a threaded-MKL
result is rewritten to `mkl_sequential` (`COHERE_MKL=ON` exempt), and
`test-mel-blas-parity` guards it (00822744, 817d6bf3).

### Server

- **Command injection, fixed.** The ffmpeg fallback put a filename derived from
  the client's upload into an `sh -c` line; an unauthenticated
  `/v1/audio/transcriptions` request could run a command on the server.
  `crispasr-server` passed the upload's *bytes* as the path, which also made
  every upload fail. Subprocesses now run from an argv vector
  (`posix_spawnp` / `CreateProcessW`), never a shell; the zonos espeak-ng
  fallback uses the same path (6409647a). Under WASM the fallback is
  unavailable (630bd13d).
- **Chunked uploads work** (#452, contributed): `Transfer-Encoding: chunked`
  bodies are read and bounded by the same 512 MB cap instead of refused with
  411 (1aa0d55f, docs/server.md).
- **SIGTERM shuts down.** The WebSocket and realtime listeners blocked in
  `accept()`, which `close()` does not wake on Linux, so shutdown hung on the
  join. Before: still alive 15 s after SIGTERM. After: clean exit, rc 0, in
  0.72 s (c56b9a2e).

### Smaller

- **`crispasr-diff parakeet` never compared the last encoder layer** and scored
  an all-zero candidate as cos 1.0 PASS. Both fixed; existing parakeet parity
  claims rest on `encoder_output`, which was always really measured (4ee1b38a).
- **btc-chords** gets the same allocator hoist as O&F. **Not validated at
  runtime** — no checkpoint on the dev box; the commit asks anyone with one to
  run `tools/btc_torch_parity.py` (995bf240).

---

## Session ABI: three TTS defects the CLI never showed

Found from CrisperWeaver driving the bindings (c97fc1aa):

- `crispasr_session_set_voice` loaded an f5-tts reference at a hard-coded
  24 kHz. **Raon-OpenTTS** is the f5-tts runtime with a 16 kHz front-end, so
  bindings built a 747-frame reference mel against the CLI's 680 (cos 0.94). It
  now loads at the model's rate; F5-TTS proper is unchanged.
- `crispasr_session_output_sample_rate` had no **bt2-tts** arm and reported
  0 Hz; it now returns 24 kHz.
- The session's f5-tts arm rejected **`raon` / `raon-1b`**, which the CLI
  accepts, so an open from a binding returned null.

Guarded by a structural test (every synthesize arm needs a rate arm) and a live
reference-mel diff via `CRISPASR_F5_DUMP_REFMEL` — 4 s, against 1 h 50 min for
the first output-length version (c97fc1aa, HISTORY.md).

---

## ggml's CPU repack buffer type

`core_gguf` can now load matmul weights into ggml's CPU repack buffer type,
which reaches the interleaved int8 GEMM kernels (10e95422). **hFT-Transformer is
the only adopter**; a model has to name its matmul weights (10e95422). If no
tensor is accepted it falls back to the ordinary loader and keeps mmap
(10e95422). The buffer type is selected by name, since on Sapphire/Emerald
Rapids ggml lists AMX first (78bbfcb8).

Whole-model hFT, clean CI runners, CPU seconds relative to f32 (15393a45,
0c1916d1):

| | EPYC 7763 (AVX2) | EPYC 9V74 (VNNI) | arm64 (dotprod/i8mm) |
|---|---|---|---|
| q8_0 generic | 1.09× | 0.84× | 0.50× |
| q8_0 + repack | no x86 kernel | declined | 0.37× |
| q4_0 generic | 1.22× | 0.91× | 0.55× |
| q4_0 + repack | 0.72× | 0.74× | 0.35× |

- **On x86 ggml has no repacked q8_0 kernel**, so it helps only q4_0 / q4_K
  (10e95422).
- **On arm64 it speeds up q8_0 GGUFs as they ship:** 1.37× over generic
  (b0e44221). At kernel level: 3.1–3.4× on macos-14, 2.3–2.4× on Linux arm64
  (e837d5cb).
- **VNNI narrows the lead** (1.24× vs 1.69×), because it speeds up the generic
  path more (0c1916d1). An accidental AMX run was *slower* for every quantised
  arm (1.42–1.51× f32); one unintended run, not a finding yet (b0e44221).
- "Quantisation does not make CPU inference faster" holds only for pre-VNNI
  x86: hFT q8_0 vs f32 is 1.29× on Skylake-SP, 0.50× on arm64 (0c1916d1).
- For 22 MB hFT, giving up mmap cost nothing measurable (233–234 MiB RSS in
  every arm). No measurement exists for a large model (15393a45).

`CRISPASR_GGUF_REPACK=0` disables it; `CRISPASR_GGUF_EXTRA_BUFT` overrides the
chosen buffer type (10e95422, 78bbfcb8). Reproduced within 1.4% on a second run
(9f6d616c). Full write-up in `docs/ggml-repack-buft-evaluation.md`, alongside
the new `docs/ggml-optimisation-playbook.md` (ee1457a3).

---

## CI and lint

Lint and Lint Deep had been red on main; both are green again (2570762d):
161 clang-format violations fixed, the registry-URL checker now joins adjacent
string literals (all 265 URLs resolve), and two cppcheck findings resolved
(dolphin beam comparator, Qwen3 realtime session constructor). Also:
`crispasr-repack-probe` no longer breaks `GGML_BACKEND_DL` builds (254d8750),
CI and feature-matrix regeneration can be dispatched on a feature branch
(a7c230f5, 909405c0), and new repack and CPU-vs-Metal A/B workflows each
check that the arm they measured is the arm they meant to measure
(425c345e, 8cdd9daa, 9e3b2898).

---

## Behaviour changes worth knowing

- **Kaldi-fbank backends** (sensevoice, funasr, paraformer, wespeaker, CAM++,
  dolphin, firered-asr, firered-vad) now use mel-domain triangles. There is no
  env var to switch back; `FbankParams::mel_domain_triangles = false` keeps the
  Hz form for a C++ caller that sets it explicitly (fb64dc12).
- **SenseVoice** applies CMVN from the GGUF. An old GGUF still runs without it
  and warns; re-download for upstream-matching features and rich tags
  (ad26dac7).
- **Paraformer** encoder input changed: √d scaling plus sinusoidal PE
  (242fbb2d).
- **hFT-Transformer** uses the repack buffer type by default on CPU;
  `CRISPASR_GGUF_REPACK=0` restores the plain mmap loader (10e95422).
- **hFT and O&F use a GPU when one is available**; `CRISPASR_HFT_NO_GPU=1` /
  `CRISPASR_OAF_NO_GPU=1` force CPU (3d81244a).
- **O&F's graph BiLSTM is opt-in**: `CRISPASR_OAF_GRAPH_LSTM=1` (663c6867).
- **hojo-asr decodes greedily** unless `-bs 4` is passed (9b6f4ffc).
- **Server**: chunked uploads are accepted (were 411) (1aa0d55f). The
  `crispasr --server` temp file keeps the upload's extension only if it matches
  `.[A-Za-z0-9]{1,8}` (6409647a).
- **BLAS**: a threaded-MKL link is rewritten to `mkl_sequential` unless
  `COHERE_MKL=ON` (817d6bf3).
