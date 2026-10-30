# MioTTS and Index-Echo integration — 2026-10-03

The MioTTS fixes, opt-in Echo scheduler experiment, and Q4 preparation/validation
tools are integrated with the Qwen3 hotword and MiMo language-prompt fixes.
Echo Q4 model acceptance is still pending; merging its tools does not approve
candidate weights or change the public/default quantization.

## Behavior

- MioTTS CLI/server and session ABI report the loaded codec rate: public v2 is
  44100 Hz; missing legacy metadata defaults to 24000 Hz. Native preset loading,
  resident request overrides/restoration, temperature and seed controls reach
  the runtime. The registry downloads the tokenizer companion. Binding docs
  explain preset embeddings and using the session output rate for PCM.
- `CRISPASR_LLAMA_PIPELINE_DISABLE=1` remains an opt-in embedded-llama scheduler
  experiment. `INDEX_ECHO_BENCH=1` emits stage timing and reuse counters. The
  earlier 48-call AB/BA result on two physical T4s shows 2.5–3.6% improvement;
  no default flip, CUDA graph-capture claim, or broader hardware claim follows.
- Q4 tools define four recipes, audit actual tensor types/bytes, and preserve all
  177 original F32 tensors including 24 recurrent convolution matrices. Hosted
  CPU prepares artifacts; Kaggle only runs pinned candidates on actual CUDA
  hardware. The GPU wrapper refuses to launch with pending preparation pins.

## Integration evidence

| Gate | Evidence |
|---|---|
| MioTTS ARM speech/rate | [37137394514](https://github.com/CrispStrobe/CrispASR/actions/runs/37137394514): 18 assertions/4 cases; both CLI and session return 132300 samples at 44100 Hz, 3.000 seconds; both ASR readbacks have 0% WER. Published v0.8.41's 24000 Hz result is rejected. Metadata-copy checks prove dispatch only. |
| Qwen3/MiMo combined speech | [37137396334](https://github.com/CrispStrobe/CrispASR/actions/runs/37137396334): all 12 Qwen3 off/on/clear/CLI recognitions remain exact; MiMo forced/auto English/Chinese CLI and ABI agree, without external LID. Prompt/MiMo/provenance units pass. |
| Broad CI | [37137392582](https://github.com/CrispStrobe/CrispASR/actions/runs/37137392582): all 13 jobs, including 1988 unit tests, platform builds, sanitizers, dynamic backends and fuzz smoke. |
| Echo 2B F16 and nightly | [37138488949](https://github.com/CrispStrobe/CrispASR/actions/runs/37138488949): three F32-reference clips each pass 68 checks with 16/16 cached greedy IDs; exact direct text/timestamps, five full-file cases and the pinned Q8 nightly driver pass. |
| Lint | [37137535938](https://github.com/CrispStrobe/CrispASR/actions/runs/37137535938): all 10 jobs, including full-tree clang-tidy. |
| Binding checks | [Go 37137535761](https://github.com/CrispStrobe/CrispASR/actions/runs/37137535761), [Dart 37137535765](https://github.com/CrispStrobe/CrispASR/actions/runs/37137535765), [Rust 37137392967](https://github.com/CrispStrobe/CrispASR/actions/runs/37137392967); Go cgo linkage drift check passes. |
| Generated wiring | ARM and x86 hosted CLI generators produce identical feature matrix and ABI capability table, retaining both MiMo and MioTTS declarations. |

The structured [integration receipt](miotts-echo-integration-2026-10-03.json)
records each source pin, test result, artifact hash and the final Echo/lint gates.
Original feature source is archived as `archive/miotts-echo-q4-source-20261003`.
Integrated runtime files are byte-identical to that proven feature; new Qwen3/MiMo
ABI behavior is tested together above. Historical GPU timing is retained in
[index-echo-scheduler-ab-2026-10-02.json](index-echo-scheduler-ab-2026-10-02.json).
This integration does not add a new GPU performance measurement.

The first Echo integration dispatch selected the original mixed-precision
full-file oracle and was cancelled before acceptance. The corrected job uses
separate forced-F32 stage/full-file references and the existing pinned nightly
driver; no thresholds or expected outputs were changed. Its cancelled log is
retained with the successful proof artifacts under
`${CRISPASR_STORAGE}/crispasr/miotts-echo-integration-20261003/`.

## Validation on merged main

The merged source `17dc2a81e70839b79ac8d905acaa375459e07435` also passes
[main CI 37140082182](https://github.com/CrispStrobe/CrispASR/actions/runs/37140082182)
(13 jobs), [main lint 37140082038](https://github.com/CrispStrobe/CrispASR/actions/runs/37140082038)
(10 jobs), [selected regression 37140082052](https://github.com/CrispStrobe/CrispASR/actions/runs/37140082052)
(11 jobs), all five WASM variants, all triggered language bindings, Docker smoke
and Windows Piper live checks. The structured receipt records each workflow.

Fresh [9B ARM CPU validation 37140082238](https://github.com/CrispStrobe/CrispASR/actions/runs/37140082238)
passes 76 checks on each of three F32-reference clips, exact decoded output,
five full-file cases, three Piper roundtrips and the pinned nightly driver.
Minimum stage-row cosine is **0.999997**; maximum relative L2 is **0.1883%**.
This validates the public F16 pair, not an experimental Q4 candidate.

[Cppcheck 37140082070](https://github.com/CrispStrobe/CrispASR/actions/runs/37140082070)
continues in the separate non-cancelling deep-analysis workflow; its snapshot
is still running, so no completed cppcheck verdict is claimed. The cold Git
repository now has its own 197 MiB object pack and passes connectivity checks,
with no dependency on the original checkout's object database.

## Still pending

Corrected Q4 CPU preparation [37046153440](https://github.com/CrispStrobe/CrispASR/actions/runs/37046153440)
produced a 5.05 GB plain decoder but its HF upload commit failed with HTTP 400
(private repository storage quota). The original GPU v1 F16 control passed,
then repository creation failed with HTTP 403 before any candidate ran. No Q4
candidate has successful uploaded artifact pins or GPU acceptance. Private
staging was a transfer choice; it is not required by the runtime.

Establish a feasible transfer route, prepare/pin the candidates, and run the
unchanged stage/magnitude/cache, complete decoded-output and TTS→ASR gates before
publishing weights or considering a default. GPU-only v2 has not been pushed.
Issue #488 still needs the reporter's Windows/Vulkan retest; this CPU integration
proof does not reproduce their unavailable six recordings. No new tag is cut.
