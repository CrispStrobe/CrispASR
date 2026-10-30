# Dia opt-in serial head — current-main integration

The CPU F32 head diagnostic is integrated at9384c3a5. All production source,
CMake, bindings and example bytes match tested candidate d198d7a4; rebase adds
only the intervening main documentation/diagnostic-tool commits. All13 general
CI jobs38874305718 and10 lint jobs38874302965 pass at that immutable candidate.
Hosted build/replay38874191736 and independent57-payload audit pass, including
12 fresh graph replays against returned actual static libraries.
[Complete hosted receipt](dia-current-main-head-hosted-2026-10-16.json).

`CRISPASR_DIA_SERIAL_HEAD` accepts `off`, `identity` and `serial`. Unset/empty
means OFF and returns the original graph tensor. Identity copies native logits
through the same custom-node topology. Serial uses actual native weights and
current hidden, separately rounded F32 products and sequential F32 addition.
It runs generically at every head/step without a token override, external
reference input or position-dependent branch. Compute/apply function bytes
match causal candidate3c2fbb5a. Opt-in modes reject GPU/non-F32/invalid head
shapes before the encoder. The ordinary matmul remains a graph dependency, so
serial deliberately pays both paths and makes no performance claim.

The registered CMake unit passes30 actual graph/admission checks, two batches,
one/four threads, identity and a tiny fusion witness. Removing the GPU guard
makes the negative control fail. Both captured hidden states reproduce all
18,504 original-compatible serial values exactly at both thread counts; native
and identity outputs match. Whole-runtime syntax and format18 pass. The failed
CIFS shared-library symlink build remains intact; accepted static proof is at
${CRISPASR_STORAGE}/crispasr/dia-main-head-20261016/local-v2/preflight-v2.

This accepts a disabled-by-default diagnostic graph, not current-main speech.
Current main retains F16 cross K/V, std::discrete_distribution, repaired top-k
and stop/crop behavior, and existing ggml RoPE. The accepted pinned8d0ad886/C36
speech experiment instead used diagnostic F32 cross cache and verified portable
noise. [Its speech proof](dia-serial-head-results-2026-10-16.md) remains scoped
to one prompt/seed and cannot be transferred to these different defaults.

Current-main full speech A/B38880025781 completed and failed quality. Its
independent170-payload audit confirms quiet output and empty ASR/WER1 for all
native arms through both DACs and CLI; original positive WER0 and stage gates
pass. [Audited result](dia-current-main-speech-results-2026-10-16.json).
Next isolate F16/F32 cache from standard/portable noise with defaults preserved.
Reference/diff composition, broader prompts/seeds, GPU/Q4 and profiling remain
in PLAN.md. No model pin, production default or release tag is promoted.
