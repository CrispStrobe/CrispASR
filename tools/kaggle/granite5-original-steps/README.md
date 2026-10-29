# Granite Speech5 original forward-step diagnostic

This GPU-only script complements the [accepted paired T4 receipt](../../../docs/granite5-original-t4-benchmark-2026-10-29.json). It reuses that run's pinned JFK WAV, original model revision, Transformers 4.57.3, F16 CUDA placement, chat prompt, greedy one-beam generation, 96-token cap and four CPU threads. It does not load the native Q4_K runtime and does not produce a cross-engine speed ratio.

The script downloads the original checkpoint only on the GPU worker. It rejects P100 before downloading weights and checks worker RAM and scratch space. It saves `receipt.json` in `/kaggle/working` as each result arrives. The private kernel is prepared for `tools/kaggle/kpush.py`; **it has not been submitted**.

## Timed calls

The timed arm registers `with_kwargs=True` pre/post hooks on `GraniteSpeechForConditionalGeneration`, its `encoder`, `projector` and `language_model`. Each hook boundary calls `torch.cuda.synchronize()` and uses `perf_counter_ns()`. The root prefill forward must carry audio features and run one encoder, one projector and one language-model forward. Each later cached root forward must have exactly one input token, no audio features, and one language-model forward. Those conditions are checked from actual hook calls. Every forward call and duration is retained in the receipt; totals and medians are derived from these rows.

One warmed hooks-off and hooks-on call precedes three alternating measured pairs on the **same loaded model**. `generate_ms` starts immediately before `model.generate` and ends after a CUDA synchronization, excluding model loading, processor work and transcript decoding. The hooks-off arm has no per-forward counters. The measured hooks-on/off difference reveals profiling disturbance; it is not a correction to subtract from a particular step. The root duration includes synchronization in nested stage hooks, so root/encoder/projector/language-model durations are **not additive**. Sampling, cache preparation and generation-loop Python work lie outside individual forward durations but inside `generate_ms`.

`generated_token_count` counts output IDs after the prompt, including any special token. `actual_single_token_decode_calls` counts only observed cached root forwards; a prefill forward may itself emit the first output token. The script never infers a step count from output length and makes no assumption that the two counts are equal. These counts also have different boundaries from the native receipt's 26 post-prefill decoder calls.

The gate requires all eight calls to have identical token-ID hashes and transcripts, zero JFK word error rate, at least ten generated IDs, F16 CUDA parameters, and observation of the script's PID with at least 1 GiB of GPU allocation. The resulting numbers describe one clip on one worker, with hook overhead explicitly measured.

## Model terms

The [pinned IBM model card](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/c075cf19e041a2e4d16c5ee24763e5b037d8f479/README.md) declares **Apache-2.0**. The diagnostic verifies that declaration in the downloaded card and records the card hash. This directory contains no IBM weights, generated output or model code. Redistribution of the model must carry the applicable [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) and any required notices; this diagnostic keeps only hashes, short transcript text and timing metadata in the worker receipt.
