# Dia broader CPU diagnostic — 2026-10-16

The numerical diagnostic passes; speech quality fails on all four cases,
including the original Python positives. This does not accept broader speech,
F16/Q4/GPU, production defaults or a release.

Frozen worker17a7aedc uses runtime5e04955e, full F32 neural weights, critical
F32 DAC output weights, serial head, F32 cross-cache and portable sampling.
Original source4a9e29b1 runs on PyTorch2.6 CPU with MKL_COMPATIBLE. Prompts,
seeds42/123, temperature1.2, top_p0.95, top_k45, CFG3 and capacity3072 were
declared before execution. No failed case was replaced.

[Hosted run38913941927](https://github.com/CrispStrobe/CrispASR/actions/runs/38913941927)
completed all four cases and correctly exited with failure. Official artifact
12057273677 contains3022635432 bytes, SHA256
`75d9f3ec9d4fa96f1e284d2cf602afcf8b288d5cbdaa5bea5bf88532eda26c90`.
Independent audit verifies1829 payloads,352 complete aligned stage comparisons,
actual original sampler replay of20556 decisions, source pins, physical tensor
audits, own-code DAC replay and CLI/Session equality. All input histories and
generated code grids exactly match original Python in all four cases.

| Case | Decoder calls | Original inputs/codes | Worst stage cosine | Largest relative L2 | Original/native ASR WER |
| --- | ---: | --- | ---: | ---: | --- |
| hello-42 | 401 | exact / exact | 0.9999741433 | 0.00719130 | 1 / 1 (empty) |
| hello-123 | 266 | exact / exact | 0.9999710707 | 0.00760665 | 1 / 1 (empty) |
| fox-42 | 455 | exact / exact | 0.9999626558 | 0.00864579 | 1 / 1 (empty) |
| fox-123 | 1162 | exact / exact | 0.9999620016 | 0.00872113 | 1 / 1 (empty) |

All352 rows satisfy cosine>=0.9995 and per-row norm error<2%. Actual ASR
also returns empty text on the original DAC decoding native codes and on CLI
PCM. Crossed [recognizer replay38930760374](https://github.com/CrispStrobe/CrispASR/actions/runs/38930760374)
now passes independent audit:54 manifest payloads,8 upload-dereferenced SONAME
aliases, six exact PCM inputs and all12 recognition cells. Both actual retained
ELF sets transcribe the prior accepted original PCM and JFK at WER0, and both
return empty text/WER1 for all four expanded original clips. Every selected
parent PCM, source snapshot, child result and actual mapped-library hash checks.
This argues against a recognizer-library regression; original loading/capture,
model/recipe quality and recognizer domain sensitivity remain distinct hypotheses.
The next control uses the direct original constructor and untouched generate(),
with original DAC WeightNorm hooks preserved and added observers rejected.

A second architecture, **original OpenAI Whisper base.en**, now also fails all16
expanded audio views while transcribing legacy and JFK at WER0. Source
`31243bad24cc746f07d4c8bfdd2d974872cb1803` (20250625), checkpoint
`25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead`,
F32 CPU, four threads, English, temperature0, no prompt or previous-text
conditioning. The original ffmpeg loader is unchanged. Independent audit
verifies77 payloads, all18 exact original-loader resampling replays, selected
sealed inputs, source/checkpoint hashes, raw outputs and WER.

| Case | Whisper WER in original / native / original-DAC / CLI | Recognizer output |
| --- | --- | --- |
| hello-42 | 1 / 1 / 1 / 1 | Empty |
| hello-123 | 1 / 1 / 1 / 1 | Unrelated short phrase |
| fox-42 | 24.7778 / 24.7778 / 24.7778 / 24.7778 | Repetitive hallucination |
| fox-123 | 1 / 1 / 1 / 1 | Punctuation only |

Whisper's fox-42 WER is223/9, including insertions; WER is not capped at1.
All four views of each case have the same Whisper transcript. This makes a
Nemotron-only explanation less likely; it does not yet identify an original
loading/capture, model, recipe or remaining recognizer-sensitivity cause.
No existing quality gate is replaced or relaxed.

[Complete independent Whisper proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/eeadb3f56a3db3ba0232bafb55ef8bc7d1389801/dia-20261016-broader-speech/independent-whisper-base-en-v2/proof.zip)
retains all inputs, actual16k loader arrays, raw outputs, source/license,
executable auditor, results and full logs. Revision`eeadb3f56a3db3ba0232bafb55ef8bc7d1389801`,
36241894bytes, SHA256`5ad0e46849c53a97fdc04e6a7e220093ab717f9f30c16497931be28a9832a9bc`;
full anonymous download is verified. Model weights are excluded.

Each table entry below is the worst value across the four cases; extrema in
different columns can come from different cases. Full per-case arrays/metrics
and output transcripts are indexed in the [machine receipt](dia-broader-speech-results-2026-10-16.json).
Original decoder-stage comparisons use aligned startup inputs; DAC stages
compare the actual original/native decoders on each native run's own codes.
These are not cosine comparisons of diverged autoregressive trajectories.

| Intermediate | Minimum row cosine | Maximum relative L2 | Maximum row norm error |
| --- | ---: | ---: | ---: |
| cond_logits | 1.0000000000 | 9.6537897e-07 | 6.1045447e-07 |
| cross_k0 | 1.0000000000 | 5.7754703e-07 | 6.7553882e-07 |
| cross_v0 | 1.0000000000 | 6.6066462e-07 | 4.1136042e-07 |
| dac_block_0_own_codes | 0.9999973650 | 0.00029779522 | 0.0012841443 |
| dac_block_1_own_codes | 0.9999987727 | 0.00034644882 | 0.0014505793 |
| dac_block_2_own_codes | 0.9999989740 | 0.00016937629 | 0.00085563404 |
| dac_block_3_own_codes | 0.9999999671 | 0.00014413947 | 0.00030952944 |
| dac_input_conv_own_codes | 0.9999988793 | 0.00021541249 | 0.0009041444 |
| dac_latent_own_codes | 0.9999984670 | 0.00028892945 | 0.00094911579 |
| dac_output_conv_own_codes | 0.9999995765 | 0.00095582665 | 0.00025868177 |
| dac_output_snake_own_codes | 0.9999999066 | 0.00020871954 | 0.00016592919 |
| decoded_audio_own_codes | 0.9999620016 | 0.0087211278 | 0.0002884323 |
| decoder_block0_ca_residual | 1.0000000000 | 6.9610044e-07 | 1.8964575e-07 |
| decoder_block0_cross_attention | 1.0000000000 | 6.7306265e-07 | 2.3114083e-06 |
| decoder_block0_cross_output | 1.0000000000 | 7.692358e-07 | 2.9850198e-07 |
| decoder_block0_cross_q | 1.0000000000 | 4.9301207e-07 | 4.4956351e-07 |
| decoder_block0_input | 1.0000000000 | 0 | 0 |
| decoder_block0_mlp_gate | 1.0000000000 | 5.6287759e-07 | 2.9072001e-07 |
| decoder_block0_mlp_hidden | 1.0000000000 | 1.3888669e-06 | 1.360916e-06 |
| decoder_block0_mlp_output | 1.0000000000 | 2.0016278e-06 | 1.7492912e-06 |
| decoder_block0_mlp_up | 1.0000000000 | 9.8749511e-07 | 2.2565216e-07 |
| decoder_block0_output | 1.0000000000 | 1.7746813e-06 | 1.574536e-06 |
| decoder_block0_pre_ca_norm | 1.0000000000 | 5.2518083e-07 | 2.7294555e-08 |
| decoder_block0_pre_mlp_norm | 1.0000000000 | 5.9966487e-07 | 1.3833942e-07 |
| decoder_block0_pre_sa_norm | 1.0000000000 | 0 | 0 |
| decoder_block0_sa_attention | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_block0_sa_output | 1.0000000000 | 7.3707194e-07 | 1.6220913e-07 |
| decoder_block0_sa_residual | 1.0000000000 | 7.2680475e-07 | 2.1868524e-07 |
| decoder_block17_ca_residual | 1.0000000000 | 1.1071419e-06 | 2.4000356e-07 |
| decoder_block17_cross_attention | 1.0000000000 | 7.969088e-07 | 1.0547769e-06 |
| decoder_block17_cross_output | 1.0000000000 | 9.3679212e-07 | 2.9994332e-07 |
| decoder_block17_cross_q | 1.0000000000 | 5.8631834e-07 | 5.0451273e-07 |
| decoder_block17_input | 1.0000000000 | 1.0383495e-06 | 2.6220317e-07 |
| decoder_block17_mlp_gate | 1.0000000000 | 1.0438305e-06 | 3.9624028e-07 |
| decoder_block17_mlp_hidden | 1.0000000000 | 1.2321132e-06 | 1.2494723e-06 |
| decoder_block17_mlp_output | 1.0000000000 | 1.7032307e-06 | 7.7678345e-07 |
| decoder_block17_mlp_up | 1.0000000000 | 1.2308213e-06 | 4.2208303e-07 |
| decoder_block17_output | 1.0000000000 | 1.3063162e-06 | 5.1299304e-07 |
| decoder_block17_pre_ca_norm | 1.0000000000 | 8.572428e-07 | 7.9473558e-08 |
| decoder_block17_pre_mlp_norm | 1.0000000000 | 1.5307719e-06 | 2.4805906e-07 |
| decoder_block17_pre_sa_norm | 1.0000000000 | 1.4365122e-06 | 2.6199664e-07 |
| decoder_block17_sa_attention | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| decoder_block17_sa_output | 1.0000000000 | 1.50574e-06 | 5.2097743e-07 |
| decoder_block17_sa_residual | 1.0000000000 | 1.1285995e-06 | 2.6057973e-07 |
| decoder_ca0 | 1.0000000000 | 7.692358e-07 | 2.9850198e-07 |
| decoder_hidden | 1.0000000000 | 1.8380892e-06 | 4.0691019e-07 |
| decoder_layer_0 | 1.0000000000 | 1.7746813e-06 | 1.574536e-06 |
| decoder_layer_1 | 1.0000000000 | 1.306964e-06 | 1.2273675e-06 |
| decoder_layer_10 | 1.0000000000 | 9.272386e-07 | 7.4886168e-07 |
| decoder_layer_11 | 1.0000000000 | 7.9781131e-07 | 3.7403662e-07 |
| decoder_layer_12 | 1.0000000000 | 7.983949e-07 | 3.5918458e-07 |
| decoder_layer_13 | 1.0000000000 | 9.2078181e-07 | 1.713914e-07 |
| decoder_layer_14 | 1.0000000000 | 9.9719363e-07 | 1.8674085e-07 |
| decoder_layer_15 | 1.0000000000 | 9.9379901e-07 | 2.3833043e-07 |
| decoder_layer_16 | 1.0000000000 | 1.0383495e-06 | 2.6220317e-07 |
| decoder_layer_17 | 1.0000000000 | 1.3063162e-06 | 5.1299304e-07 |
| decoder_layer_2 | 1.0000000000 | 1.2528395e-06 | 1.2492338e-06 |
| decoder_layer_3 | 1.0000000000 | 1.3620571e-06 | 1.4454731e-06 |
| decoder_layer_4 | 1.0000000000 | 1.3018485e-06 | 1.3392993e-06 |
| decoder_layer_5 | 1.0000000000 | 1.3785857e-06 | 1.4338201e-06 |
| decoder_layer_6 | 1.0000000000 | 1.2573791e-06 | 1.194366e-06 |
| decoder_layer_7 | 1.0000000000 | 1.1171336e-06 | 1.0494289e-06 |
| decoder_layer_8 | 1.0000000000 | 1.0005388e-06 | 8.291357e-07 |
| decoder_layer_9 | 1.0000000000 | 8.7868714e-07 | 6.9319724e-07 |
| decoder_sa0_attention | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_sa0_k | 1.0000000000 | 6.416451e-07 | 2.1665104e-07 |
| decoder_sa0_q | 1.0000000000 | 7.4945458e-07 | 7.5837883e-07 |
| decoder_sa0_v | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_sa17_attention | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| decoder_sa17_k | 1.0000000000 | 8.6986787e-07 | 7.4593745e-07 |
| decoder_sa17_q | 1.0000000000 | 6.6979039e-07 | 6.7107188e-07 |
| decoder_sa17_v | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| encoder_embedding | 1.0000000000 | 0 | 0 |
| encoder_layer_0 | 1.0000000000 | 1.481601e-07 | 1.8951192e-07 |
| encoder_layer_1 | 1.0000000000 | 1.7434047e-07 | 2.563124e-07 |
| encoder_layer_10 | 1.0000000000 | 5.0904957e-07 | 4.4571551e-07 |
| encoder_layer_11 | 1.0000000000 | 2.7332257e-07 | 1.7847408e-06 |
| encoder_layer_2 | 1.0000000000 | 2.5611121e-07 | 2.2293614e-07 |
| encoder_layer_3 | 1.0000000000 | 2.8721647e-07 | 2.433244e-07 |
| encoder_layer_4 | 1.0000000000 | 3.8390665e-07 | 2.6916194e-07 |
| encoder_layer_5 | 1.0000000000 | 3.8042075e-07 | 2.6548046e-07 |
| encoder_layer_6 | 1.0000000000 | 3.8472525e-07 | 2.9106552e-07 |
| encoder_layer_7 | 1.0000000000 | 4.0144007e-07 | 5.9063515e-07 |
| encoder_layer_8 | 1.0000000000 | 3.4536787e-07 | 4.4507828e-07 |
| encoder_layer_9 | 1.0000000000 | 4.2496899e-07 | 4.6437124e-07 |
| encoder_output | 1.0000000000 | 7.316707e-07 | 3.7107947e-07 |
| encoder_tokens | 1.0000000000 | 0 | 0 |
| uncond_logits | 1.0000000000 | 9.4143337e-07 | 6.8383328e-07 |

The complete failed experiment is permanently public: [unmodified official
raw archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4500d64ef00a0e2902a4e5411ccdac2bc2c056c3/dia-20261016-broader-speech/complete-failed-four-case-v1/raw.zip)
and [full executable audit, results and metadata](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4500d64ef00a0e2902a4e5411ccdac2bc2c056c3/dia-20261016-broader-speech/complete-failed-four-case-v1/proof.zip).
Revision`4500d64ef00a0e2902a4e5411ccdac2bc2c056c3` retains all3022635432 raw bytes,
SHA256`75d9f3ec9d4fa96f1e284d2cf602afcf8b288d5cbdaa5bea5bf88532eda26c90`. The audit pack is131244bytes,
SHA256`9eab7483f24349991aafac4d08455c508ca5677724b513836c3094a1d102b093`. Both full anonymous downloads were hash-verified.

The [complete crossed-ASR raw archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/951fa9f714dd2fc5346f8e5e9eadc0fdf9f3fed6/dia-20261016-broader-speech/crossed-asr-v1/raw.zip)
and [independent audit pack](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/951fa9f714dd2fc5346f8e5e9eadc0fdf9f3fed6/dia-20261016-broader-speech/crossed-asr-v1/proof.zip)
are also permanently public at revision`951fa9f714dd2fc5346f8e5e9eadc0fdf9f3fed6`.
Raw50159711bytes, SHA256`fdaa4da6e39a86ca52ddca48de50bc8c93de77f8d7bcfd906aab765517fee416`;
audit16576bytes, SHA256`58e45ef0a8ab893b703ea732c363ed77406f7832529c6ea28852053cfd89a612`.
Full anonymous downloads match both digests. Publication preserves failed
quality gates and establishes no model/default/GPU/Q4 acceptance.
