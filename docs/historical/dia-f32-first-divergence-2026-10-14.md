# Dia F32 first free-running divergence

Complete unforced native CLI/C-ABI inputs and logits from GH38527734866 are
byte-identical across surfaces. Their full arrays and the original canonical
reference from38472367538 are rehashed against accepted delivery ledgers.
All979 shared-input steps are independently compared over full1028-value
vocabulary rows, including magnitudes. No new model run is needed.

The first next-input difference is step979/codebook2: original877, native904.
The sampled output causing it is at step978. At that boundary, minimum row
cosine is0.9999999999981901, maximum row norm error7.316e-7 and relative
L2 1.413e-6. Maximum relative L2 over the complete shared prefix is5.267e-6.
This F32 result is distinct from the earlier F16 step76 analysis.

The actual pinned original decoder CFG, EOS masking, temperature, top-k45
and top-p0.95 code runs unchanged on both retained boundary logits. Its final
multinomial call is replaced only with the retained exponential-race noise.
Both original Torch and portable noise slices are rehashed and tested:

| Logit owner | Torch noise selects | Portable noise selects |
|---|---:|---:|
| Original | 877 | 877 |
| Native | 904 | 904 |

Nucleus support differs at exactly two entries, both in codebook2. Original
keeps46 and excludes904; native keeps904 and excludes46. Their CFG ranks
exchange32/33 at the cutoff. Original CFG values are6.791631222/6.791584969
for46/904; native values are6.791618347/6.791645050. Both supports contain32
tokens. Once included,904 wins the native exponential race with score4.6861
versus877's2.9426; this is not an almost-tied race or a noise-stream mismatch.
Tiny neural differences change a discontinuous filter's support. No top-p,
top-k, seed, tolerance or production default is changed.

A separate downstream check compiles and executes the literal native delay
reversal on known emitted rows derived from logged next-step inputs. The
unknown final stopping row is excluded. All963 aligned code frames before
the divergence neighborhood match the original exactly at zero offset; both
plus/minus-one-frame controls disagree. The conservative484,864-sample PCM
prefix also agrees: cosine0.999963626, norm ratio0.996749141. Both original
and native prefix RMS are approximately0.000152. The original itself is
nearly silent for roughly its first eleven seconds, so a long matched prefix
does not prove that native generation has reached the eventual spoken words.
Plus/minus512-sample comparisons are explicitly retained. This excludes a
broad delay/codec fault on that prefix, not every full-length codec case.

Independent audit recalculates all979 full-vocabulary comparisons, code
prefixes, three PCM checks and all four races/supports.
[Complete 36-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/80f53c2cf181688f41f16784a245997baf5a5c42/dia-20261014-decoder-boundaries/complete-f32-first-divergence.zip)
is public and anonymously verified: 131,732,617 bytes, SHA256
`f85c5dc2419546b47539584dcead95861d2d4e080f2886a59791139288d054a2`. It contains the complete shared-prefix logits/inputs,
aligned codes, boundary probabilities/noise/scores, source expressions,
compiled reversal adapter, parent-input hashes, methods and licenses.
[Structured result](dia-f32-first-divergence-2026-10-14.json).

Next isolate error accumulation beyond block0 using actual owned layer inputs
and caches, with the measured two-token CFG margin as a diagnostic quantity.
A filter workaround cannot substitute for reproducing the original recipe.
Native full speech remains rejected (empty ASR/WER1), and original speech
remains WER0. No Q4, GPU, speed or release acceptance follows.
