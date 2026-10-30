# Dia top-k index selection: integrated correctness repair

Tested candidate `e97546f546f8e9e3afcd078e67d86efc8756d7c5`, now rebased
and integrated into main, repairs a separate original-recipe bug.
The old sampler selected a probability threshold after softmax, retaining
extra tied probabilities even when the input logits were distinct. With
logits0/1e-8 and top-k1, the actual old header picks the lower logit in34/64
seeded trials. The corrected header picks the unique maximum in64/64.

The pinned original Python sampler selects exactly k temperature-scaled logit
indices. The candidate follows that rule and the finite-score CPU selection
strategy in [Torch2.6 TopKImpl](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/TopKImpl.h),
then retains the existing probability, nucleus and RNG operations. No public
ABI or default parameter changes. The shared runtime serves CLI and C ABI;
two semantic regressions are added to the existing unit test target.

Four actual original Torch2.6 CPU oracle fixtures cover a unique tiny maximum,
rounded probability ties, exact logit ties and both partial/nth selection paths.
Every old fixture has incorrect support; all corrected supports match the
original. The original function runs unchanged; only its multinomial call is
observed to retain final probabilities. This is a probability component check,
not a newly generated spoken-word result.

Baseline/candidate normalized sampling probabilities and actual seeded native
IDs remain byte-identical for all18,288 captured rows:666 earlier CFG rows plus
17,622 rows over both979-step F32 shared-history prefixes. Original supports
match exactly; maximum measured probability error is4.673e-7. The known
rank32/33 nucleus swap remains unchanged under top-k45, so this fix does not
resolve the recorded quiet-speech failure. Full-prefix inputs and masking
execute the actual original decoder-step expressions.

[Complete74-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/a07cb3e4d3f504e8be508a67adfbf8938032ff94/dia-20261014-sampler/complete-topk-selection-proof.zip) is public
and anonymously verified: 86,340,668 bytes, SHA256
`03debc238cd2c75cac4c19ac7cd991c8724a8e333dd30875394fe7ce7d9b9218`. It includes both actual headers, compiled observers/guards,
source/config/input hashes, original probabilities, all before/after distributions
and samples, methods, independent audit and notices. The committed header's
hash matches the one tested.

[General CI38637379126](https://github.com/CrispStrobe/CrispASR/actions/runs/38637379126),
[lint38637383035](https://github.com/CrispStrobe/CrispASR/actions/runs/38637383035),
and [secret scan38636413855](https://github.com/CrispStrobe/CrispASR/actions/runs/38636413855)
all passed on the exact candidate: 13 general jobs, 10 lint checks and the
secret scan. Linux unit testing passed 2,026 tests, including the two new
regressions. The runtime repair was rebased as
`bd9c9a2d38a00c39032654a6d2448ac572652f78`; header and test source remain byte-identical
to the tested candidate. It is integrated into main; main-tip CI is checked
separately. No internal PR is created. Native speech/Q4/GPU/performance and
release acceptance remain separate gates.
[Structured receipt](dia-topk-selection-2026-10-14.json).
