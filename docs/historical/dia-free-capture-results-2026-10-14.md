# Dia free-running capture transparency

[GH run38527734866](https://github.com/CrispStrobe/CrispASR/actions/runs/38527734866)
completed successfully on source `eb94d724bc26b7643b5474456dc4fdd1e0200602`,
ggmlc36dab89. Complete official artifact digest, all archive members, 267
delivery entries, 261 producer payload hashes and six immutable source files
verify. Independent audit recomputes controls and recognizer WER from raw
outputs, rather than accepting producer booleans.

CLI and C ABI each run five arms without forcing: plain, logging-only,
common step978 capture, extended block-zero capture, plain repeat. All five
complete PCM arrays within each surface match byte-for-byte. All three
logged arms retain 1,115 complete steps; logits, input histories and sampled
histories are byte-exact within and across both surfaces. Step978 selected
logits match their full-history rows, and exactly16 extended snapshots occur
only in each extended arm. The plain arms have no debug dump flags; logging
adds no graph output marks.

The recipe is fixed diagnostic F32 model, F32 cross cache, portable RNG,
source-RoPE off, CPU/four threads, seed42, reported short phrase, unchanged
model capacity3072. This is CPU transparency for that recipe and phrase; it
is not a newly accepted production default, GPU, Q4 or performance result.
Model/codec/library hashes are producer receipts: their payloads are not in
this delivery and were not independently rehashed here. Existing physical
coefficient audits remain separate evidence.

| Surface baseline | PCM samples | Duration | RMS | ASR transcript | WER |
| --- | ---: | ---: | ---: | --- | ---: |
| C ABI | 563,200 | 12.770975s | 0.000155668 | empty | 1.0 |
| CLI | 563,200 | 12.770975s | 0.000155902 | empty | 1.0 |

CLI PCM is an int16 WAV representation; C ABI PCM retains floats. Within-
surface equality is exact; cross-surface raw PCM equality is not claimed.
Both surfaces have identical logged model histories. Speech remains rejected.
The result excludes these debug output marks as the cause of the current
CPU failure; it does not turn high stage cosine into spoken-word acceptance.
Further numerical/recipe isolation remains in the Dia roadmap; no seed,
filter, threshold or quality-gate change follows.

[Complete raw 262-payload archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/5feaff1755599305758f3f43248a63e71bd87b4b/dia-20261014-free-capture/artifact-11878953458.zip)
and [complete audit, method and logs](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/5feaff1755599305758f3f43248a63e71bd87b4b/dia-20261014-free-capture/complete-free-capture-audit.zip)
are public and fully anonymously verified. RawZIP478,636,694 bytes, SHA256
`ae9626d4020f45125bca0741c166e5e59f73d39781e4f6ea80250fbe6a5bf24e`;
auditZIP51,064 bytes, SHA256
`8d48d643a1a2e15118171336bbe40273dd60875fb013c77061e478b4e55beb37`.
[Structured independent result](dia-free-capture-results-2026-10-14.json).
