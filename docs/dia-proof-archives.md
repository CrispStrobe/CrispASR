# Reproducing the retained Dia proofs

The [public archive receipt](historical/dia-public-proof-archives-2026-10-13.json) pins four
complete proof ZIPs in `cstr/crispasr-regression-fixtures` at immutable revision
`6693718d8112b62425dfab5b941e61fb7b0334eb`. All 77 original files retain their
recorded hashes. All four public ZIPs were downloaded anonymously and verified
in full; their combined size is 619,724,990 bytes.

Use a **new destination** on storage. The standard-library helper downloads
without HF credentials, verifies the pinned ZIP size/SHA, rejects unsafe or
duplicate members and unexpected coverage, checks every extracted file against
the manifest, then returns the original `heavy-<run>` proof folder. It preserves
existing paths and removes its temporary download after validation.

```bash
python3 tools/dia_proof_archives.py portable-rng ${CRISPASR_STORAGE}/dia-proof/rng
python3 tools/dia_proof_archives.py canonical-original ${CRISPASR_STORAGE}/dia-proof/original
```

Both actual layouts have been validated: the public RNG proof through the real
anonymous downloader, and the canonical ZIP against its previously anonymously
verified immutable archive. Eight small contracts cover corruption, missing
manifest, incorrect payload hashes, unsafe/duplicate/extra members and existing
paths. The path-filtered `dia-proof-archives.yml` repeats those contracts and
retrieves the full public RNG proof in CI.

| Name | Preserved scope |
| --- | --- |
| `canonical-original` | Original short speech WER 0; native F16 stage replay 40/40; Q4 replay fails 13/40. |
| `portable-rng` | Complete compiled RNG/native-filter component replay: 11,034 draws and 1,210 codec frames exact. |
| `native-speech-rejection` | Original neural model with portable sampler gives byte-identical canonical audio and WER 0; native F16 CLI/C-ABI audio remains quiet with WER 1. |
| `dac-converter` | Real critical-DAC conversion: +1,312 bytes, exact validated candidate, all 134 physical tensor rows. |

These archives preserve the original references and negative controls. They do
not establish native free-speech, Q4 or GPU acceptance. The current neural
precision investigation is tracked in `PLAN.md` and the
[completed precision-control receipt](historical/dia-neural-precision-hosted-2026-10-13.json).

Two supplementary archives at public revision
`68873c45c08a36372667fa6b62e746072b9bf1f5`, folder `dia-20261013-neural`,
preserve the 41-file neural precision artifact plus its complete hosted log
and all four original tiny-cache probe files. Full anonymous size/SHA checks
pass for both ZIPs (245,203,012 bytes combined). The
[supplementary public receipt](historical/dia-neural-public-proof-2026-10-13.json) provides
pinned download URLs and complete per-file manifests. The neural archive
retains both 40/40 stage passes and all four rejected free-speech results; the
tiny probe establishes cache-rounding effects only.


## Cache and late-layer controls

These five supplemental archives use their own immutable revision. The original
retrieval-helper aliases keep their existing pins. Every full ZIP has passed
anonymous size/SHA verification; each includes its tracked evidence and payload
manifest. Public account placeholders have explicit original/public hashes;
private originals remain preserved. These diagnostic scopes do not accept native
speech or performance.

| Archive | Scope |
| --- | --- |
| [cpu-cross-cache-rejection.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/13cfd81146f17b746a47ce154013603aeabef658/dia-20261013-cache/cpu-cross-cache-rejection.zip) | Complete CPU F16/F32/F16-repeat cache controls: 120 replay stages pass, six speech checks fail empty/WER 1; exact OFF/repeat arrays. |
| [cuda-cross-cache-rejection.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/13cfd81146f17b746a47ce154013603aeabef658/dia-20261013-cache/cuda-cross-cache-rejection.zip) | Complete actual T4 v2 cache controls: actual CUDA allocation/splits, 120 replay stages pass, six speech checks fail empty/WER 1; no speed/model acceptance. |
| [late-layer-partial-rejection.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/13cfd81146f17b746a47ce154013603aeabef658/dia-20261013-cache/late-layer-partial-rejection.zip) | Partial hosted teacher replay at step 978: original speech passes, two comparisons pass; logging counter rejects and repeat/negative controls not executed. |
| [rotary-coefficient-probe.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/13cfd81146f17b746a47ce154013603aeabef658/dia-20261013-cache/rotary-coefficient-probe.zip) | Tiny actual-source rotary cache coefficient controls including negative; no full graph/model/GPU/speech acceptance. |
| [cache-first-draw-boundary.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/13cfd81146f17b746a47ce154013603aeabef658/dia-20261013-cache/cache-first-draw-boundary.zip) | Weight-free actual-original/native filter replay from retained logits and unchanged portable noise; identical-logit supports agree, neural differences change nucleus membership. |

[Complete publication and per-file verification receipt](historical/dia-cache-public-proof-2026-10-13.json).
