# Reproducing the retained Dia proofs

The [public archive receipt](dia-public-proof-archives-2026-10-13.json) pins four
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
python3 tools/dia_proof_archives.py portable-rng /mnt/storage/dia-proof/rng
python3 tools/dia_proof_archives.py canonical-original /mnt/storage/dia-proof/original
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
[completed precision-control receipt](dia-neural-precision-hosted-2026-10-13.json).

Two supplementary archives at public revision
`68873c45c08a36372667fa6b62e746072b9bf1f5`, folder `dia-20261013-neural`,
preserve the 41-file neural precision artifact plus its complete hosted log
and all four original tiny-cache probe files. Full anonymous size/SHA checks
pass for both ZIPs (245,203,012 bytes combined). The
[supplementary public receipt](dia-neural-public-proof-2026-10-13.json) provides
pinned download URLs and complete per-file manifests. The neural archive
retains both 40/40 stage passes and all four rejected free-speech results; the
tiny probe establishes cache-rounding effects only.
