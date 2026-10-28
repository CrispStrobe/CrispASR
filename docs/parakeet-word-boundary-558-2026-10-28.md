# Parakeet standalone word-boundary repair — #558

Read the complete #558 report and all three comments on linked #15. The new
report concerns occasional English word joins on natural 4–10 minute speech,
including F16/Q8/Q4 and CLI/server results. The reporter's private audio and
matched original token traces are unavailable. Synthetic speech did not reproduce
their failure. Issue #558 remains open; this is a demonstrated text-grouping
repair, not an assertion that the reported recording now transcribes correctly.

## Demonstrated defect and repair

`spiece_to_text` retains a standalone SentencePiece `▁` as a space in full text.
The shared word-range helper discarded that marker's boundary when followed by
an unprefixed token. Word grouping and downstream word-based rendering could
therefore disagree with the already decoded full token text.

| Emitted visible token pieces | Before word grouping | After |
|---|---|---|
| ` on`, ` the`, ` `, `3090`, `,`, ` because` | `on / the3090, / because` | `on / the / 3090, / because` |
| ` but`, ` we`, ` could`, ` `, `util`, `ize`, `.`, ` that` | `but / we / couldutilize. / that` | `but / we / could / utilize. / that` |

`src/core/parakeet_word_ranges.h` now carries an interior standalone-space
boundary to the next non-punctuation token. Leading BOS and trailing markers
still create no word; punctuation attaches backward. Markers remain excluded
from word timestamps/confidence, and raw token ownership remains a partition.
Both native grouping and CLI/session orchestration use this same range helper.
No model, encoder, decoder, duration choice, weights or sampling changes.

## Validation and limits

The new production-function fixture failed all three initial cases before the
repair and passes afterward. The expanded fixture also checks repeated/empty
markers, punctuation between a marker and a lexical token, and a leading BOS
followed by punctuation. The full seven-test window/metadata suite passes,
including Japanese slicing, overlap/deduplication, equal-frame repair and
missing-token controls. All 13 session-metadata tests also pass, exercising
production C ABI getters and Python result handling. Its immutable historical
grouping fixture is unchanged;
2,000 randomized comparisons cover behavior without interior standalone spaces,
whose intentionally corrected behavior has separate explicit expectations.

This is a small compiled, model-free production grouping/orchestration test.
It does not establish physical CUDA acceptance, NeMo alignment on the reporter's
clip, or the provenance of their emitted token stream. Published v0.8.46 and the
already frozen v0.8.47 tag do not contain this later source repair.

## Bounded remaining work

1. Obtain a consented failing excerpt with enough original context, exact model
   bytes/settings, and matched NeMo output/token IDs. Keep personal audio and
   host/account details out of public artifacts.
2. Compare original versus native emitted pieces/IDs first. If both emit the
   standalone marker, check full text versus grouped/rendered text across CLI,
   C ABI/Python and server. If native never emits it, localize the first
   encoder/predictor/joint or duration divergence instead; inserting spaces
   heuristically cannot repair missing model tokens.
3. Reproduce with F16 first under identical original/native input and decoder
   settings; only then compare quantized profiles. CPU probes can isolate text
   formatting; any CUDA math claim needs supported actual CUDA evidence under
   the maintainer's Kaggle rules. No large local build or inference is needed.
4. Require the reporter's failing boundaries and broader original-reference
   controls before closing #558 or describing the whole report as fixed.
