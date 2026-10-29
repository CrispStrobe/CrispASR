// core/whisper_special_tokens.h — which special-token layout a whisper vocab uses.
//
// A whisper `.bin` either serializes its special tokens by NAME (`<|endoftext|>`,
// `<|startoftranscript|>`, `<|0.00|>`) or it does not, and the loader must pick one
// of two mutually exclusive paths:
//
//   serialized  -> read every id out of the vocab map by name
//   legacy      -> keep the compiled-in English-layout defaults and shift them
//                  (`token_eot++`, `token_sot++`, `+= dt`) for a multilingual model
//
// Choosing wrongly is silent and expensive, so the choice lives here as a PURE
// FUNCTION over ids that have already been looked up. That shape is the point:
//
//   * It cannot mutate anything. #322 was exactly a probe that assigned its
//     destination and then reported "not found" — `set_token_id()` returns true
//     AFTER writing, and sat on the left of a short-circuit `&&`, so a
//     half-serialized vocab left `token_eot` resolved while the flag said legacy.
//     The legacy fixup then incremented an already-correct id (50257 -> 50258,
//     aliasing `token_sot`). A predicate that takes plain ints has nothing to
//     corrupt.
//   * It is testable without a model. tests/test-whisper-special-tokens.cpp pins
//     all four vocab shapes that exist in the wild, which nothing did before —
//     whisper is absent from the regression manifest and no test anywhere
//     asserted a token id.
//
// The four shapes, measured across eight real whisper `.bin` files:
//
//   prebuilt multilingual   eot absent,  sot absent   -> legacy   (tiny/base/large-v3)
//   English-only (.en)      eot 50256,   sot absent   -> legacy   (not multilingual,
//                                                        so the fixup never runs)
//   HF-converted multiling. eot 50257,   sot absent   -> legacy   <-- the #322 trap
//   CrispASR-converted      eot/sot/beg all present   -> serialized
//
// Weight-free and header-only.

#pragma once

#include <cstdint>

namespace core_whisper_specials {

// Ids read out of the serialized vocab, or `kAbsent` when the name is not present.
inline constexpr int kAbsent = -1;

struct Serialized {
    int eot = kAbsent; // <|endoftext|>
    int sot = kAbsent; // <|startoftranscript|>
    int beg = kAbsent; // <|0.00|> — the pivot for every timestamp rule
};

// Should the loader read the special ids by name?
//
// `<|0.00|>` is REQUIRED, not merely nice to have. The serialized branch resolves
// `token_beg` from it, and `token_beg` is what `whisper_process_logits()` compares
// against for every timestamp decision. A vocab carrying eot+sot but not `<|0.00|>`
// would otherwise take the branch and leave `token_beg` at the English-layout
// default 50363 with no `dt` shift applied — worse than either path taken whole.
// Requiring it costs nothing for the models the serialized branch exists to serve:
// a CrispASR-converted vocab writes all three.
inline bool use_serialized(const Serialized& s) {
    return s.eot != kAbsent && s.sot != kAbsent && s.beg != kAbsent;
}

// Only mandatory IDs are checked here. Optional serialized specials may be
// absent; Tiron speaker tokens can legitimately follow the timestamp range.
inline bool valid_logits_indices(int vocab_size, int eot, int sot, int beg, bool has_speakers, int speaker_beg,
                                 int n_speakers) {
    const auto in_range = [vocab_size](int id) { return id >= 0 && id < vocab_size; };
    if (vocab_size <= 0 || !in_range(eot) || !in_range(sot) || !in_range(beg))
        return false;
    return !has_speakers || (n_speakers > 0 && in_range(speaker_beg) &&
                             static_cast<std::int64_t>(speaker_beg) + n_speakers <= vocab_size);
}

} // namespace core_whisper_specials
