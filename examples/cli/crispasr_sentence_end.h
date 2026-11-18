// crispasr_sentence_end.h — shared "does this '.' end a sentence?" check.
//
// The TTS chunker (crispasr_tts_chunking.cpp) and the subtitle splitter
// (--split-on-punct, crispasr_output.cpp) used to treat every ". " as a
// sentence end. German ordinals and abbreviations then split mid-phrase:
// "am 24. | November", "Herr Dr. | Sieben", "z. | B.". In TTS each false split
// is an audible pause; in subtitles a broken cue.
//
// Live translation already had a conservative, unit-tested rule for exactly
// this (lt_ends_sentence in crispasr_live_translate.h: one- and two-digit
// ordinals, single letters/initials, inner dots, ellipses and a DE/EN
// abbreviation list). This header lets the other two splitters reuse it for
// the word that ends in '.'; '!', '?' and CJK/Devanagari terminators keep
// their existing handling at the call sites. Two deliberate differences from
// live translation: an ellipsis still ends the piece (historical behaviour of
// both splitters), and a closing bracket/quote before the dot always ends it.
//
// Measured on 1,776 human-segmented UD German GSD r2.14 dev+test sentences
// joined into 356 five-sentence paragraphs (1,420 internal boundaries):
// false splits 72 -> 7 for both splitters, with the same boundaries found
// otherwise. CRISPASR_SENTENCE_ABBREV=0 restores the bare ". " rule.
#pragma once

#include "core/env_gate.h"
#include "crispasr_live_translate.h"

#include <cstddef>
#include <string>

// `dot` indexes a '.' in `text`. Returns true when the whitespace-delimited
// word ending at that '.' closes a sentence.
inline bool crispasr_period_ends_sentence(const std::string& text, std::size_t dot) {
    if (dot >= text.size() || text[dot] != '.')
        return false;
    if (core_env::explicitly_off("CRISPASR_SENTENCE_ABBREV"))
        return true;
    // Unlike live translation (which waits on a hesitation), these splitters
    // keep their historical break after an ellipsis ("... Diese Firma").
    if (dot > 0 && text[dot - 1] == '.')
        return true;
    // A closing bracket or quote before the dot is never an ordinal or an
    // abbreviation: "(Nr. 216/94).", ":-).", "sagte er".".
    if (dot > 0) {
        const char prev = text[dot - 1];
        if (prev == ')' || prev == ']' || prev == '"' || prev == '\'')
            return true;
    }
    std::size_t begin = dot;
    while (begin > 0) {
        const char c = text[begin - 1];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            break;
        --begin;
    }
    const auto words = crispasr::lt_split_words(text.substr(begin, dot + 1 - begin));
    return !words.empty() && crispasr::lt_ends_sentence(words, words.size() - 1);
}
