// phonemizer.h — pluggable text-to-phoneme interface.
//
// Provides a common abstraction for phonemization backends:
//   1. espeak-ng (via dlopen or popen) — GPLv3, loaded at runtime
//   2. [future] CMUdict lookup — public domain, English-only
//   3. [future] Neural G2P — MIT/Apache, multilingual
//   4. [future] GGUF-embedded dictionary — zero dependencies
//
// Each backend implements the same interface. The runtime tries them
// in priority order until one succeeds.

#pragma once

#include <set>
#include <string>
#include <vector>
#include <functional>

namespace crispasr {

// Phonemizer backend interface.
// text  = UTF-8 input text (e.g. "Hello world")
// lang  = espeak-ng voice name or BCP-47 tag (e.g. "en-us")
// out   = IPA phoneme string (e.g. "həlˈoʊ wˈɜːld")
// Returns true on success.
using phonemize_fn = std::function<bool(const std::string& lang, const std::string& text, std::string& out)>;

// Built-in backend: espeak-ng via dlopen (MIT-clean, loads GPL at runtime).
// Returns false if libespeak-ng is not available.
bool phonemize_espeak_dlopen(const std::string& lang, const std::string& text, std::string& out);

// Built-in backend: espeak-ng via popen subprocess.
// Returns false if the espeak-ng binary is not on $PATH.
bool phonemize_espeak_popen(const std::string& lang, const std::string& text, std::string& out);

// Built-in English G2P: LTS rules (always available, zero deps) +
// optional CMUdict (134K words, auto-loaded from ~/.cache/crispasr/cmudict.dict
// or CRISPASR_CMUDICT_PATH env var) + optional neural G2P (GRU seq2seq).
// Produces IPA directly via ARPAbet→IPA conversion table.
// For non-English, returns false and falls through.
// Emits the ESPEAK-IPA dialect (`tʃ`, `oʊ`, `ɜː`, length marks) — see
// core/phoneme_dialect.h. That is what piper expects and what g2p_en's ARPAbet
// table is tuned for. A backend trained on a different spelling (Kokoro was
// trained on misaki's) converts the result with core_phoneme::convert() rather
// than changing what this function emits, which would regress the others.
// `misaki_style` selects the CONSUMER's output conventions (contextual function
// words, punctuation carried through, hyphenated compounds kept whole) rather
// than a different dictionary — Kokoro needs them even on this CMUdict fallback
// path, piper must not have them. See g2p_en::style.
bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out,
                          bool misaki_style = false);

// #435: the three output conventions a CONSUMER picks, mirroring g2p_en::style
// without dragging core/g2p_en.h (1700 lines, Kokoro-shaped) into this header.
// They are not properties of the dictionary — one dictionary serves several
// consumers with different conventions, which is why they are a parameter and
// not a build flag.
struct g2p_style {
    // Contextual function-word rules (the/to/a/an/in) + the capitalisation
    // stress rule. Needs the NEXT word's phonemes. Off for espeak-derived
    // dictionaries: those already encode their own reductions.
    bool context_words = false;
    // Carry `,.;:!?` through into the phoneme string. Required by any model
    // whose symbol table contains punctuation — that is how it pauses, and
    // dropping it delivers a paragraph in one breath. Zonos is one: its Python
    // reference phonemizes with `preserve_punctuation=True`.
    bool emit_punctuation = false;
    // Keep a hyphenated compound as one word.
    bool join_hyphenated = false;
};

// Same English G2P, with the conventions stated explicitly instead of chosen by
// a bool. The `bool misaki_style` overload above remains the Kokoro spelling and
// is unchanged; this one exists because "espeak conventions BUT keep the
// punctuation" is a real combination that the bool cannot express.
bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out, const g2p_style& style);

// Which built-in G2P covers `lang`, or nullptr when none does.
// Returns one of the literals "en", "de", "fr", "es", "ru".
//
// Matches the PRIMARY SUBTAG exactly ("en-us" -> en, "es-419" -> es), which the
// individual phonemize_builtin_* functions do NOT: they test
// `lang.find("de") != npos`, so "sv" is safe but a voice name like "nl-de" or a
// three-letter code containing the digraph would be claimed by the wrong
// language. Callers that dispatch by language should ask here first.
const char* builtin_g2p_language(const std::string& lang);

// #435: one entry point for a phoneme-conditioned TTS backend — dispatch to the
// built-in G2P for `lang` with the TTS conventions applied (punctuation carried
// through). Returns false when no built-in covers `lang`, so the caller falls
// through to espeak rather than shipping silence.
//
// This is the whole point of phonemizer.cpp living in crispasr-core: any backend
// that links the core lib can now reach a non-GPL G2P for en/de/fr/es/ru
// without depending on kokoro.
bool phonemize_builtin_tts(const std::string& lang, const std::string& text, std::string& out);

// #316: same, but Tier 0 is misaki's own lexicon (Kokoro's G2P, Apache-2.0)
// instead of CMUdict — ~94% phoneme agreement with misaki on ordinary prose vs
// ~58% for the CMUdict path. Output is still in the espeak dialect for words
// that miss the lexicon, so the caller applies core_phoneme::convert() exactly
// as it does for phonemize_builtin_en (the conversion is idempotent on
// lexicon hits: misaki never emits ɚ/ɝ/r/ː). Returns false when the lexicon is
// not installed — callers must fall back rather than ship worse pronunciation.
bool phonemize_misaki_en(const std::string& lang, const std::string& text, std::string& out);
bool misaki_lexicon_available();

// Built-in German G2P: LTS rules (always available) + optional IPA
// dictionary (787K words, auto-loaded from ~/.cache/crispasr/ipa_dict_de.txt
// or CRISPASR_DE_DICT_PATH env var, CC-BY-SA from open-dict-data).
// For non-German, returns false and falls through.
// `tts_punctuation` keeps `,.;:!?` in the phoneme string. Kokoro's vocabulary
// has them and they are how it pauses; piper's espeak inventory has never been
// fed them. See g2p_en::style — same split, one language further on.
bool phonemize_builtin_de(const std::string& lang, const std::string& text, std::string& out,
                          bool tts_punctuation = false);

// Built-in French G2P: LTS rules (always available) + optional IPA dictionary.
bool phonemize_builtin_fr(const std::string& lang, const std::string& text, std::string& out,
                          bool tts_punctuation = false);

// Built-in Spanish G2P: LTS rules (seseo, lenition, yeísmo) + optional dict.
bool phonemize_builtin_es(const std::string& lang, const std::string& text, std::string& out,
                          bool tts_punctuation = false);

// Built-in Russian G2P: a 813K-entry IPA dictionary with lexical stress already
// resolved (bene-ges/ru_g2p_ipa_bert_large, CC-BY-4.0) in front of letter-to-
// sound rules that cover palatalisation, voicing assimilation, final devoicing
// and stress-driven vowel reduction.
//
// TWO LIMITS, because a caller that does not know them will read the output as
// a bug rather than as the state of the art here:
//
//  * HETERONYMS. The upstream project shipped 17,359 words it judged genuinely
//    ambiguous and REMOVED them from the vocabulary — the two files are
//    disjoint. So `замок`, `мука`, `все`, `уже`, `потом` have no dictionary
//    reading at all and go to the rules, which pick ONE of the readings from
//    spelling alone. A dictionary cannot carry sentence context and this one
//    does not pretend to. `CRISPASR_G2P_RU_HETERONYM_WARN=1` names them on
//    stderr as they occur.
//  * `ё` WRITTEN AS `е`. Russian text routinely omits the diaeresis, and the
//    dictionary's own keys fold it away as well. When the input writes `ё` it
//    is used (it is both a stress and a vowel-quality signal); when the input
//    writes `е` for it, neither path can recover it.
//
// For non-Russian, returns false and falls through.
bool phonemize_builtin_ru(const std::string& lang, const std::string& text, std::string& out,
                          bool tts_punctuation = false);

// Try all available phonemizers in priority order.
// Order: builtin_{en,de,fr,es,ru} → espeak_dlopen → espeak_popen
inline bool phonemize(const std::string& lang, const std::string& text, std::string& out) {
    if (phonemize_builtin_en(lang, text, out))
        return true;
    if (phonemize_builtin_de(lang, text, out))
        return true;
    if (phonemize_builtin_fr(lang, text, out))
        return true;
    if (phonemize_builtin_es(lang, text, out))
        return true;
    if (phonemize_builtin_ru(lang, text, out))
        return true;
    if (phonemize_espeak_dlopen(lang, text, out))
        return true;
    if (phonemize_espeak_popen(lang, text, out))
        return true;
    return false;
}

// Strip espeak-ng language-switch markers like (en), (it), (de-AT) from
// IPA output. These parenthesized ISO 639 codes are inserted by espeak
// when it detects a mid-sentence language switch. TTS backends don't
// understand them and read them literally (#169).
inline void strip_espeak_lang_markers(std::string& ipa) {
    size_t out = 0;
    size_t len = ipa.size();
    for (size_t i = 0; i < len;) {
        if (ipa[i] == '(' && i + 3 < len) {
            size_t j = i + 1;
            size_t alpha_start = j;
            while (j < len && j - alpha_start < 3 && ipa[j] >= 'a' && ipa[j] <= 'z')
                j++;
            if (j - alpha_start >= 2) {
                size_t before_region = j;
                if (j < len && ipa[j] == '-') {
                    j++;
                    size_t reg_start = j;
                    while (j < len && j - reg_start < 4 &&
                           ((ipa[j] >= 'a' && ipa[j] <= 'z') || (ipa[j] >= 'A' && ipa[j] <= 'Z') ||
                            (ipa[j] >= '0' && ipa[j] <= '9')))
                        j++;
                    if (j - reg_start < 2)
                        j = before_region;
                }
                if (j < len && ipa[j] == ')') {
                    j++;
                    if (j < len && ipa[j] == ' ')
                        j++;
                    i = j;
                    continue;
                }
            }
        }
        ipa[out++] = ipa[i++];
    }
    ipa.resize(out);
}

// Filter IPA output to only contain characters present in a model's
// phoneme inventory. Silently drops unmapped chars (combining marks,
// tie bars, etc.) that would cause garbled output.
//
// `valid_chars` should contain every single Unicode codepoint (as a
// UTF-8 string per char) that the model's phoneme_id_map accepts.
// Pass the keys of piper's phoneme_id_map JSON.
inline std::string filter_to_inventory(const std::string& ipa, const std::set<std::string>& valid_chars) {
    std::string out;
    size_t i = 0;
    while (i < ipa.size()) {
        // Decode one UTF-8 codepoint
        unsigned char c = (unsigned char)ipa[i];
        int cp_len = 1;
        if (c >= 0xF0)
            cp_len = 4;
        else if (c >= 0xE0)
            cp_len = 3;
        else if (c >= 0xC0)
            cp_len = 2;
        if (i + cp_len > ipa.size())
            break;
        std::string ch = ipa.substr(i, cp_len);
        if (valid_chars.count(ch) || ch == " ") {
            out += ch;
        }
        // else: silently drop unmapped char
        i += cp_len;
    }
    return out;
}

} // namespace crispasr
