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
bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out);

// Built-in German G2P: LTS rules (always available) + optional IPA
// dictionary (787K words, auto-loaded from ~/.cache/crispasr/ipa_dict_de.txt
// or CRISPASR_DE_DICT_PATH env var, CC-BY-SA from open-dict-data).
// For non-German, returns false and falls through.
bool phonemize_builtin_de(const std::string& lang, const std::string& text, std::string& out);

// Built-in French G2P: LTS rules (always available) + optional IPA dictionary.
bool phonemize_builtin_fr(const std::string& lang, const std::string& text, std::string& out);

// Built-in Spanish G2P: LTS rules (seseo, lenition, yeísmo) + optional dict.
bool phonemize_builtin_es(const std::string& lang, const std::string& text, std::string& out);

// Try all available phonemizers in priority order.
// Order: builtin_{en,de,fr,es} → espeak_dlopen → espeak_popen
inline bool phonemize(const std::string& lang, const std::string& text, std::string& out) {
    if (phonemize_builtin_en(lang, text, out))
        return true;
    if (phonemize_builtin_de(lang, text, out))
        return true;
    if (phonemize_builtin_fr(lang, text, out))
        return true;
    if (phonemize_builtin_es(lang, text, out))
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
