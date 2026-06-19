// src/core/lang_names.h — ISO-639-1 → English language-name map.
//
// Single source of truth for the language-name lookup used when injecting
// a "transcribe/translate in <language>" instruction into an audio-LLM
// prompt. A bare two-letter code is unreliable in an LLM prompt — "de"
// reads as the English word "of" and steers the model to Spanish — so the
// spelled-out name is what gets sent (see PLAN §174).
//
// Header-only `inline` so it works across the CLI/library module boundary
// (`examples/cli/` adapters and the `src/` C ABI both include it without a
// link dependency). Previously copy-pasted into four places:
//   examples/cli/crispasr_backend_utils.h   (crispasr_iso_to_english_lang)
//   examples/cli/crispasr_backend_voxtral.cpp
//   src/crispasr_c_api.cpp                   (ca_iso_to_english_lang)
//   src/gemma4_e2b.cpp                       (g4e_lang_name, with extra
//                                             auto/allow_original handling)
//
// Coverage is the UNION of all four former maps (15 + uk/vi) so migrating
// any caller preserves its existing mappings exactly; the only change is
// that codes a given caller previously passed through raw now get spelled
// out — a strict improvement, never a different name. Unknown codes (or
// already-spelled-out names) pass through verbatim.
#pragma once

#include <string>

namespace core_lang {

inline std::string iso_to_english(const std::string& code) {
    if (code == "en")
        return "English";
    if (code == "de")
        return "German";
    if (code == "fr")
        return "French";
    if (code == "es")
        return "Spanish";
    if (code == "it")
        return "Italian";
    if (code == "pt")
        return "Portuguese";
    if (code == "ru")
        return "Russian";
    if (code == "ja")
        return "Japanese";
    if (code == "ko")
        return "Korean";
    if (code == "zh")
        return "Chinese";
    if (code == "nl")
        return "Dutch";
    if (code == "pl")
        return "Polish";
    if (code == "tr")
        return "Turkish";
    if (code == "ar")
        return "Arabic";
    if (code == "hi")
        return "Hindi";
    if (code == "uk")
        return "Ukrainian";
    if (code == "vi")
        return "Vietnamese";
    return code;
}

} // namespace core_lang
