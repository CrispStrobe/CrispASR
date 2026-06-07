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

#include <string>
#include <vector>
#include <functional>

namespace crispasr {

// Phonemizer backend interface.
// text  = UTF-8 input text (e.g. "Hello world")
// lang  = espeak-ng voice name or BCP-47 tag (e.g. "en-us")
// out   = IPA phoneme string (e.g. "həlˈoʊ wˈɜːld")
// Returns true on success.
using phonemize_fn = std::function<bool(const std::string& lang,
                                         const std::string& text,
                                         std::string& out)>;

// Built-in backend: espeak-ng via dlopen (MIT-clean, loads GPL at runtime).
// Returns false if libespeak-ng is not available.
bool phonemize_espeak_dlopen(const std::string& lang, const std::string& text, std::string& out);

// Built-in backend: espeak-ng via popen subprocess.
// Returns false if the espeak-ng binary is not on $PATH.
bool phonemize_espeak_popen(const std::string& lang, const std::string& text, std::string& out);

// Stub: CMUdict lookup (English dictionary-based phonemizer).
// MeloTTS already has a full CMUdict + ARPAbet→IPA pipeline in melotts.cpp.
// This stub is for extracting it into a shared, model-independent module.
// Currently always returns false.
bool phonemize_cmudict(const std::string& lang, const std::string& text, std::string& out);

// Stub: Neural G2P (grapheme-to-phoneme model for OOV words).
// MeloTTS already has a GRU seq2seq neural G2P (29 graphemes → 74 ARPAbet
// phonemes, ~4 KB weights in GGUF metadata) in melotts.cpp. This stub is
// for extracting it into a shared module usable by piper/kokoro too.
// Currently always returns false.
bool phonemize_neural_g2p(const std::string& lang, const std::string& text, std::string& out);

// Try all available phonemizers in priority order.
// Order: cmudict → neural_g2p → espeak_dlopen → espeak_popen
inline bool phonemize(const std::string& lang, const std::string& text, std::string& out) {
    if (phonemize_cmudict(lang, text, out)) return true;
    if (phonemize_neural_g2p(lang, text, out)) return true;
    if (phonemize_espeak_dlopen(lang, text, out)) return true;
    if (phonemize_espeak_popen(lang, text, out)) return true;
    return false;
}

} // namespace crispasr
