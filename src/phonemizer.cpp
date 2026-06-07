// phonemizer.cpp — pluggable text-to-phoneme backends.

#include "phonemizer.h"
#include "espeak_dlopen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace crispasr {

// ── espeak-ng via dlopen ─────────────────────────────────────────────

static std::mutex g_espeak_mu;
static bool g_espeak_inited = false;
static bool g_espeak_init_failed = false;
static std::string g_espeak_voice;

bool phonemize_espeak_dlopen(const std::string& lang, const std::string& text, std::string& out) {
    std::lock_guard<std::mutex> g(g_espeak_mu);
    if (g_espeak_init_failed) return false;

    auto& dl = espeak_dl_get();
    if (!g_espeak_inited) {
        if (!dl.load()) return false;
        const char* data_path = std::getenv("CRISPASR_ESPEAK_DATA_PATH");
        int sr = dl.Initialize(
            CRISPASR_ESPEAK_AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path,
            CRISPASR_ESPEAK_INITIALIZE_PHONEME_IPA | CRISPASR_ESPEAK_INITIALIZE_DONT_EXIT);
        if (sr < 0) {
            g_espeak_init_failed = true;
            return false;
        }
        g_espeak_inited = true;
    }
    if (!dl.loaded) return false;
    if (g_espeak_voice != lang) {
        if (dl.SetVoiceByName(lang.c_str()) != 0) return false;
        g_espeak_voice = lang;
    }
    out.clear();
    const void* tp = text.c_str();
    while (tp) {
        const char* chunk = dl.TextToPhonemes(&tp, CRISPASR_ESPEAK_CHARS_UTF8, 0x02);
        if (chunk && *chunk) { if (!out.empty()) out += ' '; out += chunk; }
    }
    return !out.empty();
}

// ── espeak-ng via popen ──────────────────────────────────────────────

bool phonemize_espeak_popen(const std::string& lang, const std::string& text, std::string& out) {
#ifdef _WIN32
#define PHON_POPEN _popen
#define PHON_PCLOSE _pclose
    const char* redir = " 2>NUL";
#else
#define PHON_POPEN popen
#define PHON_PCLOSE pclose
    const char* redir = " 2>/dev/null";
#endif
    // Shell-quote the text (basic — wraps in single quotes, escapes embedded quotes)
    std::string cmd = "espeak-ng -q --ipa=3 -v ";
    cmd += lang;
    cmd += " '";
    for (char c : text) {
        if (c == '\'') cmd += "'\\''";
        else cmd += c;
    }
    cmd += "'";
    cmd += redir;
    FILE* fp = PHON_POPEN(cmd.c_str(), "r");
    if (!fp) return false;
    out.clear();
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) len--;
        if (!out.empty() && len > 0) out += ' ';
        out.append(buf, len);
    }
    PHON_PCLOSE(fp);
    return !out.empty();
#undef PHON_POPEN
#undef PHON_PCLOSE
}

// ── Stubs for future backends ────────────────────────────────────────

bool phonemize_cmudict(const std::string& /*lang*/, const std::string& /*text*/, std::string& /*out*/) {
    // TODO: implement CMU Pronouncing Dictionary lookup.
    // Load ~134K word→ARPAbet entries, convert ARPAbet→IPA via static table.
    // English-only; return false for non-English langs.
    return false;
}

bool phonemize_neural_g2p(const std::string& /*lang*/, const std::string& /*text*/, std::string& /*out*/) {
    // TODO: implement GGML-based neural grapheme-to-phoneme model.
    // Multilingual; ~5-20 MB model. DeepPhonemizer or similar.
    return false;
}

} // namespace crispasr
