// phonemizer.cpp — pluggable text-to-phoneme backends.

#include "phonemizer.h"
#include "espeak_dlopen.h"
#include "core/g2p_en.h"
#include "core/g2p_de.h"
#include "core/g2p_fr.h"
// Auto-download support — only available when linked with crispasr-lib
#ifdef CRISPASR_HAS_CACHE
#include "crispasr_cache.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace crispasr {

// ── Built-in English G2P (LTS rules + optional CMUdict/neural) ───────

static g2p_en::context g_g2p_ctx;
static std::mutex g_g2p_mu;
static bool g_g2p_cmudict_tried = false;

// Try to auto-load CMUdict on first use.
static void ensure_neural_g2p_loaded() {
    if (g_g2p_ctx.neural.loaded) return;
    const char* env = std::getenv("CRISPASR_G2P_MODEL_PATH");
    if (env && *env) {
        if (g2p_en::load_neural_g2p_file(g_g2p_ctx.neural, env))
            fprintf(stderr, "g2p: loaded neural G2P model from %s\n", env);
        return;
    }
    // Try cache dir
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
        std::string p = std::string(home) + "/.cache/crispasr/g2p_en.json";
        if (g2p_en::load_neural_g2p_file(g_g2p_ctx.neural, p))
            fprintf(stderr, "g2p: loaded neural G2P model from %s\n", p.c_str());
    }
}

static void ensure_cmudict_loaded() {
    if (g_g2p_ctx.dict.loaded || g_g2p_cmudict_tried) return;
    g_g2p_cmudict_tried = true;

    // Check env var first
    const char* env = std::getenv("CRISPASR_CMUDICT_PATH");
    if (env && *env) {
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, env);
        if (n > 0) { fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, env); return; }
    }

    // Try local cache dir
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
        std::string cache_path = std::string(home) + "/.cache/crispasr/cmudict.dict";
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, cache_path);
        if (n > 0) { fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, cache_path.c_str()); return; }
    }
#ifdef CRISPASR_HAS_CACHE
    // Auto-download (BSD license, public domain data)
    static const char* CMUDICT_URL =
        "https://raw.githubusercontent.com/cmusphinx/cmudict/refs/heads/master/cmudict.dict";
    std::string path = crispasr_cache::ensure_cached_file(
        "cmudict.dict", CMUDICT_URL, /*quiet=*/true, "crispasr", "");
    if (!path.empty()) {
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, path);
        if (n > 0) { fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, path.c_str()); return; }
    }
#endif
}

bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out) {
    // Only handles English
    if (!lang.empty() && lang.find("en") == std::string::npos && lang != "auto")
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_mu);
        ensure_cmudict_loaded();
        ensure_neural_g2p_loaded();
    }
    out = g2p_en::text_to_ipa(g_g2p_ctx, text);
    return !out.empty();
}

// ── Built-in German G2P (LTS rules + optional IPA dictionary) ────────

static g2p_de::context g_g2p_de_ctx;
static std::mutex g_g2p_de_mu;
static bool g_g2p_de_tried = false;

static void ensure_de_dict_loaded() {
    if (g_g2p_de_ctx.dict.loaded || g_g2p_de_tried) return;
    g_g2p_de_tried = true;
    const char* env = std::getenv("CRISPASR_DE_DICT_PATH");
    if (env && *env) {
        int n = g2p_de::load_ipa_dict_file(g_g2p_de_ctx.dict, env);
        if (n > 0) { fprintf(stderr, "g2p: loaded German IPA dict (%d entries) from %s\n", n, env); return; }
    }
    // Try local cache
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
        std::string p = std::string(home) + "/.cache/crispasr/ipa_dict_de.txt";
        int n = g2p_de::load_ipa_dict_file(g_g2p_de_ctx.dict, p);
        if (n > 0) { fprintf(stderr, "g2p: loaded German IPA dict (%d entries) from %s\n", n, p.c_str()); return; }
    }
#ifdef CRISPASR_HAS_CACHE
    // Auto-download (CC-BY-SA, requires attribution — separate data file)
    static const char* DE_DICT_URL =
        "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/de.txt";
    std::string path = crispasr_cache::ensure_cached_file(
        "ipa_dict_de.txt", DE_DICT_URL, /*quiet=*/true, "crispasr", "");
    if (!path.empty()) {
        int n = g2p_de::load_ipa_dict_file(g_g2p_de_ctx.dict, path);
        if (n > 0) { fprintf(stderr, "g2p: loaded German IPA dict (%d entries) from %s\n", n, path.c_str()); return; }
    }
#endif
}

bool phonemize_builtin_de(const std::string& lang, const std::string& text, std::string& out) {
    if (!lang.empty() && lang.find("de") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_de_mu);
        ensure_de_dict_loaded();
    }
    out = g2p_de::text_to_ipa(g_g2p_de_ctx, text);
    return !out.empty();
}

// ── Built-in French G2P (LTS rules + optional IPA dictionary) ────────

static g2p_fr::context g_g2p_fr_ctx;
static std::mutex g_g2p_fr_mu;
static bool g_g2p_fr_tried = false;

static void ensure_fr_dict_loaded() {
    if (g_g2p_fr_ctx.dict.loaded || g_g2p_fr_tried) return;
    g_g2p_fr_tried = true;
    const char* env = std::getenv("CRISPASR_FR_DICT_PATH");
    if (env && *env) {
        int n = g2p_fr::load_ipa_dict_file(g_g2p_fr_ctx.dict, env);
        if (n > 0) { fprintf(stderr, "g2p: loaded French IPA dict (%d entries) from %s\n", n, env); return; }
    }
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
        std::string p = std::string(home) + "/.cache/crispasr/ipa_dict_fr.txt";
        int n = g2p_fr::load_ipa_dict_file(g_g2p_fr_ctx.dict, p);
        if (n > 0) { fprintf(stderr, "g2p: loaded French IPA dict (%d entries) from %s\n", n, p.c_str()); return; }
    }
#ifdef CRISPASR_HAS_CACHE
    static const char* FR_DICT_URL =
        "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/fr.txt";
    std::string path = crispasr_cache::ensure_cached_file(
        "ipa_dict_fr.txt", FR_DICT_URL, /*quiet=*/true, "crispasr", "");
    if (!path.empty()) {
        int n = g2p_fr::load_ipa_dict_file(g_g2p_fr_ctx.dict, path);
        if (n > 0) { fprintf(stderr, "g2p: loaded French IPA dict (%d entries) from %s\n", n, path.c_str()); return; }
    }
#endif
}

bool phonemize_builtin_fr(const std::string& lang, const std::string& text, std::string& out) {
    if (!lang.empty() && lang.find("fr") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_fr_mu);
        ensure_fr_dict_loaded();
    }
    out = g2p_fr::text_to_ipa(g_g2p_fr_ctx, text);
    return !out.empty();
}

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

} // namespace crispasr
