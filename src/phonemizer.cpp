// phonemizer.cpp — pluggable text-to-phoneme backends.

#include "phonemizer.h"
#include "espeak_dlopen.h"
#include "core/g2p_en.h"
#include "core/g2p_de.h"
#include "core/g2p_fr.h"
#include "core/g2p_es.h"
// Auto-download support — only available when linked with crispasr-lib
#ifdef CRISPASR_HAS_CACHE
#include "crispasr_cache.h"
#endif

// OLaPh (MIT) and open-dict-data (CC-BY-SA) URL templates.
// CRISPASR_G2P_DICT_SOURCE env var selects provider:
//   "olaph"      → OLaPh (MIT, iisys-hof/olaph) — default
//   "open-dict"  → open-dict-data (CC-BY-SA, Wiktionary-sourced)
struct g2p_dict_urls {
    const char* olaph_file;
    const char* olaph_url;
    const char* opendict_file;
    const char* opendict_url;
};

static const g2p_dict_urls G2P_URLS_DE = {
    "olaph_de.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/de/de.txt",
    "ipa_dict_de.txt",
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/de.txt",
};
static const g2p_dict_urls G2P_URLS_FR = {
    "olaph_fr.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/fr/fr.txt",
    "ipa_dict_fr.txt",
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/fr.txt",
};
static const g2p_dict_urls G2P_URLS_ES = {
    "olaph_es.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/es/es.txt",
    "ipa_dict_es.txt",
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/es.txt",
};
static const g2p_dict_urls G2P_URLS_IT = {
    "olaph_it.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/it/it.txt",
    nullptr, nullptr,
};
static const g2p_dict_urls G2P_URLS_NL = {
    "olaph_nl.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/nl/nl.txt",
    nullptr, nullptr,
};
static const g2p_dict_urls G2P_URLS_PT = {
    // OLaPh doesn't have Portuguese yet; open-dict-data does
    nullptr, nullptr,
    "ipa_dict_pt.txt",
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/pt.txt",
};

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Check if user prefers open-dict-data over OLaPh
static bool prefer_opendict() {
    const char* src = std::getenv("CRISPASR_G2P_DICT_SOURCE");
    return src && std::string(src) == "open-dict";
}

// Try loading a dict from cache, with auto-download fallback.
// Returns number of entries loaded (0 = not found).
template<typename Dict>
static int try_load_dict(Dict& dict, const char* env_var, const g2p_dict_urls& urls,
                         int (*loader)(Dict&, const std::string&)) {
    // 1. Env var override
    const char* env = std::getenv(env_var);
    if (env && *env) {
        int n = loader(dict, env);
        if (n > 0) return n;
    }
    // 2. Local cache (check both providers)
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
        std::string base = std::string(home) + "/.cache/crispasr/";
        if (urls.olaph_file) {
            int n = loader(dict, base + urls.olaph_file);
            if (n > 0) return n;
        }
        if (urls.opendict_file) {
            int n = loader(dict, base + urls.opendict_file);
            if (n > 0) return n;
        }
    }
#ifdef CRISPASR_HAS_CACHE
    // 3. Auto-download
    bool use_od = prefer_opendict();
    const char* file = nullptr;
    const char* url = nullptr;
    if (use_od && urls.opendict_url) { file = urls.opendict_file; url = urls.opendict_url; }
    else if (urls.olaph_url)         { file = urls.olaph_file; url = urls.olaph_url; }
    else if (urls.opendict_url)      { file = urls.opendict_file; url = urls.opendict_url; }
    if (file && url) {
        std::string path = crispasr_cache::ensure_cached_file(
            file, url, /*quiet=*/true, "crispasr", "");
        if (!path.empty()) return loader(dict, path);
    }
#endif
    return 0;
}

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
    int n = try_load_dict(g_g2p_de_ctx.dict, "CRISPASR_DE_DICT_PATH", G2P_URLS_DE, g2p_de::load_ipa_dict_file);
    if (n > 0) fprintf(stderr, "g2p: loaded German IPA dict (%d entries)\n", n);
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
    int n = try_load_dict(g_g2p_fr_ctx.dict, "CRISPASR_FR_DICT_PATH", G2P_URLS_FR, g2p_fr::load_ipa_dict_file);
    if (n > 0) fprintf(stderr, "g2p: loaded French IPA dict (%d entries)\n", n);
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

// ── Built-in Spanish G2P (LTS rules + optional IPA dictionary) ───────

static g2p_es::context g_g2p_es_ctx;
static std::mutex g_g2p_es_mu;
static bool g_g2p_es_tried = false;

static void ensure_es_dict_loaded() {
    if (g_g2p_es_ctx.dict.loaded || g_g2p_es_tried) return;
    g_g2p_es_tried = true;
    int n = try_load_dict(g_g2p_es_ctx.dict, "CRISPASR_ES_DICT_PATH", G2P_URLS_ES, g2p_es::load_ipa_dict_file);
    if (n > 0) fprintf(stderr, "g2p: loaded Spanish IPA dict (%d entries)\n", n);
}

bool phonemize_builtin_es(const std::string& lang, const std::string& text, std::string& out) {
    if (!lang.empty() && lang.find("es") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_es_mu);
        ensure_es_dict_loaded();
    }
    out = g2p_es::text_to_ipa(g_g2p_es_ctx, text);
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
