// phonemizer.cpp — pluggable text-to-phoneme backends.

#include "phonemizer.h"
#include "espeak_dlopen.h"
#include "core/g2p_en.h"
#include "core/g2p_inflect.h" // #316: pronounce regular inflections from their stems
#include "core/g2p_de.h"
#include "core/g2p_fr.h"
#include "core/g2p_es.h"
#include "core/g2p_ru.h"
// Auto-download support — only when compiled as part of crispasr-lib.
// Unit tests compile phonemizer.cpp standalone without the cache library.
#ifdef CRISPASR_BUILD
#include "crispasr_cache.h"
#define CRISPASR_HAS_CACHE 1
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

// Dict URLs: espeak-generated dicts (piper-compatible IPA, primary) +
// OLaPh MIT dicts (fallback) + open-dict-data CC-BY-SA (alt fallback).
// espeak dicts are pre-generated IPA from espeak-ng — factual phonetic data,
// not GPL-covered (same as GCC output not being GPL).
static const g2p_dict_urls G2P_URLS_DE = {
    "espeak_de.tsv",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/espeak_de.tsv",
    "olaph_de.txt",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/olaph_de.txt",
};
static const g2p_dict_urls G2P_URLS_FR = {
    "espeak_fr.tsv",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/espeak_fr.tsv",
    "olaph_fr.txt",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/olaph_fr.txt",
};
static const g2p_dict_urls G2P_URLS_ES = {
    "espeak_es.tsv",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/espeak_es.tsv",
    "olaph_es.txt",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/olaph_es.txt",
};
// Russian. Derived from bene-ges/ru_g2p_ipa_bert_large, **CC-BY-4.0** —
// attribution is a licence CONDITION, not a courtesy, so it is stated here
// beside the URL as well as on the dataset card:
//
//     ru_g2p_ipa.tsv / ru_heteronyms.txt
//       derived from https://huggingface.co/bene-ges/ru_g2p_ipa_bert_large
//       by bene-ges, licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/).
//       Changes: the ` (U+0060) stress marker was rewritten to ˈ (U+02C8),
//       the IPA was wrapped in /…/, and 1,952 `(по)…` keys were dropped.
//
// Unlike the other languages here there is no second provider: this is the only
// free, stress-resolved Russian pronunciation dictionary at this scale that we
// found, so both slots point at it.
static const g2p_dict_urls G2P_URLS_RU = {
    "ru_g2p_ipa.tsv",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/ru_g2p_ipa.tsv",
    nullptr,
    nullptr,
};
// The heteronym list travels the same road. It is NOT a second dictionary — see
// ensure_ru_dict_loaded() for what it is and is not good for.
static const g2p_dict_urls G2P_URLS_RU_HET = {
    "ru_heteronyms.txt",
    "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/ru_heteronyms.txt",
    nullptr,
    nullptr,
};
static const g2p_dict_urls G2P_URLS_IT = {
    "olaph_it.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/it/it.txt",
    nullptr,
    nullptr,
};
static const g2p_dict_urls G2P_URLS_NL = {
    "olaph_nl.txt",
    "https://raw.githubusercontent.com/iisys-hof/olaph/main/src/olaph/dictionaries/nl/nl.txt",
    nullptr,
    nullptr,
};
static const g2p_dict_urls G2P_URLS_PT = {
    nullptr,
    nullptr,
    "ipa_dict_pt.txt",
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/pt.txt",
};

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Global dict source override (set by CLI --g2p-dict or env var)
static std::string g_dict_source_override;

void phonemizer_set_dict_source(const std::string& source) {
    g_dict_source_override = source;
}

// Check if user prefers open-dict-data over OLaPh
static bool prefer_opendict() {
    if (!g_dict_source_override.empty())
        return g_dict_source_override == "open-dict";
    const char* src = std::getenv("CRISPASR_G2P_DICT_SOURCE");
    return src && std::string(src) == "open-dict";
}

// Try loading a dict from cache, with auto-download fallback.
// Returns number of entries loaded (0 = not found).
template <typename Dict>
static int try_load_dict(Dict& dict, const char* env_var, const g2p_dict_urls& urls,
                         int (*loader)(Dict&, const std::string&)) {
    // 1. Env var override
    const char* env = std::getenv(env_var);
    if (env && *env) {
        int n = loader(dict, env);
        if (n > 0)
            return n;
    }
    // 2. Local cache (check both providers)
    const char* home = std::getenv("HOME");
    if (!home)
        home = std::getenv("USERPROFILE");
    if (home) {
        std::string base = std::string(home) + "/.cache/crispasr/";
        if (urls.olaph_file) {
            int n = loader(dict, base + urls.olaph_file);
            if (n > 0)
                return n;
        }
        if (urls.opendict_file) {
            int n = loader(dict, base + urls.opendict_file);
            if (n > 0)
                return n;
        }
    }
#ifdef CRISPASR_HAS_CACHE
    // 3. Auto-download
    bool use_od = prefer_opendict();
    const char* file = nullptr;
    const char* url = nullptr;
    if (use_od && urls.opendict_url) {
        file = urls.opendict_file;
        url = urls.opendict_url;
    } else if (urls.olaph_url) {
        file = urls.olaph_file;
        url = urls.olaph_url;
    } else if (urls.opendict_url) {
        file = urls.opendict_file;
        url = urls.opendict_url;
    }
    if (file && url) {
        std::string path = crispasr_cache::ensure_cached_file(file, url, /*quiet=*/true, "crispasr", "");
        if (!path.empty())
            return loader(dict, path);
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
    if (g_g2p_ctx.neural.loaded)
        return;
    const char* env = std::getenv("CRISPASR_G2P_MODEL_PATH");
    if (env && *env) {
        if (g2p_en::load_neural_g2p_file(g_g2p_ctx.neural, env))
            fprintf(stderr, "g2p: loaded neural G2P model from %s\n", env);
        return;
    }
    // Try cache dir
    const char* home = std::getenv("HOME");
    if (!home)
        home = std::getenv("USERPROFILE");
    if (home) {
        std::string p = std::string(home) + "/.cache/crispasr/g2p_en.json";
        if (g2p_en::load_neural_g2p_file(g_g2p_ctx.neural, p))
            fprintf(stderr, "g2p: loaded neural G2P model from %s\n", p.c_str());
    }
}

static void ensure_cmudict_loaded() {
    if (g_g2p_ctx.dict.loaded || g_g2p_cmudict_tried)
        return;
    g_g2p_cmudict_tried = true;

    // Check env var first
    const char* env = std::getenv("CRISPASR_CMUDICT_PATH");
    if (env && *env) {
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, env);
        if (n > 0) {
            fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, env);
            return;
        }
    }

    // Try local cache dir
    const char* home = std::getenv("HOME");
    if (!home)
        home = std::getenv("USERPROFILE");
    if (home) {
        std::string cache_path = std::string(home) + "/.cache/crispasr/cmudict.dict";
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, cache_path);
        if (n > 0) {
            fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, cache_path.c_str());
            return;
        }
    }
#ifdef CRISPASR_HAS_CACHE
    // Auto-download (BSD license, public domain data).
    //
    // TWO hosts, upstream first, and the second one is not redundancy for its
    // own sake: measured on Kaggle 2026-09-14, English was the ONLY language
    // whose dictionary failed to load, and it is the only one fetched from
    // raw.githubusercontent.com -- the de/fr/es dicts below come from
    // huggingface.co and all three arrived. GitHub access from a Kaggle worker
    // is documented-flaky, so English silently dropped to the letter-to-sound
    // rules and phonemised "quick brown" as `kˈʌɪk bɹˈoʊn`, which an ASR
    // roundtrip read back as "cook bone". That is not a G2P limitation being
    // measured, it is a download being measured, and the two are indistinguishable
    // from the output.
    //
    // piper_tts.cpp has fetched the same file from the HF mirror all along, so
    // the mirror is not new infrastructure -- this path simply never used it.
    // Upstream stays FIRST: when it is reachable nothing about the result
    // changes, so this can only turn a silent degradation into a working
    // dictionary.
    static const char* CMUDICT_URLS[] = {
        "https://raw.githubusercontent.com/cmusphinx/cmudict/refs/heads/master/cmudict.dict",
        "https://huggingface.co/datasets/cstr/g2p-dicts/resolve/main/cmudict.dict",
    };
    for (const char* url : CMUDICT_URLS) {
        std::string path = crispasr_cache::ensure_cached_file("cmudict.dict", url, /*quiet=*/true, "crispasr", "");
        if (path.empty())
            continue;
        int n = g2p_en::load_cmudict_file(g_g2p_ctx.dict, path);
        if (n > 0) {
            fprintf(stderr, "g2p: loaded CMUdict (%d entries) from %s\n", n, path.c_str());
            return;
        }
    }
    fprintf(stderr, "g2p: WARNING: no English CMUdict could be loaded — falling back to "
                    "letter-to-sound rules, which mispronounce common words. Set "
                    "CRISPASR_CMUDICT_PATH to a local cmudict.dict to avoid this.\n");
#endif
}

// ── misaki lexicon (Kokoro) ──────────────────────────────────────────
//
// #316: Kokoro was trained on misaki's output, and our CMUdict-based G2P agrees
// with misaki on only ~58% of words — not because the conversion is wrong but
// because CMUdict makes different stress and unstressed-vowel choices. misaki
// ships its own lexicon (Apache-2.0); loading it as Tier 0 of a SEPARATE
// context takes agreement to ~94% on ordinary prose. Separate because piper
// must keep the espeak pronunciations: same G2P, different consumer.
//
// Generate the file with tools/convert-misaki-lexicon.py.
static g2p_en::context g_g2p_misaki_ctx;
static std::mutex g_g2p_misaki_mu;
static bool g_g2p_misaki_tried = false;

static void ensure_misaki_lexicon_loaded() {
    if (g_g2p_misaki_tried)
        return;
    g_g2p_misaki_tried = true;
    // Kokoro reads misaki's output, so this context follows misaki's rules:
    // contextual function words, punctuation carried through, hyphenated
    // compounds kept whole. Nothing set these before, so the whole
    // contextual-word layer shipped inert — "the" stayed `ði` everywhere and
    // the article "a" was read as the LETTER, `ˈA` (#316).
    g2p_en::configure_for_misaki(g_g2p_misaki_ctx);
    std::string path;
    if (const char* env = std::getenv("CRISPASR_MISAKI_DICT_PATH"); env && *env) {
        path = env;
    } else {
        const char* home = std::getenv("HOME");
        if (!home)
            home = std::getenv("USERPROFILE");
        if (home)
            path = std::string(home) + "/.cache/crispasr/misaki-us.txt";
    }
    if (!path.empty()) {
        // A .json path is misaki's own file; anything else is the TSV that
        // tools/convert-misaki-lexicon.py emits.
        const bool is_json = path.size() > 5 && path.compare(path.size() - 5, 5, ".json") == 0;
        int n = is_json ? g2p_en::load_misaki_json(g_g2p_misaki_ctx.espeak_ipa, g_g2p_misaki_ctx.phrase_final, path,
                                                   &g_g2p_misaki_ctx.letters)
                        : g2p_en::load_ipa_dict_file(g_g2p_misaki_ctx.espeak_ipa, path);
        if (n > 0) {
            g_g2p_misaki_ctx.espeak_ipa.loaded = true;
            g_g2p_misaki_ctx.phrase_final.loaded = !g_g2p_misaki_ctx.phrase_final.entries.empty();
            fprintf(stderr, "g2p: loaded misaki lexicon (%d entries) from %s\n", n, path.c_str());
        }
    }
#ifdef CRISPASR_HAS_CACHE
    if (!g_g2p_misaki_ctx.espeak_ipa.loaded) {
        // Fetch from UPSTREAM, not from a CrispASR mirror. The user receives the
        // lexicon from hexgrad/misaki under hexgrad's own terms, so CrispASR
        // redistributes nothing and no relicensing question arises — the same
        // route ensure_cmudict_loaded() already uses for cmusphinx/cmudict.
        //
        // Pinned to a commit: `main` can change a pronunciation under us, and a
        // G2P that shifts silently between runs is not reproducible.
        //
        // misaki is Apache-2.0. Its lexicon is largely espeak-ng-generated
        // (measured 2026-07-28: silver 87% identical to espeak `en-us` output,
        // gold 48%) — the same category as the espeak_*.tsv dicts above, which
        // this file already treats as factual phonetic data rather than
        // GPL-covered.
        static const char* MISAKI_REV = "fba1236595f2d2bf21d414ba6e57d25256afada3";
        const std::string base =
            std::string("https://raw.githubusercontent.com/hexgrad/misaki/") + MISAKI_REV + "/misaki/data/";
        int total = 0;
        // gold FIRST so it wins: load_misaki_json keeps the first entry seen.
        for (const char* which : {"us_gold.json", "us_silver.json"}) {
            std::string p2 = crispasr_cache::ensure_cached_file(std::string("misaki-") + which, base + which,
                                                                /*quiet=*/true, "crispasr", "");
            if (p2.empty())
                continue;
            total += g2p_en::load_misaki_json(g_g2p_misaki_ctx.espeak_ipa, g_g2p_misaki_ctx.phrase_final, p2,
                                              &g_g2p_misaki_ctx.letters);
        }
        if (total > 0) {
            g_g2p_misaki_ctx.espeak_ipa.loaded = true;
            g_g2p_misaki_ctx.phrase_final.loaded = !g_g2p_misaki_ctx.phrase_final.entries.empty();
            fprintf(stderr, "g2p: misaki lexicon %d entries (%zu phrase-final) from hexgrad/misaki@%.7s\n", total,
                    g_g2p_misaki_ctx.phrase_final.entries.size(), MISAKI_REV);
        }
    }
#endif
    // #316: the lexicon stores STEMS — only 46% of inflected forms are listed
    // verbatim (CMUdict lists 100%, which is why the espeak path does not need
    // this). Without the fallback every plural and past tense dropped to
    // CMUdict and lost the agreement with Kokoro's training data. Worth +9.7
    // points of whole-word phoneme agreement with misaki.
    g_g2p_misaki_ctx.inflect_fallback = [](const std::string& w) -> std::string {
        core_g2p_inflect::Params p;
        p.reduced_vowel = "ᵻ"; // misaki's reduced vowel
        p.flap = "T";          // misaki's flap
        return core_g2p_inflect::inflect(
            w,
            [](const std::string& stem) -> std::string {
                auto it = g_g2p_misaki_ctx.espeak_ipa.entries.find(stem);
                return it == g_g2p_misaki_ctx.espeak_ipa.entries.end() ? std::string() : it->second;
            },
            p);
    };

    // Words outside the lexicon still need SOME pronunciation; reuse the same
    // CMUdict + LTS tiers the default path uses.
    g2p_en::load_cmudict_file(g_g2p_misaki_ctx.dict, [] {
        if (const char* e = std::getenv("CRISPASR_CMUDICT_PATH"); e && *e)
            return std::string(e);
        const char* home = std::getenv("HOME");
        if (!home)
            home = std::getenv("USERPROFILE");
        return home ? std::string(home) + "/.cache/crispasr/cmudict.dict" : std::string();
    }());
}

// True when the misaki lexicon is actually available — callers fall back to
// phonemize_builtin_en() otherwise rather than silently using worse data.
bool misaki_lexicon_available() {
    std::lock_guard<std::mutex> g(g_g2p_misaki_mu);
    ensure_misaki_lexicon_loaded();
    return g_g2p_misaki_ctx.espeak_ipa.loaded;
}

bool phonemize_misaki_en(const std::string& lang, const std::string& text, std::string& out) {
    if (!lang.empty() && lang.find("en") == std::string::npos && lang != "auto")
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_misaki_mu);
        ensure_misaki_lexicon_loaded();
        if (!g_g2p_misaki_ctx.espeak_ipa.loaded)
            return false;
        out = g2p_en::text_to_ipa(g_g2p_misaki_ctx, text);
    }
    return !out.empty();
}

// Shared body for both public English entry points. `st` is already resolved to
// the consumer's conventions by the time it gets here.
static bool builtin_en_with_style(const std::string& lang, const std::string& text, std::string& out,
                                  const g2p_en::style& st) {
    // Only handles English
    if (!lang.empty() && lang.find("en") == std::string::npos && lang != "auto")
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_mu);
        ensure_cmudict_loaded();
        ensure_neural_g2p_loaded();
    }
    out = g2p_en::text_to_ipa(g_g2p_ctx, text, st);
    return !out.empty();
}

bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out, bool misaki_style) {
    // #316: this ONE context serves two consumers — piper, which wants espeak's
    // conventions, and Kokoro's fallback for when the misaki lexicon could not
    // be fetched. The dictionary is the same; the output conventions are not,
    // and Kokoro needs its punctuation whichever dictionary it ended up with.
    return builtin_en_with_style(lang, text, out, misaki_style ? g2p_en::misaki_style() : g_g2p_ctx.consumer());
}

bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out, const g2p_style& style) {
    g2p_en::style st;
    st.context_words = style.context_words;
    st.emit_punctuation = style.emit_punctuation;
    st.join_hyphenated = style.join_hyphenated;
    return builtin_en_with_style(lang, text, out, st);
}

// ── Built-in German G2P (LTS rules + optional IPA dictionary) ────────

static g2p_de::context g_g2p_de_ctx;
static std::mutex g_g2p_de_mu;
static bool g_g2p_de_tried = false;

static void ensure_de_dict_loaded() {
    if (g_g2p_de_ctx.dict.loaded || g_g2p_de_tried)
        return;
    g_g2p_de_tried = true;
    int n = try_load_dict(g_g2p_de_ctx.dict, "CRISPASR_DE_DICT_PATH", G2P_URLS_DE, g2p_de::load_ipa_dict_file);
    if (n > 0)
        fprintf(stderr, "g2p: loaded German IPA dict (%d entries)\n", n);
}

bool phonemize_builtin_de(const std::string& lang, const std::string& text, std::string& out, bool tts_punctuation) {
    if (!lang.empty() && lang.find("de") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_de_mu);
        ensure_de_dict_loaded();
    }
    g_g2p_de_ctx.emit_punctuation = tts_punctuation;
    // #316: read the German closed class the way espeak reads it in a SENTENCE,
    // not the citation form our per-word dictionary stores. Default ON — the
    // German Kokoro training recipe phonemizes whole sentences through espeak,
    // so the citation form is a spelling the model never saw.
    static const bool de_unstress = [] {
        const char* v = std::getenv("CRISPASR_G2P_DE_UNSTRESS");
        return !(v && *v && std::strcmp(v, "0") == 0);
    }();
    g_g2p_de_ctx.unstress_function_words = de_unstress;
    out = g2p_de::text_to_ipa(g_g2p_de_ctx, text);
    return !out.empty();
}

// ── Built-in French G2P (LTS rules + optional IPA dictionary) ────────

static g2p_fr::context g_g2p_fr_ctx;
static std::mutex g_g2p_fr_mu;
static bool g_g2p_fr_tried = false;

static void ensure_fr_dict_loaded() {
    if (g_g2p_fr_ctx.dict.loaded || g_g2p_fr_tried)
        return;
    g_g2p_fr_tried = true;
    int n = try_load_dict(g_g2p_fr_ctx.dict, "CRISPASR_FR_DICT_PATH", G2P_URLS_FR, g2p_fr::load_ipa_dict_file);
    if (n > 0)
        fprintf(stderr, "g2p: loaded French IPA dict (%d entries)\n", n);
}

bool phonemize_builtin_fr(const std::string& lang, const std::string& text, std::string& out, bool tts_punctuation) {
    if (!lang.empty() && lang.find("fr") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_fr_mu);
        ensure_fr_dict_loaded();
    }
    g_g2p_fr_ctx.emit_punctuation = tts_punctuation;
    out = g2p_fr::text_to_ipa(g_g2p_fr_ctx, text);
    return !out.empty();
}

// ── Built-in Spanish G2P (LTS rules + optional IPA dictionary) ───────

static g2p_es::context g_g2p_es_ctx;
static std::mutex g_g2p_es_mu;
static bool g_g2p_es_tried = false;

static void ensure_es_dict_loaded() {
    if (g_g2p_es_ctx.dict.loaded || g_g2p_es_tried)
        return;
    g_g2p_es_tried = true;
    int n = try_load_dict(g_g2p_es_ctx.dict, "CRISPASR_ES_DICT_PATH", G2P_URLS_ES, g2p_es::load_ipa_dict_file);
    if (n > 0)
        fprintf(stderr, "g2p: loaded Spanish IPA dict (%d entries)\n", n);
}

bool phonemize_builtin_es(const std::string& lang, const std::string& text, std::string& out, bool tts_punctuation) {
    if (!lang.empty() && lang.find("es") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_es_mu);
        ensure_es_dict_loaded();
    }
    g_g2p_es_ctx.emit_punctuation = tts_punctuation;
    out = g2p_es::text_to_ipa(g_g2p_es_ctx, text);
    return !out.empty();
}

// ── Built-in Russian G2P (dictionary + LTS rules) ────────────────────

static g2p_ru::context g_g2p_ru_ctx;
static std::mutex g_g2p_ru_mu;
static bool g_g2p_ru_tried = false;

static void ensure_ru_dict_loaded() {
    if (g_g2p_ru_ctx.dict.loaded || g_g2p_ru_tried)
        return;
    g_g2p_ru_tried = true;
    int n = try_load_dict(g_g2p_ru_ctx.dict, "CRISPASR_RU_DICT_PATH", G2P_URLS_RU, g2p_ru::load_ipa_dict_file);
    if (n > 0)
        fprintf(stderr, "g2p: loaded Russian IPA dict (%d entries)\n", n);

    // The heteronym list is DIAGNOSTIC, not a lookup tier. Upstream removed
    // these 17,359 words from the vocabulary because it could not choose a
    // reading for them, so there is nothing here to look up — they are exactly
    // the words the letter-to-sound rules have to handle, and the rules pick
    // ONE reading from spelling with no sentence context. Fetching the list
    // buys the ability to SAY so when it happens instead of leaving a
    // mispronunciation looking like a rule bug.
    //
    // Only fetched when someone asked to be told, so the default path does not
    // pull a file it will never read.
    const char* warn = std::getenv("CRISPASR_G2P_RU_HETERONYM_WARN");
    g_g2p_ru_ctx.warn_heteronyms = warn && *warn && std::strcmp(warn, "0") != 0;
    if (g_g2p_ru_ctx.warn_heteronyms) {
        int h = try_load_dict(g_g2p_ru_ctx.het, "CRISPASR_RU_HETERONYMS_PATH", G2P_URLS_RU_HET,
                              g2p_ru::load_heteronyms_file);
        if (h > 0)
            fprintf(stderr, "g2p: loaded Russian heteronym list (%d words)\n", h);
    }

    // The stress-analogy tier. On by default and worth 47 points: measured by
    // holding each of 10,000 dictionary words OUT of the dictionary and asking
    // the rules for it, the stressed-syllable index is right 93.9% of the time
    // with analogy and 47.1% without (exact IPA match 79.4% vs 40.9%). The
    // escape hatch exists because it is the one tier that can take a stress
    // from a word that merely LOOKS related.
    if (const char* v = std::getenv("CRISPASR_G2P_RU_ANALOGY"); v && *v && std::strcmp(v, "0") == 0)
        g_g2p_ru_ctx.stress_analogy = false;
}

bool phonemize_builtin_ru(const std::string& lang, const std::string& text, std::string& out, bool tts_punctuation) {
    if (!lang.empty() && lang.find("ru") == std::string::npos)
        return false;
    {
        std::lock_guard<std::mutex> g(g_g2p_ru_mu);
        ensure_ru_dict_loaded();
    }
    g_g2p_ru_ctx.emit_punctuation = tts_punctuation;
    out = g2p_ru::text_to_ipa(g_g2p_ru_ctx, text);
    return !out.empty();
}

// ── language dispatch for the built-in G2P (#435) ────────────────────

const char* builtin_g2p_language(const std::string& lang) {
    // Primary subtag only, lowercased: "en-us" -> "en", "es_419" -> "es".
    // An EXACT compare, not a substring test — see the header for why.
    std::string p;
    for (char c : lang) {
        if (c == '-' || c == '_')
            break;
        p += (char)std::tolower((unsigned char)c);
    }
    if (p == "en")
        return "en";
    if (p == "de")
        return "de";
    if (p == "fr")
        return "fr";
    if (p == "es")
        return "es";
    if (p == "ru")
        return "ru";
    return nullptr;
}

bool phonemize_builtin_tts(const std::string& lang, const std::string& text, std::string& out) {
    const char* fam = builtin_g2p_language(lang);
    if (!fam)
        return false;
    // Punctuation ON for every language here: a phoneme-conditioned TTS model
    // has `,.;:!?` in its symbol table and uses them for prosody. Contextual
    // function words and hyphen-joining stay OFF — those are misaki/Kokoro
    // conventions, and the consumers of this entry point are trained on
    // espeak-shaped phonemes.
    if (std::strcmp(fam, "en") == 0) {
        g2p_style st;
        st.emit_punctuation = true;
        return phonemize_builtin_en(lang, text, out, st);
    }
    if (std::strcmp(fam, "de") == 0)
        return phonemize_builtin_de(lang, text, out, /*tts_punctuation=*/true);
    if (std::strcmp(fam, "fr") == 0)
        return phonemize_builtin_fr(lang, text, out, /*tts_punctuation=*/true);
    if (std::strcmp(fam, "ru") == 0)
        return phonemize_builtin_ru(lang, text, out, /*tts_punctuation=*/true);
    return phonemize_builtin_es(lang, text, out, /*tts_punctuation=*/true);
}

// ── espeak-ng via dlopen ─────────────────────────────────────────────

static std::mutex g_espeak_mu;
static bool g_espeak_inited = false;
static bool g_espeak_init_failed = false;
static std::string g_espeak_voice;

bool phonemize_espeak_dlopen(const std::string& lang, const std::string& text, std::string& out) {
    std::lock_guard<std::mutex> g(g_espeak_mu);
    if (g_espeak_init_failed)
        return false;

    auto& dl = espeak_dl_get();
    if (!g_espeak_inited) {
        if (!dl.load())
            return false;
        const char* data_path = std::getenv("CRISPASR_ESPEAK_DATA_PATH");
        int sr = dl.Initialize(CRISPASR_ESPEAK_AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path,
                               CRISPASR_ESPEAK_INITIALIZE_PHONEME_IPA | CRISPASR_ESPEAK_INITIALIZE_DONT_EXIT);
        if (sr < 0) {
            g_espeak_init_failed = true;
            return false;
        }
        g_espeak_inited = true;
    }
    if (!dl.loaded)
        return false;
    if (g_espeak_voice != lang) {
        if (dl.SetVoiceByName(lang.c_str()) != 0)
            return false;
        g_espeak_voice = lang;
    }
    out.clear();
    const void* tp = text.c_str();
    while (tp) {
        const char* chunk = dl.TextToPhonemes(&tp, CRISPASR_ESPEAK_CHARS_UTF8, 0x02);
        if (chunk && *chunk) {
            if (!out.empty())
                out += ' ';
            out += chunk;
        }
    }
    strip_espeak_lang_markers(out);
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
        if (c == '\'')
            cmd += "'\\''";
        else
            cmd += c;
    }
    cmd += "'";
    cmd += redir;
    FILE* fp = PHON_POPEN(cmd.c_str(), "r");
    if (!fp)
        return false;
    out.clear();
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            len--;
        if (!out.empty() && len > 0)
            out += ' ';
        out.append(buf, len);
    }
    PHON_PCLOSE(fp);
    strip_espeak_lang_markers(out);
    return !out.empty();
#undef PHON_POPEN
#undef PHON_PCLOSE
}

} // namespace crispasr
