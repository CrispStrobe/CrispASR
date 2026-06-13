#ifndef CRISPASR_PUNC_MODEL_H
#define CRISPASR_PUNC_MODEL_H

#include <string>

// Pure `--punc-model` resolution, shared across every front-end: the CLI
// one-shot path, the HTTP server, and the C-ABI session layer. Keeping the
// alias → (cache filename, download URL) table in one dependency-free place
// (only <string>) stops the front-ends from drifting on which model a flag
// value selects, and lets it be unit tested without loading any model.
//
// Lives in src/ so src/crispasr_c_api.cpp can include it; the CLI-layer
// header examples/cli/crispasr_punc_loader.h re-exports it for the CLI/server.

enum class crispasr_punc_kind {
    none,        // disabled / no model
    fireredpunc, // FireRedPunc-family GGUF (auto|firered|fullstop|punctuate-all|path)
    pcs,         // XLM-R punctuation + capitalization + segmentation GGUF
};

struct crispasr_punc_spec {
    crispasr_punc_kind kind = crispasr_punc_kind::none;
    std::string cache_filename; // download target filename ("" for a direct path)
    std::string url;            // download URL              ("" for a direct path)
    std::string direct_path;    // explicit on-disk path     ("" for an alias)
};

// Map a `--punc-model` value to a model spec. Pure: no I/O, no model load.
inline crispasr_punc_spec crispasr_resolve_punc_model(const std::string& punc_model) {
    crispasr_punc_spec s;
    const std::string& m = punc_model;

    if (m.empty() || m == "none" || m == "off")
        return s; // disabled

    if (m == "auto" || m == "firered") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "fireredpunc-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k.gguf";
        return s;
    }
    if (m == "fullstop") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "fullstop-punc-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q4_k.gguf";
        return s;
    }
    if (m == "punctuate-all") {
        s.kind = crispasr_punc_kind::fireredpunc;
        s.cache_filename = "punctuate-all-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/punctuate-all-GGUF/resolve/main/punctuate-all-q4_k.gguf";
        return s;
    }
    if (m == "pcs") {
        s.kind = crispasr_punc_kind::pcs;
        s.cache_filename = "pcs-xlmr-base-q4_k.gguf";
        s.url = "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k.gguf";
        return s;
    }
    if (m.find("pcs") != std::string::npos) {
        // A path or keyword that mentions "pcs": only a concrete .gguf is
        // loadable as a PCS model. Anything else (e.g. a bare "pcs-de") matched
        // neither front-end's load path historically, so it stays disabled.
        if (m.find(".gguf") != std::string::npos) {
            s.kind = crispasr_punc_kind::pcs;
            s.direct_path = m;
        }
        return s;
    }

    // Otherwise: a direct path to a FireRedPunc-family GGUF.
    s.kind = crispasr_punc_kind::fireredpunc;
    s.direct_path = m;
    return s;
}

#endif // CRISPASR_PUNC_MODEL_H
