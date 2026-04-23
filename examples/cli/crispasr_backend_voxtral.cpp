// crispasr_backend_voxtral.cpp — adapter for Mistral Voxtral-Mini-3B-2507.
//
// The pipeline logic (mel -> encoder -> prompt splice -> KV decode) is in
// crispasr_llm_pipeline.h. This file only provides the voxtral-specific
// traits: function name mapping, audio_pad/EOS token IDs, and the Tekken
// prompt template.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_llm_pipeline.h"
#include "whisper_params.h"

#include "voxtral.h"

#include <cstdio>
#include <string>

namespace {

struct VoxtralOps {
    using CtxT = voxtral_context;

    static const char* name() { return "voxtral"; }

    // Tekken audio placeholder token id (special token <audio_pad> = 24).
    static constexpr int audio_pad_id = 24;
    // Mistral Tekken EOS.
    static constexpr int eos_id = 2;

    static CtxT* init(const char* path, int n_threads, int verbosity, bool use_gpu) {
        auto cp = voxtral_context_default_params();
        cp.n_threads = n_threads;
        cp.verbosity = verbosity;
        cp.use_gpu = use_gpu;
        return voxtral_init_from_file(path, cp);
    }
    static void free_ctx(CtxT* ctx) { voxtral_free(ctx); }

    static float* compute_mel(CtxT* ctx, const float* s, int n, int* n_mels, int* T_mel) {
        return voxtral_compute_mel(ctx, s, n, n_mels, T_mel);
    }

    static float* run_encoder(CtxT* ctx, const float* mel, int n_mels, int T_mel, int* N_enc, int* enc_dim) {
        return voxtral_run_encoder(ctx, mel, n_mels, T_mel, N_enc, enc_dim);
    }

    static int32_t* tokenize(CtxT* ctx, const char* text, int* n) { return voxtral_tokenize(ctx, text, n); }

    static float* embed_tokens(CtxT* ctx, const int32_t* ids, int n) { return voxtral_embed_tokens(ctx, ids, n); }

    static bool kv_init(CtxT* ctx, int max_ctx) { return voxtral_kv_init(ctx, max_ctx); }
    static void kv_reset(CtxT* ctx) { voxtral_kv_reset(ctx); }

    static float* run_llm_kv(CtxT* ctx, const float* embeds, int n_tokens, int n_past, int* out_n_tokens,
                             int* out_vocab) {
        return voxtral_run_llm_kv(ctx, embeds, n_tokens, n_past, out_n_tokens, out_vocab);
    }

    static const uint8_t* token_text(CtxT* ctx, int id, int* out_len) { return voxtral_token_text(ctx, id, out_len); }

    // Voxtral Tekken template:
    //   <s>[INST][BEGIN_AUDIO] <audio_pad>×N [/INST]lang:LANG[TRANSCRIBE]
    // For --translate we swap [TRANSCRIBE] for an instruction prompt
    // because Voxtral doesn't have a dedicated translate control token.
    static std::string build_prefix(const whisper_params& /*p*/) { return "<s>[INST][BEGIN_AUDIO]"; }
    static std::string build_suffix(const whisper_params& p) {
        const std::string lang = p.language.empty() ? std::string("en") : p.language;
        if (!p.ask.empty()) {
            // Audio understanding / Q&A mode: instead of transcribing,
            // ask a question about the audio content. The model responds
            // with free-form text.
            return "[/INST]" + p.ask;
        }
        if (p.translate) {
            // Voxtral handles translation as an instruction. We keep
            // the lang: marker so the model knows the source language
            // and append a plain English directive in the user turn.
            // Map ISO codes to full English names so the model gets
            // an unambiguous target ("de" alone reads as Spanish "of").
            auto to_eng = [](const std::string& c) -> std::string {
                if (c == "en")
                    return "English";
                if (c == "de")
                    return "German";
                if (c == "fr")
                    return "French";
                if (c == "es")
                    return "Spanish";
                if (c == "it")
                    return "Italian";
                if (c == "pt")
                    return "Portuguese";
                if (c == "ru")
                    return "Russian";
                if (c == "ja")
                    return "Japanese";
                if (c == "zh")
                    return "Chinese";
                if (c == "nl")
                    return "Dutch";
                return c;
            };
            const std::string tgt = p.target_lang.empty() ? std::string("English") : to_eng(p.target_lang);
            return "[/INST]lang:" + lang + " Translate the audio to " + tgt + ".[TRANSCRIBE]";
        }
        return "[/INST]lang:" + lang + "[TRANSCRIBE]";
    }
};

class VoxtralBackend : public CrispasrBackend {
public:
    VoxtralBackend() = default;
    ~VoxtralBackend() override { VoxtralBackend::shutdown(); }

    const char* name() const override { return "voxtral"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN |
               CAP_TOKEN_CONFIDENCE | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_BEAM_SEARCH | CAP_DIARIZE |
               CAP_PARALLEL_PROCESSORS;
    }

    bool init(const whisper_params& p) override {
        ctx_ =
            VoxtralOps::init(p.model.c_str(), p.n_threads, p.no_prints ? 0 : 1, crispasr_backend_should_use_gpu(p));
        if (!ctx_) {
            fprintf(stderr, "crispasr[voxtral]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        return crispasr_run_voxtral_style_pipeline<VoxtralOps>(ctx_, samples, n_samples, t_offset_cs, params);
    }

    void shutdown() override {
        if (ctx_) {
            VoxtralOps::free_ctx(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    voxtral_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_backend() {
    return std::unique_ptr<CrispasrBackend>(new VoxtralBackend());
}
