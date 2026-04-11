// crispasr_backend_voxtral.cpp — adapter for Mistral Voxtral-Mini-3B-2507.
//
// The pipeline logic (mel -> encoder -> prompt splice -> KV decode) is in
// crispasr_llm_pipeline.h. This file only provides the voxtral-specific
// traits: function name mapping, audio_pad/EOS token IDs, and the Tekken
// prompt template.

#include "crispasr_backend.h"
#include "crispasr_llm_pipeline.h"
#include "whisper_params.h"

#include "voxtral.h"

#include <cstdio>
#include <string>

namespace {

struct VoxtralOps {
    using CtxT = voxtral_context;

    static const char * name() { return "voxtral"; }

    // Tekken audio placeholder token id (special token <audio_pad> = 24).
    static constexpr int audio_pad_id = 24;
    // Mistral Tekken EOS.
    static constexpr int eos_id       = 2;

    static CtxT * init(const char * path, int n_threads, int verbosity) {
        auto cp = voxtral_context_default_params();
        cp.n_threads = n_threads;
        cp.verbosity = verbosity;
        return voxtral_init_from_file(path, cp);
    }
    static void free_ctx(CtxT * ctx) { voxtral_free(ctx); }

    static float * compute_mel(CtxT * ctx, const float * s, int n,
                               int * n_mels, int * T_mel)
    { return voxtral_compute_mel(ctx, s, n, n_mels, T_mel); }

    static float * run_encoder(CtxT * ctx, const float * mel, int n_mels, int T_mel,
                               int * N_enc, int * enc_dim)
    { return voxtral_run_encoder(ctx, mel, n_mels, T_mel, N_enc, enc_dim); }

    static int32_t * tokenize(CtxT * ctx, const char * text, int * n)
    { return voxtral_tokenize(ctx, text, n); }

    static float * embed_tokens(CtxT * ctx, const int32_t * ids, int n)
    { return voxtral_embed_tokens(ctx, ids, n); }

    static bool kv_init(CtxT * ctx, int max_ctx) { return voxtral_kv_init(ctx, max_ctx); }
    static void kv_reset(CtxT * ctx)             { voxtral_kv_reset(ctx); }

    static float * run_llm_kv(CtxT * ctx, const float * embeds, int n_tokens,
                              int n_past, int * out_n_tokens, int * out_vocab)
    { return voxtral_run_llm_kv(ctx, embeds, n_tokens, n_past, out_n_tokens, out_vocab); }

    static const uint8_t * token_text(CtxT * ctx, int id, int * out_len)
    { return voxtral_token_text(ctx, id, out_len); }

    // Voxtral Tekken template:
    //   <s>[INST][BEGIN_AUDIO] <audio_pad>×N [/INST]lang:LANG[TRANSCRIBE]
    static std::string build_prefix(const std::string & /*lang*/) {
        return "<s>[INST][BEGIN_AUDIO]";
    }
    static std::string build_suffix(const std::string & lang) {
        return "[/INST]lang:" + lang + "[TRANSCRIBE]";
    }
};

class VoxtralBackend : public CrispasrBackend {
public:
    VoxtralBackend() = default;
    ~VoxtralBackend() override { shutdown(); }

    const char * name() const override { return "voxtral"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE
             | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN;
    }

    bool init(const whisper_params & p) override {
        ctx_ = VoxtralOps::init(p.model.c_str(), p.n_threads, p.no_prints ? 0 : 1);
        if (!ctx_) {
            fprintf(stderr, "crispasr[voxtral]: failed to load model '%s'\n",
                    p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & params) override
    {
        return crispasr_run_voxtral_style_pipeline<VoxtralOps>(
            ctx_, samples, n_samples, t_offset_cs, params);
    }

    void shutdown() override {
        if (ctx_) {
            VoxtralOps::free_ctx(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    voxtral_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_backend() {
    return std::unique_ptr<CrispasrBackend>(new VoxtralBackend());
}
