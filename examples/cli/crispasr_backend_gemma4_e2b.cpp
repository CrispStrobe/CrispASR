// crispasr_backend_gemma4_e2b.cpp — Gemma-4-E2B ASR backend adapter.

#include "crispasr_backend.h"
#include "gemma4_e2b.h"
#include "whisper_params.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class Gemma4E2BBackend : public CrispasrBackend {
public:
    Gemma4E2BBackend() = default;

    const char* name() const override { return "gemma4-e2b"; }

    uint32_t capabilities() const override {
        // Verified against src/gemma4_e2b.{h,cpp} as of v0.5.7:
        //   CAP_AUTO_DOWNLOAD        registry entry in src/crispasr_model_registry.cpp
        //   CAP_DIARIZE              framework post-step on segment list
        //   CAP_TIMESTAMPS_CTC       framework post-step via -am aligner.gguf
        //   CAP_FLASH_ATTN           uses ggml_flash_attn_ext in attention graph
        //   CAP_PARALLEL_PROCESSORS  shared session-level dispatcher
        //   CAP_TEMPERATURE          params.temperature → ctx->temperature → decode cfg
        //
        // CAP_LANGUAGE_DETECT intentionally NOT declared: gemma4_e2b has no
        // native LID code path. The cap is a "this backend handles LID
        // itself" signal — declaring it would disable the framework's
        // pre-step LID gate (crispasr_run.cpp:`!has_native_lid`), so users
        // wanting LID would get nothing. With the cap absent, `-dl`
        // routes through the whisper-tiny pre-step.
        //
        // Now declared:
        //   CAP_TRANSLATE        audio AST + text translation path
        //   CAP_SRC_TGT_LANGUAGE separate source/target language hints
        //
        // Not yet declared (would need code changes elsewhere):
        //   CAP_TOKEN_CONFIDENCE — gemma4_e2b_transcribe_with_probs exists in
        //     the C-ABI but transcribe() below only calls the plain text variant
        //   CAP_PUNCTUATION_TOGGLE — no toggle exposed
        return CAP_AUTO_DOWNLOAD | CAP_DIARIZE | CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN | CAP_PARALLEL_PROCESSORS |
               CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE;
    }

    bool init(const whisper_params& params) override {
        gemma4_e2b_context_params cp = gemma4_e2b_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || getenv("GEMMA4_E2B_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = params.use_gpu;
        // Honor -tp / --temperature so CAP_TEMPERATURE is real, not just a
        // declaration — gemma4_e2b.cpp already plumbs ctx->temperature to
        // the decode config in run_llm_decode (line 2115).
        cp.temperature = params.temperature;
        ctx_ = gemma4_e2b_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // PLAN #125 P4: gemma4-e2b is trained on ~30 s windows. Beyond
        // that the encoder embeddings collapse and the LM hits <eos>
        // immediately after the prompt, then continues autoregressively
        // into unrelated commentary (issue #125 report 02).
        //
        // Default: refuse the slice with a clear error so the user routes
        // via --vad. Opt-in: set CRISPASR_GEMMA4_AUTO_CHUNK=1 to chunk
        // internally at the training-window boundary. Auto-chunking is
        // off by default because we haven't validated quality at chunk
        // boundaries on long audio yet.
        constexpr int kMaxSamples = 30 * 16000;
        if (n_samples > kMaxSamples) {
            const bool auto_chunk = std::getenv("CRISPASR_GEMMA4_AUTO_CHUNK") != nullptr;
            if (!auto_chunk) {
                fprintf(stderr,
                        "crispasr[gemma4-e2b]: input is %.1f s (> %.0f s training window). "
                        "Use --vad to segment, chunk externally, or set "
                        "CRISPASR_GEMMA4_AUTO_CHUNK=1 to chunk internally "
                        "(quality at chunk boundaries not yet validated). Aborting this slice.\n",
                        (double)n_samples / 16000.0, (double)kMaxSamples / 16000.0);
                return out;
            }
            for (int start = 0; start < n_samples; start += kMaxSamples) {
                const int this_n = std::min(kMaxSamples, n_samples - start);
                const int64_t chunk_offset_cs = t_offset_cs + (int64_t)((double)start / 16000.0 * 100.0);
                auto chunk_segs = transcribe_one(samples + start, this_n, chunk_offset_cs, params);
                for (auto& s : chunk_segs)
                    out.push_back(std::move(s));
            }
            return out;
        }
        return transcribe_one(samples, n_samples, t_offset_cs, params);
    }

    std::vector<crispasr_segment> transcribe_one(const float* samples, int n_samples, int64_t t_offset_cs,
                                                 const whisper_params& params) {
        std::vector<crispasr_segment> out;
        gemma4_e2b_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        const std::string src =
            !params.source_lang.empty() ? params.source_lang : (!params.language.empty() ? params.language : "");
        const std::string tgt = !params.target_lang.empty() ? params.target_lang : std::string("en");
        char* text =
            gemma4_e2b_transcribe_ex(ctx_, samples, n_samples, params.translate ? 1 : 0, src.c_str(), tgt.c_str());
        if (!text || !text[0]) {
            free(text);
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        free(text);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    std::string translate_text(const std::string& text, const std::string& src_lang, const std::string& tgt_lang,
                               const whisper_params& /*params*/) override {
        if (!ctx_ || text.empty())
            return {};
        char* out = gemma4_e2b_translate_text(ctx_, text.c_str(), src_lang.c_str(), tgt_lang.c_str());
        if (!out || !out[0]) {
            free(out);
            return {};
        }
        std::string result(out);
        free(out);
        return result;
    }

    void shutdown() override {
        if (ctx_) {
            gemma4_e2b_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~Gemma4E2BBackend() override { Gemma4E2BBackend::shutdown(); }

private:
    gemma4_e2b_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_gemma4_e2b_backend() {
    return std::make_unique<Gemma4E2BBackend>();
}
