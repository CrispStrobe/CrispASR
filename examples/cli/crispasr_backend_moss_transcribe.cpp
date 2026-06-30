// crispasr_backend_moss_transcribe.cpp — adapter for MOSS-Transcribe-preview-2B.
//
// Pipeline: mel → Qwen3-Omni audio encoder (conv stem + 32L windowed attn +
// proj head) → GatedMLP adapter → masked_scatter into the ChatML audio prompt
// (chat_template_default.py framing) → Qwen3-1.7B decode. ASR-only.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/bpe.h"

#include "moss_transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string decode_token(const std::string& s) {
    return core_bpe::token_bytes_to_utf8(s);
}

class MossTranscribeBackend : public CrispasrBackend {
public:
    MossTranscribeBackend() = default;
    ~MossTranscribeBackend() override { MossTranscribeBackend::shutdown(); }

    const char* name() const override { return "moss-transcribe"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_PUNCTUATION_NATIVE | CAP_BEAM_SEARCH; }

    bool init(const whisper_params& p) override {
        auto cp = moss_transcribe_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = moss_transcribe_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[moss-transcribe]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        if (!ctx_)
            return {};
        moss_transcribe_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        char* result = moss_transcribe_transcribe(ctx_, samples, n_samples);
        if (!result)
            return {};
        std::string text(result);
        free(result);

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        int64_t dur_cs = (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.t1 = t_offset_cs + dur_cs;
        return {seg};
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }
        std::string accumulated;
        bool first_tok = true;
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = moss_transcribe_token_text(ctx_, tok_id);
            if (!raw)
                return;
            std::string piece = decode_token(std::string(raw));
            if (first_tok) {
                size_t sp = 0;
                while (sp < piece.size() && (piece[sp] == ' ' || piece[sp] == '\n'))
                    sp++;
                piece = piece.substr(sp);
                if (!piece.empty())
                    first_tok = false;
            }
            accumulated += piece;
            if (!accumulated.empty())
                on_text(accumulated.c_str(), false);
        };
        auto cb_fn = [](int tok_id, float prob, void* ud) { (*static_cast<decltype(cb)*>(ud))(tok_id, prob, nullptr); };
        moss_transcribe_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(accumulated.c_str(), true);
    }

    void shutdown() override {
        if (ctx_) {
            moss_transcribe_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    moss_transcribe_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_moss_transcribe_backend() {
    return std::unique_ptr<CrispasrBackend>(new MossTranscribeBackend());
}
