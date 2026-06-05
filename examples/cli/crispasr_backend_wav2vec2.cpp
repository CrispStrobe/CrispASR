// crispasr_backend_wav2vec2.cpp — adapter for Wav2Vec2ForCTC models.
//
// Hosts any wav2vec2-architecture CTC model (standard HF wav2vec2, omniASR,
// XLS-R, MMS, etc) via the existing src/wav2vec2-ggml.cpp runtime. Models
// are loaded from GGUF files produced by convert-wav2vec2-to-gguf.py or
// convert-omniasr-ctc-to-gguf.py.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "wav2vec2-ggml.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

class Wav2Vec2Backend : public CrispasrBackend {
public:
    Wav2Vec2Backend() = default;
    ~Wav2Vec2Backend() override { Wav2Vec2Backend::shutdown(); }

    const char* name() const override { return "wav2vec2"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_DIARIZE |
               CAP_UNBOUNDED_INPUT;
    }

    bool init(const whisper_params& p) override {
        model_ = std::make_unique<wav2vec2_model>();
        if (!wav2vec2_load(p.model.c_str(), *model_)) {
            fprintf(stderr, "crispasr[wav2vec2]: failed to load '%s'\n", p.model.c_str());
            model_.reset();
            return false;
        }
        n_threads_ = p.n_threads;
        if (!p.no_prints) {
            const auto& hp = model_->hparams;
            fprintf(stderr, "wav2vec2: hidden=%u layers=%u heads=%u ff=%u vocab=%u\n", hp.hidden_size,
                    hp.num_hidden_layers, hp.num_attention_heads, hp.intermediate_size, hp.vocab_size);
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!model_)
            return out;

        auto logits = wav2vec2_compute_logits(*model_, samples, n_samples, n_threads_);
        if (logits.empty()) {
            fprintf(stderr, "crispasr[wav2vec2]: compute_logits failed\n");
            return out;
        }

        const int V = (int)model_->hparams.vocab_size;
        const int T = (int)(logits.size() / V);

        auto token_probs = wav2vec2_greedy_decode_with_probs(*model_, logits.data(), T);

        // Reassemble text from token-level emissions (matches greedy_decode
        // output: "|" → " ", trim leading/trailing spaces).
        std::string text;
        text.reserve(token_probs.size());
        for (const auto& tp : token_probs)
            text += tp.text;
        auto lo = text.find_first_not_of(' ');
        auto hi = text.find_last_not_of(' ');
        text = (lo == std::string::npos) ? "" : text.substr(lo, hi - lo + 1);

        if (getenv("WAV2VEC2_BENCH"))
            fprintf(stderr, "wav2vec2: decoded %d frames → %zu chars (%zu emissions)\n", T, text.size(),
                    token_probs.size());

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        const float frame_dur_s = wav2vec2_frame_dur(*model_);
        seg.tokens.reserve(token_probs.size());
        for (const auto& tp : token_probs) {
            crispasr_token tok;
            tok.text = tp.text;
            tok.id = tp.id;
            tok.confidence = tp.prob;
            tok.t0 = t_offset_cs + (int64_t)(tp.frame_start * frame_dur_s * 100.0);
            tok.t1 = t_offset_cs + (int64_t)((tp.frame_end + 1) * frame_dur_s * 100.0);
            seg.tokens.push_back(std::move(tok));
        }
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override { model_.reset(); }

private:
    std::unique_ptr<wav2vec2_model> model_;
    int n_threads_ = 4;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_wav2vec2_backend() {
    return std::unique_ptr<CrispasrBackend>(new Wav2Vec2Backend());
}
