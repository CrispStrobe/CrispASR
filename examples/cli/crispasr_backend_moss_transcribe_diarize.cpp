// crispasr_backend_moss_transcribe_diarize.cpp — adapter for MOSS-Transcribe-Diarize-0.9B.
//
// Pipeline: mel → stock Whisper encoder (24L, 1024d, 80 mel) →
// 4x temporal merge → VQAdaptor → time-marker injection → Qwen3-0.6B decode
// → parse [timestamp][Sxx]text[timestamp] segments. Joint ASR + diarization.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/bpe.h"

#include "moss_transcribe_diarize.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class MossTranscribeDiarizeBackend : public CrispasrBackend {
public:
    MossTranscribeDiarizeBackend() = default;
    ~MossTranscribeDiarizeBackend() override { MossTranscribeDiarizeBackend::shutdown(); }

    const char* name() const override { return "moss-diarize"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_DIARIZE | CAP_AUTO_DOWNLOAD | CAP_PUNCTUATION_NATIVE | CAP_BEAM_SEARCH |
               CAP_TEMPERATURE;
    }

    bool init(const whisper_params& p) override {
        auto cp = moss_diarize_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = moss_diarize_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[moss-diarize]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        if (!p.hotwords.empty())
            moss_diarize_set_hotwords(ctx_, p.hotwords.c_str());
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        if (!ctx_)
            return {};
        moss_diarize_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        // #564: same resolution as the session path. --temperature > 0 samples;
        // --top-k / --top-p filter; --seed fixes the draw. Greedy by default.
        moss_diarize_set_sampling(ctx_, params.temperature, params.tts_top_k, params.tts_top_p, params.seed);
        // #292: forward --max-new-tokens only when the user set it explicitly;
        // 0 keeps the backend's 1024 default so the CLI's 512 default (too low
        // for diarize) never shrinks it.
        moss_diarize_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);
        if (!params.hotwords.empty())
            moss_diarize_set_hotwords(ctx_, params.hotwords.c_str());
        // Language hint not injected by default — the model auto-detects language.
        // Injecting the LID-resolved language disrupts the model's timestamp generation
        // (the model wasn't trained with language hints). Users who need it can pass
        // an explicit --language via the C API set_language() instead.

        // Use the segment-based API for native timestamps + diarization
        std::vector<moss_diarize_segment> raw_segs(256);
        int n_segs = moss_diarize_transcribe_segments(ctx_, samples, n_samples, raw_segs.data(), (int)raw_segs.size());

        std::vector<crispasr_segment> result;
        for (int i = 0; i < n_segs; i++) {
            crispasr_segment seg;
            seg.t0 = raw_segs[i].t0_cs + t_offset_cs;
            seg.t1 = raw_segs[i].t1_cs + t_offset_cs;
            seg.text = raw_segs[i].text;
            if (raw_segs[i].speaker_id > 0) {
                char spk[32];
                snprintf(spk, sizeof(spk), "(Speaker %d) ", raw_segs[i].speaker_id);
                seg.speaker = spk;
            }
            result.push_back(std::move(seg));
        }

        // Fallback: if parsing failed, return the raw text as a single segment
        if (result.empty()) {
            char* raw = moss_diarize_transcribe(ctx_, samples, n_samples);
            if (raw) {
                crispasr_segment seg;
                seg.text = raw;
                seg.t0 = t_offset_cs;
                int64_t dur_cs = (int64_t)((double)n_samples / 16000.0 * 100.0);
                seg.t1 = t_offset_cs + dur_cs;
                result.push_back(std::move(seg));
                free(raw);
            }
        }
        return result;
    }

    void shutdown() override {
        if (ctx_) {
            moss_diarize_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    moss_diarize_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_moss_transcribe_diarize_backend() {
    return std::unique_ptr<CrispasrBackend>(new MossTranscribeDiarizeBackend());
}
