// crispasr_backend_hft_transformer.cpp — CLI adapter for hFT-Transformer.
//
// The real surface for this model is `--piano` (note events in, note events
// out; see crispasr_piano_cli.h). This adapter exists so `--backend
// hft-transformer` and architecture auto-detection reach it on the legacy
// transcribe() path too, rendering each note as a timestamped segment — the
// same lossy-but-useful shape crispasr_backend_onsets_and_frames.cpp emits.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "hft_transformer.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class HftTransformerBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto op = hft_transformer_default_params();
        op.n_threads = p.n_threads;
        op.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        op.use_gpu = p.use_gpu;
        ctx_ = hft_transformer_init_from_file(p.model.c_str(), op);
        return ctx_ != nullptr;
    }

    void shutdown() override {
        if (ctx_) {
            hft_transformer_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "hft-transformer"; }
    uint32_t capabilities() const override { return CAP_PIANO | CAP_TIMESTAMPS_NATIVE | CAP_AUTO_DOWNLOAD; }
    int input_sample_rate() const override { return 16000; }

    std::vector<crispasr_segment> transcribe(const float* pcm, int n_samples, int64_t /*t0_ms*/,
                                             const whisper_params& /*p*/) override {
        if (!ctx_ || !pcm || n_samples <= 0)
            return {};

        hft_transformer_result result = {};
        if (hft_transformer_transcribe(ctx_, pcm, n_samples, &result) != 0)
            return {};

        static const char* note_names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        std::vector<crispasr_segment> segs;
        segs.reserve((size_t)result.n_notes);
        for (int i = 0; i < result.n_notes; i++) {
            const auto& ev = result.note_events[i];
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%d v=%d", note_names[ev.midi_note % 12], (ev.midi_note / 12) - 1,
                     ev.velocity);
            crispasr_segment seg;
            seg.t0 = (int64_t)(ev.onset_time * 1000);
            seg.t1 = (int64_t)(ev.offset_time * 1000);
            seg.text = buf;
            segs.push_back(seg);
        }
        hft_transformer_result_free(&result);
        return segs;
    }

private:
    hft_transformer_ctx* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_hft_transformer_backend() {
    return std::make_unique<HftTransformerBackend>();
}
