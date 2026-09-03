// crispasr_backend_mt3.cpp — CLI adapter for Magenta MT3.
//
// CAP_PIANO is the task marker (audio → note events), shared with
// piano-transcription and basic-pitch; --piano is the real surface and
// dispatches on the GGUF's architecture (see crispasr_piano_cli.cpp). MT3 is
// neither piano-specific nor monophonic — it is multi-instrument, and every
// note carries a General-MIDI program — but the task and the note-event shape
// are the same, which is what the capability names. The program lands in the
// segment text here because `crispasr_segment` has no field for it.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "mt3.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class Mt3Backend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto mp = mt3_default_params();
        mp.n_threads = p.n_threads;
        mp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        mp.use_gpu = p.use_gpu;
        ctx_ = mt3_init_from_file(p.model.c_str(), mp);
        return ctx_ != nullptr;
    }

    void shutdown() override {
        if (ctx_) {
            mt3_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "mt3"; }
    uint32_t capabilities() const override { return CAP_PIANO | CAP_TIMESTAMPS_NATIVE | CAP_AUTO_DOWNLOAD; }
    int input_sample_rate() const override { return 16000; }

    std::vector<crispasr_segment> transcribe(const float* pcm, int n_samples, int64_t /*t0_ms*/,
                                             const whisper_params& /*p*/) override {
        if (!ctx_ || !pcm || n_samples <= 0)
            return {};

        mt3_result result = {};
        if (mt3_transcribe(ctx_, pcm, n_samples, &result) != 0)
            return {};

        static const char* note_names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        std::vector<crispasr_segment> segs;
        segs.reserve(result.n_notes);
        for (int i = 0; i < result.n_notes; i++) {
            const auto& ev = result.notes[i];
            char buf[96];
            if (ev.is_drum)
                snprintf(buf, sizeof(buf), "%s%d drum v=%d", note_names[ev.pitch % 12], (ev.pitch / 12) - 1,
                         ev.velocity);
            else
                snprintf(buf, sizeof(buf), "%s%d prog=%d v=%d", note_names[ev.pitch % 12], (ev.pitch / 12) - 1,
                         ev.program, ev.velocity);
            crispasr_segment seg;
            seg.t0 = (int64_t)(ev.start_time * 1000);
            seg.t1 = (int64_t)(ev.end_time * 1000);
            seg.text = buf;
            segs.push_back(seg);
        }
        mt3_result_free(&result);
        return segs;
    }

private:
    mt3_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_mt3_backend() {
    return std::make_unique<Mt3Backend>();
}
