// crispasr_backend_sensevoice.cpp — FunAudioLLM/SenseVoiceSmall adapter.
//
// Encoder-only multi-task ASR (transcript + LID + emotion + audio-event
// tags in one CTC forward pass). Output includes a 4-token rich-annotation
// prefix from the model's special-token vocab before the transcript.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "sensevoice.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <vector>

class SenseVoiceBackend : public CrispasrBackend {
public:
    SenseVoiceBackend() = default;

    const char* name() const override { return "sensevoice"; }

    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_PUNCTUATION_TOGGLE | CAP_DIARIZE | CAP_TIMESTAMPS_CTC |
               CAP_LANGUAGE_DETECT;
    }

    bool init(const whisper_params& p) override {
        sensevoice_context_params cp = sensevoice_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = sensevoice_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[sensevoice]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Map crispasr's language code to SenseVoice's known set. SenseVoice
        // supports zh / en / yue / ja / ko / nospeech; everything else falls
        // back to "auto" which lets the model pick via its language query
        // embedding.
        const char* lang = nullptr;
        const std::string& lspec = params.language;
        if (lspec == "zh" || lspec == "en" || lspec == "yue" || lspec == "ja" || lspec == "ko" || lspec == "nospeech")
            lang = lspec.c_str();

        sensevoice_result* r = sensevoice_transcribe_structured(ctx_, samples, n_samples, lang,
                                                                /*use_itn*/ params.punctuation);
        if (!r) {
            fprintf(stderr, "crispasr[sensevoice]: transcribe failed\n");
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = r->text ? r->text : "";
        if (r->language)
            seg.lang_id = r->language;
        if (r->emotion)
            seg.emotion = r->emotion;
        if (r->audio_event)
            seg.audio_event = r->audio_event;
        if (r->itn)
            seg.itn_flag = r->itn;
        sensevoice_result_free(r);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            sensevoice_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~SenseVoiceBackend() override { SenseVoiceBackend::shutdown(); }

private:
    sensevoice_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_sensevoice_backend() {
    return std::make_unique<SenseVoiceBackend>();
}
