// crispasr_backend_granite.cpp — adapter for ibm-granite/granite-4.0-1b-speech.
//
// The library's granite_speech_transcribe() is currently a stub, so this
// backend implements the full pipeline manually using the stage-level API:
// mel -> encoder -> Q-Former projector -> LLM prompt splice -> KV prefill
// -> greedy decode -> GPT-2 byte detokenize.
//
// This code is a direct port of examples/granite-main/main.cpp's pipeline,
// wrapped in the CrispasrBackend interface. Once granite_speech_transcribe
// gains a real implementation, or once the shared LLM decode loop lands in
// src/core/, this file should shrink dramatically.
//
// Granite does not expose native timestamps. Word-level timestamps via a
// CTC aligner second pass will be wired up once crispasr_aligner lands.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"

#include "granite_speech.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Granite chat-template tokens (GPT-2 BPE IDs). Same values as used in
// examples/granite-main/main.cpp — captured from the HF chat template.
constexpr int kAudioTok = 100352;
constexpr int kEos      = 100257;
// "USER: "
constexpr int32_t kPrefix[]   = { 6584, 25, 220 };
constexpr int     kNumPrefix  = 3;
// "can you transcribe the speech into a written format?\n ASSISTANT:"
constexpr int32_t kSuffix[]   = {
    4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198,
    36660, 3931, 2891, 25
};
constexpr int     kNumSuffix  = 16;

class GraniteBackend : public CrispasrBackend {
public:
    GraniteBackend() = default;
    ~GraniteBackend() override { shutdown(); }

    const char * name() const override { return "granite"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params & p) override {
        granite_speech_context_params cp = granite_speech_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = granite_speech_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[granite]: failed to load model '%s'\n",
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
        std::vector<crispasr_segment> out;
        if (!ctx_) return out;

        // ---- Mel spectrogram ----
        int n_mels = 0, T_mel = 0;
        float * mel = granite_speech_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel) {
            fprintf(stderr, "crispasr[granite]: mel failed\n");
            return out;
        }

        // ---- Encoder ----
        int N_enc = 0, enc_dim = 0;
        float * enc = granite_speech_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &enc_dim);
        free(mel);
        if (!enc) {
            fprintf(stderr, "crispasr[granite]: encoder failed\n");
            return out;
        }

        // ---- Q-Former projector ----
        int N_proj = 0, proj_dim = 0;
        float * proj = granite_speech_run_projector(ctx_, enc, N_enc, enc_dim, &N_proj, &proj_dim);
        free(enc);
        if (!proj) {
            fprintf(stderr, "crispasr[granite]: projector failed\n");
            return out;
        }

        // ---- Build prompt: [prefix] + [audio placeholders] + [suffix] ----
        const int total_prompt = kNumPrefix + N_proj + kNumSuffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int i = 0; i < kNumPrefix; i++) prompt_ids.push_back(kPrefix[i]);
        for (int i = 0; i < N_proj; i++)    prompt_ids.push_back(kAudioTok);
        for (int i = 0; i < kNumSuffix; i++) prompt_ids.push_back(kSuffix[i]);

        float * all_embeds = granite_speech_embed_tokens(ctx_, prompt_ids.data(), total_prompt);
        if (!all_embeds) {
            free(proj);
            fprintf(stderr, "crispasr[granite]: embed failed\n");
            return out;
        }

        // Splice projector output into the audio positions.
        for (int i = 0; i < N_proj; i++) {
            std::memcpy(all_embeds + (size_t)(kNumPrefix + i) * proj_dim,
                        proj + (size_t)i * proj_dim,
                        proj_dim * sizeof(float));
        }
        free(proj);

        // ---- KV cache + prefill ----
        if (!granite_speech_kv_init(ctx_, 4096)) {
            free(all_embeds);
            fprintf(stderr, "crispasr[granite]: kv init failed\n");
            return out;
        }
        granite_speech_kv_reset(ctx_);

        int vocab = 0;
        float * logits = granite_speech_run_llm_kv(ctx_, all_embeds, total_prompt, 0,
                                                    nullptr, &vocab);
        free(all_embeds);
        if (!logits) {
            fprintf(stderr, "crispasr[granite]: prefill failed\n");
            return out;
        }

        // Greedy argmax over the final token.
        int next = 0;
        {
            float mx = -1e30f;
            for (int k = 0; k < vocab; k++) {
                if (logits[k] > mx) { mx = logits[k]; next = k; }
            }
        }
        free(logits);

        // ---- Decode loop via src/core/greedy_decode.h ----
        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 200;
        dec_cfg.eos_id         = kEos;
        dec_cfg.vocab_size     = vocab;
        auto gen_ids = core_greedy_decode::run(
            ctx_,
            /*first_token=*/next,
            /*initial_n_past=*/total_prompt,
            granite_speech_embed_tokens,
            granite_speech_run_llm_kv,
            dec_cfg);

        // Strip EOS from generated IDs before detokenizing.
        std::vector<int32_t> text_ids;
        text_ids.reserve(gen_ids.size());
        for (int32_t id : gen_ids) if (id != kEos) text_ids.push_back(id);

        char * text = granite_speech_decode_tokens(ctx_, text_ids.data(), (int)text_ids.size());

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text ? text : "";
        if (text) free(text);

        // Trim leading whitespace emitted by the chat template.
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n')) {
            seg.text.erase(seg.text.begin());
        }

        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            granite_speech_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    granite_speech_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend() {
    return std::unique_ptr<CrispasrBackend>(new GraniteBackend());
}
