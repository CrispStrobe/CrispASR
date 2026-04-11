// crispasr_backend_voxtral4b.cpp — adapter for Voxtral-Mini-4B-Realtime-2602.
//
// Note: despite the shared "voxtral" naming, the 4B Realtime variant is a
// STREAMING architecture that does NOT fit the voxtral 3B prompt template.
// Differences from voxtral 3B:
//
//   * Prompt is just BOS + STREAMING_PAD×(32+delay_tokens), not Tekken text
//   * Audio encoder output is ADDED element-wise to token embeddings at
//     each position (not spliced as a replacement)
//   * The decode loop continues to add the next adapter frame to every
//     generated token's embedding — this is the streaming mechanism
//   * Output tokens with id < 1000 are control tokens (STREAMING_PAD,
//     STREAMING_WORD, etc.) and must be filtered from the transcript
//
// Because of these differences, this backend cannot use the shared
// crispasr_llm_pipeline.h template. The pipeline is implemented inline here,
// following examples/voxtral4b-main/main.cpp.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"

#include "voxtral4b.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class Voxtral4bBackend : public CrispasrBackend {
public:
    Voxtral4bBackend() = default;
    ~Voxtral4bBackend() override { shutdown(); }

    const char * name() const override { return "voxtral4b"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params & p) override {
        auto cp = voxtral4b_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = voxtral4b_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[voxtral4b]: failed to load model '%s'\n",
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

        // ---- Pad audio for the streaming model ----
        // Each "token" = hop_length * conv_stride * downsample_factor =
        // 160 * 2 * 4 = 1280 samples. The 4B Realtime encoder expects left
        // padding of 32 tokens plus right-alignment + 17 tokens of right
        // padding; without this the encoder graph reshape fails.
        constexpr int SAMPLES_PER_TOKEN    = 1280;
        constexpr int N_LEFT_PAD_TOKENS    = 32;
        constexpr int N_RIGHT_PAD_TOKENS   = 17;
        const int left_pad   = N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN;
        const int right_align =
            (SAMPLES_PER_TOKEN - (n_samples % SAMPLES_PER_TOKEN)) % SAMPLES_PER_TOKEN;
        const int right_pad  = right_align + N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN;

        std::vector<float> padded(left_pad + (size_t)n_samples + right_pad, 0.0f);
        std::memcpy(padded.data() + left_pad, samples, (size_t)n_samples * sizeof(float));

        // ---- Mel ----
        int n_mels = 0, T_mel = 0;
        float * mel = voxtral4b_compute_mel(ctx_, padded.data(), (int)padded.size(), &n_mels, &T_mel);
        if (!mel) { fprintf(stderr, "crispasr[voxtral4b]: mel failed\n"); return out; }

        // ---- Encoder ----
        int N_enc = 0, pdim = 0;
        float * audio_embeds = voxtral4b_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &pdim);
        free(mel);
        if (!audio_embeds) {
            fprintf(stderr, "crispasr[voxtral4b]: encoder failed\n");
            return out;
        }

        // ---- Prompt: BOS + STREAMING_PAD × (32 + delay_tokens) ----
        // The 4B Realtime encoder produces adapter frames that are ADDED to
        // the prompt token embeddings (and later to the tail embedding each
        // decode step — that's the streaming mechanism).
        const int delay_tokens = 6;                     // 480 ms default
        const int T_prompt = 1 + 32 + delay_tokens;     // 39

        std::vector<int32_t> prompt_ids(T_prompt);
        prompt_ids[0] = 1;                              // BOS
        for (int i = 1; i < T_prompt; i++) prompt_ids[i] = 32; // STREAMING_PAD

        float * prompt_embeds = voxtral4b_embed_tokens(ctx_, prompt_ids.data(), T_prompt);
        if (!prompt_embeds) {
            free(audio_embeds);
            fprintf(stderr, "crispasr[voxtral4b]: embed failed\n");
            return out;
        }

        const int n_fill = std::min(N_enc, T_prompt);
        for (int i = 0; i < n_fill; i++) {
            for (int j = 0; j < pdim; j++) {
                prompt_embeds[(size_t)i * pdim + j] += audio_embeds[(size_t)i * pdim + j];
            }
        }

        // ---- KV cache + prefill ----
        if (!voxtral4b_kv_init(ctx_, 4096)) {
            free(prompt_embeds); free(audio_embeds);
            fprintf(stderr, "crispasr[voxtral4b]: kv_init failed\n");
            return out;
        }
        voxtral4b_kv_reset(ctx_);

        int n_t = 0, vocab = 0;
        float * logits = voxtral4b_run_llm_kv(ctx_, prompt_embeds, T_prompt, 0, &n_t, &vocab);
        free(prompt_embeds);
        if (!logits) {
            free(audio_embeds);
            fprintf(stderr, "crispasr[voxtral4b]: prefill failed\n");
            return out;
        }

        int next = 0;
        {
            float mx = -1e30f;
            for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; next = k; }
        }
        free(logits);

        // ---- Streaming decode via src/core/greedy_decode.h ----
        // The 4B-Realtime variant is a "streaming" audio-LLM: on every
        // decode step we add the next adapter frame to the tail
        // embedding BEFORE the forward pass, and we stop the loop when
        // we run out of adapter frames. That's what the pre-forward
        // hook is for.
        constexpr int EOS = 2;
        int adapter_pos = T_prompt;
        auto pre_hook = [&](int /*step*/, float * tail) -> bool {
            if (adapter_pos >= N_enc) return false;  // audio exhausted
            for (int j = 0; j < pdim; j++) {
                tail[j] += audio_embeds[(size_t)adapter_pos * pdim + j];
            }
            adapter_pos++;
            return true;
        };

        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 512;
        dec_cfg.eos_id         = EOS;
        dec_cfg.vocab_size     = vocab;
        auto gen = core_greedy_decode::run(
            ctx_,
            /*first_token=*/next,
            /*initial_n_past=*/T_prompt,
            voxtral4b_embed_tokens,
            voxtral4b_run_llm_kv,
            pre_hook,
            dec_cfg);

        free(audio_embeds);

        // ---- Detokenize, filtering streaming control tokens (id < 1000) ----
        std::string transcript;
        for (int32_t id : gen) {
            if (id == EOS) break;
            if (id < 1000) continue; // skip STREAMING_PAD / STREAMING_WORD / etc.
            int len = 0;
            const uint8_t * bytes = voxtral4b_token_text(ctx_, id, &len);
            if (bytes && len > 0) transcript.append((const char *)bytes, (size_t)len);
        }
        while (!transcript.empty() &&
               (transcript.front() == ' ' || transcript.front() == '\t')) {
            transcript.erase(transcript.begin());
        }
        while (!transcript.empty() &&
               (transcript.back() == ' ' || transcript.back() == '\t')) {
            transcript.pop_back();
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = transcript;
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            voxtral4b_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    voxtral4b_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voxtral4b_backend() {
    return std::unique_ptr<CrispasrBackend>(new Voxtral4bBackend());
}
