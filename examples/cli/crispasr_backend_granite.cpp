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
#include <random>
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
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE
             | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TOKEN_CONFIDENCE
             | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE;
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
        // Default suffix is the hardcoded "can you transcribe..." prompt
        // tokenized once at dev time. For --translate we re-tokenize an
        // alternative suffix at runtime via granite_speech_tokenize().
        // Falls back gracefully when the GGUF predates the merges-table
        // commit (we keep the original suffix and warn).
        std::vector<int32_t> suffix_ids;
        if (params.translate) {
            auto iso_to_eng = [](const std::string & c) -> std::string {
                if (c == "en") return "English";
                if (c == "de") return "German";
                if (c == "fr") return "French";
                if (c == "es") return "Spanish";
                if (c == "it") return "Italian";
                if (c == "pt") return "Portuguese";
                if (c == "ru") return "Russian";
                if (c == "ja") return "Japanese";
                if (c == "zh") return "Chinese";
                return c;
            };
            const std::string tgt = params.target_lang.empty()
                                  ? std::string("English")
                                  : iso_to_eng(params.target_lang);
            const std::string instr =
                "can you translate the speech to " + tgt + "?\n ASSISTANT:";
            int n_tok = 0;
            int32_t * arr = granite_speech_tokenize(ctx_, instr.c_str(), &n_tok);
            if (arr && n_tok > 0) {
                suffix_ids.assign(arr, arr + n_tok);
                free(arr);
            } else {
                fprintf(stderr,
                        "crispasr[granite]: tokenize failed (re-convert with the "
                        "newer models/convert-granite-speech-to-gguf.py to enable "
                        "--translate); falling back to plain transcribe\n");
            }
        }

        const int n_suffix = suffix_ids.empty() ? kNumSuffix : (int)suffix_ids.size();
        const int total_prompt = kNumPrefix + N_proj + n_suffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int i = 0; i < kNumPrefix; i++) prompt_ids.push_back(kPrefix[i]);
        for (int i = 0; i < N_proj; i++)    prompt_ids.push_back(kAudioTok);
        if (suffix_ids.empty()) {
            for (int i = 0; i < kNumSuffix; i++) prompt_ids.push_back(kSuffix[i]);
        } else {
            for (int i = 0; i < n_suffix; i++) prompt_ids.push_back(suffix_ids[i]);
        }

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

        // ---- Decode loop via src/core/greedy_decode.h ----
        // Temperature sampling kicks in when --temperature > 0; otherwise
        // we stay on the historical bit-identical greedy path.
        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 200;
        dec_cfg.eos_id         = kEos;
        dec_cfg.vocab_size     = vocab;
        dec_cfg.temperature    = params.temperature;

        int   next   = 0;
        float next_p = 1.0f;
        if (dec_cfg.temperature > 0.0f) {
            std::mt19937_64 seed_rng(dec_cfg.seed != 0 ? dec_cfg.seed
                                       : (uint64_t)std::random_device{}());
            next = core_greedy_decode::sample_temp(
                logits, vocab, dec_cfg.temperature, seed_rng);
        } else {
            next = core_greedy_decode::argmax(logits, vocab);
        }
        next_p = core_greedy_decode::softmax_of(
            logits, vocab, next, logits[next]);
        free(logits);

        auto dec = core_greedy_decode::run_with_probs(
            ctx_,
            /*first_token=*/next,
            /*first_prob=*/next_p,
            /*initial_n_past=*/total_prompt,
            granite_speech_embed_tokens,
            granite_speech_run_llm_kv,
            dec_cfg);
        const std::vector<int32_t> & gen_ids = dec.tokens;
        const std::vector<float>   & probs   = dec.probs;

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

        // Per-token entries with decode-loop confidences. granite uses
        // its own batch detokenizer (granite_speech_decode_tokens) for
        // the segment text, but we still surface per-token id + prob so
        // downstream consumers (JSON full, confidence filters) have the
        // raw signal. Token text is intentionally left empty here to
        // avoid duplicating the batch detokenizer's merging logic.
        seg.tokens.reserve(gen_ids.size());
        for (size_t i = 0; i < gen_ids.size(); i++) {
            if (gen_ids[i] == kEos) break;
            crispasr_token ct;
            ct.id         = gen_ids[i];
            ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
            seg.tokens.push_back(std::move(ct));
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
