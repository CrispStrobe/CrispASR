// Shared Qwen3 prefix-stream model call for CLI and session bindings.
#pragma once
#include "qwen3_asr.h"
#include "core/bpe.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace core_qwen3_stream_model {
inline int special_id(qwen3_asr_context* ctx, const char* token) {
    int n = 0;
    int32_t* ids = qwen3_asr_tokenize(ctx, token, &n);
    const int id = ids && n == 1 ? ids[0] : -1;
    free(ids);
    return id;
}
inline std::string generate(qwen3_asr_context* ctx_, bool raon_, const std::vector<float>& audio,
                            const std::string& assistant_suffix, int max_new, bool* success = nullptr) {
    if (success)
        *success = false;
    if (!ctx_ || audio.empty())
        return {};
    int N_enc = 0, pdim = 0;
    float* audio_embeds = nullptr;
    if (raon_) {
        audio_embeds = qwen3_asr_raon_encode(ctx_, audio.data(), (int)audio.size(), &N_enc, &pdim);
    } else {
        int nm = 0, nt = 0;
        float* mel = qwen3_asr_compute_mel(ctx_, audio.data(), (int)audio.size(), &nm, &nt);
        if (mel) {
            audio_embeds = qwen3_asr_run_encoder(ctx_, mel, nm, nt, &N_enc, &pdim);
            free(mel);
        }
    }
    if (!audio_embeds)
        return {};
    // qwen_asr _build_text_prompt with context="" (the chat template puts
    // the context in the system turn; R2T2's example passes none).
    std::string text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
    for (int i = 0; i < N_enc; i++)
        text += "<|audio_pad|>";
    text += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    text += assistant_suffix;
    if (raon_)
        text = "<|im_start|>user\n<|audio_start|>";
    if (raon_) {
        for (int i = 0; i < N_enc; ++i)
            text += "<|audio_pad|>";
        text += "<|audio_end|>Transcribe the audio into text<|im_end|>\n<|im_start|>assistant\n" + assistant_suffix;
    }
    int n_prompt = 0;
    int32_t* raw_ids = qwen3_asr_tokenize(ctx_, text.c_str(), &n_prompt);
    if (!raw_ids) {
        free(audio_embeds);
        return {};
    }
    std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
    free(raw_ids);
    const int audio_pad_id = special_id(ctx_, "<|audio_pad|>");
    float* emb = qwen3_asr_embed_tokens(ctx_, ids.data(), (int)ids.size());
    if (!emb || audio_pad_id < 0) {
        free(emb);
        free(audio_embeds);
        return {};
    }
    int spliced = 0;
    for (size_t i = 0; i < ids.size() && spliced < N_enc; i++)
        if (ids[i] == audio_pad_id)
            std::memcpy(emb + i * pdim, audio_embeds + (size_t)(spliced++) * pdim, pdim * sizeof(float));
    free(audio_embeds);

    const int prompt_len = (int)ids.size();
    if (!qwen3_asr_kv_init(ctx_, std::max(4096, prompt_len + max_new + 16))) {
        free(emb);
        return {};
    }
    qwen3_asr_kv_reset(ctx_);
    int n_t = 0, vocab = 0;
    float* logits = qwen3_asr_run_llm_kv(ctx_, emb, prompt_len, 0, &n_t, &vocab);
    free(emb);
    if (!logits)
        return {};
    // generation_config.eos_token_id = [<|endoftext|>, <|im_end|>]
    const int eos_a = special_id(ctx_, "<|im_end|>"), eos_b = special_id(ctx_, "<|endoftext|>");
    auto argmax = [vocab](const float* row) {
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (row[v] > row[best])
                best = v;
        return best;
    };
    std::string bytes;
    static const bool trace_ids =
        getenv("CRISPASR_QWEN3_STREAM_TRACE") && atoi(getenv("CRISPASR_QWEN3_STREAM_TRACE")) >= 2;
    if (trace_ids)
        fprintf(stderr, "QWEN3_STREAM_IDS n_audio=%zu n_enc=%d prompt_len=%d ids=", audio.size(), N_enc, prompt_len);
    int id = argmax(logits + (size_t)(n_t - 1) * vocab);
    free(logits);
    int n_past = prompt_len;
    for (int step = 0; step < max_new; step++) {
        if (trace_ids)
            fprintf(stderr, "%d,", id);
        if (id == eos_a || id == eos_b)
            break;
        const char* piece = qwen3_asr_token_text(ctx_, id);
        // skip_special_tokens: the special-flagged added tokens are the
        // <|...|> family; <asr_text> / <non_speech> are NOT special and stay.
        if (piece && !(piece[0] == '<' && piece[1] == '|'))
            bytes += core_bpe::token_bytes_to_utf8(piece);
        if (step + 1 == max_new)
            break;
        float* te = qwen3_asr_embed_tokens(ctx_, &id, 1);
        if (!te)
            return {};
        int nt2 = 0, v2 = 0;
        float* lg = qwen3_asr_run_llm_kv(ctx_, te, 1, n_past, &nt2, &v2);
        free(te);
        if (!lg)
            return {};
        n_past++;
        id = argmax(lg);
        free(lg);
    }
    if (trace_ids)
        fprintf(stderr, "\n");
    if (success)
        *success = true;
    return bytes;
}

} // namespace core_qwen3_stream_model
