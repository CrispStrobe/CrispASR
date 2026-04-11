// crispasr_llm_pipeline.h — templated audio-LLM pipeline shared across
// voxtral and voxtral4b (and a likely candidate for qwen3 and granite too).
//
// All of our audio-LLM backends follow the same pattern:
//
//   mel spectrogram  -> audio encoder (+ projector)  ->
//   build text prompt with audio-pad placeholders    ->
//   embed tokens                                      ->
//   splice audio-encoder output into audio-pad slots  ->
//   KV-cache prefill                                   ->
//   greedy argmax decode loop                          ->
//   detokenize via the backend's token_text function.
//
// They differ only in (a) the C function names and (b) model-specific
// constants like the audio_pad token ID, the EOS token, and the prompt
// template. By parameterising on an Ops traits struct we write the
// pipeline once and each backend provides a ~40-line traits class.
//
// This is the first piece of the model-level DRY work that will later move
// to src/core/ once the underlying models share their mel/attention
// primitives too. Keeping it in examples/cli/ for now means the refactor
// lands incrementally without touching src/*.cpp.

#pragma once

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Ops traits requirements (duck-typed, no inheritance):
//
//   typedef CtxT;                                     // backend context
//
//   static const char * name();                        // "voxtral" etc.
//
//   static CtxT * init(const char * path, int n_threads, int verbosity);
//   static void   free(CtxT * ctx);
//
//   static float * compute_mel(CtxT *, const float *, int,
//                              int * n_mels, int * T_mel);
//   static float * run_encoder(CtxT *, const float * mel, int n_mels, int T_mel,
//                              int * N_enc, int * enc_dim);
//
//   static int32_t * tokenize(CtxT *, const char * text, int * n);
//   static float  * embed_tokens(CtxT *, const int32_t * ids, int n);
//
//   static bool kv_init(CtxT *, int max_ctx);
//   static void kv_reset(CtxT *);
//   static float * run_llm_kv(CtxT *, const float * embeds, int n_tokens,
//                              int n_past, int * out_n_tokens, int * out_vocab);
//
//   static const uint8_t * token_text(CtxT *, int id, int * out_len);
//
//   static int audio_pad_id;                          // e.g. 24 for voxtral
//   static int eos_id;                                // e.g. 2
//   static std::string build_prefix(const whisper_params & p);
//   static std::string build_suffix(const whisper_params & p);
// (Both receive the full params struct so they can read e.g. .translate
//  or .target_lang in addition to .language.)
//

// Run the shared audio-LLM pipeline end-to-end for one audio slice.
// Returns a vector with exactly one crispasr_segment (text + offset times)
// on success, empty on failure. Prints errors to stderr.
template <typename Ops>
std::vector<crispasr_segment> crispasr_run_voxtral_style_pipeline(
    typename Ops::CtxT * ctx,
    const float * samples, int n_samples,
    int64_t t_offset_cs,
    const whisper_params & params)
{
    std::vector<crispasr_segment> out;
    if (!ctx) return out;

    const char * BE = Ops::name();

    // ---- Mel spectrogram ----
    int n_mels = 0, T_mel = 0;
    float * mel = Ops::compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "crispasr[%s]: mel failed\n", BE);
        return out;
    }

    // ---- Audio encoder (+ projector) ----
    int N_enc = 0, pdim = 0;
    float * audio_embeds = Ops::run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) {
        fprintf(stderr, "crispasr[%s]: encoder failed\n", BE);
        return out;
    }

    // ---- Build prompt via the backend's tokenizer ----
    const std::string prefix = Ops::build_prefix(params);
    const std::string suffix = Ops::build_suffix(params);

    int n_prefix = 0, n_suffix = 0;
    int32_t * pid = Ops::tokenize(ctx, prefix.c_str(), &n_prefix);
    int32_t * sid = Ops::tokenize(ctx, suffix.c_str(), &n_suffix);
    if (!pid || !sid) {
        fprintf(stderr, "crispasr[%s]: tokenize failed\n", BE);
        free(pid); free(sid); free(audio_embeds);
        return out;
    }

    std::vector<int32_t> ids;
    ids.reserve((size_t)n_prefix + N_enc + n_suffix);
    ids.insert(ids.end(), pid, pid + n_prefix);
    for (int i = 0; i < N_enc; i++) ids.push_back(Ops::audio_pad_id);
    ids.insert(ids.end(), sid, sid + n_suffix);
    free(pid);
    free(sid);

    const int T_prompt = (int)ids.size();

    // ---- Embed and splice audio frames into audio_pad positions ----
    float * text_embeds = Ops::embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        fprintf(stderr, "crispasr[%s]: embed failed\n", BE);
        free(audio_embeds);
        return out;
    }

    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == Ops::audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // ---- KV cache + prefill ----
    if (!Ops::kv_init(ctx, 4096)) {
        free(text_embeds);
        fprintf(stderr, "crispasr[%s]: kv_init failed\n", BE);
        return out;
    }
    Ops::kv_reset(ctx);

    int n_tokens_out = 0, vocab = 0;
    float * logits = Ops::run_llm_kv(ctx, text_embeds, T_prompt, 0,
                                     &n_tokens_out, &vocab);
    free(text_embeds);
    if (!logits) {
        fprintf(stderr, "crispasr[%s]: prefill failed\n", BE);
        return out;
    }

    // ---- First-token selection ----
    // Pure greedy (temperature <= 0) is the historical bit-identical path.
    // When --temperature > 0 we draw the first token too, so the whole
    // decode run is seeded from params.temperature consistently.
    core_greedy_decode::Config dec_cfg;
    dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 512;
    dec_cfg.eos_id         = Ops::eos_id;
    dec_cfg.vocab_size     = vocab;
    dec_cfg.temperature    = params.temperature;

    int   next    = 0;
    float next_p  = 1.0f;
    if (dec_cfg.temperature > 0.0f) {
        std::mt19937_64 seed_rng(dec_cfg.seed != 0 ? dec_cfg.seed
                                  : (uint64_t)std::random_device{}());
        next = core_greedy_decode::sample_temp(
            logits, vocab, dec_cfg.temperature, seed_rng);
    } else {
        next = core_greedy_decode::argmax(logits, vocab);
    }
    next_p = core_greedy_decode::softmax_of(logits, vocab, next, logits[next]);
    free(logits);

    // ---- Greedy / temperature-sampled decode loop with per-token probs ----
    auto dec = core_greedy_decode::run_with_probs(
        ctx,
        /*first_token=*/next,
        /*first_prob=*/next_p,
        /*initial_n_past=*/T_prompt,
        Ops::embed_tokens,
        Ops::run_llm_kv,
        dec_cfg);
    const std::vector<int32_t> & gen   = dec.tokens;
    const std::vector<float>   & probs = dec.probs;

    // ---- Detokenize + attach per-token confidence to the segment ----
    std::string transcript;
    crispasr_segment seg;
    seg.tokens.reserve(gen.size());
    for (size_t i = 0; i < gen.size(); i++) {
        const int32_t id = gen[i];
        if (id == Ops::eos_id) break;
        int len = 0;
        const uint8_t * bytes = Ops::token_text(ctx, id, &len);
        crispasr_token ct;
        ct.id         = id;
        ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
        if (bytes && len > 0) {
            ct.text.assign((const char *)bytes, (size_t)len);
            transcript.append((const char *)bytes, (size_t)len);
        }
        seg.tokens.push_back(std::move(ct));
    }

    seg.t0 = t_offset_cs;
    seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
    seg.text = transcript;

    // Trim leading whitespace if the prompt template bled one in.
    while (!seg.text.empty() &&
           (seg.text.front() == ' ' || seg.text.front() == '\n')) {
        seg.text.erase(seg.text.begin());
    }

    out.push_back(std::move(seg));
    return out;
}
