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
std::vector<crispasr_segment> crispasr_run_voxtral_style_pipeline(typename Ops::CtxT* ctx, const float* samples,
                                                                  int n_samples, int64_t t_offset_cs,
                                                                  const whisper_params& params) {
    std::vector<crispasr_segment> out;
    if (!ctx)
        return out;

    const char* BE = Ops::name();

    // ---- Mel spectrogram ----
    int n_mels = 0, T_mel = 0;
    float* mel = Ops::compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "crispasr[%s]: mel failed\n", BE);
        return out;
    }

    // ---- Audio encoder (+ projector) ----
    int N_enc = 0, pdim = 0;
    float* audio_embeds = Ops::run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) {
        fprintf(stderr, "crispasr[%s]: encoder failed\n", BE);
        return out;
    }

    // ---- Build prompt via the backend's tokenizer ----
    const std::string prefix = Ops::build_prefix(params);
    const std::string suffix = Ops::build_suffix(params);

    int n_prefix = 0, n_suffix = 0;
    int32_t* pid = Ops::tokenize(ctx, prefix.c_str(), &n_prefix);
    int32_t* sid = Ops::tokenize(ctx, suffix.c_str(), &n_suffix);
    if (!pid || !sid) {
        fprintf(stderr, "crispasr[%s]: tokenize failed\n", BE);
        free(pid);
        free(sid);
        free(audio_embeds);
        return out;
    }

    std::vector<int32_t> ids;
    ids.reserve((size_t)n_prefix + N_enc + n_suffix);
    ids.insert(ids.end(), pid, pid + n_prefix);
    for (int i = 0; i < N_enc; i++)
        ids.push_back(Ops::audio_pad_id);
    ids.insert(ids.end(), sid, sid + n_suffix);
    free(pid);
    free(sid);

    const int T_prompt = (int)ids.size();

    // ---- Embed and splice audio frames into audio_pad positions ----
    float* text_embeds = Ops::embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        fprintf(stderr, "crispasr[%s]: embed failed\n", BE);
        free(audio_embeds);
        return out;
    }

    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == Ops::audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // ---- KV cache + best-of-N decode ----
    if (!Ops::kv_init(ctx, 4096)) {
        free(text_embeds);
        fprintf(stderr, "crispasr[%s]: kv_init failed\n", BE);
        return out;
    }

    // Best-of-N is only meaningful with sampling. With temperature == 0
    // every run is deterministic, so n_runs collapses to 1 and we keep
    // the bit-identical historical path. With temperature > 0 and
    // params.best_of > 1, we run N independent decodes and keep the one
    // with the highest mean per-token probability (a cheap proxy for
    // mean log-prob ranking that doesn't need the noisy first-token
    // step). Each candidate gets its own prefill + greedy_decode pass;
    // the KV cache is reset between candidates so they don't share
    // history.
    const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;

    core_greedy_decode::Config dec_cfg;
    dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 512;
    dec_cfg.eos_id = Ops::eos_id;
    dec_cfg.vocab_size = 0; // filled after first prefill
    dec_cfg.temperature = params.temperature;
    dec_cfg.frequency_penalty = params.frequency_penalty;
    dec_cfg.seed = params.seed;

    core_greedy_decode::Result best_dec;
    double best_score = -1.0;
    int vocab = 0;

    for (int run = 0; run < n_runs; run++) {
        Ops::kv_reset(ctx);

        int n_tokens_out = 0;
        float* logits = Ops::run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_tokens_out, &vocab);
        if (!logits) {
            fprintf(stderr, "crispasr[%s]: prefill failed (run %d/%d)\n", BE, run + 1, n_runs);
            free(text_embeds);
            return out;
        }
        if (run == 0)
            dec_cfg.vocab_size = vocab;

        int next = 0;
        float next_p = 1.0f;
        if (dec_cfg.temperature > 0.0f) {
            // Different seed per run so the N runs actually diverge.
            // Mix in run index so they don't all collapse to the same
            // sample sequence on a deterministic seed.
            std::mt19937_64 seed_rng((dec_cfg.seed != 0 ? dec_cfg.seed : (uint64_t)std::random_device{}()) ^
                                     (uint64_t)(run * 0x9E3779B97F4A7C15ull));
            next = core_greedy_decode::sample_temp(logits, vocab, dec_cfg.temperature, seed_rng);
        } else {
            next = core_greedy_decode::argmax(logits, vocab);
        }
        next_p = core_greedy_decode::softmax_of(logits, vocab, next, logits[next]);
        free(logits);

        auto dec = core_greedy_decode::run_with_probs(ctx,
                                                      /*first_token=*/next,
                                                      /*first_prob=*/next_p,
                                                      /*initial_n_past=*/T_prompt, Ops::embed_tokens, Ops::run_llm_kv,
                                                      dec_cfg);

        // Score: arithmetic mean of per-token softmax probabilities,
        // skipping the trailing EOS if present. Equivalent ranking to
        // mean-log-prob for our purposes and avoids one log() per token.
        double sum = 0.0;
        int cnt = 0;
        for (size_t i = 0; i < dec.probs.size(); i++) {
            if ((int32_t)dec.tokens[i] == Ops::eos_id)
                break;
            sum += (double)dec.probs[i];
            cnt++;
        }
        const double score = (cnt > 0) ? (sum / cnt) : 0.0;

        if (run == 0 || score > best_score) {
            best_score = score;
            best_dec = std::move(dec);
        }
    }
    free(text_embeds);

    if (!params.no_prints && n_runs > 1) {
        fprintf(stderr, "crispasr[%s]: best-of-%d picked score=%.4f\n", BE, n_runs, best_score);
    }

    const std::vector<int32_t>& gen = best_dec.tokens;
    const std::vector<float>& probs = best_dec.probs;

    // ---- Detokenize + attach per-token confidence to the segment ----
    std::string transcript;
    crispasr_segment seg;
    seg.tokens.reserve(gen.size());
    for (size_t i = 0; i < gen.size(); i++) {
        const int32_t id = gen[i];
        if (id == Ops::eos_id)
            break;
        int len = 0;
        const uint8_t* bytes = Ops::token_text(ctx, id, &len);
        crispasr_token ct;
        ct.id = id;
        ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
        if (bytes && len > 0) {
            ct.text.assign((const char*)bytes, (size_t)len);
            transcript.append((const char*)bytes, (size_t)len);
        }
        seg.tokens.push_back(std::move(ct));
    }

    seg.t0 = t_offset_cs;
    seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
    seg.text = transcript;

    // Trim leading whitespace if the prompt template bled one in.
    while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n')) {
        seg.text.erase(seg.text.begin());
    }

    out.push_back(std::move(seg));
    return out;
}

// ============================================================================
// Long-form variant — Mistral upstream pattern (PLAN #114, issue #89)
// ============================================================================
//
// The single-chunk pipeline above pads its input to exactly 30 s before
// compute_mel + run_encoder. For audio longer than 30 s our previous
// approach was to slice at 30 s in `crispasr_run.cpp` and call the
// single-chunk pipeline N times — each call gets a fresh LLM context,
// the AR decoder cold-starts at every chunk boundary, and on long audio
// the model drops or duplicates content at boundaries (issue #89 follow-
// up: 600 s voxtral coverage was 9 % with 545 s of audio dropped in the
// middle).
//
// transformers.models.voxtral.processing_voxtral.apply_transcription_
// request handles this differently: it pads audio to a multiple of 30 s,
// reshapes mel into (n_chunks, n_mels, 3000), feeds the whole batch to
// the encoder, then concatenates the per-chunk audio embeds into a
// single prompt for the LLM to decode in one AR pass. The LLM sees one
// contiguous audio sequence — no per-chunk cold-start.
//
// We replicate that shape here. The mel + encoder calls are sequential
// rather than batched (CrispASR's runtime is single-graph), but the
// produced audio_embeds buffer has the same logical layout as upstream's
// batched call, and the downstream prompt-build / splice / LLM-decode
// path is the same code as the single-chunk variant.
//
// Context budget: 30 s -> 375 audio tokens (3000 mel frames / 8x conv
// subsampling). 600 s -> 7500 audio tokens. Voxtral-mini-3B's training
// context is 32 768 tokens, so this fits up to ~30 min before we'd need
// to revisit. For longer audio the CLI's VAD or explicit chunking path
// remains available as a fallback.
template <typename Ops>
std::vector<crispasr_segment> crispasr_run_voxtral_style_pipeline_streamed(typename Ops::CtxT* ctx,
                                                                           const float* samples, int n_samples,
                                                                           int64_t t_offset_cs,
                                                                           const whisper_params& params) {
    std::vector<crispasr_segment> out;
    if (!ctx)
        return out;

    const char* BE = Ops::name();
    constexpr int kSampleRate = 16000;
    constexpr int kChunkSamples = 30 * kSampleRate; // 480000 — matches upstream `pad_to_multiple_of`

    // For short audio (≤ one chunk), defer to the single-chunk pipeline.
    // That preserves the bit-identical behaviour on JFK / short clips.
    if (n_samples <= kChunkSamples) {
        return crispasr_run_voxtral_style_pipeline<Ops>(ctx, samples, n_samples, t_offset_cs, params);
    }

    // ---- Sequential per-chunk encode, concatenate audio_embeds --------------
    const int n_chunks = (n_samples + kChunkSamples - 1) / kChunkSamples;
    std::vector<float> audio_embeds_all;
    int per_chunk_N = 0;
    int per_chunk_dim = 0;

    for (int c = 0; c < n_chunks; c++) {
        const int start = c * kChunkSamples;
        const int len = std::min(kChunkSamples, n_samples - start);

        int n_mels = 0, T_mel = 0;
        float* mel = Ops::compute_mel(ctx, samples + start, len, &n_mels, &T_mel);
        if (!mel) {
            fprintf(stderr, "crispasr[%s][streamed]: mel failed on chunk %d/%d\n", BE, c + 1, n_chunks);
            return out;
        }

        int N_enc = 0, pdim = 0;
        float* audio_embeds = Ops::run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
        free(mel);
        if (!audio_embeds) {
            fprintf(stderr, "crispasr[%s][streamed]: encoder failed on chunk %d/%d\n", BE, c + 1, n_chunks);
            return out;
        }

        if (c == 0) {
            per_chunk_N = N_enc;
            per_chunk_dim = pdim;
            audio_embeds_all.reserve((size_t)n_chunks * (size_t)N_enc * (size_t)pdim);
        } else if (N_enc != per_chunk_N || pdim != per_chunk_dim) {
            fprintf(stderr, "crispasr[%s][streamed]: chunk %d shape (%d,%d) != chunk 0 (%d,%d)\n", BE, c, N_enc, pdim,
                    per_chunk_N, per_chunk_dim);
            free(audio_embeds);
            return out;
        }

        audio_embeds_all.insert(audio_embeds_all.end(), audio_embeds, audio_embeds + (size_t)N_enc * (size_t)pdim);
        free(audio_embeds);
    }

    const int N_enc_total = (int)(audio_embeds_all.size() / (size_t)per_chunk_dim);

    if (!params.no_prints) {
        fprintf(stderr, "crispasr[%s][streamed]: %d chunk(s), %d audio tokens, then single LLM decode\n", BE, n_chunks,
                N_enc_total);
    }

    // ---- Build prompt --------------------------------------------------------
    const std::string prefix = Ops::build_prefix(params);
    const std::string suffix = Ops::build_suffix(params);

    int n_prefix = 0, n_suffix = 0;
    int32_t* pid = Ops::tokenize(ctx, prefix.c_str(), &n_prefix);
    int32_t* sid = Ops::tokenize(ctx, suffix.c_str(), &n_suffix);
    if (!pid || !sid) {
        fprintf(stderr, "crispasr[%s][streamed]: tokenize failed\n", BE);
        free(pid);
        free(sid);
        return out;
    }

    std::vector<int32_t> ids;
    ids.reserve((size_t)n_prefix + (size_t)N_enc_total + (size_t)n_suffix);
    ids.insert(ids.end(), pid, pid + n_prefix);
    for (int i = 0; i < N_enc_total; i++)
        ids.push_back(Ops::audio_pad_id);
    ids.insert(ids.end(), sid, sid + n_suffix);
    free(pid);
    free(sid);

    const int T_prompt = (int)ids.size();

    // ---- Embed + splice ------------------------------------------------------
    float* text_embeds = Ops::embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        fprintf(stderr, "crispasr[%s][streamed]: embed failed\n", BE);
        return out;
    }

    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc_total; i++) {
        if (ids[i] == Ops::audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * per_chunk_dim,
                        audio_embeds_all.data() + (size_t)spliced * per_chunk_dim,
                        (size_t)per_chunk_dim * sizeof(float));
            spliced++;
        }
    }

    // ---- KV cache + decode budget proportional to audio duration ------------
    // `whisper_params::max_new_tokens` defaults to 512 (whisper_params.h:134),
    // plenty for ≤30 s of speech. The streamed variant decodes all chunks in
    // one AR pass — 300 s needs ~2000 tokens, 600 s ~4000. If we honoured the
    // 512 default verbatim the long-form transcript would cap mid-sentence
    // (empirically confirmed on the 300 s lenhone clip: output capped at
    // exactly 512 tokens at "...今年は4日が今年と言えば20" instead of running
    // to the end).
    //
    // Heuristic: take the larger of the user-provided value and a
    // duration-scaled budget at ~8 tokens / second (generous; Japanese
    // verbatim runs ~5-6 tok/s, English ~4 tok/s). A user who really wants
    // to cap below the scaled budget will need to bump the threshold via
    // a future explicit flag — calling out the rare case in the message
    // below if we extend the default. The KV cache is sized to match.
    const int audio_seconds = (n_samples + kSampleRate - 1) / kSampleRate;
    const int max_new_scaled = std::max(512, audio_seconds * 8);
    const int max_new = std::max(params.max_new_tokens, max_new_scaled);
    if (!params.no_prints && max_new > params.max_new_tokens) {
        fprintf(stderr,
                "crispasr[%s][streamed]: scaling max_new_tokens %d -> %d for %d s of audio\n",
                BE, params.max_new_tokens, max_new, audio_seconds);
    }
    const int kv_budget = T_prompt + max_new + 64;
    if (!Ops::kv_init(ctx, kv_budget)) {
        free(text_embeds);
        fprintf(stderr, "crispasr[%s][streamed]: kv_init(%d) failed\n", BE, kv_budget);
        return out;
    }
    Ops::kv_reset(ctx);

    // ---- Single greedy decode over the concatenated audio embeds ------------
    int n_tokens_out = 0, vocab = 0;
    float* logits = Ops::run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_tokens_out, &vocab);
    if (!logits) {
        fprintf(stderr, "crispasr[%s][streamed]: prefill failed\n", BE);
        free(text_embeds);
        return out;
    }

    core_greedy_decode::Config dec_cfg;
    dec_cfg.max_new_tokens = max_new;
    dec_cfg.eos_id = Ops::eos_id;
    dec_cfg.vocab_size = vocab;
    dec_cfg.temperature = params.temperature;
    dec_cfg.frequency_penalty = params.frequency_penalty;
    dec_cfg.seed = params.seed;

    int next = 0;
    float next_p = 1.0f;
    if (dec_cfg.temperature > 0.0f) {
        std::mt19937_64 rng(dec_cfg.seed != 0 ? dec_cfg.seed : (uint64_t)std::random_device{}());
        next = core_greedy_decode::sample_temp(logits, vocab, dec_cfg.temperature, rng);
    } else {
        next = core_greedy_decode::argmax(logits, vocab);
    }
    next_p = core_greedy_decode::softmax_of(logits, vocab, next, logits[next]);
    free(logits);

    auto dec = core_greedy_decode::run_with_probs(ctx, next, next_p, T_prompt, Ops::embed_tokens, Ops::run_llm_kv,
                                                  dec_cfg);
    free(text_embeds);

    // ---- Detokenize ----------------------------------------------------------
    std::string transcript;
    crispasr_segment seg;
    seg.tokens.reserve(dec.tokens.size());
    for (size_t i = 0; i < dec.tokens.size(); i++) {
        const int32_t id = dec.tokens[i];
        if (id == Ops::eos_id)
            break;
        int len = 0;
        const uint8_t* bytes = Ops::token_text(ctx, id, &len);
        crispasr_token ct;
        ct.id = id;
        ct.confidence = (i < dec.probs.size()) ? dec.probs[i] : -1.0f;
        if (bytes && len > 0) {
            ct.text.assign((const char*)bytes, (size_t)len);
            transcript.append((const char*)bytes, (size_t)len);
        }
        seg.tokens.push_back(std::move(ct));
    }

    seg.t0 = t_offset_cs;
    seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
    seg.text = transcript;
    while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n')) {
        seg.text.erase(seg.text.begin());
    }

    out.push_back(std::move(seg));
    return out;
}
