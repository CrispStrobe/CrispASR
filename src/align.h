/**
 * align.h  —  CTC forced alignment using Viterbi DP.
 *
 * Takes pre-computed CTC logits [T × V] and a word sequence, and returns
 * per-word timestamps by constraining the Viterbi path to the given labels.
 *
 * Usage:
 *   std::vector<float> logits = wav2vec2_compute_logits(...);
 *   int T = logits.size() / model.hparams.vocab_size;
 *
 *   auto stamps = ctc_forced_align(
 *       logits.data(), T, model.hparams.vocab_size,
 *       words,                        // vector of words from Cohere
 *       model.vocab,                  // wav2vec2 id → token string
 *       model.hparams.pad_token_id,   // CTC blank
 *       wav2vec2_frame_dur(model));   // seconds per frame (~0.02 s)
 *
 *   for (auto & ws : stamps)
 *       printf("[%.2f --> %.2f]  %s\n", ws.t0, ws.t1, ws.word.c_str());
 */

#pragma once

#include <string>
#include <vector>

// Per-word timestamp produced by ctc_forced_align().
struct ctc_word_stamp {
    std::string word; // original word text (not normalised)
    float t0;         // start time, seconds
    float t1;         // end time, seconds (exclusive: start of next frame)
};

/**
 * CTC forced alignment.
 *
 * @param logits     Raw LM-head output, shape [T × V] row-major (log-softmax applied internally).
 * @param T          Number of encoder frames.
 * @param V          Vocabulary size.
 * @param words      Ordered transcript words (as produced by the transcription model).
 * @param vocab      CTC model vocabulary: vocab[id] = token string (e.g. "a", "|", "<pad>").
 * @param blank_id   CTC blank token ID (== pad_token_id from GGUF metadata).
 * @param frame_dur  Duration of one encoder frame in seconds (e.g. 320/16000 = 0.02 s).
 *
 * @return Per-word timestamps, same order as @p words.
 *         Returns an empty vector if the alignment fails (e.g. zero vocab overlap).
 *         Words whose characters are entirely absent from the CTC vocab get t0 == t1 == 0.
 */
std::vector<ctc_word_stamp> ctc_forced_align(const float* logits, int T, int V, const std::vector<std::string>& words,
                                             const std::vector<std::string>& vocab, int blank_id, float frame_dur);
