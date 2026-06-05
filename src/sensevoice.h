// sensevoice.h — FunAudioLLM/SenseVoiceSmall ggml runtime
//
// Multi-task ASR: transcript + language ID + emotion + audio-event tags
// in one forward pass through a single CTC head. Encoder-only (no AR
// decode); same SenseVoiceEncoderSmall topology as Fun-ASR-Nano-2512
// but with an attached CTC head instead of an LLM decoder.
//
// Architecture (matches funasr/models/sense_voice/model.py exactly):
//
//   16 kHz mono PCM
//     → kaldi-fbank (hamming, 80 mels, 25/10ms, *32768) + LFR(7, 6)
//                                                      → (T_lfr, 560)
//     → prepend 4 query embeds from sensevoice.query_embed.w (16, 560):
//         idx 0 — language: {0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko,
//                            13=nospeech}
//         idx 1 — event:    always 1
//         idx 2 — emotion:  always 2
//         idx 3 — textnorm: {14=withitn, 15=woitn}
//       sequence: [lang_q, event_q, emo_q, textnorm_q, fbank_lfr]
//                                                       (T_lfr+4, 560)
//     → SenseVoiceEncoderSmall — 70 SANM blocks
//         (1 entry block @ 560→512 + 49 main blocks + 20 tp blocks,
//          after_norm between, tp_norm at end)         (T_lfr+4, 512)
//     → CTC head: linear (512 → 25055) + bias         (T_lfr+4, 25055)
//     → greedy CTC: argmax + unique_consecutive + drop blank (id=0)
//     → SentencePiece detokenize via stored token list
//
// Models loaded from GGUF files produced by:
//   `python models/convert-sensevoice-to-gguf.py --input <hf_dir>
//        --bpemodel <chn_jpn_yue_eng_ko_spectok.bpe.model>
//        --output X.gguf`
//
// Reference dumper:
//   `python tools/dump_reference.py --backend sensevoice ...`
//
// Stage names (for sensevoice_extract_stage + the diff harness):
//   mel_features                (T_lfr, 560)
//   encoder_input               (T_lfr+4, 560)  — after query-embed prepend
//   encoder_layer_{0..69}       (T_lfr+4, 512)
//   encoder_main_out            (T_lfr+4, 512)  — after_norm output
//   encoder_output              (T_lfr+4, 512)  — tp_norm output
//   ctc_logits                  (T_lfr+4, 25055)
//   generated_text              UTF-8 (transcript + rich-annotation prefix)

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sensevoice_context;

struct sensevoice_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
};

struct sensevoice_context_params sensevoice_context_default_params(void);

struct sensevoice_context* sensevoice_init_from_file(const char* path_model, struct sensevoice_context_params params);

void sensevoice_free(struct sensevoice_context* ctx);

// CTC beam search. beam_size == 1 (default) is greedy; > 1 runs prefix
// beam search with optional gamma-threshold pruning (gamma == 0 = off).
void sensevoice_set_beam_size(struct sensevoice_context* ctx, int beam_size, float gamma);

// Transcribe 16 kHz mono PCM. Returns malloc'd UTF-8 string (caller
// owns; free with free()). The string includes the 4-token rich-annotation
// prefix as special tokens like `<|en|><|HAPPY|><|Speech|><|withitn|>`
// followed by the SentencePiece-decoded transcript. Pass `language` to
// pin the language hint (`auto` / `zh` / `en` / `yue` / `ja` / `ko` /
// `nospeech`); pass nullptr or "auto" for auto-detect.
char* sensevoice_transcribe(struct sensevoice_context* ctx, const float* samples, int n_samples, const char* language,
                            bool use_itn);

// Structured result. Same forward pass as sensevoice_transcribe() but
// with the 4-token rich-annotation prefix parsed out into its own
// fields and removed from `text`. Empty strings mean the model did
// not emit that token (e.g. degenerate audio). Upstream emits the
// prefix in fixed order `[language, event, emotion, itn]`.
struct sensevoice_result {
    char* language;    // e.g. "en", "zh", "yue", "ja", "ko", "nospeech"
    char* emotion;     // e.g. "HAPPY", "NEUTRAL", "ANGRY", "SAD", "EMO_UNKNOWN"
    char* audio_event; // e.g. "Speech", "Music", "BGM", "Laughter", "Cough"
    char* itn;         // "withitn" or "woitn"
    char* text;        // transcript with the prefix stripped
    char* raw;         // original transcribe() output, prefix included
};

// Caller owns; release with sensevoice_result_free(). nullptr on failure.
struct sensevoice_result* sensevoice_transcribe_structured(struct sensevoice_context* ctx, const float* samples,
                                                           int n_samples, const char* language, bool use_itn);

void sensevoice_result_free(struct sensevoice_result* r);

// Pull an intermediate activation for diff testing. Caller free()s.
// Recognised stage names: see top-of-file comment.
float* sensevoice_extract_stage(struct sensevoice_context* ctx, const float* samples, int n_samples,
                                const char* language, bool use_itn, const char* stage_name, int* n_out);

#ifdef __cplusplus
}
#endif
