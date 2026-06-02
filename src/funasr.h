// funasr.h — FunAudioLLM/Fun-ASR-Nano-2512 ggml runtime
//
// Architecture (LLM-decoder path — the only path the published checkpoint
// supports; CTC weights are absent from model.pt):
//
//   16 kHz mono PCM
//     → WavFrontend  : kaldi-fbank (hamming, 80 mels, 25/10 ms,
//                       *32768 input scale) + LFR(m=7, n=6) →   (T_lfr, 560)
//     → Encoder      : SenseVoiceEncoderSmall, 70 SANM blocks
//                       (1 in_size=560→size=512 entry, 49 main,
//                        20 tp; after_norm between, tp_norm at end) → (T_lfr, 512)
//     → audio_adaptor: linear(512→2048) + ReLU + linear(2048→1024)
//                       + 2× Transformer blocks (separate Q/K/V,
//                        FFN inner=256, LN eps=1e-12)            → (T_lfr, 1024)
//     → ChatML prompt + splice : embed the prompt text via Qwen3
//                       token_embd; overwrite the placeholder slots
//                       with adaptor_out[:fake_token_len].
//     → Qwen3-0.6B LLM AR decode (28 layers, GQA 16/8, RoPE θ=1e6).
//     → Detokenize via Qwen3 GPT-2-style BPE.
//
// Models loaded from GGUF files produced by:
//   `python models/convert-funasr-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Reference dumper (Python ground-truth):
//   `python tools/dump_reference.py --backend funasr ...`
//
// Stage names (for funasr_extract_stage and the diff harness):
//   mel_features              (T_lfr, 560)
//   encoder_layer_{0..69}     (T_lfr, 512)
//   encoder_main_out          (T_lfr, 512)   // after_norm output
//   encoder_output            (T_lfr, 512)   // tp_norm output (final encoder)
//   audio_adaptor_layer_{0,1} (T_lfr, 1024)
//   audio_adaptor_output      (T_lfr, 1024)
//   generated_text            UTF-8 bytes

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct funasr_context;

struct funasr_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
    bool use_gpu;      // false => force CPU backend
    float temperature; // 0 = greedy argmax (default)
};

struct funasr_context_params funasr_context_default_params(void);

struct funasr_context* funasr_init_from_file(const char* path_model, struct funasr_context_params params);

void funasr_free(struct funasr_context* ctx);

// Beam search width. 1 = greedy (default); >1 = replay-from-prefix beam.
void funasr_set_beam_size(struct funasr_context* ctx, int beam_size);

// Transcribe 16 kHz mono PCM. Returns malloc'd UTF-8 string; caller frees with free().
char* funasr_transcribe(struct funasr_context* ctx, const float* samples, int n_samples);

// Variant that additionally returns per-emitted-token ids + softmax probs.
struct funasr_result {
    char* text;
    int32_t* token_ids;
    float* token_probs;
    int n_tokens;
};
struct funasr_result* funasr_transcribe_with_probs(struct funasr_context* ctx, const float* samples, int n_samples);
void funasr_result_free(struct funasr_result* r);

// Single-id detokenize.
const char* funasr_token_text(struct funasr_context* ctx, int id);

// Pull one intermediate activation out of the pipeline for diff testing.
// Returns malloc'd F32 buffer; caller frees with free(). *n_out is set
// to the number of float elements in the returned buffer.
//
// Recognised stage names (see top-of-file comment for shapes):
//   "mel_features"
//   "encoder_layer_K"        K in [0, 70)
//   "encoder_main_out"
//   "encoder_output"
//   "audio_adaptor_layer_K"  K in [0, 2)
//   "audio_adaptor_output"
//   "generated_text"         (bytes — pointer cast to char* gives the UTF-8 string)
//
// Returns nullptr (with *n_out = 0) when the stage name is unrecognized
// or the pipeline failed.
float* funasr_extract_stage(struct funasr_context* ctx, const float* samples, int n_samples, const char* stage_name,
                            int* n_out);

#ifdef __cplusplus
}
#endif
