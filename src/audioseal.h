#pragma once

// audioseal.h — AudioSeal watermark generator & detector (Meta, MIT license).
//
// AudioSeal uses a SEANet encoder-decoder architecture to embed an
// imperceptible watermark into speech audio, and a matching detector
// to recover it. The generator is ~5M parameters; the detector is ~4M.
//
// Architecture (from audioseal/builder.py + models.py):
//
//   Generator:
//     1. SEANet encoder: Conv1d(1, C, k=7) → N EncoderBlocks → LSTM(2) → Conv1d(C, D, k=7)
//     2. Message projection: Linear(nbits, D) added to encoder output
//     3. SEANet decoder: Conv1d(D, C, k=7) → N DecoderBlocks → Conv1d(C, 1, k=7) → Tanh
//     4. Output = input_audio + decoder_output (additive watermark)
//
//   Detector:
//     1. SEANet encoder: Conv1d(1, C, k=7) → N EncoderBlocks → LSTM(2)
//     2. Linear(D, 2) for watermark detection (binary: present/absent)
//     3. Linear(D, nbits) for message decoding
//
//   EncoderBlock(Cin, Cout, stride):
//     Residual units (dilation 1,3,9) + downsample Conv1d(Cin, Cout, k=2*s, s)
//
//   DecoderBlock(Cin, Cout, stride):
//     ConvTranspose1d(Cin, Cout, k=2*s, s) + Residual units (dilation 1,3,9)
//
//   Residual unit: ELU → Conv1d(d,d,k=3,dil) → ELU → Conv1d(d,d,k=1) + skip
//
// GGUF tensor names: audioseal.gen.enc.*, audioseal.gen.dec.*, audioseal.det.*
//
// This module loads a GGUF emitted by `models/convert-audioseal-to-gguf.py`.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct audioseal_ctx;

struct audioseal_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct audioseal_params audioseal_default_params(void);

// Load an AudioSeal GGUF (generator, detector, or combined).
struct audioseal_ctx* audioseal_init_from_file(const char* path, struct audioseal_params params);

void audioseal_free(struct audioseal_ctx* ctx);

// Returns the expected sample rate (16000 Hz for AudioSeal).
uint32_t audioseal_sample_rate(const struct audioseal_ctx* ctx);

// Returns the number of message bits embedded (default 16).
uint32_t audioseal_nbits(const struct audioseal_ctx* ctx);

// Embed watermark into float32 mono PCM at 16 kHz. The watermark is
// additive: output = input + generator(input, message).
//
// `message` is a bit array of length `audioseal_nbits()`. If nullptr,
// a default all-ones message is used.
//
// Returns a malloc'd float32 buffer of `n_samples` elements. Caller
// frees with `free()`. Returns nullptr on error.
float* audioseal_embed(struct audioseal_ctx* ctx, const float* pcm, int n_samples, const uint8_t* message);

// Detect watermark in float32 mono PCM at 16 kHz.
//
// Returns per-sample detection probability in [0,1] (malloc'd, length
// n_samples / hop_length). Caller frees with `free()`.
// `out_n` receives the number of detection frames.
// `out_message` (if non-null, length >= nbits) receives the decoded
// message bits.
//
// Returns nullptr on error.
float* audioseal_detect(struct audioseal_ctx* ctx, const float* pcm, int n_samples, int* out_n, uint8_t* out_message);

#ifdef __cplusplus
}
#endif
