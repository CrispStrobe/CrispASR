// chatterbox_ve.h — internal VoiceEncoder (VE) bits for native voice cloning.
//
// The chatterbox voice-clone path needs four model components to turn a
// reference WAV into the conditioning bundle T3 + S3Gen consume. This file
// hosts module 2 (VoiceEncoder, a 3-layer LSTM that emits a 256-d L2-normed
// speaker embedding). Modules 3 (S3Tokenizer) and 4 (CAMPPlus) land in
// future PRs; the integration glue stays in chatterbox.cpp.
//
// Pipeline (mirrors `chatterbox.models.voice_encoder.embeds_from_wavs`):
//   - 16 kHz mono PCM in
//   - Slaney mel: n_fft=400, hop=160, win=400, n_mels=40, fmin=0, fmax=8000,
//     mel_power=2.0, mel_type='amp' (NO log step), preemph=0,
//     normalized_mels=False — produces (T, 40) row-major
//   - Partial extraction: 160-frame windows, frame_step = round((sr/rate)/
//     partial_frames) = round(16000/1.3/160) = 77 (rate=1.3 Resemble default)
//   - 3 stacked unidirectional LSTMs (40→256→256→256), take h[T-1, :] of L2
//   - proj.weight (256, 256) + proj.bias, ReLU, L2-norm per partial
//   - Mean across partials, then a final L2-norm — shape (1, 256)
//
// Silence trim (`librosa.effects.trim(top_db=20)`) is intentionally NOT
// applied here yet; the parity dump in `tools/reference_backends/
// chatterbox.py` also bypasses it so the comparison stays apples-to-apples.
// Trimming is a follow-up; for typical pre-trimmed reference WAVs the
// embedding quality is unchanged.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

// Tensor handles for the VoiceEncoder, populated by `bind_ve` in
// chatterbox.cpp from the T3 GGUF's `ve.*` tensors. Lives outside the
// anonymous namespace so chatterbox_ve.cpp can take it by const-ref.
struct cb_ve_model {
    // 3-layer LSTM
    ggml_tensor* lstm_ih_w[3] = {}; // weight_ih_l{i}: (4*H, in)   F16/F32
    ggml_tensor* lstm_hh_w[3] = {}; // weight_hh_l{i}: (4*H, H)    F16/F32
    ggml_tensor* lstm_ih_b[3] = {}; // bias_ih_l{i}:   (4*H,)      F32
    ggml_tensor* lstm_hh_b[3] = {}; // bias_hh_l{i}:   (4*H,)      F32
    ggml_tensor* proj_w = nullptr;  // (256, 256)                  F16/F32
    ggml_tensor* proj_b = nullptr;  // (256,)                       F32
};

namespace chatterbox_ve {

// Compute the (T, 40) raw-amp Slaney mel for a 16 kHz mono PCM buffer.
// Returns a flat row-major vector of length T*40 with T = 1 + n_samples/160
// (librosa center=True). Bit-equivalent to upstream `melspectrogram` for
// the VE config; verified against
// `chatterbox/models/voice_encoder/melspec.py`.
std::vector<float> compute_mel(const float* pcm_16k, int n_samples, int& T_out);

// Run the LSTM + proj + ReLU + L2-norm chain over each partial of the
// (T, 40) mel and return the (n_partials, 256) per-partial L2-normed
// embeddings. Walks the mel with `ve_partial_frames=160` windows at
// stride 77 (rate=1.3). If T < 160, zero-pads to one partial. Builds a
// fresh ggml graph per partial onto `sched`; reuses the caller's
// `compute_meta` scratch.
//
// Returns flat (n_partials * 256). On error returns an empty vector and
// leaves *n_partials_out at zero.
std::vector<float> compute_partial_embeds(const cb_ve_model& ve, ggml_backend_sched_t sched,
                                          std::vector<uint8_t>& compute_meta, const float* mel_tm, int T,
                                          int& n_partials_out);

// Full VE chain: PCM → 256-d L2-normed speaker embedding. Writes to
// `out_emb` (must hold 256 f32). Returns false on error.
bool compute_speaker_emb(const cb_ve_model& ve, ggml_backend_sched_t sched, std::vector<uint8_t>& compute_meta,
                         const float* pcm_16k, int n_samples, float out_emb[256]);

} // namespace chatterbox_ve
