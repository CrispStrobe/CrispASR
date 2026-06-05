// chatterbox_s3tok.h — internal S3Tokenizer V2 forward for native voice cloning.
//
// Module 3 of the chatterbox WAV→cond port. Mirrors upstream
// `chatterbox.models.s3tokenizer.S3Tokenizer.forward(wavs, max_len=...)`,
// which is `S3TokenizerV2.quantize` on a 16 kHz mono input.
//
// Pipeline:
//   - log-mel features: n_fft=400, hop=160, n_mels=128, librosa Slaney mel,
//     log10 + max-clip(max-8) + (x+4)/4 normalisation. Drops the LAST STFT
//     frame (`stft[..., :-1]`). Channel-first (n_mels, T) row-major.
//   - AudioEncoderV2:
//       conv1: (128, 1280, k=3, stride=2, p=1) + GELU
//       conv2: (1280, 1280, k=3, stride=2, p=1) + GELU
//       permute → (B, T/4, 1280)
//       6 × ResidualAttentionBlock:
//         x += FSMNMultiHeadAttention(LN(x))
//         x += MLP(LN(x))      # Linear(1280→5120) + GELU + Linear(5120→1280)
//   - FSQVectorQuantization (codebook size = 3^8 = 6561):
//       project_down: Linear(1280, 8)
//       h = tanh(h) * 0.999 → round → +1   (values in {0,1,2})
//       token = Σ_i 3^i * h_i               (integer in [0, 6561))
//
// Tensor names (s3.tok.* in the chatterbox-s3gen GGUF):
//   _mel_filters                 (201, 128) F16
//   encoder.conv{1,2}.{weight,bias}
//   enc.b.{0..5}.attn.{query,key,value,out}.{weight,bias}
//   enc.b.{0..5}.attn.fsmn.weight                     (31, 1, 1280) F16 — DW conv1d k=31
//   enc.b.{0..5}.attn_ln.{weight,bias}
//   enc.b.{0..5}.mlp.{0,2}.{weight,bias}              # 0=up, 2=down (Sequential)
//   enc.b.{0..5}.mlp_ln.{weight,bias}
//   quant.cb.pd.{weight,bias}                          # project_down
//
// Hyperparameters (`s3tokenizer.model_v2.ModelConfig`):
//   n_mels=128, n_audio_state=1280, n_audio_head=20, head_dim=64,
//   n_audio_layer=6, conv stride=2, RoPE θ=10000, FSMN kernel=31,
//   attn_ln eps=1e-6, mlp_ln eps=1e-5 (nn.LayerNorm default),
//   max position end = 1024 * 2 = 2048, codebook_size=3^8=6561.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

struct cb_s3tok_block {
    ggml_tensor* attn_ln_w = nullptr; // (1280,) F32
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* attn_q_w = nullptr; // (1280, 1280) F16/Q8_0
    ggml_tensor* attn_q_b = nullptr; // (1280,) F32 — Q is biased
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr; // K has NO bias (Whisper convention) — always nullptr
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr; // V is biased
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* attn_fsmn_w = nullptr; // (31, 1, 1280) F16 — DW conv1d kernel
    ggml_tensor* mlp_ln_w = nullptr;
    ggml_tensor* mlp_ln_b = nullptr;
    ggml_tensor* mlp_up_w = nullptr; // (1280, 5120)
    ggml_tensor* mlp_up_b = nullptr; // (5120,)
    ggml_tensor* mlp_dn_w = nullptr; // (5120, 1280)
    ggml_tensor* mlp_dn_b = nullptr; // (1280,)
};

struct cb_s3tok_model {
    ggml_tensor* mel_filters = nullptr; // (201, 128) F16 — librosa Slaney mel basis
    ggml_tensor* conv1_w = nullptr;     // (3, 128, 1280) F16
    ggml_tensor* conv1_b = nullptr;     // (1280,) F32
    ggml_tensor* conv2_w = nullptr;     // (3, 1280, 1280) F16
    ggml_tensor* conv2_b = nullptr;     // (1280,) F32
    std::vector<cb_s3tok_block> blocks; // 6 blocks
    ggml_tensor* quant_pd_w = nullptr;  // (1280, 8) — project_down
    ggml_tensor* quant_pd_b = nullptr;  // (8,)
};

namespace chatterbox_s3tok {

// Compute (n_mels=128, T) channel-first log-mel features matching
// upstream's `log_mel_spectrogram`. T = floor(n_samples / 160) (the
// last STFT frame from the librosa center=True grid is dropped — that
// matches `magnitudes = stft[..., :-1].abs()**2`).
//
// Returns flat row-major (n_mels * T). On error returns empty + sets
// T_out to 0.
std::vector<float> compute_log_mel(const float* pcm_16k, int n_samples, int& T_out);

// Run the conv front-end + 6 transformer blocks + project_down on a
// (n_mels=128, T) log-mel tensor and return the post-project_down
// pre-quantize (T/4, 8) tensor — the float input to the FSQ codebook.
//
// `max_tokens > 0` truncates the mel to (max_tokens * 4) frames before
// the conv; this matches `S3Tokenizer.forward(..., max_len=plen)` which
// truncates `mel = mel[..., :max_len * 4]`.
std::vector<float> encode_to_proj(const cb_s3tok_model& m, ggml_backend_sched_t sched,
                                  std::vector<uint8_t>& compute_meta, const float* mel_n_t, int T, int max_tokens,
                                  int& T_tok_out);

// Run the encoder + project_down + FSQ quantize chain on a (n_mels, T)
// log-mel tensor and return T/4 int32 tokens. Tokens are in [0, 6561).
std::vector<int32_t> encode_tokens(const cb_s3tok_model& m, ggml_backend_sched_t sched,
                                   std::vector<uint8_t>& compute_meta, const float* mel_n_t, int T, int max_tokens);

// Convenience: PCM → log-mel → tokens.
std::vector<int32_t> tokenize(const cb_s3tok_model& m, ggml_backend_sched_t sched, std::vector<uint8_t>& compute_meta,
                              const float* pcm_16k, int n_samples, int max_tokens);

} // namespace chatterbox_s3tok
