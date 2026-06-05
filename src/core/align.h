// src/core/align.h — duration-based alignment helpers for TTS pipelines.
//
// Every TTS with explicit duration prediction (FastSpeech, VITS,
// StyleTTS, Kokoro, qwen3-tts talker) needs the same primitive: take a
// per-token feature matrix and a per-token integer duration vector,
// and return a per-frame feature matrix where each token's features
// are repeated `dur[i]` times. PyTorch reference:
//
//   indices = torch.repeat_interleave(torch.arange(L), durations)
//   en      = features.transpose(-1, -2) @ one_hot(indices, L)
//
// The matmul-with-one-hot formulation is what the reference dumpers
// emit, but it's a wasteful way to compute it on CPU (O(L · T_frames)
// memory for the alignment matrix) when the same answer comes from a
// straight memcpy loop.
//
// This helper does the memcpy version. Output buffer is malloc'd —
// caller frees with `free()`.

#pragma once

#include <cstdlib>
#include <cstring>

namespace core_align {

// Repeat-interleave a (D, L) channel-major feature matrix according to
// per-token integer durations. Returns a malloc'd (D, T_frames) F32
// buffer with T_frames = sum(durations); writes T_frames to
// *out_T_frames. Returns nullptr (and *out_T_frames=0) on empty input
// or allocation failure.
//
// Layout: features[i] is the i-th token's column at offset i·D in the
// flat buffer, matching ggml's ne=(D, L) storage. The output preserves
// the same channel-major layout.
static inline float* repeat_interleave(const float* features, int D, int L, const int* durations, int* out_T_frames) {
    int T = 0;
    for (int i = 0; i < L; i++)
        T += durations[i];
    if (out_T_frames)
        *out_T_frames = T;
    if (T <= 0)
        return nullptr;
    float* out = (float*)std::malloc((size_t)D * T * sizeof(float));
    if (!out)
        return nullptr;
    int j = 0;
    for (int i = 0; i < L; i++) {
        const int n = durations[i];
        const float* col = features + (size_t)i * D;
        for (int k = 0; k < n; k++) {
            std::memcpy(out + (size_t)j * D, col, (size_t)D * sizeof(float));
            j++;
        }
    }
    return out;
}

} // namespace core_align
