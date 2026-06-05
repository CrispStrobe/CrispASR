// src/core/rvq.h — Euclidean residual vector quantization encode (CPU).
//
// Hoists the iterative argmin loop that kyutai_stt and mimo_tokenizer both
// need: for each codebook stage, find the nearest entry to the running
// residual, append the index, and subtract the matched centroid before
// moving on to the next stage.
//
// Both consumers store codebook embeddings as ggml tensors with
// ne[0]=codebook_dim and ne[1]=codebook_size (the GGUF on-disk layout).
// This helper takes them as already-extracted F32 row-major (K, D) and
// pre-computed ||embed[k]||^2 buffers, which avoids re-reading the GGUF
// memory on every call. Callers stage that conversion once at init time
// (F16 codebooks are promoted to F32 on first use, matching upstream's
// `quantizer.float()` cast).

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace core_rvq {

// Per-stage codebook: F32 row-major (codebook_size, dim) plus cached
// ||embed[k]||^2 (codebook_size,). The pre-computed norms turn the
// inner argmin into a 2 x·E[k] − ||E[k]||² shootout that skips the
// constant ||x||² term.
struct Codebook {
    const float* embed = nullptr;         // (codebook_size, dim) row-major
    const float* embed_norm_sq = nullptr; // (codebook_size,)
    int codebook_size = 0;
    int dim = 0;
};

// Encode `T` D-dimensional vectors against `n_stages` codebooks.
//
//   features : (T, dim) row-major; not mutated. The helper makes its own
//              residual copy.
//   stages   : pointer to Codebook[n_stages]. All codebooks must share
//              the same `dim`, which must equal the feature dim.
//   codes    : output, sized T * n_stages, row-major (T, n_stages).
//
// Returns true on success, false if any codebook is malformed.
bool encode_euclidean(const float* features, int T, int dim, const Codebook* stages, int n_stages, int32_t* codes_out);

} // namespace core_rvq
