// src/core/hojo_asr_frames.h — weight-free frame arithmetic for the Hojo-ASR
// audio tower, plus the decode-side logit transform.
//
// Everything here is invisible to crispasr-diff: it decides SHAPES and the
// tiling schedule, not activation values, so a wrong answer shows up as a
// plausible-looking transcript computed over the wrong frames (HARD RULE 3b).
// It lives in its own header so `tests/test-hojo-asr-frames.cpp` can pin it
// without a model file.
//
// The conv stack is 3 x Conv2d(k=3, stride=2, pad=1), so in the time axis
// output frame `o` reads input frames [8o-7, 8o+7] and nothing else. That one
// fact is what makes the tiled conv schedule EXACT rather than approximate,
// and `tile_window()` below is its only consumer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace core_hojo_frames {

// Conv2d(k=3, stride=2, pad=1) output length.
inline int conv_out_len(int L) {
    return (L - 1) / 2 + 1;
}

// Time downsample of the whole 3-conv stem.
inline int conv_stem_out_len(int L) {
    return conv_out_len(conv_out_len(conv_out_len(L)));
}

// Number of input frames on each side of a kept output that the stem reads.
inline constexpr int kConvHalo = 8;

// `_get_feat_extract_output_lengths(L, n_window)` from
// hojo_asr/qwen3_omni_audioencoder.py, verbatim:
//
//   leave  = L % n_window
//   feat   = (leave - 1) // 2 + 1
//   out    = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (L // n_window) * ceil(n_window / 8)
//
// Python's `//` floors toward -inf, but every intermediate here is >= -1 and
// C's truncation agrees with flooring on -1/2 == 0 vs -1... it does NOT: in
// Python (-1)//2 == -1, in C -1/2 == 0. The only way to reach a negative
// numerator is leave == 0, where Python yields feat == 0 and C would yield 1.
// So the leave == 0 case is handled explicitly instead of relying on either.
inline int feat_output_len(int L, int n_window) {
    if (n_window <= 0)
        return 0;
    const int leave = L % n_window;
    const int per_window = (n_window + 7) / 8; // int(np.ceil(n_window / 8))
    const int whole = (L / n_window) * per_window;
    if (leave == 0)
        return whole;
    const int feat = (leave - 1) / 2 + 1;
    return ((feat - 1) / 2 + 1 - 1) / 2 + 1 + whole;
}

struct Chunk {
    int t0;    // first mel frame of the chunk
    int len;   // mel frames in the chunk
    int valid; // encoder frames the chunk contributes
};

struct ChunkPlan {
    std::vector<Chunk> chunks;
    int win_T = 0; // conv width: pad_sequence pads to the LONGEST chunk
    int T_enc = 0; // total encoder frames
};

// `forward()` in ModifyQwen3OmniMoeAudioEncoder: chunk_num = ceil(L / chunk_T),
// every chunk is chunk_T long except the tail, which takes L % chunk_T (or a
// full chunk when it divides exactly).
inline ChunkPlan plan_chunks(int T_mel, int chunk_T, int n_window_infer) {
    ChunkPlan p;
    if (T_mel <= 0 || chunk_T <= 0)
        return p;
    int num = (T_mel + chunk_T - 1) / chunk_T;
    if (num < 1)
        num = 1;
    int t0 = 0;
    for (int c = 0; c < num; c++) {
        Chunk ch;
        ch.t0 = t0;
        ch.len = (c == num - 1 && (T_mel % chunk_T) != 0) ? (T_mel % chunk_T) : chunk_T;
        ch.valid = 0;
        p.chunks.push_back(ch);
        t0 += ch.len;
        if (ch.len > p.win_T)
            p.win_T = ch.len;
    }
    const int cap = conv_stem_out_len(p.win_T);
    for (auto& ch : p.chunks) {
        ch.valid = feat_output_len(ch.len, n_window_infer);
        if (ch.valid > cap)
            ch.valid = cap;
        p.T_enc += ch.valid;
    }
    return p;
}

struct TileWindow {
    int mel_offset; // first mel frame of the window, RELATIVE to the chunk start
    int width;      // window width in mel frames
    int keep_from;  // first output frame of the window to keep
    int keep_count; // how many to keep
    int out_offset; // where they land in the chunk's output
};

// One tile of the conv schedule, covering output frames [o0, o1).
//
// The window is [8*o0 - 8, 8*o1 + 8) INTERSECTED with the real array
// [0, win_T). Both halves of that matter, and the intersection is the whole
// subtlety:
//
//  * The 8-frame halo covers every input a kept output reads, so in the
//    interior no kept output ever touches a padded position.
//  * The clamp is what makes the EDGES exact rather than merely plausible.
//    A first draft shifted the window to -8 and zero-filled the halo, on the
//    reasoning that the conv pads with zeros anyway. It does not: the conv
//    pads by injecting a literal zero vector at EACH level, whereas feeding
//    zeros as INPUT produces gelu(conv(0,0,0) + bias) = gelu(bias) at level 1
//    and propagates that non-zero value upward. The two differ exactly where
//    the untiled conv would have hit its own boundary -- the first and last
//    output frame of the chunk. Clamping puts the tile's boundary where the
//    full array's boundary is, so the same padding lands in the same place.
//
// mel_offset is always a multiple of 8 (either 0 or 8*(o0-1)), which is what
// lets keep_from be o0 - mel_offset/8, and what makes
// conv_stem_out_len(win_T - mel_offset) == conv_stem_out_len(win_T) - mel_offset/8.
inline TileWindow tile_window(int o0, int o1, int win_T) {
    int s = 8 * o0 - kConvHalo;
    if (s < 0)
        s = 0;
    int e = 8 * o1 + kConvHalo;
    if (e > win_T)
        e = win_T;
    TileWindow w;
    w.mel_offset = s;
    w.width = e - s;
    w.keep_from = o0 - s / 8;
    w.keep_count = o1 - o0;
    w.out_offset = o0;
    return w;
}

// transformers' RepetitionPenaltyLogitsProcessor, in place.
//   s = logits[t];  logits[t] = (s < 0) ? s * penalty : s / penalty
// Applied only to already-generated tokens: HOJO_ASR generates from
// inputs_embeds, so transformers' `input_ids` holds the generated suffix and
// nothing else — the BOS embedding and the speech frames are not tokens.
inline void apply_repetition_penalty(float* logits, int vocab, const int32_t* generated, int n_generated,
                                     float penalty) {
    if (!logits || penalty == 1.0f || n_generated <= 0)
        return;
    for (int i = 0; i < n_generated; i++) {
        const int t = generated[i];
        if (t < 0 || t >= vocab)
            continue;
        const float s = logits[t];
        logits[t] = (s < 0.0f) ? (s * penalty) : (s / penalty);
    }
}

// HOJO_ASR.infer:
//   max_new_tokens = min(cfg.max_new_tokens, T_enc * 2 + 10); then max(.., 10)
inline int max_new_for_frames(int cfg_max_new, int T_enc) {
    int n = T_enc * 2 + 10;
    if (cfg_max_new < n)
        n = cfg_max_new;
    if (n < 10)
        n = 10;
    return n;
}

} // namespace core_hojo_frames
