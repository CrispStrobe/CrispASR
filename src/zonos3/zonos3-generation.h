// The generation registry model/src/zonos3/registries/generation.json (Python: zonos3.generation): the request
// defaults and limits every server shares, embedded at build time and parsed on first use.
#pragma once

#include <cstddef>

struct zonos3_generation_registry {
    int sample_rate; // the API's audio rate (Hz)
    struct {         // the v2 sampling defaults (servers/api/openapi.yaml repeats them)
        float temperature, top_p, min_p, repetition_penalty;
        int topk, repetition_window;
        int repetition_codebooks; // < 0 in a request means every codebook
    } sampling;
    int silence_prefill_default_frames; // server default (ZONOS3_SILENCE_PREFILL_FRAMES overrides)
    int silence_prefill_max_frames;     // the longest silence_prefill_frames a request may ask for
    int decode_context_frames;          // decode_context_frames unset or 0
    int reference_max_frames;           // the reference prefix; longer codes keep their first frames
    int max_frames;                     // the default --max frame budget
    int stream_codebooks_default;       // prefix streaming: codebooks a frame needs to decode
    bool stream_codebooks_first_only;   // ... on the first chunk only (scope "first") or every one
    double reference_audio_max_seconds, upload_max_seconds, edit_max_seconds; // the longest reference clip, upload
                                                                              // and speech-edit clip
    double output_limit_dbfs; // the soft limiter's knee after the output gain
};

const zonos3_generation_registry& zonos3_generation();

// The output stage's soft limiter (zonos3.tts.audio.output_pcm), in place, after the decoder's output gain: samples
// at or below the knee k = 10^(output_limit_dbfs / 20) pass unchanged, louder ones become
// sign(x) * (k + (1 - k) * tanh((|x| - k) / (1 - k))): slope 1 at the knee, bounded by full scale. Per sample.
void zonos3_soft_limit(float* pcm, size_t n);
