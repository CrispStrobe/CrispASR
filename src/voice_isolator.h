// Inworld Voice Isolator v1 (Apache-2.0): 48 kHz voice isolation.
// Blueprint: github.com/inworld-ai/voice-isolator (model.py / isolator.py),
// weights huggingface.co/inworld/voice-isolator, pinned by the converter.
//
// A band-split dual-path network over the STFT (n_fft 2048, hop 480): 46 band
// tokens per frame, seven blocks alternating time attention (rotary, exclusive
// self attention) + band attention with bidirectional Mamba along time + Mamba
// across bands; per-band heads predict a bounded complex mask plus a bounded
// residual that are added to the input spectrum. Output is the isolated voice;
// the removed background is input - voice (upstream's own definition).
//
// Two checkpoints share the runtime: `voice-isolator-v1` (offline, two-sided)
// and `voice-isolator-v1-realtime` (causal, trained look-ahead 0/2/4 frames).
// The realtime model runs here as a causal render: every frame sees only the
// past plus its look-ahead, exactly as the upstream network's causal forward.
// A push/pull streaming session (upstream streaming.Stream) is not implemented.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct voice_isolator_context;

struct voice_isolator_params {
    int n_threads; // <= 0: runtime default
    bool use_gpu;  // prefer a GPU; unsupported ops fall back to the CPU
    int gpu_device;
    int lookahead_frames;  // causal checkpoints only; < 0: the checkpoint's default (2 = 20 ms)
    float chunk_seconds;   // <= 0: 8 s (the offline checkpoint's trained chunk)
    float overlap_seconds; // < 0: 2 s
};

struct voice_isolator_result {
    int n_sources;  // 2: voice, background
    int n_channels; // 1 (mono, as upstream)
    int n_samples;  // exactly the input length, at sample_rate
    int sample_rate;
    float** sources;
    const char** source_names;
};

struct voice_isolator_params voice_isolator_default_params(void);
struct voice_isolator_context* voice_isolator_init_from_file(const char* path, struct voice_isolator_params params);
void voice_isolator_free(struct voice_isolator_context* ctx);

// Isolate the voice of `n_samples` frames of interleaved `in_channels` audio at
// the model rate (48 kHz). Channels are averaged to mono like upstream to_mono.
struct voice_isolator_result* voice_isolator_separate(struct voice_isolator_context* ctx, const float* pcm,
                                                      int n_samples, int in_channels);
void voice_isolator_result_free(struct voice_isolator_result* result);

// Convenience for ASR pre-processing: mono audio at any rate in, isolated voice
// at the same rate out (torchaudio kaiser_best resampling both ways, as
// upstream VoiceIsolator.isolate). Writes exactly n_samples floats. Returns 0 on success.
int voice_isolator_isolate_mono(struct voice_isolator_context* ctx, const float* pcm, int n_samples, int sample_rate,
                                float* out);

int voice_isolator_sample_rate(const struct voice_isolator_context* ctx);
int voice_isolator_n_sources(const struct voice_isolator_context* ctx);
const char* voice_isolator_source_name(const struct voice_isolator_context* ctx, int idx);
int voice_isolator_is_causal(const struct voice_isolator_context* ctx);
const char* voice_isolator_backend_name(const struct voice_isolator_context* ctx);

// Stage diff against a tools/reference_backends/voice_isolator.py archive.
// Runs the chained forward from the reference's input_48k and reports cosine
// and norms per stage, plus a teacher-forced pass (each block fed the
// reference's previous block). Returns 0 when every stage passes the gate for
// the model's weight type (F32 > 0.9999, F16 > 0.999, Q8_0 > 0.99).
int voice_isolator_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif
