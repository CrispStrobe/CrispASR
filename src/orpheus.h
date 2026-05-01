#pragma once

// Orpheus public C ABI.
//
// Orpheus is canopylabs/orpheus-3b-0.1-ft (Llama-3.2-3B-Instruct
// finetuned, llama3.2 license — "Built with Llama") + the SNAC 24 kHz
// codec (hubertsiuzdak/snac_24khz, MIT). Same architecture covers the
// SebastianBodza/Kartoffel_Orpheus DE finetunes and the lex-au
// language-specific Q8_0 GGUFs as drop-in checkpoint swaps.
//
// The talker emits a stream of <custom_token_N> LM tokens; every 7
// emitted tokens form one "super-frame" that de-interleaves into 1
// codes_0 / 2 codes_1 / 4 codes_2 entries (from
// orpheus_tts_pypi/orpheus_tts/decoder.py:convert_to_audio). 4
// super-frames cover 16 SNAC frames (× 512-sample hop = 8192 PCM
// samples at 24 kHz); the streaming protocol emits the middle 2048
// samples of each 4-super-frame sliding window.
//
// Status: talker-side wired up; SNAC decoder C++ side is the next
// slice. orpheus_synthesize_codes works once SNAC weights are loaded
// via orpheus_set_codec_path; orpheus_synthesize is gated on the
// codec landing.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct orpheus_context;

struct orpheus_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy
    int max_audio_tokens; // upper bound on AR decode steps; 0 = use built-in default (8192)
};

struct orpheus_context_params orpheus_context_default_params(void);

// Initialise from the talker LM GGUF (arch="orpheus", produced by
// `models/convert-orpheus-to-gguf.py`).
struct orpheus_context* orpheus_init_from_file(const char* path_model, struct orpheus_context_params params);

// Point the runtime at the SNAC codec GGUF (cstr/snac-24khz-GGUF — to
// be published; for now the SNAC codec runs from the canonical
// hubertsiuzdak/snac_24khz cache directory at the Python-reference
// rate). Required before the first orpheus_synthesize call. Returns 0
// on success.
int orpheus_set_codec_path(struct orpheus_context* ctx, const char* path);

// Returns the number of fixed speakers baked into the GGUF metadata
// (0 if the model isn't a fixed-speaker variant). Pass into
// orpheus_get_speaker_name to enumerate them.
int orpheus_n_speakers(struct orpheus_context* ctx);

// Returns the i-th fixed speaker name. Buffer is owned by ctx; do not
// free. Returns nullptr for out-of-range indices.
const char* orpheus_get_speaker_name(struct orpheus_context* ctx, int i);

// Select a fixed Orpheus speaker by NAME. The Orpheus prompt convention
// is the literal `f"{name}: {text}"` prefix — there is no
// embedding-table dispatch (in contrast to Qwen3-TTS-CustomVoice).
// Returns 0 on success, -2 if the name is unknown.
int orpheus_set_speaker_by_name(struct orpheus_context* ctx, const char* name);

// Returns true if the loaded model is a fixed-speaker variant
// (orpheus.tts_model_type == "fixed_speaker"). Default for canopylabs
// orpheus-3b-0.1-ft and the Kartoffel finetunes.
int orpheus_is_fixed_speaker(struct orpheus_context* ctx);

// Synthesise text → 24 kHz mono float32 PCM. Caller frees with
// `orpheus_pcm_free`. *out_n_samples is set on success.
//
// Returns nullptr until the SNAC C++ decoder lands. The current
// implementation can drive the talker AR loop end-to-end and emit
// custom_token_N LM token IDs via orpheus_synthesize_codes; the
// C++ SNAC half is in flight (slice b).
float* orpheus_synthesize(struct orpheus_context* ctx, const char* text, int* out_n_samples);

// Run the talker on `text`, AR-decode until the audio_end token (or
// max_audio_tokens), and return the LM token IDs that fell in the
// <custom_token_N> codec block. *out_n is set to the count. Caller
// frees with `orpheus_codes_free`.
//
// These codes are valid SNAC inputs once de-interleaved per the 7-slot
// super-frame layout — you can render them to audio via the canonical
// SNAC python decoder (tools/reference_backends/orpheus_snac.py).
int32_t* orpheus_synthesize_codes(struct orpheus_context* ctx, const char* text, int* out_n);

void orpheus_codes_free(int32_t* codes);
void orpheus_pcm_free(float* pcm);

void orpheus_free(struct orpheus_context* ctx);

void orpheus_set_n_threads(struct orpheus_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
