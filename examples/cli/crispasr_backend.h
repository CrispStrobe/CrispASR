// crispasr_backend.h — abstract backend interface for the unified crispasr CLI.
//
// Each model in src/ (parakeet, canary, cohere, qwen3-asr, voxtral, voxtral4b,
// granite_speech) is wrapped by a backend that converts its native result type
// into the common crispasr_segment vector. The whisper backend is a thin
// adapter over whisper_full_parallel() that reads whisper_context segments out
// into the same vector.
//
// The main CLI (cli.cpp) parses args into whisper_params, then either takes
// the historical whisper code path (when params.backend == "" or "whisper")
// or dispatches to crispasr_run_backend() which drives the pipeline:
//   load audio -> VAD slice -> backend->transcribe() -> write outputs.

#pragma once

#include "crispasr_speaker_identity_models.h" // declared_speaker_identity() — the researched verdicts

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declaration — defined in cli.cpp. We intentionally reuse the
// existing whisper_params struct (extended with a few new fields) instead of
// introducing a parallel crispasr_params, so users keep the same interface
// they already know from crispasr.
struct whisper_params;

// ---------------------------------------------------------------------------
// Common result types
// ---------------------------------------------------------------------------

struct crispasr_token_alt {
    std::string text;
    float prob = 0.0f; // probability [0,1]
    int32_t id = -1;
};

struct crispasr_token {
    std::string text;
    float confidence = -1.0f; // [0,1], -1 if unavailable
    int64_t t0 = -1;          // centiseconds, absolute; -1 if unavailable
    int64_t t1 = -1;
    int32_t id = -1;                      // backend-specific token id, -1 if unavailable
    int64_t t_dtw = -1;                   // whisper DTW token time, -1 if unused
    bool is_special = false;              // whisper: token id >= eot; skipped by wts
    std::vector<crispasr_token_alt> alts; // top-N alternatives (--alt mode)
};

struct crispasr_word {
    std::string text;
    int64_t t0 = 0; // centiseconds, absolute
    int64_t t1 = 0;
};

struct crispasr_segment {
    std::string text;
    int64_t t0 = 0; // centiseconds, absolute
    int64_t t1 = 0;
    std::string speaker;            // empty if no diarization
    bool speaker_turn_next = false; // whisper tinydiarize
    // #292: which chunk/slice this segment came from, or -1 for a single-pass
    // (unchunked) run. Speaker labels like "(speaker 1)" are CHUNK-LOCAL — they
    // restart per chunk — so a consumer needs chunk_id to tell "same speaker,
    // same chunk" (continuity) from "speaker 1 in chunk 0" vs "speaker 1 in
    // chunk 2" (which may be different people). Only set when there is >1 chunk.
    int chunk_id = -1;
    std::vector<crispasr_word> words;   // may be empty
    std::vector<crispasr_token> tokens; // may be empty
    // Multi-task ASR metadata (SenseVoice and similar). Empty when the
    // backend doesn't emit them. There is deliberately no `emotion` field:
    // inferring emotions from voice is an emotion recognition system under
    // EU AI Act Art. 3(39) — prohibited in workplace/education (Art. 5(1)(f))
    // and high-risk otherwise (Annex III(1)(c)). See docs/eu-ai-act.md.
    std::string lang_id;     // e.g. "en", "zh", "ja"
    std::string audio_event; // e.g. "Speech", "Music"
    std::string itn_flag;    // "withitn" or "woitn"
};

struct crispasr_ctc_logits {
    int n_frames = 0;
    int n_vocab = 0;
    std::vector<float> data;        // frame-major: data[t * n_vocab + v]
    std::string normalization;      // "logits" or "log_probs"
    std::vector<std::string> vocab; // optional raw token pieces indexed by id
};

// ---------------------------------------------------------------------------
// Capability bitmask
// ---------------------------------------------------------------------------

enum crispasr_capability : uint32_t {
    CAP_TIMESTAMPS_NATIVE = 1u << 0,    // model produces segment timestamps natively
    CAP_TIMESTAMPS_CTC = 1u << 1,       // can use CTC aligner for timestamps
    CAP_WORD_TIMESTAMPS = 1u << 2,      // word-level timestamps available
    CAP_TOKEN_CONFIDENCE = 1u << 3,     // per-token probability
    CAP_LANGUAGE_DETECT = 1u << 4,      // auto language detection
    CAP_TRANSLATE = 1u << 5,            // speech translation
    CAP_DIARIZE = 1u << 6,              // speaker diarization
    CAP_GRAMMAR = 1u << 7,              // GBNF grammar constraints
    CAP_TEMPERATURE = 1u << 8,          // temperature/sampling control
    CAP_BEAM_SEARCH = 1u << 9,          // beam search
    CAP_FLASH_ATTN = 1u << 10,          // flash attention toggle
    CAP_PUNCTUATION_TOGGLE = 1u << 11,  // can enable/disable punctuation
    CAP_SRC_TGT_LANGUAGE = 1u << 12,    // separate source/target language (canary)
    CAP_AUTO_DOWNLOAD = 1u << 13,       // supports -m auto via HF hub
    CAP_PARALLEL_PROCESSORS = 1u << 14, // whisper-style n_processors
    CAP_VAD_INTERNAL = 1u << 15,        // backend handles VAD internally (whisper)
    CAP_TTS = 1u << 16,                 // text-to-speech synthesis
    CAP_S2S = 1u << 21,                 // speech-to-speech (audio in → audio out)
    CAP_VOICE_CLONING = 1u << 17,       // TTS: synthesise with --voice <reference.wav>
    CAP_PUNCTUATION_NATIVE = 1u << 18,  // backend already emits punctuation by default
    CAP_UNBOUNDED_INPUT = 1u << 19,     // encoder handles arbitrary-length audio without chunking
                                        // (FastConformer, CTC-only encoders). LLM-based backends
                                        // and whisper's fixed-window encoder do NOT set this.
    CAP_SEPARATE = 1u << 22,            // audio source separation (stems)
    CAP_INTERNAL_CHUNKING = 1u << 20,   // backend handles its own long-audio chunking internally
                                        // (PLAN #104: parakeet uses chunked-encode + single-decode).
                                        // Skip the crispasr_run.cpp auto-chunk fallback for these.
    CAP_STREAMING = 1u << 23,           // backend supports true token-level streaming output
    CAP_PITCH = 1u << 24,               // monophonic F0 / pitch-track estimation (audio in ->
                                        // pitch frames out). Like CAP_SEPARATE this is a task
                                        // marker for --list-backends, NOT a transcribe() path:
                                        // routing happens in the --pitch dispatcher.
    CAP_CHORDS = 1u << 25,              // chord recognition (audio in -> chord timeline out).
                                        // Same task-marker role as CAP_PITCH/CAP_SEPARATE;
                                        // routing happens in the --chords dispatcher.
    CAP_PIANO = 1u << 27,               // polyphonic piano transcription (audio in -> note
                                        // events out). Task marker like CAP_CHORDS/CAP_PITCH;
                                        // routing happens in the --piano dispatcher.
    CAP_BEATS = 1u << 26,               // beat / downbeat tracking (audio in -> beat grid out).
                                        // Same task-marker role as CAP_CHORDS; routing happens
                                        // in the --beats dispatcher.
    CAP_TAB = 1u << 28,                 // guitar tablature emission scoring (audio in ->
                                        // per-frame per-string fret SCORES out). Task marker
                                        // like CAP_CHORDS; routing happens in the --tab
                                        // dispatcher. Note this backend emits scores, not a
                                        // decided tablature: the constrained Viterbi/DP that
                                        // picks a playable fingering belongs to the caller.
    CAP_STREAM_DELTA = 1u << 29,        // supports streaming delta encoding
    CAP_STREAM_UTTERANCE = 1u << 30, // utterance-level delta (save/restore)
                                        // (cohere_set_stream_delta). Avoids re-encoding
                                        // the overlapping portion of sequential rolling
                                        // windows in JSON+VAD streaming mode.
};

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

class CrispasrBackend {
public:
    virtual ~CrispasrBackend() = default;

    // Human-readable name ("whisper", "parakeet", "canary", ...).
    virtual const char* name() const = 0;

    // Bitmask of crispasr_capability flags.
    virtual uint32_t capabilities() const = 0;

    // Load the model and prepare internal state. Returns false on failure.
    // Params are passed by const-ref — backends should only read the fields
    // they care about.
    virtual bool init(const whisper_params& params) = 0;

    // Sample rate the backend expects for input PCM (default 16000).
    // The CLI loads audio at this rate via crispasr_audio_load_at_rate,
    // avoiding the lossy down-then-up resample for non-16 kHz backends.
    virtual int input_sample_rate() const { return 16000; }

    // Transcribe a single audio slice of mono PCM samples at the rate
    // returned by input_sample_rate() (16 kHz unless overridden).
    // t_offset_cs is the absolute start of this slice in centiseconds; all
    // returned segment/word/token timestamps must be absolute (include the
    // offset).
    virtual std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                                     const whisper_params& params) = 0;

    // Optional per-frame dense CTC grid captured by the most recent transcribe()
    // when params.return_logits is true. Backends that do not produce a dense
    // CTC grid return nullptr. The pointer is owned by the backend and valid
    // until the next transcribe() / shutdown().
    virtual const crispasr_ctc_logits* last_ctc_logits() const { return nullptr; }

    // ---- Optional split transcribe (encode ∥ decode pipelining) ----
    //
    // Encoder-decoder backends typically run the encoder on the GPU and the
    // decoder on the CPU. Processing N slices with transcribe() serialises the
    // two, leaving one processor idle at all times. A backend that can hand out
    // its intermediate encoder state lets the caller overlap the encode of
    // slice N+1 with the decode of slice N — the same throughput win as a
    // worker pool, without the N-times model memory.
    //
    // Contract: encode_slice() may be called on a WORKER thread while
    // decode_slice() runs on another, so a backend may only advertise this when
    // its encode and decode paths touch disjoint state. Calls are issued in
    // slice order and decodes happen in the same order. The handle is opaque
    // and must be consumed by exactly one decode_slice(), which frees it;
    // release_encoded() frees an unconsumed handle on an error path.
    struct encoded_slice {
        void* h = nullptr; // backend-owned; null = encode failed
    };
    virtual bool supports_split_transcribe() const { return false; }
    // Whether THIS slice length can go through encode_slice/decode_slice. A
    // backend may route long inputs through a multi-window path that the split
    // pair does not reproduce. The caller must check every slice up front and
    // fall back to the fully sequential path if any slice says no — mixing
    // transcribe() into a running pipeline would encode on two threads at once.
    virtual bool can_split_slice(int /*n_samples*/, const whisper_params& /*params*/) const { return false; }
    virtual encoded_slice encode_slice(const float* /*samples*/, int /*n_samples*/, const whisper_params& /*params*/) {
        return {};
    }
    virtual std::vector<crispasr_segment> decode_slice(encoded_slice /*enc*/, int64_t /*t_offset_cs*/,
                                                       const whisper_params& /*params*/) {
        return {};
    }
    virtual void release_encoded(encoded_slice /*enc*/) {}
    // Called on the CALLER's thread once before a run of encode_slice/
    // decode_slice pairs, never concurrently with them.
    //
    // transcribe() typically applies per-call settings (sampling, beam,
    // hotwords, attention context) to backend state on every call; the split
    // pair cannot, because encode_slice runs on a worker thread and mutating
    // decode state there would race the decoder. Without this hook those flags
// Optional streaming delta-encoding hint.  When the caller is about to
    // transcribe sequential overlapping audio windows (e.g. JSON+VAD
    // streaming), it may call this before each transcribe() to convey how
    // many samples in the next buffer are NEW (not present in the previous
    // call).  A backend that supports delta encoding (CAP_STREAM_DELTA) can
    // use this to avoid re-encoding the overlapping portion of the audio.
    // Default no-op.
    virtual void set_stream_delta(int /*delta_new_samples*/) {}
    virtual void save_utterance_cross_kv(int64_t /*utterance_start*/, int64_t /*utterance_end*/, int64_t /*window_start*/, int /*sample_rate*/, int64_t /*generation*/) {}
    virtual void restore_utterance_cross_kv(int /*n_new_samples*/, int64_t /*generation*/) {}
    virtual void clear_utterance_cross_kv() {}

    // ---- Language detection ----
    // are silently dropped whenever the pipeline engages — measured on
    // parakeet: `--vad --beam-size 4` decoded greedily.
    virtual void begin_split_run(const whisper_params& /*params*/) {}
    // Post-decode repair that needs the model AND the raw audio, so it cannot
    // run inside decode_slice: it re-encodes, and the producer is still
    // encoding at that point. The caller invokes it after the pipeline has
    // joined, before the segments are trimmed and stored, which is where
    // transcribe() applies it. Default is no-op.
    virtual void repair_slice(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                              std::vector<crispasr_segment>& /*segs*/, const whisper_params& /*params*/) {}

    // Optional stereo-aware overload for backends that can split stereo
    // channels for diarization (currently: whisper). Default
    // implementation falls through to mono transcribe(); override when
    // your backend can use stereo. The two channel buffers each have
    // n_samples_per_channel elements at 16 kHz.
    virtual std::vector<crispasr_segment> transcribe_stereo(const float* left_samples, const float* right_samples,
                                                            int n_samples_per_channel, int64_t t_offset_cs,
                                                            const whisper_params& params) {
        // Mono fallback: average L+R into a temporary buffer and dispatch
        // through the main transcribe(). Backends that don't override
        // this method get sane behaviour without any extra wiring.
        std::vector<float> mono((size_t)n_samples_per_channel);
        for (int i = 0; i < n_samples_per_channel; i++) {
            mono[(size_t)i] = 0.5f * (left_samples[i] + right_samples[i]);
        }
        return transcribe(mono.data(), n_samples_per_channel, t_offset_cs, params);
    }

    // TTS: synthesize speech from text. Returns mono PCM samples at the
    // backend's native rate (see `tts_sample_rate()`). Default returns empty
    // (not supported). Only backends with CAP_TTS override.
    virtual std::vector<float> synthesize(const std::string& text, const whisper_params& /*params*/) {
        (void)text;
        return {};
    }

    // Streaming TTS callback: invoked once per generated PCM chunk as the
    // backend produces audio, with the last chunk flagged is_final=true.
    // `pcm` is mono float32 at `tts_sample_rate()`.
    using crispasr_pcm_stream_callback = std::function<void(const float* pcm, int n_samples, bool is_final)>;

    // Synthesize with streaming output. Default implementation falls back to
    // the whole-clip synthesize() and emits it as a single final chunk.
    // Backends with CAP_STREAMING override to emit incrementally.
    virtual void synthesize_streaming(const std::string& text, const whisper_params& p,
                                      crispasr_pcm_stream_callback cb) {
        auto v = synthesize(text, p);
        if (!v.empty())
            cb(v.data(), (int)v.size(), true);
    }

    // Whose voice this backend's BUILT-IN preset voices belong to.
    //
    // The default is a lookup in crispasr_speaker_identity_models.h, keyed on
    // (backend name, checkpoint) — because one backend serves many checkpoints
    // with different answers (`orpheus` runs both Canopy's base model and
    // Kartoffel's German fine-tune), and because these are research results
    // that belong in one reviewable table rather than scattered across 50
    // adapters. A backend only overrides this if it can do better from its own
    // loaded metadata.
    //
    // Unresearched models resolve to Unknown, and that claims nothing: it means
    // nobody has read the provider's card yet, not that the voice is synthetic.
    // Guessing Synthetic is the costly direction — it silently removes an
    // Art. 50(4) disclosure.
    //
    // A RealPerson preset is disclosed but NOT consent-gated; see
    // crispasr_speaker_identity.h for why those are different duties.
    //
    // A pack or bank entry that declares its own crispasr.voice.speaker_identity
    // outranks this, and --speaker-identity outranks both.
    virtual crispasr_voice::SpeakerIdentity declared_speaker_identity(const std::string& model_path) const {
        return crispasr_voice::identity_for_model(name(), model_path);
    }

    // Path to the multi-voice BANK this backend selects `--voice` entries from,
    // or empty when `--voice` names a file directly (the usual case).
    //
    // This exists for the voice-clone gate, not for synthesis. cosyvoice3 keeps
    // every voice inside one bundle discovered as a sibling of the model, so
    // `--voice fleurs-en` names no file on disk; the gate read no metadata and
    // classified a zero-shot voice clone as a preset, on every surface. The
    // backend is the only thing that knows which bundle it resolved, so it has
    // to hand the path over. See crispasr_voice_provenance.h.
    //
    // Any future backend that selects voices by name from a container MUST
    // override this, or its clones ship unattested and undisclosed.
    virtual std::string voice_bank_path() const { return {}; }

    // Sample rate of `synthesize()` output PCM. Defaults to 24 kHz since most
    // TTS backends (kokoro, qwen3-tts, vibevoice, chatterbox, orpheus, indextts)
    // produce 24 kHz. Backends that emit a different rate (e.g. voxcpm2-tts at
    // 48 kHz) override this.
    virtual int tts_sample_rate() const { return 24000; }

    // S2S: speech-to-speech transform. Takes 16 kHz mono PCM input, returns
    // PCM output at `tts_sample_rate()`. If out_text is non-null, writes the
    // intermediate transcript. Default returns empty (not supported). Only
    // backends with CAP_S2S override.
    virtual std::vector<float> speech_to_speech(const float* samples, int n_samples, std::string* out_text,
                                                const whisper_params& /*params*/) {
        (void)samples;
        (void)n_samples;
        (void)out_text;
        return {};
    }

    // Text-to-text translation. m2m100 and any future translate-only
    // backend overrides this. Default returns empty (not supported).
    // src_lang / tgt_lang are ISO-639-1 codes ("en", "de", …).
    virtual std::string translate_text(const std::string& text, const std::string& src_lang,
                                       const std::string& tgt_lang, const whisper_params& /*params*/) {
        (void)text;
        (void)src_lang;
        (void)tgt_lang;
        return {};
    }

    // Whether the backend should auto-enable VAD for long audio when the
    // user didn't explicitly set --chunk-seconds or --vad. Backends whose
    // encoder degenerates on arbitrary-length chunks (e.g. parakeet-ja)
    // override this to get silence-bounded segments that match training.
    virtual bool prefers_vad() const { return false; }

    // Maximum VAD slice duration (seconds) the backend can decode reliably
    // in one pass; 0 = unbounded. VAD merges continuous speech into slices
    // as long as the speech runs (40 s+ on podcast audio) — issue #89:
    // parakeet-ja's encoder collapses past ~12 s, so slices are re-split at
    // energy minima down to this cap before transcription. Only applied on
    // the VAD path when the user didn't pass an explicit --chunk-seconds.
    virtual int vad_slice_cap_seconds() const { return 0; }

    // ISO-639-1 code of the ONLY language this backend can produce, or nullptr
    // if it is multilingual / language-agnostic (the default). A monolingual
    // backend (e.g. moonshine, English-only) returns its code so the CLI can
    // skip external language-ID on `-l auto` — running a whisper-tiny LID pass
    // (and downloading `ggml-tiny.bin`) to "detect" the language of a backend
    // that can only ever emit one is pointless (#227). Must NOT be set by a
    // multilingual backend — that would wrongly force its output language.
    virtual const char* sole_language() const { return nullptr; }

    // Detect the spoken language using the ALREADY-LOADED model, when the
    // backend can do so without a second one. Returns false (the default) when
    // it cannot, and the caller falls back to external LID (whisper-tiny etc.).
    //
    // This is NOT the same as CAP_LANGUAGE_DETECT. That cap means "detection
    // happens inside transcribe(), the CLI need not resolve a language at all";
    // declaring it on a backend that in fact needs a language in its prompt
    // leaves `-l auto` unresolved. This hook is for the opposite case: the
    // backend REQUIRES a language but can work one out from its own weights.
    //
    // A backend whose language set is smaller than the external detector's also
    // has a correctness reason to prefer this: external LID can return a
    // language the model does not support, which is unfixable after the fact.
    virtual bool detect_language(const float* /*samples*/, int /*n_samples*/, const whisper_params& /*params*/,
                                 std::string& /*out_lang*/, float& /*out_confidence*/) {
        return false;
    }

    // Streaming transcription callback type.
    // Called with partial text (empty string counts as keep-alive)
    // and is_final flag. When is_final is true, partial_text is the
    // complete final result.
    using crispasr_stream_callback = std::function<void(const std::string& partial_text, bool is_final)>;

    // Transcribe with streaming output. Default implementation falls back
    // to non-streaming transcribe().
    virtual void transcribe_streaming(const float* samples, int n_samples, int64_t t_offset_cs,
                                      const whisper_params& params, crispasr_stream_callback on_text) {
        (void)samples;
        (void)n_samples;
        (void)t_offset_cs;
        (void)params;
        (void)on_text;
        // Fallback: run non-streaming, then push result at once.
        auto segments = transcribe(samples, n_samples, t_offset_cs, params);
        std::string full;
        for (const auto& seg : segments) {
            if (!seg.text.empty()) {
                full += seg.text;
            }
        }
        if (!full.empty()) {
            on_text(full, false); // partial
        }
        on_text(full, true); // final
    }

    // Warmup: run a short dummy transcribe to amortize first-call
    // overhead (graph allocation, GPU kernel compilation, gallocr shape
    // setup).  Called once after init(), before the first real audio.
    // Default is a no-op.  Backends override when the first-call cost
    // is user-visible (50-200 ms on GPU, <5 ms on CPU — worth it for
    // server mode and the Python Session API).  PLAN #80e.
    virtual void warmup() {}

    // Release all resources.
    virtual void shutdown() = 0;
};

// ---------------------------------------------------------------------------
// Factory + auto-detection
// ---------------------------------------------------------------------------

// Create a backend by name. Returns nullptr if the name is not recognised or
// the backend was not compiled in. Caller owns the returned pointer.
std::unique_ptr<CrispasrBackend> crispasr_create_backend(const std::string& name);

// Detect the backend from GGUF metadata. Reads the "general.architecture"
// key using gguf_init_from_file() and maps it to a backend name. Returns
// an empty string if detection fails.
std::string crispasr_detect_backend_from_gguf(const std::string& model_path);

// True if the GGUF at model_path is a pure-CTC FastConformer model (no RNN-T
// decoder/joint tensors) — i.e. a parakeet-ctc / stt_*_fastconformer_ctc that must
// run on the fastconformer-ctc backend, not the parakeet (transducer) backend.
// Cheap: reads tensor infos only (no weights). Returns false on any read failure.
bool crispasr_gguf_is_pure_ctc(const std::string& model_path);

// List the backend names that were compiled into this binary.
std::vector<std::string> crispasr_list_backends();

// Print a human-readable capability matrix for the available backends.
// Called by --list-backends in cli.cpp.
void crispasr_print_backend_matrix();

// Print the same capability matrix as JSON for tooling consumption.
// Called by --list-backends-json. Output shape:
//   {"backends":[
//     {"name":"voxtral4b",
//      "caps_bitmask":12345,
//      "caps":["timestamps-ctc","auto-download","temperature",...]},
//     ...]}
void crispasr_print_backend_matrix_json();

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

// Drive the non-whisper pipeline end-to-end: resolve model path, create
// backend, load audio, segment via VAD (or fixed chunks), transcribe,
// print to stdout, write output files. Returns a process exit code.
//
// Invoked from cli.cpp main() when params.backend is set to a non-whisper
// backend. The whisper path in cli.cpp is left completely untouched.
int crispasr_run_backend(const whisper_params& params);
