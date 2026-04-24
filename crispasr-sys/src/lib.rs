//! Raw FFI bindings to CrispASR (whisper.cpp fork).
//! Mirrors the public C API in include/whisper.h.

use std::ffi::{c_char, c_float, c_int, c_void};

/// Opaque context handle.
#[repr(C)]
pub struct WhisperContext(c_void);

/// Opaque state handle.
#[repr(C)]
pub struct WhisperState(c_void);

/// Opaque params handle (allocated by whisper_full_default_params_by_ref).
#[repr(C)]
pub struct WhisperFullParams(c_void);

/// Opaque context params handle.
#[repr(C)]
pub struct WhisperContextParams(c_void);

/// Sampling strategy.
pub const WHISPER_SAMPLING_GREEDY: c_int = 0;
pub const WHISPER_SAMPLING_BEAM_SEARCH: c_int = 1;

extern "C" {
    // --- Lifecycle ---
    pub fn whisper_init_from_file_with_params(
        path: *const c_char,
        params: *const WhisperContextParams,
    ) -> *mut WhisperContext;

    pub fn whisper_context_default_params_by_ref() -> *mut WhisperContextParams;
    pub fn whisper_free(ctx: *mut WhisperContext);
    pub fn whisper_free_params(params: *mut WhisperFullParams);
    pub fn whisper_free_context_params(params: *mut WhisperContextParams);

    // --- Inference ---
    pub fn whisper_full(
        ctx: *mut WhisperContext,
        params: *const WhisperFullParams,
        samples: *const c_float,
        n_samples: c_int,
    ) -> c_int;

    pub fn whisper_full_default_params_by_ref(
        strategy: c_int,
    ) -> *mut WhisperFullParams;

    // --- Results ---
    pub fn whisper_full_n_segments(ctx: *mut WhisperContext) -> c_int;

    pub fn whisper_full_get_segment_text(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> *const c_char;

    pub fn whisper_full_get_segment_t0(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> i64;

    pub fn whisper_full_get_segment_t1(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> i64;

    pub fn whisper_full_get_segment_no_speech_prob(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> c_float;

    // --- Language ---
    pub fn whisper_full_lang_id(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_lang_str(id: c_int) -> *const c_char;
    pub fn whisper_lang_id(lang: *const c_char) -> c_int;

    // --- 0.4.2: VAD + tdrz setters on whisper_full_params ---
    pub fn crispasr_params_set_vad(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_vad_model_path(
        p: *mut WhisperFullParams,
        path: *const c_char,
    );
    pub fn crispasr_params_set_vad_threshold(
        p: *mut WhisperFullParams,
        threshold: c_float,
    );
    pub fn crispasr_params_set_vad_min_speech_ms(
        p: *mut WhisperFullParams,
        ms: c_int,
    );
    pub fn crispasr_params_set_vad_min_silence_ms(
        p: *mut WhisperFullParams,
        ms: c_int,
    );
    pub fn crispasr_params_set_tdrz(p: *mut WhisperFullParams, v: c_int);
}

// =========================================================================
// Unified session FFI (CrispASR 0.4.0+) — multi-backend dispatch
// =========================================================================
//
// Open any CrispASR-supported GGUF (Whisper, Parakeet, Canary, Cohere,
// Qwen3-ASR, Granite Speech, FastConformer-CTC, Canary-CTC, Voxtral,
// Voxtral4B, Wav2Vec2) through one handle. Backend auto-detected from
// `general.architecture` metadata unless overridden.

/// Opaque handle returned by `crispasr_session_open`.
#[repr(C)]
pub struct CrispasrSession(c_void);

/// Opaque result handle returned by `crispasr_session_transcribe`.
/// Must be freed with `crispasr_session_result_free`.
#[repr(C)]
pub struct CrispasrSessionResult(c_void);

/// Opaque result handle for `crispasr_align_words_abi`. Must be freed
/// with `crispasr_align_result_free`.
#[repr(C)]
pub struct CrispasrAlignResult(c_void);

/// Tunables for [`crispasr_session_transcribe_vad`]. Mirrors whisper.cpp's
/// `whisper_vad_params` plus the max-chunk fallback used to bound encoder
/// cost on long audio. Pass a null pointer to use defaults.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrVadAbiOpts {
    pub threshold: c_float,
    pub min_speech_duration_ms: c_int,
    pub min_silence_duration_ms: c_int,
    pub speech_pad_ms: c_int,
    pub chunk_seconds: c_int,
    pub n_threads: c_int,
}

impl Default for CrispasrVadAbiOpts {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            chunk_seconds: 30,
            n_threads: 4,
        }
    }
}

/// ABI segment for [`crispasr_diarize_segments_abi`]. Caller fills
/// `t0_cs` / `t1_cs`; the diarizer writes `speaker` (-1 if unassigned).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrDiarizeSegAbi {
    pub t0_cs: i64,
    pub t1_cs: i64,
    pub speaker: c_int,
    pub _pad: c_int,
}

/// ABI options for [`crispasr_diarize_segments_abi`]. `method` is a
/// value in 0..3: 0 = Energy, 1 = Xcorr, 2 = VadTurns, 3 = Pyannote.
/// `pyannote_model_path` is required for Pyannote, ignored otherwise.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrDiarizeOptsAbi {
    pub method: c_int,
    pub n_threads: c_int,
    pub slice_t0_cs: i64,
    pub pyannote_model_path: *const c_char,
}

extern "C" {
    pub fn crispasr_session_open(
        model_path: *const c_char,
        n_threads: c_int,
    ) -> *mut CrispasrSession;

    pub fn crispasr_session_open_explicit(
        model_path: *const c_char,
        backend_name: *const c_char,
        n_threads: c_int,
    ) -> *mut CrispasrSession;

    pub fn crispasr_session_backend(s: *mut CrispasrSession) -> *const c_char;

    /// Write a comma-separated list of backend names the loaded dylib
    /// was built with. Returns the number of bytes written (not counting
    /// NUL) or a negative error.
    pub fn crispasr_session_available_backends(
        out_csv: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

    pub fn crispasr_session_transcribe(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
    ) -> *mut CrispasrSessionResult;

    /// 0.4.9+: language-aware session transcribe. `language` is an
    /// ISO 639-1 code or null/empty to keep the backend's historical
    /// default. Backends that accept a source-language hint (whisper,
    /// canary, cohere, voxtral, voxtral4b) honour it; others ignore
    /// silently.
    pub fn crispasr_session_transcribe_lang(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        language: *const c_char,
    ) -> *mut CrispasrSessionResult;

    /// VAD-driven session transcribe. Runs Silero VAD on the PCM buffer,
    /// merges short / overlong speech slices, stitches them into one
    /// contiguous buffer with 0.1s silence gaps, calls the backend once,
    /// then remaps segment + word timestamps back to original-audio
    /// positions.
    ///
    /// `vad_model_path` must point to a Silero GGUF on disk. Pass a null
    /// or empty `opts` pointer to use defaults (mirrors whisper.cpp's
    /// `whisper_vad_default_params`).
    pub fn crispasr_session_transcribe_vad(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        sample_rate: c_int,
        vad_model_path: *const c_char,
        opts: *const CrispasrVadAbiOpts,
    ) -> *mut CrispasrSessionResult;

    /// 0.4.9+: language-aware VAD transcribe (same semantics as the
    /// language kwarg on `crispasr_session_transcribe_lang`).
    pub fn crispasr_session_transcribe_vad_lang(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        sample_rate: c_int,
        vad_model_path: *const c_char,
        opts: *const CrispasrVadAbiOpts,
        language: *const c_char,
    ) -> *mut CrispasrSessionResult;

    /// Shared speaker diarization (0.4.5+). Writes a zero-based speaker
    /// index into each `segs[i].speaker`. Returns 0 on success, 1 on
    /// Pyannote model load failure, -1 on invalid args.
    pub fn crispasr_diarize_segments_abi(
        left_pcm: *const c_float,
        right_pcm: *const c_float,
        n_samples: c_int,
        is_stereo: c_int,
        segs: *mut CrispasrDiarizeSegAbi,
        n_segs: c_int,
        opts: *const CrispasrDiarizeOptsAbi,
    ) -> c_int;

    /// Shared language identification (0.4.6+). `method` is 0 for
    /// whisper, 1 for silero. `model_path` is required. Fills
    /// `out_lang_buf` with a null-terminated ISO 639-1 code. Returns 0
    /// on success, -1 on invalid args, 1 on model / detect failure,
    /// 2 when the output buffer is too small.
    pub fn crispasr_detect_language_pcm(
        samples: *const c_float,
        n_samples: c_int,
        method: c_int,
        model_path: *const c_char,
        n_threads: c_int,
        use_gpu: c_int,
        gpu_device: c_int,
        flash_attn: c_int,
        out_lang_buf: *mut c_char,
        out_lang_cap: c_int,
        out_confidence: *mut c_float,
    ) -> c_int;

    /// Shared CTC / forced-aligner word timings (0.4.7+).
    /// Pass any `aligner_model` path — filenames containing
    /// "forced-aligner" / "qwen3-fa" / "qwen3-forced" go through the
    /// Qwen3-ForcedAligner path; everything else uses canary-ctc.
    /// Returns a handle the caller must free with
    /// [`crispasr_align_result_free`]. Returns null on failure.
    pub fn crispasr_align_words_abi(
        aligner_model: *const c_char,
        transcript: *const c_char,
        samples: *const c_float,
        n_samples: c_int,
        t_offset_cs: i64,
        n_threads: c_int,
    ) -> *mut CrispasrAlignResult;

    pub fn crispasr_align_result_n_words(r: *mut CrispasrAlignResult) -> c_int;
    pub fn crispasr_align_result_word_text(r: *mut CrispasrAlignResult, i: c_int) -> *const c_char;
    pub fn crispasr_align_result_word_t0(r: *mut CrispasrAlignResult, i: c_int) -> i64;
    pub fn crispasr_align_result_word_t1(r: *mut CrispasrAlignResult, i: c_int) -> i64;
    pub fn crispasr_align_result_free(r: *mut CrispasrAlignResult);

    /// Shared HF download + cache (0.4.8+). Writes the resolved path
    /// into `out_buf`. Returns 0 on success, -1 on invalid args, 1 on
    /// download failure, 2 when the output buffer is too small.
    pub fn crispasr_cache_ensure_file_abi(
        filename: *const c_char,
        url: *const c_char,
        quiet: c_int,
        cache_dir_override: *const c_char,
        out_buf: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

    /// Return the CrispASR cache directory (creating it if missing).
    pub fn crispasr_cache_dir_abi(
        cache_dir_override: *const c_char,
        out_buf: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

    /// Shared known-model registry lookup by backend. 0 = hit, 1 = miss.
    pub fn crispasr_registry_lookup_abi(
        backend: *const c_char,
        out_filename: *mut c_char,
        filename_cap: c_int,
        out_url: *mut c_char,
        url_cap: c_int,
        out_size: *mut c_char,
        size_cap: c_int,
    ) -> c_int;

    /// Shared known-model registry lookup by filename (exact then fuzzy).
    pub fn crispasr_registry_lookup_by_filename_abi(
        filename: *const c_char,
        out_filename: *mut c_char,
        filename_cap: c_int,
        out_url: *mut c_char,
        url_cap: c_int,
        out_size: *mut c_char,
        size_cap: c_int,
    ) -> c_int;

    pub fn crispasr_session_result_n_segments(r: *mut CrispasrSessionResult) -> c_int;
    pub fn crispasr_session_result_segment_text(
        r: *mut CrispasrSessionResult,
        i: c_int,
    ) -> *const c_char;
    pub fn crispasr_session_result_segment_t0(r: *mut CrispasrSessionResult, i: c_int) -> i64;
    pub fn crispasr_session_result_segment_t1(r: *mut CrispasrSessionResult, i: c_int) -> i64;

    pub fn crispasr_session_result_n_words(r: *mut CrispasrSessionResult, i_seg: c_int) -> c_int;
    pub fn crispasr_session_result_word_text(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
    ) -> *const c_char;
    pub fn crispasr_session_result_word_t0(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
    ) -> i64;
    pub fn crispasr_session_result_word_t1(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
    ) -> i64;

    pub fn crispasr_session_result_free(r: *mut CrispasrSessionResult);
    pub fn crispasr_session_close(s: *mut CrispasrSession);

    pub fn crispasr_detect_backend_from_gguf(
        path: *const c_char,
        out_name: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

    // --- FireRedPunc punctuation restoration ---
    pub fn crispasr_punc_init(model_path: *const c_char) -> *mut c_void;
    pub fn crispasr_punc_process(ctx: *mut c_void, text: *const c_char) -> *mut c_char;
    pub fn crispasr_punc_free_text(text: *mut c_char);
    pub fn crispasr_punc_free(ctx: *mut c_void);

    pub fn crispasr_c_api_version() -> *const c_char;
}
