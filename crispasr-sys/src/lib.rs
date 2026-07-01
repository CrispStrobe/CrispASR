//! Raw FFI bindings to CrispASR.
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
pub const CRISPASR_SAMPLING_GREEDY: c_int = 0;
pub const CRISPASR_SAMPLING_BEAM_SEARCH: c_int = 1;

/// Progress callback for long-form (chunked) transcription (issue #208).
/// `Option<...>` so a null pointer clears the callback (C `NULL`).
pub type CrispasrProgressCallback =
    Option<unsafe extern "C" fn(processed: c_int, total: c_int, user_data: *mut c_void)>;

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

    pub fn whisper_full_default_params_by_ref(strategy: c_int) -> *mut WhisperFullParams;

    // --- Results ---
    pub fn whisper_full_n_segments(ctx: *mut WhisperContext) -> c_int;

    pub fn whisper_full_get_segment_text(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> *const c_char;

    pub fn whisper_full_get_segment_t0(ctx: *mut WhisperContext, i_segment: c_int) -> i64;

    pub fn whisper_full_get_segment_t1(ctx: *mut WhisperContext, i_segment: c_int) -> i64;

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
    pub fn crispasr_params_set_vad_model_path(p: *mut WhisperFullParams, path: *const c_char);
    pub fn crispasr_params_set_vad_threshold(p: *mut WhisperFullParams, threshold: c_float);
    pub fn crispasr_params_set_vad_min_speech_ms(p: *mut WhisperFullParams, ms: c_int);
    pub fn crispasr_params_set_vad_min_silence_ms(p: *mut WhisperFullParams, ms: c_int);
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

/// Opaque streaming-decoder handle returned by
/// `crispasr_session_stream_open`. Must be freed with
/// `crispasr_stream_close`. (PLAN #62)
#[repr(C)]
pub struct CrispasrStream(c_void);

/// Opaque microphone handle returned by `crispasr_mic_open`.
/// Must be freed with `crispasr_mic_close`. (PLAN #62d)
#[repr(C)]
pub struct CrispasrMic(c_void);

/// Opaque result handle for `crispasr_align_words_abi`. Must be freed
/// with `crispasr_align_result_free`.
#[repr(C)]
pub struct CrispasrAlignResult(c_void);

/// Tunables for [`crispasr_session_transcribe_vad`]. Mirrors crispasr's
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
    pub fn crispasr_session_available_backends(out_csv: *mut c_char, out_cap: c_int) -> c_int;

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

    /// 0.8.7+: chunked-encode transcribe (issue #208). Forces the
    /// Parakeet backend through its bounded long-form path (overlapping
    /// short-window transcribe-and-merge for non-JA models, streamed
    /// encoder for the JA-only model) regardless of audio length, so long
    /// files transcribe in bounded time AND recover the sections a single
    /// full-length pass drops. `chunk_seconds <= 0` keeps the per-model
    /// defaults; otherwise it sets the non-JA window length / the JA
    /// streamed window. `overlap_seconds < 0` uses the default. For
    /// non-Parakeet backends the chunk params are inert and this matches
    /// `crispasr_session_transcribe_lang`.
    pub fn crispasr_session_transcribe_chunked_lang(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        chunk_seconds: c_int,
        overlap_seconds: c_int,
        language: *const c_char,
    ) -> *mut CrispasrSessionResult;

    pub fn crispasr_session_transcribe_chunked(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        chunk_seconds: c_int,
        overlap_seconds: c_int,
    ) -> *mut CrispasrSessionResult;

    /// 0.10.3+ (issue #208): register a per-session progress callback for
    /// long-form (chunked) transcription. Fired once per finished window
    /// with `(processed_samples, total_samples, user_data)`; `processed`
    /// is monotonic and reaches `total` on the last window. Invoked on the
    /// transcribe thread. Pass `None`/null `cb` to clear.
    pub fn crispasr_session_set_progress_callback(
        s: *mut CrispasrSession,
        cb: CrispasrProgressCallback,
        user_data: *mut c_void,
    );

    /// VAD-driven session transcribe. Runs Silero VAD on the PCM buffer,
    /// merges short / overlong speech slices, stitches them into one
    /// contiguous buffer with 0.1s silence gaps, calls the backend once,
    /// then remaps segment + word timestamps back to original-audio
    /// positions.
    ///
    /// `vad_model_path` must point to a Silero GGUF on disk. Pass a null
    /// or empty `opts` pointer to use defaults (mirrors crispasr's
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
    pub fn crispasr_registry_list_backends_abi(out_csv: *mut c_char, out_cap: c_int) -> c_int;

    // --- Streaming (PLAN #62) — rolling-window decoder for whisper today ---
    pub fn crispasr_session_stream_open(
        s: *mut CrispasrSession,
        n_threads: c_int,
        step_ms: c_int,
        length_ms: c_int,
        keep_ms: c_int,
        language: *const c_char,
        translate: c_int,
    ) -> *mut CrispasrStream;
    pub fn crispasr_stream_feed(
        s: *mut CrispasrStream,
        pcm: *const c_float,
        n_samples: c_int,
    ) -> c_int;
    pub fn crispasr_stream_get_text(
        s: *mut CrispasrStream,
        out_text: *mut c_char,
        out_cap: c_int,
        out_t0_s: *mut f64,
        out_t1_s: *mut f64,
        out_counter: *mut i64,
    ) -> c_int;
    pub fn crispasr_stream_flush(s: *mut CrispasrStream) -> c_int;
    pub fn crispasr_stream_close(s: *mut CrispasrStream);

    /// Toggle voxtral4b live-captions decode-during-feed (PLAN #7 phase 3).
    /// No-op for backends that don't have audio-injection prompt decode.
    /// Set BEFORE the first feed for clean semantics.
    pub fn crispasr_stream_set_live_decode(s: *mut CrispasrStream, enabled: c_int);

    // --- Mic capture (PLAN #62d) — miniaudio ma_device wrapper ---
    pub fn crispasr_mic_open(
        sample_rate: c_int,
        channels: c_int,
        cb: extern "C" fn(pcm: *const c_float, n_samples: c_int, userdata: *mut c_void),
        userdata: *mut c_void,
    ) -> *mut CrispasrMic;
    pub fn crispasr_mic_start(m: *mut CrispasrMic) -> c_int;
    pub fn crispasr_mic_stop(m: *mut CrispasrMic) -> c_int;
    pub fn crispasr_mic_close(m: *mut CrispasrMic);
    pub fn crispasr_mic_default_device_name() -> *const c_char;
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
    pub fn crispasr_session_result_word_p(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
    ) -> f32;

    pub fn crispasr_session_result_free(r: *mut CrispasrSessionResult);
    pub fn crispasr_session_close(s: *mut CrispasrSession);

    // --- TTS synthesis (vibevoice, qwen3-tts, kokoro, orpheus) ---
    pub fn crispasr_session_set_codec_path(s: *mut CrispasrSession, path: *const c_char) -> c_int;
    pub fn crispasr_session_set_voice(
        s: *mut CrispasrSession,
        path: *const c_char,
        ref_text_or_null: *const c_char,
    ) -> c_int;
    pub fn crispasr_session_set_speaker_name(s: *mut CrispasrSession, name: *const c_char)
        -> c_int;
    pub fn crispasr_session_n_speakers(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_get_speaker_name(s: *mut CrispasrSession, i: c_int) -> *const c_char;
    // qwen3-tts VoiceDesign: natural-language voice description.
    pub fn crispasr_session_set_instruct(s: *mut CrispasrSession, instruct: *const c_char)
        -> c_int;
    // qwen3-tts variant detection (returns 0/1; 0 also covers "not qwen3-tts").
    pub fn crispasr_session_is_custom_voice(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_is_voice_design(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_synthesize(
        s: *mut CrispasrSession,
        text: *const c_char,
        out_n_samples: *mut c_int,
    ) -> *mut f32;
    pub fn crispasr_pcm_free(pcm: *mut f32);
    // Drop the kokoro per-session phoneme cache. No-op for non-kokoro
    // backends. Returns 0 on success, -1 if `s` is null. (PLAN #56 #5)
    pub fn crispasr_session_kokoro_clear_phoneme_cache(s: *mut CrispasrSession) -> c_int;

    // --- Sticky session-state setters (PLAN #59 partial unblock) ---
    pub fn crispasr_session_set_source_language(
        s: *mut CrispasrSession,
        lang: *const c_char,
    ) -> c_int;
    pub fn crispasr_session_set_target_language(
        s: *mut CrispasrSession,
        lang: *const c_char,
    ) -> c_int;
    pub fn crispasr_session_set_punctuation(s: *mut CrispasrSession, enable: c_int) -> c_int;
    pub fn crispasr_session_set_punc_model(
        s: *mut CrispasrSession,
        punc_model: *const c_char,
    ) -> c_int;
    pub fn crispasr_session_set_hotwords(
        s: *mut CrispasrSession,
        hotwords: *const c_char,
        boost: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_g2p_dict(s: *mut CrispasrSession, source: *const c_char) -> c_int;
    pub fn crispasr_session_set_speaker_id(s: *mut CrispasrSession, id: c_int) -> c_int;
    pub fn crispasr_session_set_translate(s: *mut CrispasrSession, enable: c_int) -> c_int;
    // --- Text-to-text translation (m2m100 / m2m100-wmt21 / madlad / gemma4-e2b) ---
    //
    // Distinct from `crispasr_session_set_translate` above, which is the
    // *audio-side* Whisper sticky flag (PCM input → English text out).
    // This one translates an already-extracted Rust string between
    // arbitrary language pairs via whichever MT-capable backend the
    // session loaded.  Returns a malloc'd UTF-8 buffer that the caller
    // MUST release via `crispasr_session_translate_text_free` (mirrors
    // the punc-side ownership pattern).  Returns nullptr on:
    //   * any input pointer being null,
    //   * the session not having a CAP_TRANSLATE backend loaded,
    //   * the backend's internal translate routine erroring out.
    //
    // `max_tokens` caps the decoder output length.  Pass `<= 0` to
    // fall back to the C++ default (200 for m2m100).
    pub fn crispasr_session_translate_text(
        s: *mut CrispasrSession,
        text: *const c_char,
        src_lang: *const c_char,
        tgt_lang: *const c_char,
        max_tokens: c_int,
    ) -> *mut c_char;
    // Free a buffer previously returned by `crispasr_session_translate_text`.
    // No-op when `text` is null.  Calling `libc::free` directly also works
    // (the C++ side just delegates to `free()`), but routing through this
    // symbol keeps ownership symmetric and protects callers if the C++
    // side ever switches allocators.
    pub fn crispasr_session_translate_text_free(text: *mut c_char);
    pub fn crispasr_session_set_temperature(
        s: *mut CrispasrSession,
        temperature: c_float,
        seed: u64,
    ) -> c_int;
    pub fn crispasr_session_set_tts_seed(s: *mut CrispasrSession, seed: u64) -> c_int;
    pub fn crispasr_session_set_max_new_tokens(
        s: *mut CrispasrSession,
        max_new_tokens: c_int,
    ) -> c_int;
    pub fn crispasr_session_set_frequency_penalty(
        s: *mut CrispasrSession,
        penalty: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_tts_steps(s: *mut CrispasrSession, steps: c_int) -> c_int;
    pub fn crispasr_session_set_tts_num_candidates(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_top_p(s: *mut CrispasrSession, top_p: c_float) -> c_int;
    pub fn crispasr_session_set_top_k(s: *mut CrispasrSession, top_k: c_int) -> c_int;
    pub fn crispasr_session_set_do_sample(s: *mut CrispasrSession, enable: c_int) -> c_int;
    pub fn crispasr_session_set_min_p(s: *mut CrispasrSession, min_p: c_float) -> c_int;
    pub fn crispasr_session_set_repetition_penalty(s: *mut CrispasrSession, r: c_float) -> c_int;
    pub fn crispasr_session_set_cfg_weight(s: *mut CrispasrSession, cfg_weight: c_float) -> c_int;
    pub fn crispasr_session_set_tts_noise_temp(
        s: *mut CrispasrSession,
        noise_temp: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_exaggeration(
        s: *mut CrispasrSession,
        exaggeration: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_max_speech_tokens(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_length_scale(s: *mut CrispasrSession, scale: c_float) -> c_int;
    pub fn crispasr_session_set_best_of(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_beam_size(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_grammar_text(
        s: *mut CrispasrSession,
        gbnf_text: *const c_char,
        root_rule: *const c_char,
        penalty: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_fallback_thresholds(
        s: *mut CrispasrSession,
        entropy_thold: c_float,
        logprob_thold: c_float,
        no_speech_thold: c_float,
        temperature_inc: c_float,
    ) -> c_int;
    pub fn crispasr_session_set_alt_n(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_whisper_decode_extras(
        s: *mut CrispasrSession,
        suppress_nst: c_int,
        suppress_regex: *const c_char,
        carry_initial_prompt: c_int,
    ) -> c_int;
    pub fn crispasr_session_set_ask(s: *mut CrispasrSession, prompt: *const c_char) -> c_int;
    pub fn crispasr_session_detect_language(
        s: *mut CrispasrSession,
        pcm: *const c_float,
        n_samples: c_int,
        lid_model_path: *const c_char,
        method: c_int,
        out_lang: *mut c_char,
        out_lang_cap: c_int,
        out_prob: *mut c_float,
    ) -> c_int;

    // --- Text-LID (P13.5 Phase 7) ---
    //
    // Detect the language of a UTF-8 text string via the internal
    // `text_lid_dispatch` façade — routes to CLD3 (ISO 639-1, 109
    // labels) or GlotLID-V3 / LID-176 fastText (ISO 639-3 + script,
    // 2102 or 176 labels) based on the GGUF's architecture key.
    // Label format follows whichever backend the GGUF loads as —
    // see the C-API doc-comment for normalisation guidance.
    //
    // Returns:
    //   *  0 — success; `out_label_buf` + `out_confidence` populated.
    //   * -1 — invalid args (null pointer or out_label_cap <= 0).
    //   *  1 — dispatcher init / predict failure.
    //   *  2 — output buffer too small for the predicted label.
    pub fn crispasr_text_detect_language(
        text: *const c_char,
        model_path: *const c_char,
        n_threads: c_int,
        out_label_buf: *mut c_char,
        out_label_cap: c_int,
        out_confidence: *mut c_float,
    ) -> c_int;

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

    // --- Kokoro per-language model + voice routing (PLAN #56 opt 2b) ---
    // See `src/kokoro.h` for full semantics.
    pub fn crispasr_kokoro_lang_is_german_abi(lang: *const c_char) -> bool;
    pub fn crispasr_kokoro_lang_has_native_voice_abi(lang: *const c_char) -> bool;
    pub fn crispasr_kokoro_resolve_model_for_lang_abi(
        model_path: *const c_char,
        lang: *const c_char,
        out_path: *mut c_char,
        out_path_len: c_int,
    ) -> c_int;
    pub fn crispasr_kokoro_resolve_fallback_voice_abi(
        model_path: *const c_char,
        lang: *const c_char,
        out_path: *mut c_char,
        out_path_len: c_int,
        out_picked: *mut c_char,
        out_picked_len: c_int,
    ) -> c_int;

    // TitaNet speaker verification
    pub fn crispasr_titanet_init(model_path: *const c_char, n_threads: i32) -> *mut c_void;
    pub fn crispasr_titanet_free(ctx: *mut c_void);
    pub fn crispasr_titanet_embed(
        ctx: *mut c_void,
        pcm_16k: *const c_float,
        n_samples: i32,
        out: *mut c_float,
    ) -> i32;
    pub fn crispasr_titanet_cosine_sim(a: *const c_float, b: *const c_float, dim: i32) -> c_float;

    // Speaker profile database
    pub fn crispasr_speaker_db_load(dir_path: *const c_char) -> *mut c_void;
    pub fn crispasr_speaker_db_free(db: *mut c_void);
    pub fn crispasr_speaker_db_count(db: *const c_void) -> i32;
    pub fn crispasr_speaker_db_match(
        db: *const c_void,
        embedding: *const c_float,
        dim: i32,
        threshold: c_float,
        out_name: *mut c_char,
        out_cap: i32,
    ) -> c_float;
    pub fn crispasr_speaker_db_enroll(
        dir_path: *const c_char,
        name: *const c_char,
        embedding: *const c_float,
        dim: i32,
    ) -> i32;

    // Pluggable speaker embedder + agglomerative clustering + pyannote
    // cache (issue #107 P6). Same building blocks as the CLI's
    // --diarize-embedder path; expose them so Rust callers can compose
    // the diarize pipeline without round-tripping through the CLI.

    /// Build a pluggable speaker embedder. `model_spec` is one of
    /// `"auto"`, `"titanet"`, `"indextts"`, `"indextts-bigvgan"`,
    /// `"ecapa"`, or a `.gguf` path. Returns null on failure.
    pub fn crispasr_speaker_embedder_make_abi(
        model_spec: *const c_char,
        n_threads: i32,
        cache_dir: *const c_char,
    ) -> *mut c_void;

    pub fn crispasr_speaker_embedder_free_abi(embedder: *mut c_void);

    /// Output embedding dimension (e.g. 192 for TitaNet, 512 for
    /// IndexTTS-BigVGAN).
    pub fn crispasr_speaker_embedder_dim_abi(embedder: *const c_void) -> i32;

    /// Extract one embedding. `out` must hold at least `dim()` floats.
    /// Returns 1 on success, 0 if the model rejected the input.
    pub fn crispasr_speaker_embedder_embed_abi(
        embedder: *mut c_void,
        pcm_16k: *const c_float,
        n_samples: i32,
        out: *mut c_float,
    ) -> i32;

    pub fn crispasr_speaker_embedder_name_abi(embedder: *const c_void) -> *const c_char;

    /// Agglomerative single-linkage cosine clustering. `embeddings` is
    /// a row-major `n × dim` buffer of (ideally L2-normalized) vectors.
    /// `labels_out` receives one cluster ID per input in `[0, k)`.
    /// Returns the cluster count `k`, or -1 on invalid arguments.
    pub fn crispasr_speaker_cluster_abi(
        embeddings: *const c_float,
        n: i32,
        dim: i32,
        merge_threshold: c_float,
        max_speakers: i32,
        labels_out: *mut i32,
    ) -> i32;

    /// Pre-compute pyannote-seg posteriors over a full audio buffer.
    /// Returns an opaque cache or null on failure. Free with
    /// `crispasr_pyannote_cache_free_abi`.
    pub fn crispasr_pyannote_cache_compute_abi(
        full_audio: *const c_float,
        n_samples: i32,
        model_path: *const c_char,
        n_threads: i32,
    ) -> *mut c_void;

    pub fn crispasr_pyannote_cache_free_abi(cache: *mut c_void);

    /// Score `segs` against the cached posteriors. `slice_t0_cs` is the
    /// absolute centisecond at which the cache buffer starts (typically
    /// 0 — the cache covers the whole input audio).
    pub fn crispasr_pyannote_cache_apply_abi(
        cache: *const c_void,
        slice_t0_cs: i64,
        segs: *mut CrispasrDiarizeSegAbi,
        n_segs: i32,
    ) -> i32;

    // --- params_set_* on whisper_full_params (full C-ABI parity) ---
    pub fn crispasr_params_set_language(p: *mut WhisperFullParams, lang: *const c_char);
    pub fn crispasr_params_set_translate(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_detect_language(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_token_timestamps(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_n_threads(p: *mut WhisperFullParams, n: c_int);
    pub fn crispasr_params_set_max_len(p: *mut WhisperFullParams, n: c_int);
    pub fn crispasr_params_set_best_of(p: *mut WhisperFullParams, n: c_int);
    pub fn crispasr_params_set_split_on_word(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_no_context(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_single_segment(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_print_realtime(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_print_progress(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_print_timestamps(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_print_special(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_suppress_blank(p: *mut WhisperFullParams, v: c_int);
    pub fn crispasr_params_set_temperature(p: *mut WhisperFullParams, t: c_float);
    pub fn crispasr_params_set_max_tokens(p: *mut WhisperFullParams, n: c_int);
    pub fn crispasr_params_set_initial_prompt(p: *mut WhisperFullParams, prompt: *const c_char);
    pub fn crispasr_params_set_alt_n(p: *mut WhisperFullParams, n: c_int);

    // --- Token-level accessors ---
    pub fn crispasr_token_t0(ctx: *mut WhisperContext, i_seg: c_int, i_tok: c_int) -> i64;
    pub fn crispasr_token_t1(ctx: *mut WhisperContext, i_seg: c_int, i_tok: c_int) -> i64;
    pub fn crispasr_token_p(ctx: *mut WhisperContext, i_seg: c_int, i_tok: c_int) -> c_float;
    pub fn crispasr_token_n_alts(ctx: *mut WhisperContext, i_seg: c_int, i_tok: c_int) -> c_int;
    pub fn crispasr_token_alt_id(
        ctx: *mut WhisperContext,
        i_seg: c_int,
        i_tok: c_int,
        i_alt: c_int,
    ) -> i32;
    pub fn crispasr_token_alt_p(
        ctx: *mut WhisperContext,
        i_seg: c_int,
        i_tok: c_int,
        i_alt: c_int,
    ) -> c_float;
    pub fn crispasr_token_alt_text(
        ctx: *mut WhisperContext,
        i_seg: c_int,
        i_tok: c_int,
        i_alt: c_int,
        out: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

    // --- Language detection (whisper context) ---
    pub fn crispasr_detect_language(
        ctx: *mut WhisperContext,
        pcm: *const c_float,
        n_samples: c_int,
        n_threads: c_int,
        out_code: *mut c_char,
        out_cap: c_int,
    ) -> c_float;

    // --- VAD ---
    pub fn crispasr_vad_segments(
        vad_model_path: *const c_char,
        pcm: *const c_float,
        n_samples: c_int,
        sample_rate: c_int,
        threshold: c_float,
        min_speech_ms: c_int,
        min_silence_ms: c_int,
        n_threads: c_int,
        use_gpu: c_int,
        out_spans: *mut *mut c_float,
    ) -> c_int;
    pub fn crispasr_vad_slices(
        vad_model_path: *const c_char,
        pcm: *const c_float,
        n_samples: c_int,
        sample_rate: c_int,
        threshold: c_float,
        min_speech_ms: c_int,
        min_silence_ms: c_int,
        speech_pad_ms: c_int,
        max_chunk_duration_s: c_float,
        n_threads: c_int,
        out_spans: *mut *mut c_float,
    ) -> c_int;
    pub fn crispasr_vad_free(spans: *mut c_float);

    // --- LCS dedup ---
    pub fn crispasr_lcs_dedup_prefix_count(
        prev_tail_tokens: *const i32,
        n_prev: c_int,
        curr_tokens: *const i32,
        n_curr: c_int,
        min_lcs_length: c_int,
    ) -> c_int;

    // --- Streaming (whisper context) ---
    pub fn crispasr_stream_open(
        ctx: *mut WhisperContext,
        n_threads: c_int,
        step_ms: c_int,
        length_ms: c_int,
        keep_ms: c_int,
        language: *const c_char,
        translate: c_int,
    ) -> *mut CrispasrStream;

    // --- Direct Parakeet API ---
    pub fn crispasr_parakeet_init(
        model_path: *const c_char,
        n_threads: c_int,
        use_flash: c_int,
    ) -> *mut c_void;
    pub fn crispasr_parakeet_free(ctx: *mut c_void);
    pub fn crispasr_parakeet_transcribe(
        ctx: *mut c_void,
        pcm: *const c_float,
        n_samples: c_int,
        language: *const c_char,
    ) -> *mut c_void;
    pub fn crispasr_parakeet_result_text(r: *mut c_void) -> *const c_char;
    pub fn crispasr_parakeet_result_n_words(r: *mut c_void) -> c_int;
    pub fn crispasr_parakeet_result_word_text(r: *mut c_void, i: c_int) -> *const c_char;
    pub fn crispasr_parakeet_result_word_t0(r: *mut c_void, i: c_int) -> i64;
    pub fn crispasr_parakeet_result_word_t1(r: *mut c_void, i: c_int) -> i64;
    pub fn crispasr_parakeet_result_n_tokens(r: *mut c_void) -> c_int;
    pub fn crispasr_parakeet_result_token_text(r: *mut c_void, i: c_int) -> *const c_char;
    pub fn crispasr_parakeet_result_token_t0(r: *mut c_void, i: c_int) -> i64;
    pub fn crispasr_parakeet_result_token_t1(r: *mut c_void, i: c_int) -> i64;
    pub fn crispasr_parakeet_result_token_p(r: *mut c_void, i: c_int) -> c_float;
    pub fn crispasr_parakeet_result_free(r: *mut c_void);

    // --- RNNoise audio enhancement ---
    pub fn crispasr_enhance_audio_rnnoise(
        in_pcm: *const c_float,
        n_samples: i32,
        out_pcm: *mut c_float,
        out_cap: i32,
    ) -> c_int;

    // --- Session open with params ---
    pub fn crispasr_session_open_with_params(
        model_path: *const c_char,
        backend_name: *const c_char,
        params: *const c_void,
    ) -> *mut CrispasrSession;

    // --- Session result word alts ---
    pub fn crispasr_session_result_word_n_alts(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
    ) -> c_int;
    pub fn crispasr_session_result_word_alt_text(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
        i_alt: c_int,
    ) -> *const c_char;
    pub fn crispasr_session_result_word_alt_p(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
        i_word: c_int,
        i_alt: c_int,
    ) -> c_float;
}
