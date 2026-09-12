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

/// ABI speaker turn for [`crispasr_diarize_segments_turns_abi`] (#395).
///
/// A turn the METHOD derived from the audio, independent of the caller's
/// segment grid — only FoxNose produces them. `t0_cs` / `t1_cs` are
/// centiseconds on the same absolute timeline as [`CrispasrDiarizeSegAbi`]
/// (`slice_t0_cs` already added back), so turns and caller segments compare
/// directly. `speaker` is dense and zero-based, never -1.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrDiarizeTurnAbi {
    pub t0_cs: i64,
    pub t1_cs: i64,
    pub speaker: c_int,
    pub _pad: c_int,
}

/// ABI options for [`crispasr_diarize_segments_abi`]. `method` is a
/// value in 0..4: 0 = Energy, 1 = Xcorr, 2 = VadTurns, 3 = Pyannote,
/// 4 = FoxNose. `pyannote_model_path` is required for Pyannote,
/// `foxnose_embedder_path` for FoxNose; each is ignored otherwise.
///
/// This layout is hand-maintained and MUST match
/// `crispasr_diarize_opts_abi` in `src/crispasr_c_api.cpp`, which is
/// APPEND-ONLY: the C side reads every field unconditionally, so a
/// short struct here is an out-of-bounds read even for methods 0..3.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrDiarizeOptsAbi {
    pub method: c_int,
    pub n_threads: c_int,
    pub slice_t0_cs: i64,
    pub pyannote_model_path: *const c_char,
    // #324 FoxNose (method 4). Ignored by the other methods.
    pub foxnose_embedder_path: *const c_char,
    /// 0 -> 1
    pub min_speakers: c_int,
    /// 0 -> 8
    pub max_speakers: c_int,
    /// >0 pins the speaker count and skips estimation
    pub num_speakers: c_int,
    pub _pad2: c_int,
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

    // CTC vocabulary access (Omni CTC backend). `n_vocab` is the number of
    // SentencePiece pieces (0 for backends without an exposed CTC vocab);
    // `token_text` maps an id in `[0, n_vocab)` to its raw piece (U+2581 marker
    // intact), or "" when out of range / unsupported. Pairs with the result
    // logits accessor to detokenize a greedy CTC decode.
    pub fn crispasr_session_n_vocab(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_token_text(s: *mut CrispasrSession, id: c_int) -> *const c_char;

    // Acoustic language detected by the last transcribe, written into `out_buf`
    // as an ISO-639-1 code (whisper only; other backends fall back to the
    // source-language hint, then "unknown"). Returns the code length in bytes
    // (not counting NUL) or -1 on bad args. Distinct from the text-LID pass.
    pub fn crispasr_session_detected_language(
        s: *mut CrispasrSession,
        out_buf: *mut c_char,
        out_cap: c_int,
    ) -> c_int;

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

    /// Diarize AND hand back the speaker turns the method derived from the
    /// audio (0.8.30+, issue #395). Identical to
    /// [`crispasr_diarize_segments_abi`] plus the three trailing turn
    /// parameters; passing `null` / `0` / `null` for them is exactly the
    /// older call.
    ///
    /// Callers need this because labelling alone can never resolve finer
    /// than the segment grid they sent in: a segment straddling a speaker
    /// change is silently awarded to whoever holds the majority of it.
    /// Only FoxNose derives turns; the other methods report 0, which is not
    /// an error.
    ///
    /// `out_n_turns`, when non-null, always receives the TOTAL turn count —
    /// also when it exceeds `n_turns_cap`, so a caller can size and retry
    /// (at the cost of a second full pass; the ABI keeps no state).
    /// `out_turns`, when non-null, receives up to `n_turns_cap` of them.
    ///
    /// Returns 0 on success, 2 when a turn buffer was given and could not
    /// hold every turn (the segments are still fully labelled and the first
    /// `n_turns_cap` turns are still written), 1 on model load failure, -1
    /// on invalid args.
    pub fn crispasr_diarize_segments_turns_abi(
        left_pcm: *const c_float,
        right_pcm: *const c_float,
        n_samples: c_int,
        is_stereo: c_int,
        segs: *mut CrispasrDiarizeSegAbi,
        n_segs: c_int,
        opts: *const CrispasrDiarizeOptsAbi,
        out_turns: *mut CrispasrDiarizeTurnAbi,
        n_turns_cap: c_int,
        out_n_turns: *mut c_int,
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
    // #433: backend -> verb ("capability") lookup.
    pub fn crispasr_backend_caps_abi(backend: *const c_char, out_csv: *mut c_char, out_cap: c_int) -> c_int;
    pub fn crispasr_backend_caps_list_abi(out_buf: *mut c_char, out_cap: c_int) -> c_int;
    // #432: reference voice from in-memory samples.
    pub fn crispasr_session_set_voice_samples(
        s: *mut CrispasrSession,
        pcm: *const f32,
        n_samples: c_int,
        sample_rate: c_int,
        ref_text_or_null: *const c_char,
    ) -> c_int;

    /// Describe the exact canonical artifact bundle downloaded by `-m auto`.
    /// Returns its artifact count, 0 on miss, or a negative argument/buffer error.
    pub fn crispasr_registry_default_bundle_info_abi(
        backend: *const c_char,
        out_backend: *mut c_char,
        backend_cap: c_int,
        out_license: *mut c_char,
        license_cap: c_int,
        out_requires_acceptance: *mut c_int,
    ) -> c_int;

    /// Read one default-bundle artifact by index. 0 = success.
    pub fn crispasr_registry_default_bundle_artifact_abi(
        backend: *const c_char,
        index: c_int,
        out_kind: *mut c_int,
        out_filename: *mut c_char,
        filename_cap: c_int,
        out_url: *mut c_char,
        url_cap: c_int,
        out_size: *mut c_char,
        size_cap: c_int,
    ) -> c_int;

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
    // Whisper's per-segment no-speech probability (the <|nospeech|> token
    // posterior) in [0, 1]. Only the whisper backend populates it; other
    // backends and out-of-range indices return the -1.0 sentinel ("no data").
    pub fn crispasr_session_result_segment_no_speech_prob(
        r: *mut CrispasrSessionResult,
        i_seg: c_int,
    ) -> f32;

    // Raw per-frame CTC logits (Omni CTC backend, opted in via
    // `crispasr_session_set_return_logits`). Frame-major, pre-softmax:
    // `logits[t * n_logit_vocab + v]`; the pointer is NULL when none captured.
    pub fn crispasr_session_result_n_logit_frames(r: *mut CrispasrSessionResult) -> c_int;
    pub fn crispasr_session_result_n_logit_vocab(r: *mut CrispasrSessionResult) -> c_int;
    pub fn crispasr_session_result_logits(r: *mut CrispasrSessionResult) -> *const c_float;

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
    // #316: synthesize these phonemes verbatim, skipping the G2P. Empty clears.
    // -2 = the active backend has no phonemes-in call (kokoro and piper do).
    pub fn crispasr_session_set_tts_phonemes(
        s: *mut CrispasrSession,
        phonemes: *const c_char,
    ) -> c_int;
    pub fn crispasr_session_set_tts_pad_silence_ms(
        s: *mut CrispasrSession,
        ms: c_int,
    );
    // qwen3-tts variant detection (returns 0/1; 0 also covers "not qwen3-tts").
    pub fn crispasr_session_is_custom_voice(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_is_voice_design(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_synthesize(
        s: *mut CrispasrSession,
        text: *const c_char,
        out_n_samples: *mut c_int,
    ) -> *mut f32;
    // Speech-to-Speech — audio in -> audio out via a single model pass. Supported
    // on S2S-capable backends (lfm2-audio, mini-omni2, sidon, voxcpm2-vae). Returns
    // malloc'd f32 PCM (free with `crispasr_pcm_free`); `out_text`, if non-null,
    // receives the malloc'd intermediate transcript (free with
    // `crispasr_session_translate_text_free`). Returns null on failure / unsupported.
    pub fn crispasr_session_speech_to_speech(
        s: *mut CrispasrSession,
        in_samples: *const f32,
        n_in_samples: c_int,
        out_text: *mut *mut c_char,
        out_n_samples: *mut c_int,
    ) -> *mut f32;
    // Source separation (#359). Stereo interleaved PCM in at the model's own
    // rate (44100 Hz for htdemucs / mel-band-roformer); returns the stem count.
    // Stem buffers are owned by the session and valid only until the next
    // separate() call or close, so a safe wrapper must copy them out.
    pub fn crispasr_session_separate(
        s: *mut CrispasrSession,
        pcm_stereo: *const f32,
        n_samples: c_int,
    ) -> c_int;
    pub fn crispasr_session_separate_n_stems(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_separate_stem_name(
        s: *mut CrispasrSession,
        stem_idx: c_int,
    ) -> *const c_char;
    pub fn crispasr_session_separate_stem(
        s: *mut CrispasrSession,
        stem_idx: c_int,
        out_n_samples: *mut c_int,
    ) -> *const f32;
    pub fn crispasr_session_separate_sample_rate(s: *mut CrispasrSession) -> c_int;
    // UNMARKED synthesis (no watermark/disclosure). Hard-refused unless
    // `crispasr_session_accept_marking_responsibility` was called first. Returns
    // malloc'd f32 PCM (free with `crispasr_pcm_free`); null on refusal/failure.
    pub fn crispasr_session_synthesize_raw(
        s: *mut CrispasrSession,
        text: *const c_char,
        out_n_samples: *mut c_int,
    ) -> *mut f32;
    // Attest that the integrator accepts AI-content marking/disclosure
    // responsibility (EU AI Act Art. 50). REQUIRED before `synthesize_raw`.
    pub fn crispasr_session_accept_marking_responsibility(
        s: *mut CrispasrSession,
        attestation: *const c_char,
    ) -> c_int;
    // Declare whose voice a PRESET voice is: "real_person" | "synthetic" |
    // "unknown". A preset can be an identifiable individual, which makes its
    // output a deep fake under Art. 3(60) without any cloning. Returns 0, -1 on
    // a bad session, -2 on an unrecognised value.
    pub fn crispasr_session_set_speaker_identity(
        s: *mut CrispasrSession,
        identity: *const c_char,
    ) -> c_int;
    // Sample rate the backend expects for input PCM (16000 for Whisper-family,
    // the model's native rate otherwise; 0 on error). Pair with s2s/synthesize to
    // feed input at the right rate.
    pub fn crispasr_session_input_sample_rate(s: *mut CrispasrSession) -> c_int;
    // #332: output-side counterparts. output_sample_rate is the rate of the
    // PCM synthesize / speech_to_speech return (0 = backend has no audio
    // output); the channel getters are 1 (mono) for every current backend.
    pub fn crispasr_session_output_sample_rate(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_input_channels(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_session_output_channels(s: *mut CrispasrSession) -> c_int;
    pub fn crispasr_pcm_free(pcm: *mut f32);
    // Embed the AI-content watermark into f32 mono PCM, in place. The other
    // half of `synthesize_raw`: opting out of automatic marking obliges the
    // caller to mark the result, and this is what they mark it with.
    // `alpha <= 0` selects the robust, reliably detectable default.
    pub fn crispasr_watermark_embed(pcm: *mut f32, n_samples: c_int, alpha: c_float);
    // Confidence in [0, 1] that `pcm` carries the watermark. A weak diagnostic:
    // the spread-spectrum null mean is 0.5, not 0 (see docs/eu-ai-act.md §6.7).
    pub fn crispasr_watermark_detect(pcm: *const c_float, n_samples: c_int) -> c_float;
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
    pub fn crispasr_session_set_tts_reference_language(
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
    pub fn crispasr_session_set_sensitivity(
        s: *mut CrispasrSession,
        preset: *const c_char,
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
    pub fn crispasr_session_set_min_speech_tokens(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_length_scale(s: *mut CrispasrSession, scale: c_float) -> c_int;
    pub fn crispasr_session_set_best_of(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_beam_size(s: *mut CrispasrSession, n: c_int) -> c_int;
    pub fn crispasr_session_set_return_logits(s: *mut CrispasrSession, enable: c_int) -> c_int;
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

// =========================================================================
// Chat / LLM FFI — mirrors include/crispasr_chat.h
// =========================================================================
//
// Text in, text out over the private `crispasr-llama-core` (vendored
// llama.cpp) built unconditionally into libcrispasr, so these symbols are
// always present — no build flag gates them. Only POD structs and one
// opaque handle cross the boundary.

/// Opaque handle returned by `crispasr_chat_open`. Free with
/// `crispasr_chat_close`.
#[repr(C)]
pub struct CrispasrChatSession(c_void);

/// The one error code on the chat ABI with a stable, documented meaning:
/// a registered abort callback stopped the run. Every other non-zero
/// `CrispasrChatError::code` is a diagnostic aid — read `message`.
pub const CRISPASR_CHAT_ERR_ABORTED: i32 = 40;

/// Out-parameter for every chat entry point that can fail (may be null).
/// Left untouched on success; on failure `code` is non-zero and `message`
/// holds a NUL-terminated diagnostic.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrChatError {
    pub code: i32,
    pub message: [c_char; 256],
}

impl Default for CrispasrChatError {
    fn default() -> Self {
        Self {
            code: 0,
            message: [0; 256],
        }
    }
}

/// One turn of a conversation. `role` is "system", "user", "assistant" or
/// "tool"; both pointers must stay valid for the duration of the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

/// Per-session, model-level open params. Fill via
/// [`crispasr_chat_open_params_default`] before overriding fields — the C
/// side reads every one of them.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrChatOpenParams {
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    /// Context window in tokens; 0 = the model's own default.
    pub n_ctx: c_int,
    pub n_batch: c_int,
    pub n_ubatch: c_int,
    /// -1 = offload all layers, 0 = CPU only.
    pub n_gpu_layers: c_int,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub embeddings: bool,
    /// Overrides the template baked into the GGUF; null reads
    /// `tokenizer.chat_template` from the model. Copied by the callee.
    pub chat_template: *const c_char,
}

impl Default for CrispasrChatOpenParams {
    fn default() -> Self {
        Self {
            n_threads: 0,
            n_threads_batch: 0,
            n_ctx: 0,
            n_batch: 0,
            n_ubatch: 0,
            n_gpu_layers: 0,
            use_mmap: false,
            use_mlock: false,
            embeddings: false,
            chat_template: std::ptr::null(),
        }
    }
}

/// Per-call, sampler-level generate params. Fill via
/// [`crispasr_chat_generate_params_default`] before overriding fields.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CrispasrChatGenerateParams {
    pub max_tokens: c_int,
    /// 0.0 = greedy (short-circuits the rest of the sampler chain).
    pub temperature: c_float,
    pub top_k: c_int,
    pub top_p: c_float,
    pub min_p: c_float,
    pub repeat_penalty: c_float,
    pub repeat_last_n: c_int,
    /// 0xFFFFFFFF = random.
    pub seed: u32,
    /// Array of `n_stop` NUL-terminated stop strings; null = none.
    pub stop: *const *const c_char,
    pub n_stop: usize,
    /// Prefill the prompt but suppress assistant generation.
    pub prefill_only: bool,
}

impl Default for CrispasrChatGenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            repeat_penalty: 0.0,
            repeat_last_n: 0,
            seed: 0,
            stop: std::ptr::null(),
            n_stop: 0,
            prefill_only: false,
        }
    }
}

/// Fired once per detokenised UTF-8 chunk during a streaming generate. The
/// chunk pointer is valid only for the duration of the call.
/// `Option<...>` so a null pointer clears the callback (C `NULL`).
pub type CrispasrChatOnToken =
    Option<unsafe extern "C" fn(utf8_chunk: *const c_char, user: *mut c_void)>;

/// Abort hook. Returns **true to continue**, false to abort — the
/// `whisper_encoder_begin_callback` convention on the ASR surface, and the
/// opposite of ggml's own. Called on the generating thread before each
/// prompt batch and each sampled token, and (CPU backend only) from inside
/// a running compute graph, so it must be cheap and non-blocking. It must
/// not re-enter the session that registered it — the session mutex is held.
pub type CrispasrChatAbortCallback = Option<unsafe extern "C" fn(user: *mut c_void) -> bool>;

extern "C" {
    // --- Params ---
    pub fn crispasr_chat_open_params_default(out: *mut CrispasrChatOpenParams);
    pub fn crispasr_chat_generate_params_default(out: *mut CrispasrChatGenerateParams);

    // --- Session lifecycle ---
    pub fn crispasr_chat_open(
        model_path: *const c_char,
        params: *const CrispasrChatOpenParams,
        err: *mut CrispasrChatError,
    ) -> *mut CrispasrChatSession;
    pub fn crispasr_chat_close(s: *mut CrispasrChatSession);
    pub fn crispasr_chat_reset(s: *mut CrispasrChatSession, err: *mut CrispasrChatError) -> i32;

    // --- Generation ---
    /// Returns a malloc'd UTF-8 string (free with
    /// [`crispasr_chat_string_free`]) or null on failure / abort.
    pub fn crispasr_chat_generate(
        s: *mut CrispasrChatSession,
        messages: *const CrispasrChatMessage,
        n_messages: usize,
        params: *const CrispasrChatGenerateParams,
        err: *mut CrispasrChatError,
    ) -> *mut c_char;

    /// 0 on clean completion (including stop-sequence / EOG termination),
    /// [`CRISPASR_CHAT_ERR_ABORTED`] when the abort callback stopped it,
    /// other non-zero on failure.
    pub fn crispasr_chat_generate_stream(
        s: *mut CrispasrChatSession,
        messages: *const CrispasrChatMessage,
        n_messages: usize,
        params: *const CrispasrChatGenerateParams,
        on_token: CrispasrChatOnToken,
        user: *mut c_void,
        err: *mut CrispasrChatError,
    ) -> i32;

    /// Register `cb` on the session (null clears it). Takes the session
    /// lock, so calling it during a generation blocks rather than
    /// cancelling — register before starting one.
    pub fn crispasr_chat_set_abort_callback(
        s: *mut CrispasrChatSession,
        cb: CrispasrChatAbortCallback,
        user: *mut c_void,
    );

    // --- Introspection ---
    pub fn crispasr_chat_template_name(s: *mut CrispasrChatSession) -> *const c_char;
    pub fn crispasr_chat_n_ctx(s: *mut CrispasrChatSession) -> i32;

    /// Prompt tokens a FRESH session prefills for `messages` — chat
    /// template, BOS, and the trailing generation prompt included.
    /// Negative on failure, with `err` filled.
    pub fn crispasr_chat_count_tokens(
        s: *mut CrispasrChatSession,
        messages: *const CrispasrChatMessage,
        n_messages: usize,
        err: *mut CrispasrChatError,
    ) -> i32;

    /// Approximate working set in bytes for a GGUF chat model on disk, or
    /// 0 when it could not be estimated (`err` filled).
    pub fn crispasr_chat_memory_estimate(
        model_path: *const c_char,
        params: *const CrispasrChatOpenParams,
        err: *mut CrispasrChatError,
    ) -> usize;

    pub fn crispasr_chat_string_free(s: *mut c_char);

    /// Canonical EU AI Act Art. 50(1) "you are talking to an AI" wording.
    /// Static string — never null, never freed.
    pub fn crispasr_chat_ai_disclosure_text() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guards the hand-maintained mirrors of the APPEND-ONLY structs in
    // src/crispasr_c_api.cpp. The C side reads every field unconditionally,
    // so a short layout here is an out-of-bounds read even for methods
    // that ignore the trailing fields (#332).
    #[test]
    fn diarize_abi_layout() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<CrispasrDiarizeSegAbi>(), 24);
        assert_eq!(size_of::<CrispasrDiarizeTurnAbi>(), 24);
        assert_eq!(offset_of!(CrispasrDiarizeTurnAbi, t1_cs), 8);
        assert_eq!(offset_of!(CrispasrDiarizeTurnAbi, speaker), 16);
        assert_eq!(size_of::<CrispasrDiarizeOptsAbi>(), 48);
        assert_eq!(offset_of!(CrispasrDiarizeOptsAbi, slice_t0_cs), 8);
        assert_eq!(offset_of!(CrispasrDiarizeOptsAbi, pyannote_model_path), 16);
        assert_eq!(
            offset_of!(CrispasrDiarizeOptsAbi, foxnose_embedder_path),
            24
        );
        assert_eq!(offset_of!(CrispasrDiarizeOptsAbi, min_speakers), 32);
        assert_eq!(offset_of!(CrispasrDiarizeOptsAbi, num_speakers), 40);
    }

    // Same guard for the hand-maintained mirrors of the POD structs in
    // include/crispasr_chat.h. Values are what the C compiler reports for
    // that header on a 64-bit target; the trailing `bool` in each params
    // struct is what makes the tail padding easy to get wrong.
    #[test]
    fn chat_abi_layout() {
        use std::mem::{offset_of, size_of};

        assert_eq!(size_of::<CrispasrChatError>(), 260);
        assert_eq!(offset_of!(CrispasrChatError, message), 4);

        assert_eq!(size_of::<CrispasrChatMessage>(), 16);
        assert_eq!(offset_of!(CrispasrChatMessage, content), 8);

        assert_eq!(size_of::<CrispasrChatOpenParams>(), 40);
        assert_eq!(offset_of!(CrispasrChatOpenParams, n_gpu_layers), 20);
        assert_eq!(offset_of!(CrispasrChatOpenParams, use_mmap), 24);
        assert_eq!(offset_of!(CrispasrChatOpenParams, use_mlock), 25);
        assert_eq!(offset_of!(CrispasrChatOpenParams, embeddings), 26);
        assert_eq!(offset_of!(CrispasrChatOpenParams, chat_template), 32);

        assert_eq!(size_of::<CrispasrChatGenerateParams>(), 56);
        assert_eq!(offset_of!(CrispasrChatGenerateParams, seed), 28);
        assert_eq!(offset_of!(CrispasrChatGenerateParams, stop), 32);
        assert_eq!(offset_of!(CrispasrChatGenerateParams, n_stop), 40);
        assert_eq!(offset_of!(CrispasrChatGenerateParams, prefill_only), 48);
    }
}
