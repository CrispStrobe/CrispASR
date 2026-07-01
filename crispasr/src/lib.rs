//! Safe Rust wrapper for CrispASR speech recognition.
//!
//! # Quick start
//!
//! ```no_run
//! use crispasr::Session;
//!
//! let sess = Session::open("model.gguf").unwrap();
//! let pcm = vec![0.0f32; 16000]; // 1s of silence
//! let segments = sess.transcribe(&pcm).unwrap();
//! for seg in &segments {
//!     println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
//! }
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

/// A transcription segment with timing information.
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start: f64, // seconds
    pub end: f64,   // seconds
    pub no_speech_prob: f32,
}

/// Options for `transcribe_pcm_with_options`. Leave defaults for standard
/// Whisper behaviour; set `vad: true` + `vad_model_path` for built-in
/// Silero VAD, or `tdrz: true` with a `.en.tdrz` model for speaker-turn
/// markers.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    pub strategy: Option<i32>,
    pub vad: bool,
    pub vad_model_path: Option<String>,
    pub vad_threshold: Option<f32>,
    pub vad_min_speech_ms: Option<i32>,
    pub vad_min_silence_ms: Option<i32>,
    pub tdrz: bool,
}

/// A loaded CrispASR model (whisper-only, legacy API).
///
/// **Deprecated:** Use [`Session`] instead. `CrispASR` wraps `whisper_full()`
/// directly without exception safety — C++ exceptions from ggml/whisper will
/// abort the process. `Session` uses the C-ABI wrapper which catches exceptions.
///
/// Not `Sync` — do not share between threads.
#[deprecated(
    since = "0.1.6",
    note = "Use Session::open() instead — CrispASR can abort on C++ exceptions"
)]
pub struct CrispASR {
    ctx: *mut crispasr_sys::WhisperContext,
}

unsafe impl Send for CrispASR {}

impl CrispASR {
    /// Load a GGUF/GGML whisper model file.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let cparams = unsafe { crispasr_sys::whisper_context_default_params_by_ref() };
        let ctx =
            unsafe { crispasr_sys::whisper_init_from_file_with_params(path.as_ptr(), cparams) };
        unsafe { crispasr_sys::whisper_free_context_params(cparams) };
        if ctx.is_null() {
            return Err(format!("failed to load model: {model_path}"));
        }
        Ok(Self { ctx })
    }

    /// Transcribe raw PCM audio (float32, mono, 16kHz).
    ///
    /// Returns a list of segments with text and timing.
    pub fn transcribe_pcm(&self, pcm: &[f32]) -> Result<Vec<Segment>, String> {
        self.transcribe_pcm_with_strategy(pcm, crispasr_sys::CRISPASR_SAMPLING_GREEDY)
    }

    /// Transcribe with a specific sampling strategy.
    pub fn transcribe_pcm_with_strategy(
        &self,
        pcm: &[f32],
        strategy: i32,
    ) -> Result<Vec<Segment>, String> {
        self.transcribe_pcm_with_options(
            pcm,
            &TranscribeOptions {
                strategy: Some(strategy),
                ..Default::default()
            },
        )
    }

    /// Transcribe with full option control — VAD, tinydiarize, and future
    /// knobs as they land upstream. Safe against older dylibs: setters
    /// that the loaded library doesn't expose are no-ops.
    pub fn transcribe_pcm_with_options(
        &self,
        pcm: &[f32],
        opts: &TranscribeOptions,
    ) -> Result<Vec<Segment>, String> {
        let strategy = opts
            .strategy
            .unwrap_or(crispasr_sys::CRISPASR_SAMPLING_GREEDY);
        let params = unsafe { crispasr_sys::whisper_full_default_params_by_ref(strategy) };

        // VAD
        if opts.vad {
            unsafe {
                crispasr_sys::crispasr_params_set_vad(params, 1);
                if let Some(t) = opts.vad_threshold {
                    crispasr_sys::crispasr_params_set_vad_threshold(params, t);
                }
                if let Some(ms) = opts.vad_min_speech_ms {
                    crispasr_sys::crispasr_params_set_vad_min_speech_ms(params, ms);
                }
                if let Some(ms) = opts.vad_min_silence_ms {
                    crispasr_sys::crispasr_params_set_vad_min_silence_ms(params, ms);
                }
            }
            // Keep the CString alive until after whisper_full returns.
            let vad_path_cstr = opts
                .vad_model_path
                .as_ref()
                .map(|s| CString::new(s.as_str()).ok())
                .flatten();
            if let Some(cs) = &vad_path_cstr {
                unsafe {
                    crispasr_sys::crispasr_params_set_vad_model_path(params, cs.as_ptr());
                }
            }
            // vad_path_cstr stays in scope for the whisper_full call below.
            return self.run_full(pcm, params, vad_path_cstr);
        }
        if opts.tdrz {
            unsafe { crispasr_sys::crispasr_params_set_tdrz(params, 1) };
        }

        self.run_full(pcm, params, None)
    }

    fn run_full(
        &self,
        pcm: &[f32],
        params: *mut crispasr_sys::WhisperFullParams,
        _keep_alive_vad_path: Option<CString>,
    ) -> Result<Vec<Segment>, String> {
        let ret =
            unsafe { crispasr_sys::whisper_full(self.ctx, params, pcm.as_ptr(), pcm.len() as i32) };
        unsafe { crispasr_sys::whisper_free_params(params) };

        if ret != 0 {
            return Err(format!("transcription failed (error code {ret})"));
        }

        let n = unsafe { crispasr_sys::whisper_full_n_segments(self.ctx) };
        let mut segments = Vec::with_capacity(n as usize);

        for i in 0..n {
            let text_ptr = unsafe { crispasr_sys::whisper_full_get_segment_text(self.ctx, i) };
            let text = if text_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(text_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            let t0 = unsafe { crispasr_sys::whisper_full_get_segment_t0(self.ctx, i) };
            let t1 = unsafe { crispasr_sys::whisper_full_get_segment_t1(self.ctx, i) };
            let nsp = unsafe { crispasr_sys::whisper_full_get_segment_no_speech_prob(self.ctx, i) };

            segments.push(Segment {
                text,
                start: t0 as f64 / 100.0,
                end: t1 as f64 / 100.0,
                no_speech_prob: nsp,
            });
        }

        Ok(segments)
    }

    /// Get the detected language from the last transcription.
    pub fn detected_language(&self) -> String {
        let id = unsafe { crispasr_sys::whisper_full_lang_id(self.ctx) };
        let ptr = unsafe { crispasr_sys::whisper_lang_str(id) };
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for CrispASR {
    fn drop(&mut self) {
        unsafe { crispasr_sys::whisper_free(self.ctx) }
    }
}

// =========================================================================
// Unified session — any CrispASR-supported backend through one handle.
//
// Prefer `Session::open` over `CrispASR::new` for new code: it dispatches
// automatically to whichever backend (Whisper, Parakeet, Canary, Cohere,
// Qwen3-ASR, Granite, FastConformer-CTC, Voxtral family, Wav2Vec2) the
// GGUF metadata specifies. `CrispASR` stays around for low-overhead
// Whisper-specific access and ABI stability.
// =========================================================================

/// Word-level timing (populated by backends that produce it).
#[derive(Debug, Clone)]
pub struct SessionWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Per-word probability in `[0, 1]`. Backends that don't emit a real
    /// per-word probability fall through to `1.0` so consumers can render
    /// uniformly. The C-side `crispasr_session_result_word_p` returns
    /// `-1.0` for "no data" — that case is folded to `1.0` here.
    pub confidence: f32,
}

/// A segment of a unified-session transcription.
#[derive(Debug, Clone)]
pub struct SessionSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Vec<SessionWord>,
}

/// A loaded session over a CrispASR model of any backend.
pub struct Session {
    handle: *mut crispasr_sys::CrispasrSession,
    n_threads: c_int,
}

// Not `Sync` — do not share between threads without external sync.
unsafe impl Send for Session {}

impl Session {
    /// Open a GGUF model, auto-detecting the backend from metadata.
    pub fn open(model_path: &str) -> Result<Self, String> {
        Self::open_inner(model_path, None, 4)
    }

    /// Open with an explicit backend (skips auto-detect).
    pub fn open_with_backend(
        model_path: &str,
        backend: &str,
        n_threads: i32,
    ) -> Result<Self, String> {
        Self::open_inner(model_path, Some(backend), n_threads)
    }

    fn open_inner(model_path: &str, backend: Option<&str>, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let handle = if let Some(be) = backend {
            let be_c = CString::new(be).map_err(|e| format!("invalid backend: {e}"))?;
            unsafe {
                crispasr_sys::crispasr_session_open_explicit(
                    path.as_ptr(),
                    be_c.as_ptr(),
                    n_threads,
                )
            }
        } else {
            unsafe { crispasr_sys::crispasr_session_open(path.as_ptr(), n_threads) }
        };
        if handle.is_null() {
            let avail = Self::available_backends().join(",");
            return Err(format!(
                "Failed to open {model_path:?}. Library was built with: [{avail}]"
            ));
        }
        Ok(Self { handle, n_threads })
    }

    /// List of backend names the loaded CrispASR library was compiled with.
    pub fn available_backends() -> Vec<String> {
        let mut buf = vec![0i8; 256];
        let mut n = unsafe {
            crispasr_sys::crispasr_session_available_backends(buf.as_mut_ptr(), buf.len() as i32)
        };
        if n <= 0 {
            return Vec::new();
        }
        if n as usize >= buf.len() {
            buf.resize(n as usize + 1, 0);
            n = unsafe {
                crispasr_sys::crispasr_session_available_backends(
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                )
            };
            if n <= 0 {
                return Vec::new();
            }
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        cstr.to_string_lossy()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Detect the backend from a GGUF file without opening it.
    pub fn detect_backend(model_path: &str) -> Result<String, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let mut buf = [0i8; 64];
        let n = unsafe {
            crispasr_sys::crispasr_detect_backend_from_gguf(
                path.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        };
        if n <= 0 {
            return Err(format!("backend detection failed (code {n})"));
        }
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Backend name this session ended up using.
    pub fn backend(&self) -> String {
        let p = unsafe { crispasr_sys::crispasr_session_backend(self.handle) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    /// Transcribe 16 kHz mono `f32` PCM. The internal dispatcher routes
    /// to whichever backend this session was opened with.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<SessionSegment>, String> {
        self.transcribe_with_language(pcm, None)
    }

    /// Language-aware transcribe (0.4.9+). `language` is an optional
    /// ISO 639-1 code ("en", "de", "ja", …). Backends that accept a
    /// source-language hint honour it; others ignore silently. `None`
    /// preserves each backend's historical default.
    pub fn transcribe_with_language(
        &self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<Vec<SessionSegment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let lang_c = match language {
            Some(l) if !l.is_empty() => {
                Some(CString::new(l).map_err(|e| format!("language NUL: {e}"))?)
            }
            _ => None,
        };
        let res = unsafe {
            match &lang_c {
                Some(c) => crispasr_sys::crispasr_session_transcribe_lang(
                    self.handle,
                    pcm.as_ptr(),
                    pcm.len() as i32,
                    c.as_ptr(),
                ),
                None => crispasr_sys::crispasr_session_transcribe(
                    self.handle,
                    pcm.as_ptr(),
                    pcm.len() as i32,
                ),
            }
        };
        self.parse_session_result(res, "crispasr_session_transcribe")
    }

    /// Chunked-encode transcribe (issue #208). Forces the Parakeet backend
    /// through its bounded long-form path (overlapping short-window
    /// transcribe-and-merge for non-JA models, streamed encoder for the
    /// JA-only model) regardless of audio length, so long files transcribe
    /// in bounded time AND recover the sections a single full-length pass
    /// drops (the decoder loses track past ~30 s; a single pass on a 5-min
    /// clip can omit half the words).
    ///
    /// `chunk_seconds <= 0` keeps the per-model defaults; otherwise it sets
    /// the non-JA window length / the JA streamed window. `overlap_seconds
    /// < 0` uses the default. For non-Parakeet backends the chunk parameters
    /// are inert and this is equivalent to [`Self::transcribe`].
    pub fn transcribe_chunked(
        &self,
        pcm: &[f32],
        chunk_seconds: i32,
        overlap_seconds: i32,
    ) -> Result<Vec<SessionSegment>, String> {
        self.transcribe_chunked_with_language(pcm, chunk_seconds, overlap_seconds, None)
    }

    /// Language-aware chunked-encode transcribe (issue #208). See
    /// [`Self::transcribe_chunked`] for the chunking semantics and
    /// [`Self::transcribe_with_language`] for the `language` semantics.
    pub fn transcribe_chunked_with_language(
        &self,
        pcm: &[f32],
        chunk_seconds: i32,
        overlap_seconds: i32,
        language: Option<&str>,
    ) -> Result<Vec<SessionSegment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let lang_c = match language {
            Some(l) if !l.is_empty() => {
                Some(CString::new(l).map_err(|e| format!("language NUL: {e}"))?)
            }
            _ => None,
        };
        let res = unsafe {
            crispasr_sys::crispasr_session_transcribe_chunked_lang(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as i32,
                chunk_seconds,
                overlap_seconds,
                lang_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            )
        };
        self.parse_session_result(res, "crispasr_session_transcribe_chunked")
    }

    /// Chunked-encode transcribe with a per-window progress callback
    /// (issue #208). `progress(processed_samples, total_samples)` is invoked
    /// once per finished window on the calling thread; `processed` is
    /// monotonically non-decreasing and reaches `total` on the last window.
    /// The callback only fires for the duration of this call, so it need not
    /// be `Send` or `'static`. Short (single-pass) audio and non-Parakeet
    /// backends do not fire it. See [`Self::transcribe_chunked`] for the
    /// chunking semantics.
    pub fn transcribe_chunked_with_progress<F: FnMut(i32, i32)>(
        &self,
        pcm: &[f32],
        chunk_seconds: i32,
        overlap_seconds: i32,
        language: Option<&str>,
        mut progress: F,
    ) -> Result<Vec<SessionSegment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let lang_c = match language {
            Some(l) if !l.is_empty() => {
                Some(CString::new(l).map_err(|e| format!("language NUL: {e}"))?)
            }
            _ => None,
        };

        extern "C" fn trampoline<F: FnMut(i32, i32)>(processed: c_int, total: c_int, ud: *mut c_void) {
            if ud.is_null() {
                return;
            }
            // SAFETY: `ud` is the `&mut F` registered just below. The C side
            // only invokes this synchronously from within the transcribe call
            // (same thread), so the reference is live and unaliased here.
            let f = unsafe { &mut *(ud as *mut F) };
            f(processed, total);
        }

        // Register for the duration of this call only, then clear — a raw
        // pointer to a stack closure must never outlive this frame.
        unsafe {
            crispasr_sys::crispasr_session_set_progress_callback(
                self.handle,
                Some(trampoline::<F>),
                &mut progress as *mut F as *mut c_void,
            );
        }
        let res = unsafe {
            crispasr_sys::crispasr_session_transcribe_chunked_lang(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as i32,
                chunk_seconds,
                overlap_seconds,
                lang_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            )
        };
        unsafe {
            crispasr_sys::crispasr_session_set_progress_callback(
                self.handle,
                None,
                std::ptr::null_mut(),
            );
        }
        self.parse_session_result(res, "crispasr_session_transcribe_chunked")
    }

    /// Parse a raw session-result handle into [`SessionSegment`]s and free
    /// it. `ctx` names the call site for the null-result error message.
    fn parse_session_result(
        &self,
        res: *mut crispasr_sys::CrispasrSessionResult,
        ctx: &str,
    ) -> Result<Vec<SessionSegment>, String> {
        if res.is_null() {
            return Err(format!("{ctx} failed for backend {:?}", self.backend()));
        }

        let mut out = Vec::new();
        unsafe {
            let n = crispasr_sys::crispasr_session_result_n_segments(res);
            for i in 0..n {
                let tp = crispasr_sys::crispasr_session_result_segment_text(res, i);
                let text = if tp.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(tp).to_string_lossy().into_owned()
                };
                let t0 = crispasr_sys::crispasr_session_result_segment_t0(res, i) as f64 / 100.0;
                let t1 = crispasr_sys::crispasr_session_result_segment_t1(res, i) as f64 / 100.0;

                let wn = crispasr_sys::crispasr_session_result_n_words(res, i);
                let mut words = Vec::with_capacity(wn as usize);
                for j in 0..wn {
                    let wtp = crispasr_sys::crispasr_session_result_word_text(res, i, j);
                    let wt = if wtp.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(wtp).to_string_lossy().into_owned()
                    };
                    let raw_p = crispasr_sys::crispasr_session_result_word_p(res, i, j);
                    words.push(SessionWord {
                        text: wt,
                        start: crispasr_sys::crispasr_session_result_word_t0(res, i, j) as f64
                            / 100.0,
                        end: crispasr_sys::crispasr_session_result_word_t1(res, i, j) as f64
                            / 100.0,
                        confidence: if raw_p < 0.0 { 1.0 } else { raw_p },
                    });
                }
                out.push(SessionSegment {
                    text: text.trim().to_string(),
                    start: t0,
                    end: t1,
                    words,
                });
            }
            crispasr_sys::crispasr_session_result_free(res);
        }
        Ok(out)
    }

    /// Transcribe with Silero VAD segmentation + crispasr-style stitching.
    ///
    /// Runs VAD on the PCM buffer, merges short / overlong speech slices
    /// into usable chunks, stitches them into a single buffer with 0.1s
    /// silence gaps, calls the backend once, then remaps segment + word
    /// timestamps back to original-audio positions.
    ///
    /// `vad_model_path` must point to a Silero GGUF on disk. Passing
    /// `None` for `opts` uses the library defaults (mirroring
    /// crispasr's `whisper_vad_default_params`).
    ///
    /// Compared to a fixed-chunk loop, stitching preserves cross-segment
    /// decoder context, which matters for O(T²) backends such as parakeet
    /// / cohere / canary. Falls back to a plain [`Self::transcribe`] call
    /// when no speech is detected or the VAD model fails to load.
    pub fn transcribe_vad(
        &self,
        pcm: &[f32],
        vad_model_path: &str,
        opts: Option<VadOptions>,
    ) -> Result<Vec<SessionSegment>, String> {
        self.transcribe_vad_with_language(pcm, vad_model_path, opts, None)
    }

    /// Language-aware VAD transcribe (0.4.9+). Accepts an ISO 639-1
    /// code that's forwarded into the backend's source-language hint.
    /// See [`Self::transcribe_with_language`] for the full semantics.
    pub fn transcribe_vad_with_language(
        &self,
        pcm: &[f32],
        vad_model_path: &str,
        opts: Option<VadOptions>,
        language: Option<&str>,
    ) -> Result<Vec<SessionSegment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        let path_c = CString::new(vad_model_path)
            .map_err(|e| format!("vad_model_path contains NUL byte: {e}"))?;
        let abi_opts = opts.unwrap_or_default().to_abi();
        let lang_c = match language {
            Some(l) if !l.is_empty() => {
                Some(CString::new(l).map_err(|e| format!("language NUL: {e}"))?)
            }
            _ => None,
        };

        let res = unsafe {
            match &lang_c {
                Some(c) => crispasr_sys::crispasr_session_transcribe_vad_lang(
                    self.handle,
                    pcm.as_ptr(),
                    pcm.len() as i32,
                    16_000,
                    path_c.as_ptr(),
                    &abi_opts,
                    c.as_ptr(),
                ),
                None => crispasr_sys::crispasr_session_transcribe_vad(
                    self.handle,
                    pcm.as_ptr(),
                    pcm.len() as i32,
                    16_000,
                    path_c.as_ptr(),
                    &abi_opts,
                ),
            }
        };
        if res.is_null() {
            return Err(format!(
                "crispasr_session_transcribe_vad failed for backend {:?}",
                self.backend()
            ));
        }

        let mut out = Vec::new();
        unsafe {
            let n = crispasr_sys::crispasr_session_result_n_segments(res);
            for i in 0..n {
                let tp = crispasr_sys::crispasr_session_result_segment_text(res, i);
                let text = if tp.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(tp).to_string_lossy().into_owned()
                };
                let t0 = crispasr_sys::crispasr_session_result_segment_t0(res, i) as f64 / 100.0;
                let t1 = crispasr_sys::crispasr_session_result_segment_t1(res, i) as f64 / 100.0;

                let wn = crispasr_sys::crispasr_session_result_n_words(res, i);
                let mut words = Vec::with_capacity(wn as usize);
                for j in 0..wn {
                    let wtp = crispasr_sys::crispasr_session_result_word_text(res, i, j);
                    let wt = if wtp.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(wtp).to_string_lossy().into_owned()
                    };
                    let raw_p = crispasr_sys::crispasr_session_result_word_p(res, i, j);
                    words.push(SessionWord {
                        text: wt,
                        start: crispasr_sys::crispasr_session_result_word_t0(res, i, j) as f64
                            / 100.0,
                        end: crispasr_sys::crispasr_session_result_word_t1(res, i, j) as f64
                            / 100.0,
                        confidence: if raw_p < 0.0 { 1.0 } else { raw_p },
                    });
                }
                out.push(SessionSegment {
                    text: text.trim().to_string(),
                    start: t0,
                    end: t1,
                    words,
                });
            }
            crispasr_sys::crispasr_session_result_free(res);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------------
    // TTS synthesis (vibevoice, qwen3-tts)
    // ---------------------------------------------------------------------

    /// Load a separate codec GGUF (qwen3-tts only; no-op for others).
    pub fn set_codec_path(&self, path: &str) -> Result<(), String> {
        let cpath = CString::new(path).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_codec_path(self.handle, cpath.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_codec_path failed (rc={})", rc));
        }
        Ok(())
    }

    /// Load a voice prompt: a baked GGUF voice pack OR a *.wav reference
    /// (qwen3-tts requires `ref_text` for *.wav inputs).
    ///
    /// For orpheus voice selection is BY NAME — use [`set_speaker_name`]
    /// instead.
    pub fn set_voice(&self, path: &str, ref_text: Option<&str>) -> Result<(), String> {
        let cpath = CString::new(path).map_err(|e| e.to_string())?;
        let crt = match ref_text {
            Some(t) => Some(CString::new(t).map_err(|e| e.to_string())?),
            None => None,
        };
        let rt_ptr = crt.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_voice(self.handle, cpath.as_ptr(), rt_ptr)
        };
        if rc != 0 {
            return Err(format!("set_voice failed (rc={})", rc));
        }
        Ok(())
    }

    /// Select a fixed/preset speaker by NAME for backends that bake names
    /// into the GGUF (orpheus). Names are e.g. `"tara"`/`"leo"` for
    /// canopylabs English; `"Anton"`/`"Sophie"` for Kartoffel_Orpheus DE.
    /// Use [`speakers`] to enumerate.
    pub fn set_speaker_name(&self, name: &str) -> Result<(), String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_speaker_name(self.handle, cname.as_ptr()) };
        match rc {
            0 => Ok(()),
            -2 => Err(format!(
                "unknown speaker {:?}; call .speakers() to enumerate",
                name
            )),
            -3 => Err("backend has no preset speakers; use set_voice() instead".to_string()),
            _ => Err(format!("set_speaker_name failed (rc={})", rc)),
        }
    }

    /// Return the list of preset speaker names for the active backend.
    /// Empty if the backend has no preset-speaker contract.
    pub fn speakers(&self) -> Vec<String> {
        let n = unsafe { crispasr_sys::crispasr_session_n_speakers(self.handle) };
        let mut out = Vec::with_capacity(n.max(0) as usize);
        for i in 0..n {
            let ptr = unsafe { crispasr_sys::crispasr_session_get_speaker_name(self.handle, i) };
            if !ptr.is_null() {
                let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned();
                out.push(s);
            }
        }
        out
    }

    /// Synthesise `text` to 24 kHz mono PCM. Requires a TTS-capable backend
    /// (`vibevoice`, `qwen3-tts`, `kokoro`, `orpheus`).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, String> {
        let ctext = CString::new(text).map_err(|e| e.to_string())?;
        let mut n: c_int = 0;
        let ptr = unsafe {
            crispasr_sys::crispasr_session_synthesize(
                self.handle,
                ctext.as_ptr(),
                &mut n as *mut c_int,
            )
        };
        if ptr.is_null() || n <= 0 {
            return Err(format!(
                "synthesize returned no audio for backend {:?}",
                self.backend()
            ));
        }
        let out = unsafe { std::slice::from_raw_parts(ptr, n as usize).to_vec() };
        unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
        Ok(out)
    }

    /// Drop the kokoro per-session phoneme cache. No-op for non-kokoro
    /// backends. Useful for long-running daemons that resynthesize across
    /// many speakers and want bounded memory. (PLAN #56 #5)
    pub fn clear_phoneme_cache(&self) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_kokoro_clear_phoneme_cache(self.handle) };
        if rc != 0 {
            return Err(format!("clear_phoneme_cache failed (rc={})", rc));
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Sticky session-state setters (PLAN #59 partial unblock).
    // -----------------------------------------------------------------

    /// Sticky source-language hint (canary, cohere, voxtral, whisper).
    /// Empty string clears. Per-call language arg passed to transcribe
    /// methods still wins.
    pub fn set_source_language(&self, lang: &str) -> Result<(), String> {
        let c = CString::new(lang).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_source_language(self.handle, c.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_source_language failed (rc={})", rc));
        }
        Ok(())
    }

    /// Sticky target-language. When set ≠ source on canary/cohere, the
    /// backend emits a translation. For whisper, pair with
    /// [`set_translate(true)`](Session::set_translate).
    pub fn set_target_language(&self, lang: &str) -> Result<(), String> {
        let c = CString::new(lang).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_target_language(self.handle, c.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_target_language failed (rc={})", rc));
        }
        Ok(())
    }

    /// Toggle punctuation + capitalisation in the output (canary/cohere
    /// natively; LLM backends via post-process strip). Default true.
    pub fn set_punctuation(&self, enable: bool) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_punctuation(self.handle, if enable { 1 } else { 0 })
        };
        if rc != 0 {
            return Err(format!("set_punctuation failed (rc={})", rc));
        }
        Ok(())
    }

    /// Whisper sticky `--translate`. For canary/cohere/voxtral the
    /// equivalent is `set_target_language` ≠ source.
    pub fn set_translate(&self, enable: bool) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_translate(self.handle, if enable { 1 } else { 0 })
        };
        if rc != 0 {
            return Err(format!("set_translate failed (rc={})", rc));
        }
        Ok(())
    }

    /// Translate `text` from `src_lang` to `tgt_lang` via whichever
    /// MT-capable backend this session loaded (m2m100, m2m100-wmt21,
    /// madlad, gemma4-e2b).  Distinct from [`Self::set_translate`] —
    /// that one is the whisper *audio-side* EN-only translate flag,
    /// applied to PCM input; this one is text→text with arbitrary
    /// language pairs.
    ///
    /// `max_tokens` caps the decoder output length.  Pass `<= 0` to
    /// fall back to the C++ default (200 tokens for m2m100, etc.).
    ///
    /// Backend selection guidance (from the README feature matrix):
    /// - **m2m100** — 100 languages, any-to-any (default for the long
    ///   tail like Bosnian, Swahili, …).
    /// - **m2m100-wmt21** — English-paired only (EN ↔ {zh, de, fr,
    ///   ja, ru, is, ha}), direction-specific checkpoints.  Higher
    ///   quality on those pairs.
    /// - **madlad** — 419 languages via target-language prefix tag
    ///   (handled internally; caller still passes `tgt_lang`).
    /// - **gemma4-e2b** — Dual ASR+MT (140+ langs).
    ///
    /// Errors when:
    /// - `text`, `src_lang`, or `tgt_lang` contain interior NULs;
    /// - the session has no MT-capable backend loaded (returns
    ///   `nullptr` from the C-ABI, surfaced as a clear error);
    /// - the backend's internal translate routine errored out.
    pub fn translate_text(
        &self,
        text: &str,
        src_lang: &str,
        tgt_lang: &str,
        max_tokens: i32,
    ) -> Result<String, String> {
        let ctext = CString::new(text).map_err(|e| format!("text contains NUL: {e}"))?;
        let csrc = CString::new(src_lang).map_err(|e| format!("src_lang contains NUL: {e}"))?;
        let ctgt = CString::new(tgt_lang).map_err(|e| format!("tgt_lang contains NUL: {e}"))?;
        let ptr = unsafe {
            crispasr_sys::crispasr_session_translate_text(
                self.handle,
                ctext.as_ptr(),
                csrc.as_ptr(),
                ctgt.as_ptr(),
                max_tokens,
            )
        };
        if ptr.is_null() {
            return Err(format!(
                "translate_text returned no output (backend {:?} may not be MT-capable, \
                 or the pair {}→{} is unsupported)",
                self.backend(),
                src_lang,
                tgt_lang
            ));
        }
        // Same CStr → owned String pattern as `PuncModel::process` — the
        // C side malloc'd this buffer and we own it until we hand it
        // back through `crispasr_session_translate_text_free`.
        let out = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { crispasr_sys::crispasr_session_translate_text_free(ptr) };
        Ok(out)
    }

    /// Open a rolling-window streaming decoder for this session
    /// (PLAN #62). Currently whisper-only at the C-ABI level; other
    /// backends return an error. `step_ms` is how often to commit a
    /// partial transcript (default 3000); `length_ms` is the rolling
    /// window size (default 10000); `keep_ms` is the trailing audio
    /// carried over (default 200). `language` empty = auto-detect;
    /// `translate` enables EN-target speech translation (whisper).
    pub fn stream_open(
        &self,
        step_ms: i32,
        length_ms: i32,
        keep_ms: i32,
        language: &str,
        translate: bool,
    ) -> Result<Stream, String> {
        self.stream_open_ex(step_ms, length_ms, keep_ms, language, translate, false)
    }

    /// Like [`stream_open`](Session::stream_open) but with the voxtral4b
    /// live-captions toggle. When `live` is true, decode runs during
    /// `feed()` so `get_text()` returns progressive transcript as audio
    /// arrives (PLAN #7 phase 3). No-op for backends without audio-injection
    /// prompt decode.
    pub fn stream_open_ex(
        &self,
        step_ms: i32,
        length_ms: i32,
        keep_ms: i32,
        language: &str,
        translate: bool,
        live: bool,
    ) -> Result<Stream, String> {
        let lang_c = CString::new(language).map_err(|e| e.to_string())?;
        let h = unsafe {
            crispasr_sys::crispasr_session_stream_open(
                self.handle,
                self.n_threads,
                step_ms,
                length_ms,
                keep_ms,
                lang_c.as_ptr(),
                if translate { 1 } else { 0 },
            )
        };
        if h.is_null() {
            return Err(format!(
                "stream_open failed for backend {:?}",
                self.backend()
            ));
        }
        if live {
            unsafe { crispasr_sys::crispasr_stream_set_live_decode(h, 1) };
        }
        Ok(Stream { handle: h })
    }

    /// Set decoder temperature on backends that support runtime control
    /// (canary, cohere, parakeet, moonshine). Other backends silently
    /// no-op. `seed` is the RNG seed; pass 0 for time-based.
    pub fn set_temperature(&self, temperature: f32, seed: u64) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_temperature(self.handle, temperature, seed)
        };
        // rc == -2 means no backend supports it — soft no-op.
        if rc != 0 && rc != -2 {
            return Err(format!("set_temperature failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the RNG seed for sampling-capable TTS backends that expose a
    /// session-level seed override (chatterbox, vibevoice, qwen3-tts,
    /// orpheus). Other backends silently no-op.
    pub fn set_tts_seed(&self, seed: u64) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_tts_seed(self.handle, seed) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_tts_seed failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set a generated-token cap for autoregressive session backends.
    /// Pass `<= 0` to clear the override and use the backend default.
    pub fn set_max_new_tokens(&self, max_new_tokens: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_max_new_tokens(self.handle, max_new_tokens)
        };
        if rc != 0 {
            return Err(format!("set_max_new_tokens failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set an opt-in repeated generated-token penalty for autoregressive
    /// session backends. Pass `<= 0.0` to disable it.
    pub fn set_frequency_penalty(&self, penalty: f32) -> Result<(), String> {
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_frequency_penalty(self.handle, penalty) };
        if rc != 0 {
            return Err(format!("set_frequency_penalty failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the diffusion / CFM step count for diffusion-based TTS backends
    /// (chatterbox today). Other backends silently no-op.
    pub fn set_tts_steps(&self, steps: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_tts_steps(self.handle, steps) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_tts_steps failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the number of flow-matching timing candidates ranked per token
    /// (TADA). Higher = more reliable multilingual timing at higher cost.
    /// Other backends silently no-op.
    pub fn set_tts_num_candidates(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_tts_num_candidates(self.handle, n) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_tts_num_candidates failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the top-p nucleus-sampling threshold. Honoured by chatterbox.
    pub fn set_top_p(&self, top_p: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_top_p(self.handle, top_p) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_top_p failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the top-k sampling cutoff (0 = disabled). Honoured by TADA.
    pub fn set_top_k(&self, top_k: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_top_k(self.handle, top_k) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_top_k failed (rc={})", rc));
        }
        Ok(())
    }

    /// Enable/disable sampling (`false` = greedy). Honoured by TADA.
    pub fn set_do_sample(&self, enable: bool) -> Result<(), String> {
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_do_sample(self.handle, enable as i32) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_do_sample failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the min-p sampling threshold. Honoured by chatterbox.
    pub fn set_min_p(&self, min_p: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_min_p(self.handle, min_p) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_min_p failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the repetition penalty (1.0 = no penalty). Honoured by chatterbox.
    pub fn set_repetition_penalty(&self, r: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_repetition_penalty(self.handle, r) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_repetition_penalty failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the classifier-free-guidance weight (chatterbox). 0 disables CFG;
    /// 0.5 is the upstream default.
    pub fn set_cfg_weight(&self, cfg_weight: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_cfg_weight(self.handle, cfg_weight) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_cfg_weight failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the TADA flow-matching noise temperature (Python noise_temp,
    /// default 0.9).
    pub fn set_tts_noise_temp(&self, noise_temp: f32) -> Result<(), String> {
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_tts_noise_temp(self.handle, noise_temp) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_tts_noise_temp failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the emotion-exaggeration scalar (chatterbox). 0.5 is the upstream default.
    pub fn set_exaggeration(&self, exaggeration: f32) -> Result<(), String> {
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_exaggeration(self.handle, exaggeration) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_exaggeration failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the upper bound on speech tokens per synthesize call (chatterbox).
    /// Default ≈1000 tokens ≈ 20 s.
    pub fn set_max_speech_tokens(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_max_speech_tokens(self.handle, n) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_max_speech_tokens failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the per-phoneme length-scale / speaking-rate scalar. Honoured by
    /// kokoro today; other backends silently no-op. 1.0 = upstream default.
    pub fn set_length_scale(&self, scale: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_length_scale(self.handle, scale) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_length_scale failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the best-of-N sampling count for ASR backends.
    pub fn set_best_of(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_best_of(self.handle, n) };
        if rc != 0 {
            return Err(format!("set_best_of failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set the beam-search width for ASR backends that support it.
    pub fn set_beam_size(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_beam_size(self.handle, n) };
        if rc != 0 {
            return Err(format!("set_beam_size failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set a GBNF grammar for constrained whisper decoding. Pass an empty
    /// string for `gbnf_text` to clear the grammar. `penalty` defaults to 100.0.
    pub fn set_grammar_text(
        &self,
        gbnf_text: &str,
        root_rule: &str,
        penalty: f32,
    ) -> Result<(), String> {
        let cgbnf = CString::new(gbnf_text).map_err(|e| e.to_string())?;
        let croot = CString::new(root_rule).map_err(|e| e.to_string())?;
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_grammar_text(
                self.handle,
                cgbnf.as_ptr(),
                croot.as_ptr(),
                penalty,
            )
        };
        if rc == -2 {
            return Err("set_grammar_text: invalid GBNF or root rule not found".into());
        }
        if rc != 0 {
            return Err(format!("set_grammar_text failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set whisper decoder fallback thresholds. `temperature_inc = 0.0`
    /// disables fallback entirely (equivalent to `--no-fallback`).
    pub fn set_fallback_thresholds(
        &self,
        entropy_thold: f32,
        logprob_thold: f32,
        no_speech_thold: f32,
        temperature_inc: f32,
    ) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_fallback_thresholds(
                self.handle,
                entropy_thold,
                logprob_thold,
                no_speech_thold,
                temperature_inc,
            )
        };
        if rc != 0 {
            return Err(format!("set_fallback_thresholds failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set per-token top-N alternative-candidate capture for whisper greedy
    /// decode. 0 disables it.
    pub fn set_alt_n(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_alt_n(self.handle, n) };
        if rc != 0 {
            return Err(format!("set_alt_n failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set whisper-only text-suppression and prompt-carry extras.
    /// `suppress_regex` may be empty to clear any prior regex.
    pub fn set_whisper_decode_extras(
        &self,
        suppress_nst: bool,
        suppress_regex: &str,
        carry_initial_prompt: bool,
    ) -> Result<(), String> {
        let cregex = CString::new(suppress_regex).map_err(|e| e.to_string())?;
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_whisper_decode_extras(
                self.handle,
                suppress_nst as c_int,
                cregex.as_ptr(),
                carry_initial_prompt as c_int,
            )
        };
        if rc != 0 {
            return Err(format!("set_whisper_decode_extras failed (rc={})", rc));
        }
        Ok(())
    }

    /// Set a free-form prompt / question passed to the backend on the next
    /// transcribe or synthesize call (used by LLM-style backends).
    pub fn set_ask(&self, prompt: &str) -> Result<(), String> {
        let cprompt = CString::new(prompt).map_err(|e| e.to_string())?;
        let rc = unsafe { crispasr_sys::crispasr_session_set_ask(self.handle, cprompt.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_ask failed (rc={})", rc));
        }
        Ok(())
    }

    /// qwen3-tts VoiceDesign: natural-language voice description.
    pub fn set_instruct(&self, instruct: &str) -> Result<(), String> {
        let c = CString::new(instruct).map_err(|e| e.to_string())?;
        let rc = unsafe { crispasr_sys::crispasr_session_set_instruct(self.handle, c.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_instruct failed (rc={})", rc));
        }
        Ok(())
    }

    /// Select + load a punctuation-restoration model (`auto`/`firered`/`fullstop`/
    /// `punctuate-all`/`pcs`/path; `"none"`/`""` unloads). Auto-downloads on first use.
    pub fn set_punc_model(&self, punc_model: &str) -> Result<(), String> {
        let c = CString::new(punc_model).map_err(|e| e.to_string())?;
        let rc = unsafe { crispasr_sys::crispasr_session_set_punc_model(self.handle, c.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_punc_model failed (rc={})", rc));
        }
        Ok(())
    }

    /// Comma-separated hotwords for contextual biasing, boosted by `boost` per
    /// token match. Empty string clears.
    pub fn set_hotwords(&self, hotwords: &str, boost: f32) -> Result<(), String> {
        let c = CString::new(hotwords).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_hotwords(self.handle, c.as_ptr(), boost) };
        if rc != 0 {
            return Err(format!("set_hotwords failed (rc={})", rc));
        }
        Ok(())
    }

    /// Select the G2P pronunciation dictionary for TTS (`olaph`/`open-dict`/path).
    pub fn set_g2p_dict(&self, source: &str) -> Result<(), String> {
        let c = CString::new(source).map_err(|e| e.to_string())?;
        let rc = unsafe { crispasr_sys::crispasr_session_set_g2p_dict(self.handle, c.as_ptr()) };
        if rc != 0 {
            return Err(format!("set_g2p_dict failed (rc={})", rc));
        }
        Ok(())
    }

    /// Select a multi-speaker backend's speaker by index.
    pub fn set_speaker_id(&self, id: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_speaker_id(self.handle, id) };
        if rc != 0 {
            return Err(format!("set_speaker_id failed (rc={})", rc));
        }
        Ok(())
    }

    /// Auto-detect spoken language on raw 16 kHz mono PCM.
    ///
    /// `method`: 0=Whisper, 1=Silero (default), 2=Firered, 3=Ecapa.
    /// Returns `(iso2_code, confidence_in_0_to_1)`.
    pub fn detect_language(
        &self,
        pcm: &[f32],
        lid_model_path: &str,
        method: i32,
    ) -> Result<(String, f32), String> {
        let cpath = CString::new(lid_model_path).map_err(|e| e.to_string())?;
        let mut buf = [0u8; 16];
        let mut prob: c_float = 0.0;
        let rc = unsafe {
            crispasr_sys::crispasr_session_detect_language(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                cpath.as_ptr(),
                method as c_int,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                &mut prob as *mut c_float,
            )
        };
        if rc != 0 {
            return Err(format!("detect_language failed (rc={})", rc));
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
        Ok((cstr.to_string_lossy().into_owned(), prob))
    }
}

// ---------------------------------------------------------------------------
// Kokoro per-language routing (PLAN #56 opt 2b).
// ---------------------------------------------------------------------------

/// Result of [`kokoro_resolve_for_lang`]. Mirrors the Python wrapper's
/// `KokoroResolved` dataclass.
#[derive(Clone, Debug)]
pub struct KokoroResolved {
    /// Path to load — may differ from the input when a German backbone
    /// sibling (`kokoro-de-hui-base-*.gguf`) sits next to the official
    /// Kokoro-82M baseline.
    pub model_path: String,
    /// Per-language fallback voice path. `None` if `lang` already has a
    /// native Kokoro-82M voice or no candidate exists in the model dir.
    pub voice_path: Option<String>,
    /// Basename of the picked voice (e.g. "df_victoria"). Same nullity
    /// as `voice_path`.
    pub voice_name: Option<String>,
    /// True iff the model path was rewritten to the German backbone.
    pub backbone_swapped: bool,
}

/// Resolve the kokoro model + fallback voice for `lang`. Mirrors what
/// the CLI does for `--backend kokoro -l <lang>` (PLAN #56 opt 2b).
///
/// Wrappers should call this *before* opening the [`Session`] so the
/// routing kicks in even outside the CLI entry point. Identical to the
/// Python wrapper's `crispasr.kokoro_resolve_for_lang`.
pub fn kokoro_resolve_for_lang(model_path: &str, lang: &str) -> Result<KokoroResolved, String> {
    let cmodel = CString::new(model_path).map_err(|e| e.to_string())?;
    let clang = CString::new(lang).map_err(|e| e.to_string())?;
    let mut out_model = vec![0i8; 1024];
    let mut out_voice = vec![0i8; 1024];
    let mut out_picked = vec![0i8; 64];

    let mut backbone_swapped = false;
    unsafe {
        let rc = crispasr_sys::crispasr_kokoro_resolve_model_for_lang_abi(
            cmodel.as_ptr(),
            clang.as_ptr(),
            out_model.as_mut_ptr() as *mut c_char,
            out_model.len() as c_int,
        );
        if rc < 0 {
            return Err("kokoro_resolve_model_for_lang: buffer too small".into());
        }
        if rc == 0 {
            backbone_swapped = true;
        }
    }
    let model_resolved = unsafe { std::ffi::CStr::from_ptr(out_model.as_ptr() as *const c_char) }
        .to_string_lossy()
        .into_owned();
    let model_resolved = if model_resolved.is_empty() {
        model_path.to_string()
    } else {
        model_resolved
    };

    let (voice_path, voice_name) = unsafe {
        let rc = crispasr_sys::crispasr_kokoro_resolve_fallback_voice_abi(
            cmodel.as_ptr(),
            clang.as_ptr(),
            out_voice.as_mut_ptr() as *mut c_char,
            out_voice.len() as c_int,
            out_picked.as_mut_ptr() as *mut c_char,
            out_picked.len() as c_int,
        );
        if rc < 0 {
            return Err("kokoro_resolve_fallback_voice: buffer too small".into());
        }
        if rc == 0 {
            let p = std::ffi::CStr::from_ptr(out_voice.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned();
            let n = std::ffi::CStr::from_ptr(out_picked.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned();
            (Some(p), Some(n))
        } else {
            (None, None)
        }
    };

    Ok(KokoroResolved {
        model_path: model_resolved,
        voice_path,
        voice_name,
        backbone_swapped,
    })
}

/// Tunables for [`Session::transcribe_vad`]. Defaults mirror crispasr's
/// `whisper_vad_default_params` plus the max-chunk fallback the shared
/// library uses to bound encoder cost on long audio.
#[derive(Clone, Copy, Debug)]
pub struct VadOptions {
    pub threshold: f32,
    pub min_speech_duration_ms: i32,
    pub min_silence_duration_ms: i32,
    pub speech_pad_ms: i32,
    /// Max merged-segment length (seconds). 0 disables the split.
    pub chunk_seconds: i32,
    /// Threads used for Silero VAD inference only; the ASR backend keeps
    /// the count chosen at session open time.
    pub n_threads: i32,
}

impl Default for VadOptions {
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

impl VadOptions {
    fn to_abi(self) -> crispasr_sys::CrispasrVadAbiOpts {
        crispasr_sys::CrispasrVadAbiOpts {
            threshold: self.threshold,
            min_speech_duration_ms: self.min_speech_duration_ms,
            min_silence_duration_ms: self.min_silence_duration_ms,
            speech_pad_ms: self.speech_pad_ms,
            chunk_seconds: self.chunk_seconds,
            n_threads: self.n_threads,
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_session_close(self.handle) }
    }
}

// =========================================================================
// Streaming (PLAN #62) — rolling-window decoder for whisper today.
// =========================================================================

/// One commit from a streaming session — the latest concatenated text
/// plus its absolute audio-time bounds.
#[derive(Debug, Clone)]
pub struct StreamingUpdate {
    pub text: String,
    pub t0: f64,
    pub t1: f64,
    pub counter: i64,
}

/// Streaming-decoder handle returned by [`Session::stream_open`]. Feed
/// PCM with [`Stream::feed`], pull text with [`Stream::get_text`],
/// finalize with [`Stream::flush`]. Auto-closes on drop.
pub struct Stream {
    handle: *mut crispasr_sys::CrispasrStream,
}

unsafe impl Send for Stream {}

impl Stream {
    /// Toggle voxtral4b live-captions decode-during-feed (PLAN #7
    /// phase 3). When enabled, each new audio_embed produced during
    /// `feed` triggers one greedy decode step; tokens commit
    /// immediately to `out_text` and `get_text` returns progressive
    /// transcript. Set BEFORE the first feed for clean semantics.
    /// No-op on backends without audio-injection prompt decode.
    pub fn set_live_decode(&self, enabled: bool) {
        unsafe {
            crispasr_sys::crispasr_stream_set_live_decode(self.handle, if enabled { 1 } else { 0 })
        };
    }

    /// Push 16 kHz mono float32 PCM. Returns 0 if still buffering, 1
    /// if a new partial transcript is ready (call [`get_text`](Stream::get_text)).
    pub fn feed(&self, pcm: &[f32]) -> Result<i32, String> {
        let rc = unsafe {
            crispasr_sys::crispasr_stream_feed(self.handle, pcm.as_ptr(), pcm.len() as c_int)
        };
        if rc < 0 {
            return Err(format!("stream_feed failed (rc={})", rc));
        }
        Ok(rc)
    }

    /// Return the latest committed transcript + absolute audio-time
    /// bounds. `counter` increments per commit; same value = no new text.
    pub fn get_text(&self) -> Result<StreamingUpdate, String> {
        let mut buf = vec![0u8; 8192];
        let mut t0: f64 = 0.0;
        let mut t1: f64 = 0.0;
        let mut counter: i64 = 0;
        let rc = unsafe {
            crispasr_sys::crispasr_stream_get_text(
                self.handle,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                &mut t0,
                &mut t1,
                &mut counter,
            )
        };
        if rc < 0 {
            return Err(format!("stream_get_text failed (rc={})", rc));
        }
        let text = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .into_owned();
        Ok(StreamingUpdate {
            text,
            t0,
            t1,
            counter,
        })
    }

    /// Finalize any remaining buffered audio.
    pub fn flush(&self) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_stream_flush(self.handle) };
        if rc < 0 {
            return Err(format!("stream_flush failed (rc={})", rc));
        }
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_stream_close(self.handle) }
    }
}

// =========================================================================
// Microphone capture (PLAN #62d) — cross-platform via miniaudio.
// =========================================================================

use std::os::raw::c_void;
use std::sync::Mutex;

/// Library-level microphone handle. The user-supplied callback is
/// invoked from miniaudio's audio thread with mono float32 PCM in
/// [-1, 1]. Keep the callback short and non-blocking — for ASR, queue
/// the audio and feed [`Stream::feed`] from another thread.
///
/// Auto-stops + closes on drop.
pub struct Mic {
    handle: *mut crispasr_sys::CrispasrMic,
    _trampoline: Box<TrampolineState>,
}

unsafe impl Send for Mic {}

struct TrampolineState {
    cb: Mutex<Box<dyn FnMut(&[f32]) + Send + 'static>>,
}

extern "C" fn mic_trampoline(pcm: *const c_float, n_samples: c_int, userdata: *mut c_void) {
    if userdata.is_null() || pcm.is_null() || n_samples <= 0 {
        return;
    }
    unsafe {
        let state = &*(userdata as *const TrampolineState);
        let slice = std::slice::from_raw_parts(pcm, n_samples as usize);
        if let Ok(mut cb) = state.cb.lock() {
            (cb)(slice);
        }
    }
}

impl Mic {
    /// Open the default capture device. `sample_rate=16000` matches
    /// every ASR backend. Pass `channels=1` for mono (recommended);
    /// channels=2 hands the callback interleaved stereo.
    pub fn open<F>(sample_rate: i32, channels: i32, callback: F) -> Result<Mic, String>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        let trampoline = Box::new(TrampolineState {
            cb: Mutex::new(Box::new(callback)),
        });
        let userdata_ptr = trampoline.as_ref() as *const TrampolineState as *mut c_void;
        let handle = unsafe {
            crispasr_sys::crispasr_mic_open(sample_rate, channels, mic_trampoline, userdata_ptr)
        };
        if handle.is_null() {
            return Err("crispasr_mic_open failed".to_string());
        }
        Ok(Mic {
            handle,
            _trampoline: trampoline,
        })
    }

    pub fn start(&self) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_mic_start(self.handle) };
        if rc != 0 {
            return Err(format!("mic_start failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_mic_stop(self.handle) };
        if rc != 0 {
            return Err(format!("mic_stop failed (rc={})", rc));
        }
        Ok(())
    }
}

impl Drop for Mic {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_mic_close(self.handle) }
    }
}

/// Human-readable name of the default capture device, or empty string
/// if no input device is available.
pub fn mic_default_device_name() -> String {
    let p = unsafe { crispasr_sys::crispasr_mic_default_device_name() };
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

// =========================================================================
// HF download + cache + model registry (shared C-ABI, 0.4.8+)
// =========================================================================

/// Known-model registry entry.
#[derive(Clone, Debug)]
pub struct RegistryEntry {
    pub filename: String,
    pub url: String,
    pub approx_size: String,
}

/// Look up the canonical GGUF for a backend (whisper, parakeet, canary,
/// voxtral, voxtral4b, granite, granite-4.1, qwen3, cohere, wav2vec2). Returns `None`
/// on miss.
/// List every backend name in the registry, in declaration order.
///
/// Each name can be passed back to [`registry_lookup`] for full details
/// (filename, URL, approximate size).
pub fn list_known_models() -> Vec<String> {
    let mut buf = vec![0u8; 8192];
    let n = unsafe {
        crispasr_sys::crispasr_registry_list_backends_abi(
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        )
    };
    if n < 0 {
        return Vec::new();
    }
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
    cstr.to_string_lossy()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub fn registry_lookup(backend: &str) -> Result<Option<RegistryEntry>, String> {
    registry_call_inner(backend, true)
}

/// Look up by filename (exact match, then fuzzy substring).
pub fn registry_lookup_by_filename(filename: &str) -> Result<Option<RegistryEntry>, String> {
    registry_call_inner(filename, false)
}

fn registry_call_inner(key: &str, by_backend: bool) -> Result<Option<RegistryEntry>, String> {
    if key.is_empty() {
        return Ok(None);
    }
    let key_c = CString::new(key).map_err(|e| format!("key NUL: {e}"))?;
    let mut fn_buf = [0u8; 256];
    let mut url_buf = [0u8; 512];
    let mut size_buf = [0u8; 32];
    let rc = unsafe {
        if by_backend {
            crispasr_sys::crispasr_registry_lookup_abi(
                key_c.as_ptr(),
                fn_buf.as_mut_ptr() as *mut c_char,
                fn_buf.len() as i32,
                url_buf.as_mut_ptr() as *mut c_char,
                url_buf.len() as i32,
                size_buf.as_mut_ptr() as *mut c_char,
                size_buf.len() as i32,
            )
        } else {
            crispasr_sys::crispasr_registry_lookup_by_filename_abi(
                key_c.as_ptr(),
                fn_buf.as_mut_ptr() as *mut c_char,
                fn_buf.len() as i32,
                url_buf.as_mut_ptr() as *mut c_char,
                url_buf.len() as i32,
                size_buf.as_mut_ptr() as *mut c_char,
                size_buf.len() as i32,
            )
        }
    };
    if rc != 0 {
        return Ok(None);
    }
    fn slice_to_string(buf: &[u8]) -> String {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }
    Ok(Some(RegistryEntry {
        filename: slice_to_string(&fn_buf),
        url: slice_to_string(&url_buf),
        approx_size: slice_to_string(&size_buf),
    }))
}

/// Download `filename` from `url` into the CrispASR cache — or return
/// the cached path if already present. Pass `None` for
/// `cache_dir_override` to use the platform default.
pub fn cache_ensure_file(
    filename: &str,
    url: &str,
    quiet: bool,
    cache_dir_override: Option<&str>,
) -> Result<Option<String>, String> {
    if filename.is_empty() || url.is_empty() {
        return Ok(None);
    }
    let fn_c = CString::new(filename).map_err(|e| format!("filename NUL: {e}"))?;
    let url_c = CString::new(url).map_err(|e| format!("url NUL: {e}"))?;
    let ov_c = CString::new(cache_dir_override.unwrap_or(""))
        .map_err(|e| format!("cache_dir_override NUL: {e}"))?;
    let mut buf = vec![0u8; 2048];
    let rc = unsafe {
        crispasr_sys::crispasr_cache_ensure_file_abi(
            fn_c.as_ptr(),
            url_c.as_ptr(),
            if quiet { 1 } else { 0 },
            ov_c.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
        )
    };
    if rc != 0 {
        return Ok(None);
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(Some(String::from_utf8_lossy(&buf[..end]).into_owned()))
}

/// Return the CrispASR cache directory (creating it if missing).
pub fn cache_dir(override_path: Option<&str>) -> Result<Option<String>, String> {
    let ov_c =
        CString::new(override_path.unwrap_or("")).map_err(|e| format!("override NUL: {e}"))?;
    let mut buf = vec![0u8; 2048];
    let rc = unsafe {
        crispasr_sys::crispasr_cache_dir_abi(
            ov_c.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
        )
    };
    if rc != 0 {
        return Ok(None);
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(Some(String::from_utf8_lossy(&buf[..end]).into_owned()))
}

// =========================================================================
// CTC / forced-aligner word timings (shared C-ABI, 0.4.7+)
// =========================================================================

#[derive(Clone, Debug)]
pub struct AlignedWord {
    pub text: String,
    pub start: f64, // seconds
    pub end: f64,
}

/// Run CTC / forced-aligner word timings for a transcript + audio pair.
///
/// `aligner_model` filename picks the backend: paths containing
/// "forced-aligner" / "qwen3-fa" / "qwen3-forced" route to the
/// Qwen3-ForcedAligner path; everything else goes through
/// canary-ctc-aligner. `t_offset` (seconds) is added to every word
/// start/end so the returned timings are absolute against the
/// original audio.
///
/// Returns an empty vector when the aligner failed or produced no
/// output. Errors are printed to stderr by the library, since they
/// typically indicate a missing / wrong model file.
pub fn align_words(
    aligner_model: &str,
    transcript: &str,
    pcm: &[f32],
    t_offset: f64,
    n_threads: i32,
) -> Result<Vec<AlignedWord>, String> {
    if aligner_model.is_empty() || transcript.is_empty() || pcm.is_empty() {
        return Ok(Vec::new());
    }
    let model_c = CString::new(aligner_model).map_err(|e| format!("aligner_model NUL: {e}"))?;
    let trans_c = CString::new(transcript).map_err(|e| format!("transcript NUL: {e}"))?;

    let res = unsafe {
        crispasr_sys::crispasr_align_words_abi(
            model_c.as_ptr(),
            trans_c.as_ptr(),
            pcm.as_ptr(),
            pcm.len() as i32,
            (t_offset * 100.0).round() as i64,
            n_threads,
        )
    };
    if res.is_null() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    unsafe {
        let n = crispasr_sys::crispasr_align_result_n_words(res);
        for i in 0..n {
            let tp = crispasr_sys::crispasr_align_result_word_text(res, i);
            let text = if tp.is_null() {
                String::new()
            } else {
                CStr::from_ptr(tp).to_string_lossy().into_owned()
            };
            let t0 = crispasr_sys::crispasr_align_result_word_t0(res, i) as f64 / 100.0;
            let t1 = crispasr_sys::crispasr_align_result_word_t1(res, i) as f64 / 100.0;
            out.push(AlignedWord {
                text,
                start: t0,
                end: t1,
            });
        }
        crispasr_sys::crispasr_align_result_free(res);
    }
    Ok(out)
}

// =========================================================================
// Language identification (shared C-ABI, 0.4.6+)
// =========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LidMethod {
    /// Whisper encoder + language head. Needs a multilingual ggml-*.bin model.
    Whisper = 0,
    /// GGUF-packed Silero 95-language classifier.
    Silero = 1,
    /// FireRedTeam/FireRedLID — encoder + 6-layer LID Transformer, 120 langs.
    /// Wired through the same module-level `crispasr_detect_language` C-ABI
    /// as Whisper/Silero — no session required.
    Firered = 2,
    /// SpeechBrain ECAPA-TDNN VoxLingua107 — attentive statistical pooling,
    /// 107 langs.  Same module-level path as the others.
    Ecapa = 3,
}

#[derive(Clone, Debug)]
pub struct LidResult {
    /// ISO 639-1 language code (`"en"`, `"de"`, …). Empty on failure.
    pub lang_code: String,
    /// Posterior probability on the argmax language. `-1.0` on failure.
    pub confidence: f32,
}

/// Run language identification on a 16 kHz mono float PCM buffer.
///
/// `model_path` must point to a concrete model file on disk (the
/// whisper `ggml-*.bin` for [`LidMethod::Whisper`] or a Silero GGUF
/// for [`LidMethod::Silero`]). Auto-download / cache resolution is the
/// caller's responsibility; the CrispASR CLI has a helper for that,
/// wrappers can ship the model as an asset.
pub fn detect_language_pcm(
    pcm: &[f32],
    method: LidMethod,
    model_path: &str,
    n_threads: i32,
    use_gpu: bool,
    gpu_device: i32,
    flash_attn: bool,
) -> Result<LidResult, String> {
    if pcm.is_empty() || model_path.is_empty() {
        return Ok(LidResult {
            lang_code: String::new(),
            confidence: -1.0,
        });
    }
    let path_c = CString::new(model_path).map_err(|e| format!("model_path contains NUL: {e}"))?;

    let mut buf = [0u8; 16];
    let mut conf: c_float = -1.0;
    let rc = unsafe {
        crispasr_sys::crispasr_detect_language_pcm(
            pcm.as_ptr(),
            pcm.len() as i32,
            method as i32,
            path_c.as_ptr(),
            n_threads,
            if use_gpu { 1 } else { 0 },
            gpu_device,
            if flash_attn { 1 } else { 0 },
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
            &mut conf,
        )
    };
    if rc != 0 {
        return Ok(LidResult {
            lang_code: String::new(),
            confidence: -1.0,
        });
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let code = std::str::from_utf8(&buf[..end])
        .map_err(|e| format!("LID returned non-UTF8 bytes: {e}"))?
        .to_string();
    Ok(LidResult {
        lang_code: code,
        confidence: conf as f32,
    })
}

// =========================================================================
// Text-LID — P13.5 Phase 7 (C-ABI 0.5.2+)
// =========================================================================

/// Result of [`text_detect_language`].  Label format depends on the
/// loaded GGUF: CLD3 returns ISO 639-1 (`"en"`, `"de"`, `"zh-Latn"`)
/// across 109 labels; GlotLID-V3 / LID-176 fastText return ISO 639-3
/// with a script tag (`"eng_Latn"`, `"sco_Latn"`) across 2102 or 176
/// labels respectively.  Callers needing ISO 639-1 normalisation
/// must do it on their side — the dispatcher preserves the model's
/// native space because the script tag carries real information
/// (e.g. `zh-Latn` ≠ `zh-Hans`).
#[derive(Clone, Debug)]
pub struct TextLidResult {
    /// Predicted language label.  Empty on failure.  See type docs
    /// for the format details.
    pub label: String,
    /// Posterior probability on the argmax label.  `-1.0` on
    /// dispatcher failures, otherwise `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Detect the language of a UTF-8 text string via the internal
/// `text_lid_dispatch` (peek the GGUF's `general.architecture` →
/// route to CLD3 or fastText).
///
/// `model_path` must be a concrete on-disk path; auto-resolution
/// from the registry is the caller's job (use
/// `registry_lookup("lid-cld3" / "lid-glotlid" / "lid-fasttext176")`
/// + `cache_ensure_file` for the same shape the ASR side already
/// uses).
///
/// Errors when:
/// - any input string contains an interior NUL,
/// - the GGUF can't be opened or has an unsupported architecture,
/// - the predict path errors out,
/// - the output buffer (256 bytes here — fits CLD3's longest
///   `zh-Latn` and fastText's longest `<3-letter>_<4-letter>`
///   labels with room to spare) overflows.
pub fn text_detect_language(
    text: &str,
    model_path: &str,
    n_threads: i32,
) -> Result<TextLidResult, String> {
    let ctext = CString::new(text).map_err(|e| format!("text contains NUL: {e}"))?;
    let cmodel = CString::new(model_path).map_err(|e| format!("model_path contains NUL: {e}"))?;

    // 256-byte buffer comfortably fits every label format the
    // dispatcher emits — CLD3's longest is ~10 bytes (`zh-Latn`),
    // fastText's longest is ~12 bytes (`<3letter>_<4letter>`).
    let mut buf = [0u8; 256];
    let mut conf: c_float = -1.0;
    let rc = unsafe {
        crispasr_sys::crispasr_text_detect_language(
            ctext.as_ptr(),
            cmodel.as_ptr(),
            n_threads,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
            &mut conf,
        )
    };
    match rc {
        0 => {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            let label = std::str::from_utf8(&buf[..end])
                .map_err(|e| format!("text-LID returned non-UTF8 bytes: {e}"))?
                .to_string();
            Ok(TextLidResult {
                label,
                confidence: conf as f32,
            })
        }
        -1 => Err("text-LID: invalid args (null pointer or bad buffer size)".to_string()),
        1 => Err(format!(
            "text-LID dispatcher init/predict failed for model {model_path} \
             (check the GGUF's architecture key — must be `lid-cld3` or `lid-fasttext`)"
        )),
        2 => Err(
            "text-LID label exceeded 256-byte output buffer — file an issue, this shouldn't happen \
             with the dispatcher's current label spaces"
                .to_string(),
        ),
        other => Err(format!("text-LID returned unexpected status code {other}")),
    }
}

// =========================================================================
// Diarization (shared C-ABI, 0.4.5+)
// =========================================================================

/// One ASR segment passed to [`diarize_segments`]. Caller fills `t0` / `t1`
/// (seconds) from the upstream transcribe result; the diarizer writes the
/// zero-based speaker index into `speaker` (`-1` means the method had no
/// info to pick).
#[derive(Clone, Copy, Debug)]
pub struct DiarizeSegment {
    pub t0: f64,
    pub t1: f64,
    pub speaker: i32,
}

impl DiarizeSegment {
    pub fn new(t0: f64, t1: f64) -> Self {
        Self {
            t0,
            t1,
            speaker: -1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DiarizeMethod {
    /// Stereo only. |L| vs |R| energy per segment, 1.1× margin.
    Energy = 0,
    /// Stereo only. TDOA via cross-correlation, ±5 ms search window.
    Xcorr = 1,
    /// Mono-friendly. Alternates 0/1 every >600 ms gap.
    VadTurns = 2,
    /// Mono-friendly, ML-based. Runs the GGUF pyannote segmentation net;
    /// requires a model path.
    Pyannote = 3,
}

#[derive(Clone, Debug)]
pub struct DiarizeOptions {
    pub method: DiarizeMethod,
    /// GGUF path. Required for `Pyannote`, ignored otherwise.
    pub pyannote_model_path: Option<String>,
    /// Threads for pyannote inference; ignored by other methods.
    pub n_threads: i32,
    /// Absolute start (seconds) of the PCM buffer within the original
    /// audio, so the diarizer can map absolute segment timestamps back
    /// to sample indices.
    pub slice_t0: f64,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            method: DiarizeMethod::VadTurns,
            pyannote_model_path: None,
            n_threads: 4,
            slice_t0: 0.0,
        }
    }
}

/// Assign a speaker index to each of `segs`, mutating each
/// [`DiarizeSegment::speaker`] in place.
///
/// Four methods — see [`DiarizeMethod`]. `left` is mono PCM for
/// mono-only methods, otherwise the left channel of a stereo pair.
/// When `is_stereo` is true, `right` must be `Some`. All PCM is 16 kHz
/// float32.
///
/// Returns `Ok(())` on success. Only [`DiarizeMethod::Pyannote`] can
/// fail (model load failure).
pub fn diarize_segments(
    segs: &mut [DiarizeSegment],
    left: &[f32],
    right: Option<&[f32]>,
    is_stereo: bool,
    opts: &DiarizeOptions,
) -> Result<(), String> {
    if segs.is_empty() || left.is_empty() {
        return Ok(());
    }

    let path_c = match (&opts.pyannote_model_path, opts.method) {
        (Some(p), DiarizeMethod::Pyannote) => Some(
            CString::new(p.as_str())
                .map_err(|e| format!("pyannote_model_path contains NUL: {e}"))?,
        ),
        _ => None,
    };

    let abi_opts = crispasr_sys::CrispasrDiarizeOptsAbi {
        method: opts.method as i32,
        n_threads: opts.n_threads,
        slice_t0_cs: (opts.slice_t0 * 100.0).round() as i64,
        pyannote_model_path: path_c
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null()),
    };

    let mut abi_segs: Vec<crispasr_sys::CrispasrDiarizeSegAbi> = segs
        .iter()
        .map(|s| crispasr_sys::CrispasrDiarizeSegAbi {
            t0_cs: (s.t0 * 100.0).round() as i64,
            t1_cs: (s.t1 * 100.0).round() as i64,
            speaker: s.speaker,
            _pad: 0,
        })
        .collect();

    let right_ptr = match (is_stereo, right) {
        (true, Some(r)) => r.as_ptr(),
        _ => left.as_ptr(),
    };

    let rc = unsafe {
        crispasr_sys::crispasr_diarize_segments_abi(
            left.as_ptr(),
            right_ptr,
            left.len() as i32,
            if is_stereo { 1 } else { 0 },
            abi_segs.as_mut_ptr(),
            abi_segs.len() as i32,
            &abi_opts,
        )
    };
    match rc {
        0 => {
            for (i, s) in segs.iter_mut().enumerate() {
                s.speaker = abi_segs[i].speaker;
            }
            Ok(())
        }
        1 => Err("pyannote model load failed".to_string()),
        -1 => Err("invalid arguments to crispasr_diarize_segments_abi".to_string()),
        other => Err(format!("crispasr_diarize_segments_abi returned {other}")),
    }
}

// =========================================================================
// Pluggable speaker embedder + cosine clustering + pyannote cache (#107 P6)
// =========================================================================
//
// These are the same building blocks the CLI's `--diarize-embedder` path
// uses. Together they let a Rust caller compose the full diarization
// pipeline (pyannote segmentation -> per-speech-interval embeddings ->
// agglomerative clustering -> globally stable speaker IDs).

/// Pluggable speaker-embedding model. Wraps the
/// `crispasr_speaker_embedder_*_abi` family in a safe Rust struct.
pub struct SpeakerEmbedder {
    raw: *mut std::ffi::c_void,
}

impl SpeakerEmbedder {
    /// Build a speaker embedder by spec.
    ///
    /// `model_spec` accepts (case-insensitive):
    ///   - `"auto"` / `"titanet"` -> TitaNet-Large (192-d)
    ///   - `"indextts"` / `"indextts-bigvgan"` / `"ecapa"` ->
    ///     IndexTTS-BigVGAN ECAPA-TDNN (512-d)
    ///   - a `.gguf` path -> dispatched by filename
    pub fn new(model_spec: &str, n_threads: i32, cache_dir: Option<&str>) -> Result<Self, String> {
        let spec_c = std::ffi::CString::new(model_spec).map_err(|e| e.to_string())?;
        let cache_c = cache_dir
            .map(|s| std::ffi::CString::new(s))
            .transpose()
            .map_err(|e: std::ffi::NulError| e.to_string())?;
        let cache_ptr = cache_c
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());
        let raw = unsafe {
            crispasr_sys::crispasr_speaker_embedder_make_abi(spec_c.as_ptr(), n_threads, cache_ptr)
        };
        if raw.is_null() {
            return Err(format!("failed to build speaker embedder '{model_spec}'"));
        }
        Ok(Self { raw })
    }

    /// Output embedding dimension (e.g. 192 for TitaNet).
    pub fn dim(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_speaker_embedder_dim_abi(self.raw) }
    }

    /// Backend name for logging (e.g. "titanet-large").
    pub fn name(&self) -> String {
        unsafe {
            let p = crispasr_sys::crispasr_speaker_embedder_name_abi(self.raw);
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    /// Extract one embedding from mono 16 kHz f32 PCM. Returns `None`
    /// when the underlying model rejected the input (typically too
    /// short for its mel pipeline).
    pub fn embed(&self, pcm_16k: &[f32]) -> Option<Vec<f32>> {
        let dim = self.dim();
        if dim <= 0 || pcm_16k.is_empty() {
            return None;
        }
        let mut out = vec![0.0f32; dim as usize];
        let ok = unsafe {
            crispasr_sys::crispasr_speaker_embedder_embed_abi(
                self.raw,
                pcm_16k.as_ptr(),
                pcm_16k.len() as i32,
                out.as_mut_ptr(),
            )
        };
        if ok != 0 {
            Some(out)
        } else {
            None
        }
    }
}

impl Drop for SpeakerEmbedder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { crispasr_sys::crispasr_speaker_embedder_free_abi(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for SpeakerEmbedder {}

/// Agglomerative single-linkage cosine clustering on L2-normalized
/// speaker embeddings.
///
/// `embeddings` is a flat row-major `n * dim` buffer (e.g. four 192-d
/// vectors -> 768 floats). Stops at `merge_threshold` similarity or
/// when the count reaches `max_speakers`. Returns one cluster ID per
/// input in `[0, k)` assigned in first-appearance order.
pub fn agglomerative_cluster(
    embeddings: &[f32],
    n: i32,
    dim: i32,
    merge_threshold: f32,
    max_speakers: i32,
) -> Result<Vec<i32>, String> {
    if n <= 0 || dim <= 0 || embeddings.len() < (n as usize) * (dim as usize) {
        return Err("invalid arguments to agglomerative_cluster".to_string());
    }
    let mut out = vec![0i32; n as usize];
    let k = unsafe {
        crispasr_sys::crispasr_speaker_cluster_abi(
            embeddings.as_ptr(),
            n,
            dim,
            merge_threshold,
            max_speakers,
            out.as_mut_ptr(),
        )
    };
    if k < 0 {
        return Err("crispasr_speaker_cluster_abi returned -1".to_string());
    }
    Ok(out)
}

/// Pre-computed pyannote-seg posteriors over a full audio buffer.
///
/// Build once at the start of a diarize pipeline, then call
/// [`PyannoteCache::apply`] for each set of segment ranges. Gives
/// cross-slice consistency for pyannote-method diarization (#107 P2a)
/// without re-running the segmentation net per slice.
pub struct PyannoteCache {
    raw: *mut std::ffi::c_void,
}

impl PyannoteCache {
    /// Run pyannote-seg once over `pcm_16k` and cache the posteriors.
    pub fn compute(pcm_16k: &[f32], model_path: &str, n_threads: i32) -> Result<Self, String> {
        if pcm_16k.is_empty() {
            return Err("empty audio".to_string());
        }
        let model_c = std::ffi::CString::new(model_path).map_err(|e| e.to_string())?;
        let raw = unsafe {
            crispasr_sys::crispasr_pyannote_cache_compute_abi(
                pcm_16k.as_ptr(),
                pcm_16k.len() as i32,
                model_c.as_ptr(),
                n_threads,
            )
        };
        if raw.is_null() {
            return Err(format!(
                "failed to compute pyannote cache from '{model_path}'"
            ));
        }
        Ok(Self { raw })
    }

    /// Score `segs` against the cached posteriors. Each segment's
    /// `speaker` is set to `0/1/2` (local pyannote-seg track index) or
    /// `-1` for silence.
    pub fn apply(&self, segs: &mut [DiarizeSegment], slice_t0: f64) -> Result<(), String> {
        if segs.is_empty() {
            return Ok(());
        }
        let mut abi_segs: Vec<crispasr_sys::CrispasrDiarizeSegAbi> = segs
            .iter()
            .map(|s| crispasr_sys::CrispasrDiarizeSegAbi {
                t0_cs: (s.t0 * 100.0).round() as i64,
                t1_cs: (s.t1 * 100.0).round() as i64,
                speaker: s.speaker,
                _pad: 0,
            })
            .collect();
        let rc = unsafe {
            crispasr_sys::crispasr_pyannote_cache_apply_abi(
                self.raw,
                (slice_t0 * 100.0).round() as i64,
                abi_segs.as_mut_ptr(),
                abi_segs.len() as i32,
            )
        };
        if rc != 0 {
            return Err(format!("crispasr_pyannote_cache_apply_abi returned {rc}"));
        }
        for (i, s) in segs.iter_mut().enumerate() {
            s.speaker = abi_segs[i].speaker;
        }
        Ok(())
    }
}

impl Drop for PyannoteCache {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { crispasr_sys::crispasr_pyannote_cache_free_abi(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for PyannoteCache {}

// =========================================================================
// FireRedPunc — punctuation restoration post-processor
// =========================================================================

/// BERT-based punctuation restoration model (FireRedPunc).
///
/// Adds punctuation and capitalization to unpunctuated ASR output.
/// Particularly useful for CTC-based backends (wav2vec2, omniasr,
/// fastconformer-ctc, firered-asr) that output lowercase text.
///
/// ```no_run
/// use crispasr::PuncModel;
///
/// let punc = PuncModel::open("fireredpunc-q8_0.gguf").unwrap();
/// let text = punc.process("and so my fellow americans ask not");
/// println!("{text}"); // "And so my fellow americans, ask not..."
/// ```
pub struct PuncModel {
    handle: *mut std::ffi::c_void,
}

unsafe impl Send for PuncModel {}

impl PuncModel {
    /// Load a FireRedPunc GGUF model.
    pub fn open(model_path: &str) -> Result<Self, String> {
        let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
        let handle = unsafe { crispasr_sys::crispasr_punc_init(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(format!("Failed to load punc model: {model_path}"));
        }
        Ok(Self { handle })
    }

    /// Add punctuation to unpunctuated text.
    pub fn process(&self, text: &str) -> String {
        let c_text = CString::new(text).unwrap_or_default();
        let result = unsafe { crispasr_sys::crispasr_punc_process(self.handle, c_text.as_ptr()) };
        if result.is_null() {
            return text.to_string();
        }
        let out = unsafe { CStr::from_ptr(result) }
            .to_string_lossy()
            .into_owned();
        unsafe { crispasr_sys::crispasr_punc_free_text(result) };
        out
    }
}

impl Drop for PuncModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { crispasr_sys::crispasr_punc_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// =========================================================================
// Direct Parakeet API (bypasses unified session)
// =========================================================================

/// Direct Parakeet ASR context with word- and token-level timestamps.
///
/// For most use cases prefer [`Session`] which auto-dispatches to Parakeet
/// when the GGUF metadata indicates it.
pub struct Parakeet {
    handle: *mut std::ffi::c_void,
}

unsafe impl Send for Parakeet {}

/// Parakeet transcription result with word and token accessors.
pub struct ParakeetResult {
    handle: *mut std::ffi::c_void,
}

impl ParakeetResult {
    pub fn text(&self) -> String {
        let p = unsafe { crispasr_sys::crispasr_parakeet_result_text(self.handle) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
    pub fn n_words(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_parakeet_result_n_words(self.handle) }
    }
    pub fn word_text(&self, i: i32) -> String {
        let p = unsafe { crispasr_sys::crispasr_parakeet_result_word_text(self.handle, i) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
    pub fn word_t0(&self, i: i32) -> i64 {
        unsafe { crispasr_sys::crispasr_parakeet_result_word_t0(self.handle, i) }
    }
    pub fn word_t1(&self, i: i32) -> i64 {
        unsafe { crispasr_sys::crispasr_parakeet_result_word_t1(self.handle, i) }
    }
    pub fn n_tokens(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_parakeet_result_n_tokens(self.handle) }
    }
    pub fn token_text(&self, i: i32) -> String {
        let p = unsafe { crispasr_sys::crispasr_parakeet_result_token_text(self.handle, i) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
    pub fn token_t0(&self, i: i32) -> i64 {
        unsafe { crispasr_sys::crispasr_parakeet_result_token_t0(self.handle, i) }
    }
    pub fn token_t1(&self, i: i32) -> i64 {
        unsafe { crispasr_sys::crispasr_parakeet_result_token_t1(self.handle, i) }
    }
    pub fn token_p(&self, i: i32) -> f32 {
        unsafe { crispasr_sys::crispasr_parakeet_result_token_p(self.handle, i) }
    }
}

impl Drop for ParakeetResult {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { crispasr_sys::crispasr_parakeet_result_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl Parakeet {
    pub fn new(model_path: &str, n_threads: i32, use_flash: bool) -> Result<Self, String> {
        let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
        let handle = unsafe {
            crispasr_sys::crispasr_parakeet_init(
                c_path.as_ptr(),
                n_threads,
                if use_flash { 1 } else { 0 },
            )
        };
        if handle.is_null() {
            return Err(format!("Failed to load Parakeet model: {model_path}"));
        }
        Ok(Self { handle })
    }

    pub fn transcribe(
        &self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<ParakeetResult, String> {
        let lang = language.map(|l| CString::new(l).unwrap_or_default());
        let lang_ptr = lang.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let res = unsafe {
            crispasr_sys::crispasr_parakeet_transcribe(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                lang_ptr,
            )
        };
        if res.is_null() {
            return Err("crispasr_parakeet_transcribe returned null".to_string());
        }
        Ok(ParakeetResult { handle: res })
    }
}

impl Drop for Parakeet {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { crispasr_sys::crispasr_parakeet_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// =========================================================================
// Standalone helpers — full C-ABI parity
// =========================================================================

/// Chunk-boundary LCS dedup: returns the number of leading tokens
/// of `curr_tokens` to drop to remove overlap with `prev_tail_tokens`.
pub fn lcs_dedup_prefix_count(prev_tail: &[i32], curr: &[i32], min_lcs_length: i32) -> i32 {
    unsafe {
        crispasr_sys::crispasr_lcs_dedup_prefix_count(
            prev_tail.as_ptr(),
            prev_tail.len() as c_int,
            curr.as_ptr(),
            curr.len() as c_int,
            min_lcs_length,
        )
    }
}

/// Run standalone VAD returning speech spans in centiseconds.
pub fn vad_segments(
    model_path: &str,
    pcm: &[f32],
    sample_rate: i32,
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
    n_threads: i32,
    use_gpu: bool,
) -> Result<Vec<(f32, f32)>, String> {
    let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
    let mut out_spans: *mut f32 = std::ptr::null_mut();
    let n = unsafe {
        crispasr_sys::crispasr_vad_segments(
            c_path.as_ptr(),
            pcm.as_ptr(),
            pcm.len() as c_int,
            sample_rate,
            threshold,
            min_speech_ms,
            min_silence_ms,
            n_threads,
            if use_gpu { 1 } else { 0 },
            &mut out_spans,
        )
    };
    if n < 0 {
        return Err(format!("crispasr_vad_segments failed (rc={n})"));
    }
    let mut spans = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        unsafe {
            spans.push((*out_spans.offset(2 * i), *out_spans.offset(2 * i + 1)));
        }
    }
    if n > 0 {
        unsafe { crispasr_sys::crispasr_vad_free(out_spans) };
    }
    Ok(spans)
}

/// Run unified VAD dispatcher returning speech spans in seconds.
pub fn vad_slices(
    model_path: &str,
    pcm: &[f32],
    sample_rate: i32,
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
    speech_pad_ms: i32,
    max_chunk_duration_s: f32,
    n_threads: i32,
) -> Result<Vec<(f32, f32)>, String> {
    let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
    let mut out_spans: *mut f32 = std::ptr::null_mut();
    let n = unsafe {
        crispasr_sys::crispasr_vad_slices(
            c_path.as_ptr(),
            pcm.as_ptr(),
            pcm.len() as c_int,
            sample_rate,
            threshold,
            min_speech_ms,
            min_silence_ms,
            speech_pad_ms,
            max_chunk_duration_s,
            n_threads,
            &mut out_spans,
        )
    };
    if n < 0 {
        return Err(format!("crispasr_vad_slices failed (rc={n})"));
    }
    let mut spans = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        unsafe {
            spans.push((*out_spans.offset(2 * i), *out_spans.offset(2 * i + 1)));
        }
    }
    if n > 0 {
        unsafe { crispasr_sys::crispasr_vad_free(out_spans) };
    }
    Ok(spans)
}

/// RNNoise audio enhancement on 48 kHz mono PCM.
pub fn enhance_audio_rnnoise(pcm: &[f32]) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; pcm.len()];
    let rc = unsafe {
        crispasr_sys::crispasr_enhance_audio_rnnoise(
            pcm.as_ptr(),
            pcm.len() as i32,
            out.as_mut_ptr(),
            out.len() as i32,
        )
    };
    if rc != 0 {
        return Err(format!("enhance_audio_rnnoise failed (rc={rc})"));
    }
    Ok(out)
}

/// TitaNet cosine similarity between two embeddings.
pub fn titanet_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len().min(b.len()) as i32;
    unsafe { crispasr_sys::crispasr_titanet_cosine_sim(a.as_ptr(), b.as_ptr(), dim) }
}

/// Speaker database wrapper.
pub struct SpeakerDB {
    handle: *mut std::ffi::c_void,
    dir_path: String,
}

unsafe impl Send for SpeakerDB {}

impl SpeakerDB {
    pub fn load(dir_path: &str) -> Result<Self, String> {
        let c_path = CString::new(dir_path).map_err(|e| e.to_string())?;
        let handle = unsafe { crispasr_sys::crispasr_speaker_db_load(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(format!("Failed to load speaker DB: {dir_path}"));
        }
        Ok(Self {
            handle,
            dir_path: dir_path.to_string(),
        })
    }

    pub fn count(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_speaker_db_count(self.handle) }
    }

    pub fn match_embedding(&self, embedding: &[f32], threshold: f32) -> (Option<String>, f32) {
        let mut name_buf = vec![0u8; 256];
        let score = unsafe {
            crispasr_sys::crispasr_speaker_db_match(
                self.handle,
                embedding.as_ptr(),
                embedding.len() as i32,
                threshold,
                name_buf.as_mut_ptr() as *mut c_char,
                256,
            )
        };
        let name = if score >= threshold {
            let c_str = unsafe { CStr::from_ptr(name_buf.as_ptr() as *const c_char) };
            Some(c_str.to_string_lossy().into_owned())
        } else {
            None
        };
        (name, score)
    }

    pub fn enroll(&self, name: &str, embedding: &[f32]) -> Result<(), String> {
        let c_dir = CString::new(&*self.dir_path).map_err(|e| e.to_string())?;
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        let rc = unsafe {
            crispasr_sys::crispasr_speaker_db_enroll(
                c_dir.as_ptr(),
                c_name.as_ptr(),
                embedding.as_ptr(),
                embedding.len() as i32,
            )
        };
        if rc != 0 {
            return Err(format!("speaker_db_enroll failed (rc={rc})"));
        }
        Ok(())
    }
}

impl Drop for SpeakerDB {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { crispasr_sys::crispasr_speaker_db_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Whether `lang` is German (Kokoro phoneme selection).
pub fn kokoro_lang_is_german(lang: &str) -> bool {
    let c = CString::new(lang).unwrap_or_default();
    unsafe { crispasr_sys::crispasr_kokoro_lang_is_german_abi(c.as_ptr()) }
}

/// Whether `lang` has a native Kokoro voice.
pub fn kokoro_lang_has_native_voice(lang: &str) -> bool {
    let c = CString::new(lang).unwrap_or_default();
    unsafe { crispasr_sys::crispasr_kokoro_lang_has_native_voice_abi(c.as_ptr()) }
}
