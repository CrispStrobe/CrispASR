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
use std::os::raw::{c_char, c_float};

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
#[deprecated(since = "0.1.6", note = "Use Session::open() instead — CrispASR can abort on C++ exceptions")]
pub struct CrispASR {
    ctx: *mut crispasr_sys::WhisperContext,
}

unsafe impl Send for CrispASR {}

impl CrispASR {
    /// Load a GGUF/GGML whisper model file.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let path = CString::new(model_path)
            .map_err(|e| format!("invalid path: {e}"))?;
        let cparams = unsafe { crispasr_sys::whisper_context_default_params_by_ref() };
        let ctx = unsafe {
            crispasr_sys::whisper_init_from_file_with_params(path.as_ptr(), cparams)
        };
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
        self.transcribe_pcm_with_strategy(pcm, crispasr_sys::WHISPER_SAMPLING_GREEDY)
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
        let strategy = opts.strategy.unwrap_or(crispasr_sys::WHISPER_SAMPLING_GREEDY);
        let params = unsafe {
            crispasr_sys::whisper_full_default_params_by_ref(strategy)
        };

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
                    crispasr_sys::crispasr_params_set_vad_model_path(
                        params,
                        cs.as_ptr(),
                    );
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
        let ret = unsafe {
            crispasr_sys::whisper_full(
                self.ctx,
                params,
                pcm.as_ptr(),
                pcm.len() as i32,
            )
        };
        unsafe { crispasr_sys::whisper_free_params(params) };

        if ret != 0 {
            return Err(format!("transcription failed (error code {ret})"));
        }

        let n = unsafe { crispasr_sys::whisper_full_n_segments(self.ctx) };
        let mut segments = Vec::with_capacity(n as usize);

        for i in 0..n {
            let text_ptr = unsafe {
                crispasr_sys::whisper_full_get_segment_text(self.ctx, i)
            };
            let text = if text_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(text_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            let t0 = unsafe { crispasr_sys::whisper_full_get_segment_t0(self.ctx, i) };
            let t1 = unsafe { crispasr_sys::whisper_full_get_segment_t1(self.ctx, i) };
            let nsp = unsafe {
                crispasr_sys::whisper_full_get_segment_no_speech_prob(self.ctx, i)
            };

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
}

// Not `Sync` — do not share between threads without external sync.
unsafe impl Send for Session {}

impl Session {
    /// Open a GGUF model, auto-detecting the backend from metadata.
    pub fn open(model_path: &str) -> Result<Self, String> {
        Self::open_inner(model_path, None, 4)
    }

    /// Open with an explicit backend (skips auto-detect).
    pub fn open_with_backend(model_path: &str, backend: &str, n_threads: i32) -> Result<Self, String> {
        Self::open_inner(model_path, Some(backend), n_threads)
    }

    fn open_inner(model_path: &str, backend: Option<&str>, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let handle = if let Some(be) = backend {
            let be_c = CString::new(be).map_err(|e| format!("invalid backend: {e}"))?;
            unsafe {
                crispasr_sys::crispasr_session_open_explicit(path.as_ptr(), be_c.as_ptr(), n_threads)
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
        Ok(Self { handle })
    }

    /// List of backend names the loaded CrispASR library was compiled with.
    pub fn available_backends() -> Vec<String> {
        let mut buf = [0i8; 256];
        let n = unsafe {
            crispasr_sys::crispasr_session_available_backends(buf.as_mut_ptr(), buf.len() as i32)
        };
        if n <= 0 {
            return Vec::new();
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
            Some(l) if !l.is_empty() => Some(
                CString::new(l).map_err(|e| format!("language NUL: {e}"))?,
            ),
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
        if res.is_null() {
            return Err(format!(
                "crispasr_session_transcribe failed for backend {:?}",
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
                    words.push(SessionWord {
                        text: wt,
                        start: crispasr_sys::crispasr_session_result_word_t0(res, i, j) as f64
                            / 100.0,
                        end: crispasr_sys::crispasr_session_result_word_t1(res, i, j) as f64
                            / 100.0,
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

    /// Transcribe with Silero VAD segmentation + whisper.cpp-style stitching.
    ///
    /// Runs VAD on the PCM buffer, merges short / overlong speech slices
    /// into usable chunks, stitches them into a single buffer with 0.1s
    /// silence gaps, calls the backend once, then remaps segment + word
    /// timestamps back to original-audio positions.
    ///
    /// `vad_model_path` must point to a Silero GGUF on disk. Passing
    /// `None` for `opts` uses the library defaults (mirroring
    /// whisper.cpp's `whisper_vad_default_params`).
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
            Some(l) if !l.is_empty() => Some(
                CString::new(l).map_err(|e| format!("language NUL: {e}"))?,
            ),
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
                    words.push(SessionWord {
                        text: wt,
                        start: crispasr_sys::crispasr_session_result_word_t0(res, i, j) as f64
                            / 100.0,
                        end: crispasr_sys::crispasr_session_result_word_t1(res, i, j) as f64
                            / 100.0,
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
}

/// Tunables for [`Session::transcribe_vad`]. Defaults mirror whisper.cpp's
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
/// voxtral, voxtral4b, granite, qwen3, cohere, wav2vec2). Returns `None`
/// on miss.
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
                fn_buf.as_mut_ptr() as *mut c_char, fn_buf.len() as i32,
                url_buf.as_mut_ptr() as *mut c_char, url_buf.len() as i32,
                size_buf.as_mut_ptr() as *mut c_char, size_buf.len() as i32,
            )
        } else {
            crispasr_sys::crispasr_registry_lookup_by_filename_abi(
                key_c.as_ptr(),
                fn_buf.as_mut_ptr() as *mut c_char, fn_buf.len() as i32,
                url_buf.as_mut_ptr() as *mut c_char, url_buf.len() as i32,
                size_buf.as_mut_ptr() as *mut c_char, size_buf.len() as i32,
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
    let ov_c = CString::new(override_path.unwrap_or("")).map_err(|e| format!("override NUL: {e}"))?;
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
    let model_c =
        CString::new(aligner_model).map_err(|e| format!("aligner_model NUL: {e}"))?;
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
    let path_c =
        CString::new(model_path).map_err(|e| format!("model_path contains NUL: {e}"))?;

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
        let result = unsafe {
            crispasr_sys::crispasr_punc_process(self.handle, c_text.as_ptr())
        };
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
