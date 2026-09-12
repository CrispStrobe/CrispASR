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
    /// Whisper's per-segment probability that the segment is non-speech (the
    /// `<|nospeech|>` token posterior), in `[0, 1]`. Only the whisper backend
    /// produces it; every other backend leaves the `-1.0` sentinel ("no
    /// data"), so a consumer can tell "unavailable" apart from a genuine low
    /// no-speech probability.
    pub no_speech_prob: f32,
}

/// Per-frame CTC logits captured from a CTC backend.
///
/// `data` is frame-major: `data[t * n_vocab + v]` is the score for vocabulary
/// entry `v` at encoder frame `t`, so its length is `n_vocab * n_frames`.
/// Produced only by [`Session::transcribe_with_logits`] on a backend with a
/// dense CTC grid (Omni CTC, wav2vec2/hubert/data2vec, or canary-ctc); other
/// backends yield no grid. The Omni and wav2vec2 grids are raw logits
/// (pre-softmax); the canary-ctc grid is log-probabilities. Log-softmax before
/// use if you need normalized scores — it is idempotent on the canary grid.
#[derive(Debug, Clone)]
pub struct CtcLogits {
    pub n_vocab: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

/// A loaded session over a CrispASR model of any backend.
/// One stem from [`Session::separate`]: its name (`vocals`, `drums`, …) and
/// its interleaved-stereo PCM at [`Session::separate_sample_rate`].
#[derive(Debug, Clone)]
pub struct Stem {
    pub name: String,
    pub pcm: Vec<f32>,
}

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

    /// What can this backend actually *do*? (#433)
    ///
    /// [`detect_backend`](Self::detect_backend) returns a name and nothing
    /// else, but several backends serve more than one purpose — `lfm2-audio`
    /// and `mini-omni2` are both `tts` **and** `s2s`, and `gemma4-e2b` does
    /// recognition and translation. Pair the two calls to answer "what can I do
    /// with this file?".
    ///
    /// Returns the capability names, e.g. `["auto-download", "tts", "s2s"]`.
    /// An unknown backend is an error rather than an empty list, because a
    /// backend that genuinely declares no capabilities is a different answer.
    pub fn backend_caps(backend: &str) -> Result<Vec<String>, String> {
        let name = CString::new(backend).map_err(|e| format!("invalid backend: {e}"))?;
        let mut buf = [0i8; 1024];
        let n = unsafe {
            crispasr_sys::crispasr_backend_caps_abi(name.as_ptr(), buf.as_mut_ptr(), buf.len() as i32)
        };
        if n == -3 {
            return Err(format!("unknown backend '{backend}'"));
        }
        if n < 0 {
            return Err(format!("backend_caps failed (code {n})"));
        }
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
        Ok(s.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect())
    }

    /// Every backend paired with its verbs (#433).
    ///
    /// The "available backend list containing verb info" the issue asked for —
    /// without shelling out to `crispasr --list-backends-json` and parsing
    /// stdout, which was previously the only way to reach this data.
    pub fn list_backends_with_caps() -> Result<Vec<(String, Vec<String>)>, String> {
        // Ask once with an empty buffer: a negative return is the required size,
        // so the buffer is never guessed and never silently truncated.
        let need = unsafe { crispasr_sys::crispasr_backend_caps_list_abi(std::ptr::null_mut(), 0) };
        let cap = if need < 0 { (-need) as usize } else { 65536 };
        let mut buf = vec![0i8; cap.max(1024)];
        let n = unsafe {
            crispasr_sys::crispasr_backend_caps_list_abi(buf.as_mut_ptr(), buf.len() as i32)
        };
        if n < 0 {
            return Err(format!("list_backends_with_caps failed (code {n})"));
        }
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
        Ok(s.lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let mut it = l.splitn(2, '\t');
                let name = it.next().unwrap_or("").to_string();
                let caps = it
                    .next()
                    .unwrap_or("")
                    .split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| x.to_string())
                    .collect();
                (name, caps)
            })
            .collect())
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

    /// The Omni CTC vocabulary as raw SentencePiece pieces, indexed by token
    /// id (`vocab[id]`). Pieces keep the U+2581 (`▁`) word-boundary marker
    /// intact, so a consumer can group a greedy CTC decode over
    /// [`CtcLogits`] into words at `▁` boundaries and map `▁` → space.
    /// Returns `None` for backends that don't expose a CTC vocab.
    pub fn ctc_vocab(&self) -> Option<Vec<String>> {
        let n = unsafe { crispasr_sys::crispasr_session_n_vocab(self.handle) };
        if n <= 0 {
            return None;
        }
        let mut out = Vec::with_capacity(n as usize);
        for id in 0..n {
            let p = unsafe { crispasr_sys::crispasr_session_token_text(self.handle, id) };
            let piece = if p.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            };
            out.push(piece);
        }
        Some(out)
    }

    /// The acoustic language whisper detected on the last transcribe, as an
    /// ISO-639-1 code (e.g. `"en"`). Whisper-only: other backends return the
    /// session's source-language hint, or `"unknown"` when none was set — as
    /// does whisper before its first transcribe. This is the in-decode
    /// acoustic signal, distinct from a text-LID pass over the transcript.
    pub fn detected_language(&self) -> String {
        let mut buf = [0 as c_char; 32];
        let n = unsafe {
            crispasr_sys::crispasr_session_detected_language(
                self.handle,
                buf.as_mut_ptr(),
                buf.len() as c_int,
            )
        };
        if n <= 0 {
            return "unknown".to_string();
        }
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
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

    /// Opt in to capturing the per-frame CTC logits on subsequent transcribe
    /// calls (backends with a dense CTC grid: Omni CTC, wav2vec2/hubert/data2vec,
    /// canary-ctc). Off by default: capture copies `n_vocab × n_frames` floats
    /// per call, so leave it off unless a consumer (e.g. forced alignment) needs
    /// the grid. Retrieve the logits with [`Self::transcribe_with_logits`].
    pub fn set_return_logits(&self, on: bool) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_return_logits(self.handle, if on { 1 } else { 0 })
        };
        if rc != 0 {
            return Err(format!("set_return_logits failed (rc={rc})"));
        }
        Ok(())
    }

    /// Transcribe and also return the CTC logits captured for this call.
    /// Enables logit capture for the duration, so the caller need not call
    /// [`Self::set_return_logits`] first. The logits are `None` for backends
    /// that don't produce a dense CTC grid (only Omni CTC, wav2vec2/hubert/
    /// data2vec, and canary-ctc do) or when the transcript is empty.
    pub fn transcribe_with_logits(
        &self,
        pcm: &[f32],
    ) -> Result<(Vec<SessionSegment>, Option<CtcLogits>), String> {
        if pcm.is_empty() {
            return Ok((Vec::new(), None));
        }
        self.set_return_logits(true)?;
        let res = unsafe {
            crispasr_sys::crispasr_session_transcribe(self.handle, pcm.as_ptr(), pcm.len() as i32)
        };
        let parsed = self.parse_session_result_logits(res, "crispasr_session_transcribe");
        let _ = self.set_return_logits(false);
        parsed
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

        extern "C" fn trampoline<F: FnMut(i32, i32)>(
            processed: c_int,
            total: c_int,
            ud: *mut c_void,
        ) {
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
        self.parse_session_result_logits(res, ctx)
            .map(|(segs, _)| segs)
    }

    /// Like [`Self::parse_session_result`], but also lifts out any raw CTC
    /// logits the backend attached to the result (see [`CtcLogits`]) before
    /// freeing the handle. The logits are `None` unless the session opted in
    /// via [`Self::set_return_logits`] and the backend produced a grid.
    fn parse_session_result_logits(
        &self,
        res: *mut crispasr_sys::CrispasrSessionResult,
        ctx: &str,
    ) -> Result<(Vec<SessionSegment>, Option<CtcLogits>), String> {
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
                let nsp = crispasr_sys::crispasr_session_result_segment_no_speech_prob(res, i);
                out.push(SessionSegment {
                    text: text.trim().to_string(),
                    start: t0,
                    end: t1,
                    words,
                    no_speech_prob: nsp,
                });
            }
            // Lift out the raw CTC logits (if any) before the handle is freed.
            let n_frames = crispasr_sys::crispasr_session_result_n_logit_frames(res);
            let n_vocab = crispasr_sys::crispasr_session_result_n_logit_vocab(res);
            let lp = crispasr_sys::crispasr_session_result_logits(res);
            let logits = if n_frames > 0 && n_vocab > 0 && !lp.is_null() {
                let n = n_vocab as usize * n_frames as usize;
                Some(CtcLogits {
                    n_vocab: n_vocab as usize,
                    n_frames: n_frames as usize,
                    data: std::slice::from_raw_parts(lp, n).to_vec(),
                })
            } else {
                None
            };
            crispasr_sys::crispasr_session_result_free(res);
            Ok((out, logits))
        }
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
                let nsp = crispasr_sys::crispasr_session_result_segment_no_speech_prob(res, i);
                out.push(SessionSegment {
                    text: text.trim().to_string(),
                    start: t0,
                    end: t1,
                    words,
                    no_speech_prob: nsp,
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

    /// Speech-to-speech: input PCM in → output PCM out via a single model
    /// pass. Requires an S2S-capable backend (`lfm2-audio`, `mini-omni2`,
    /// `sidon`, `voxcpm2-vae`). Input PCM must be at the backend's native
    /// input rate (see [`Session::input_sample_rate`]).
    ///
    /// Returns the output PCM plus the optional intermediate transcript the
    /// model produced on the way (`None` if the backend doesn't surface one).
    /// Errors if the backend has no S2S capability or the pass fails.
    pub fn speech_to_speech(&self, pcm: &[f32]) -> Result<(Vec<f32>, Option<String>), String> {
        let mut n: c_int = 0;
        let mut text_ptr: *mut c_char = std::ptr::null_mut();
        let ptr = unsafe {
            crispasr_sys::crispasr_session_speech_to_speech(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                &mut text_ptr as *mut *mut c_char,
                &mut n as *mut c_int,
            )
        };
        if ptr.is_null() || n <= 0 {
            if !text_ptr.is_null() {
                unsafe { crispasr_sys::crispasr_session_translate_text_free(text_ptr) };
            }
            return Err(format!(
                "speech_to_speech returned no audio for backend {:?} (S2S may be unsupported). \
                 Separation models (htdemucs, mel-band-roformer) are not S2S — use \
                 Session::separate() instead (#359).",
                self.backend()
            ));
        }
        let out = unsafe { std::slice::from_raw_parts(ptr, n as usize).to_vec() };
        unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
        let transcript = if text_ptr.is_null() {
            None
        } else {
            let s = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { crispasr_sys::crispasr_session_translate_text_free(text_ptr) };
            Some(s)
        };
        Ok((out, transcript))
    }

    /// Source separation: split stereo audio into its stems (#359).
    ///
    /// This is the verb for `htdemucs` and `mel-band-roformer`. They are NOT
    /// speech-to-speech models, so [`Session::speech_to_speech`] returns no
    /// audio for them — the C ABI has always had a separate five-function
    /// surface for this, and it simply was not bound here.
    ///
    /// `pcm_stereo` is INTERLEAVED stereo at the model's own rate, which is
    /// not 16 kHz: call [`Session::separate_sample_rate`] after loading (44100
    /// for the shipped separation models). Each returned stem is interleaved
    /// stereo of the same length.
    ///
    /// The C side owns the stem buffers only until the next call, so this
    /// copies them out before returning.
    pub fn separate(&self, pcm_stereo: &[f32]) -> Result<Vec<Stem>, String> {
        // The C API counts PER-CHANNEL frames, not floats.
        let n_frames = (pcm_stereo.len() / 2) as c_int;
        if n_frames == 0 {
            return Err("separate needs interleaved stereo PCM".to_string());
        }
        let n_stems = unsafe {
            crispasr_sys::crispasr_session_separate(self.handle, pcm_stereo.as_ptr(), n_frames)
        };
        if n_stems <= 0 {
            return Err(format!(
                "separate returned no stems for backend {:?} (is it a separation model?)",
                self.backend()
            ));
        }
        let mut stems = Vec::with_capacity(n_stems as usize);
        for i in 0..n_stems {
            let name_ptr =
                unsafe { crispasr_sys::crispasr_session_separate_stem_name(self.handle, i) };
            let name = if name_ptr.is_null() {
                format!("stem{}", i)
            } else {
                unsafe { CStr::from_ptr(name_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            let mut n_out: c_int = 0;
            let ptr = unsafe {
                crispasr_sys::crispasr_session_separate_stem(
                    self.handle,
                    i,
                    &mut n_out as *mut c_int,
                )
            };
            if ptr.is_null() || n_out <= 0 {
                return Err(format!("stem {} ({}) came back empty", i, name));
            }
            // n_out is per-channel; the buffer is interleaved stereo.
            let pcm =
                unsafe { std::slice::from_raw_parts(ptr, (n_out as usize) * 2).to_vec() };
            stems.push(Stem { name, pcm });
        }
        Ok(stems)
    }

    /// Sample rate (Hz) of the stems from [`Session::separate`], and the rate
    /// its input must be at. `0` before a separation backend is loaded — which
    /// is what a caller sees if they reach for the ASR-side rate accessors on
    /// a separation model.
    pub fn separate_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_separate_sample_rate(self.handle) as i32 }
    }

    /// The sample rate (Hz) the backend expects for input PCM — `16000` for
    /// Whisper-family backends, the model's native rate otherwise, `0` on
    /// error. Feed [`Session::speech_to_speech`] (and TTS voice-clone input)
    /// at this rate rather than resampling twice.
    pub fn input_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_input_sample_rate(self.handle) as i32 }
    }

    /// The sample rate (Hz) of the PCM that [`Session::synthesize`] /
    /// [`Session::speech_to_speech`] produce for this backend — the
    /// "backend-native rate" their docs refer to. `0` when the backend
    /// produces no audio output (ASR-only). (#332)
    pub fn output_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_output_sample_rate(self.handle) as i32 }
    }

    /// Channel count for audio input (transcribe / s2s / voice references):
    /// `1` (mono) for every current backend. Source separation is the stereo
    /// exception and has its own surface. `0` on error. (#332)
    pub fn input_channels(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_input_channels(self.handle) as i32 }
    }

    /// Channel count for synthesized / s2s output audio: `1` (mono) for every
    /// current backend, `0` when the backend produces no audio output. (#332)
    pub fn output_channels(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_output_channels(self.handle) as i32 }
    }

    /// Attest that the integrator accepts AI-content marking/disclosure
    /// responsibility (EU AI Act Art. 50). **Required** before
    /// [`Session::synthesize_raw`] will return unmarked audio; the default
    /// [`Session::synthesize`] is watermarked and needs no attestation.
    /// `attestation` is a free-text acknowledgement recorded for audit.
    pub fn accept_marking_responsibility(&self, attestation: &str) -> Result<(), String> {
        let c = CString::new(attestation).map_err(|e| e.to_string())?;
        // The C side records the attestation; the return is informational
        // (mirrors the Python/Dart bindings, which ignore it).
        unsafe {
            crispasr_sys::crispasr_session_accept_marking_responsibility(self.handle, c.as_ptr())
        };
        Ok(())
    }

    /// Declare whose voice a PRESET voice is: `"real_person"`,
    /// `"synthetic"` or `"unknown"`.
    ///
    /// Cloning is not the only way to produce a deep fake: a preset voice
    /// shipped inside a model can be an identifiable individual — a named
    /// donor, or a corpus speaker such as VCTK's `p225` — and EU AI Act
    /// Art. 3(60) attaches to the audio resembling that person, not to which
    /// pipeline produced it. Setting `real_person` makes the Art. 50(4)
    /// reminder fire for a non-cloned voice.
    ///
    /// It does **not** require a consent attestation: whether that donor
    /// agreed to the model being trained is a licensing matter settled
    /// upstream, which you cannot attest to.
    ///
    /// Returns `Err` on an unrecognised value rather than silently
    /// downgrading it to `unknown`.
    pub fn set_speaker_identity(&self, identity: &str) -> Result<(), String> {
        let c = CString::new(identity).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_speaker_identity(self.handle, c.as_ptr()) };
        match rc {
            0 => Ok(()),
            -2 => Err(format!(
                "unrecognised speaker_identity {identity:?} (expected real_person, synthetic or unknown)"
            )),
            _ => Err(format!("set_speaker_identity failed (rc={rc})")),
        }
    }

    /// Embed the AI-content watermark into f32 mono PCM, in place.
    ///
    /// The other half of [`Session::synthesize_raw`]: opting out of automatic
    /// marking makes marking the result *your* duty (EU AI Act Art. 50(2)), and
    /// this is what discharges it. Do the post-processing you opted out for —
    /// resample, mix, concatenate — then call this on the finished buffer.
    ///
    /// Uses the robust, reliably detectable default strength; AudioSeal instead
    /// if a model was loaded. Associated function, not a method: marking is a
    /// property of the samples, not of the session that produced them.
    pub fn watermark_embed(pcm: &mut [f32]) {
        if pcm.is_empty() {
            return;
        }
        unsafe {
            crispasr_sys::crispasr_watermark_embed(pcm.as_mut_ptr(), pcm.len() as c_int, -1.0)
        };
    }

    /// Confidence in `[0, 1]` that `pcm` carries the watermark.
    ///
    /// A weak diagnostic, not proof: the spread-spectrum detector's null mean is
    /// 0.5, not 0, and a negative result on a short clip is mostly evidence that
    /// the clip was short. See `docs/eu-ai-act.md` §6.7 before reading anything
    /// into a number from here.
    pub fn watermark_detect(pcm: &[f32]) -> f32 {
        if pcm.is_empty() {
            return 0.0;
        }
        unsafe { crispasr_sys::crispasr_watermark_detect(pcm.as_ptr(), pcm.len() as c_int) }
    }

    /// UNMARKED synthesis (no watermark / disclosure), for callers that embed
    /// the mark themselves after post-processing. Hard-refused (returns `Err`)
    /// unless [`Session::accept_marking_responsibility`] was called first.
    /// Prefer [`Session::synthesize`] for the default watermarked output.
    /// Mark the result with [`Session::watermark_embed`].
    pub fn synthesize_raw(&self, text: &str) -> Result<Vec<f32>, String> {
        let ctext = CString::new(text).map_err(|e| e.to_string())?;
        let mut n: c_int = 0;
        let ptr = unsafe {
            crispasr_sys::crispasr_session_synthesize_raw(
                self.handle,
                ctext.as_ptr(),
                &mut n as *mut c_int,
            )
        };
        if ptr.is_null() || n <= 0 {
            return Err(format!(
                "synthesize_raw returned no audio for backend {:?} (call accept_marking_responsibility first?)",
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

    /// Language a voice-cloning reference clip is spoken in (issue #329).
    ///
    /// Cross-lingual TTS backends (cosyvoice3) compare it to the requested
    /// output language — [`set_target_language`](Session::set_target_language),
    /// falling back to [`set_source_language`](Session::set_source_language) —
    /// and drop the reference transcript when they differ, so the clone speaks
    /// the target language instead of carrying the reference's accent.
    ///
    /// Optional: the backend otherwise infers the reference language from the
    /// voice-bank entry or the reference transcript. That inference cannot
    /// answer for a short transcript, and when it cannot, the requested target
    /// language has no effect — set this to make it explicit.
    pub fn set_tts_reference_language(&self, lang: &str) -> Result<(), String> {
        let c = CString::new(lang).map_err(|e| e.to_string())?;
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_tts_reference_language(self.handle, c.as_ptr())
        };
        if rc != 0 {
            return Err(format!("set_tts_reference_language failed (rc={})", rc));
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

    /// Set the floor on generated audio length (MOSS TTS). Units are codec frames at 12.5 Hz (80 ms each), so n=25 floors at ~2 s; other backends no-op (rc=-2).
    pub fn set_min_speech_tokens(&self, n: i32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_min_speech_tokens(self.handle, n) };
        if rc != 0 && rc != -2 {
            return Err(format!("set_min_speech_tokens failed (rc={})", rc));
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

    /// Synthesize `phonemes` verbatim instead of phonemizing the text — the
    /// seam between text processing and the acoustic model. Use it to reproduce
    /// another implementation's pronunciation exactly, or to tell a G2P bug from
    /// a model bug (#316). An empty string clears it.
    ///
    /// Honoured by `kokoro` and `piper`; other backends soft no-op (`rc = -2`).
    pub fn set_tts_phonemes(&self, phonemes: &str) -> Result<(), String> {
        let c = CString::new(phonemes).map_err(|e| e.to_string())?;
        let rc =
            unsafe { crispasr_sys::crispasr_session_set_tts_phonemes(self.handle, c.as_ptr()) };
        if rc == -2 {
            return Err("backend has no phonemes-in entry point (kokoro and piper do)".to_string());
        }
        if rc != 0 {
            return Err(format!("set_tts_phonemes failed (rc={})", rc));
        }
        Ok(())
    }

    /// Pad N ms of silence at the beginning of TTS output. Useful to bypass VLC playback bugs
    /// where it drops the first ~1.5s of audio while parsing a large C2PA chunk.
    pub fn set_tts_pad_silence_ms(&self, ms: i32) {
        unsafe { crispasr_sys::crispasr_session_set_tts_pad_silence_ms(self.handle, ms as std::os::raw::c_int) };
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

    /// Apply a named bundle of the four decoder fallback thresholds:
    /// `"conservative"`, `"balanced"` (the shipped defaults, a no-op) or
    /// `"aggressive"`. `"strict"`/`"default"`/`"loose"` are aliases. Mirrors the
    /// CLI's `--sensitivity`.
    ///
    /// The four thresholds interact — a decode is only retried when the logprob
    /// *and* no-speech bars are both crossed — so they move as a set. A later
    /// [`Session::set_fallback_thresholds`] overrides this. An unrecognised
    /// preset is rejected rather than silently treated as `"balanced"`.
    pub fn set_sensitivity(&self, preset: &str) -> Result<(), String> {
        let c = CString::new(preset).map_err(|e| e.to_string())?;
        let rc = unsafe { crispasr_sys::crispasr_session_set_sensitivity(self.handle, c.as_ptr()) };
        if rc == -2 {
            return Err(format!(
                "unknown sensitivity preset {:?} (expected: conservative, balanced, aggressive)",
                preset
            ));
        }
        if rc != 0 {
            return Err(format!("set_sensitivity failed (rc={})", rc));
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

/// Role of one artifact in a canonical model download bundle.
///
/// Mirrors the append-only `crispasr_registry_artifact_kind` C enum, so
/// new kinds may appear in minor releases — match with a `_` arm (#332).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegistryArtifactKind {
    Primary,
    Companion,
    Extra,
}

/// One file in a backend's canonical default download bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryArtifact {
    pub kind: RegistryArtifactKind,
    pub filename: String,
    pub url: String,
    pub approx_size: String,
}

/// The exact artifact bundle downloaded by `-m auto`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryBundle {
    pub backend: String,
    pub license: String,
    pub requires_acceptance: bool,
    pub artifacts: Vec<RegistryArtifact>,
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

/// Return the backend's exact canonical `-m auto` artifact bundle.
///
/// Artifacts are ordered as downloaded: primary model, inline companion,
/// then any extra companions. No preferred quant is applied. Returns `None`
/// when the backend has no registry entry.
pub fn registry_default_bundle(backend: &str) -> Result<Option<RegistryBundle>, String> {
    if backend.is_empty() {
        return Ok(None);
    }
    let backend_c = CString::new(backend).map_err(|e| format!("backend NUL: {e}"))?;
    let mut canonical_buf = [0u8; 256];
    let mut license_buf = [0u8; 1024];
    let mut requires_acceptance = 0;
    let count = unsafe {
        crispasr_sys::crispasr_registry_default_bundle_info_abi(
            backend_c.as_ptr(),
            canonical_buf.as_mut_ptr() as *mut c_char,
            canonical_buf.len() as c_int,
            license_buf.as_mut_ptr() as *mut c_char,
            license_buf.len() as c_int,
            &mut requires_acceptance,
        )
    };
    if count == 0 {
        return Ok(None);
    }
    if count < 0 {
        return Err(format!(
            "default-bundle registry lookup failed (rc={count})"
        ));
    }

    fn slice_to_string(buf: &[u8]) -> String {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    let mut artifacts = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut kind = 0;
        let mut filename_buf = [0u8; 256];
        let mut url_buf = [0u8; 2048];
        let mut size_buf = [0u8; 64];
        let rc = unsafe {
            crispasr_sys::crispasr_registry_default_bundle_artifact_abi(
                backend_c.as_ptr(),
                index,
                &mut kind,
                filename_buf.as_mut_ptr() as *mut c_char,
                filename_buf.len() as c_int,
                url_buf.as_mut_ptr() as *mut c_char,
                url_buf.len() as c_int,
                size_buf.as_mut_ptr() as *mut c_char,
                size_buf.len() as c_int,
            )
        };
        if rc != 0 {
            return Err(format!(
                "default-bundle artifact {index} lookup failed (rc={rc})"
            ));
        }
        let kind = match kind {
            0 => RegistryArtifactKind::Primary,
            1 => RegistryArtifactKind::Companion,
            2 => RegistryArtifactKind::Extra,
            value => {
                return Err(format!(
                    "default-bundle artifact {index} has unknown kind {value}"
                ))
            }
        };
        artifacts.push(RegistryArtifact {
            kind,
            filename: slice_to_string(&filename_buf),
            url: slice_to_string(&url_buf),
            approx_size: slice_to_string(&size_buf),
        });
    }

    Ok(Some(RegistryBundle {
        backend: slice_to_string(&canonical_buf),
        license: slice_to_string(&license_buf),
        requires_acceptance: requires_acceptance != 0,
        artifacts,
    }))
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

/// Mirrors the append-only C LID-method enum, so new methods may appear
/// in minor releases — match with a `_` arm (#332).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
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

/// Mirrors the append-only `CrispasrDiarizeMethod` C enum, so new methods
/// may appear in minor releases — match with a `_` arm (#332).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
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
    /// Mono-friendly, ML-based (#324): WeSpeaker embeddings + spectral
    /// clustering (the FoxNose recipe). Requires
    /// [`DiarizeOptions::foxnose_embedder_path`]. Unlike the other methods
    /// it derives speaker turns from the audio and attributes each caller
    /// segment to the turn it overlaps most.
    FoxNose = 4,
}

/// Construct via [`Default`] and set fields as needed — the struct grows
/// alongside the append-only C ABI (#332).
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    /// GGUF path for the speaker-embedding model (WeSpeaker ResNet34-LM).
    /// Required for `FoxNose`, ignored otherwise.
    pub foxnose_embedder_path: Option<String>,
    /// FoxNose speaker-count lower bound for automatic estimation (0 -> 1).
    pub min_speakers: i32,
    /// FoxNose speaker-count upper bound for automatic estimation (0 -> 8).
    pub max_speakers: i32,
    /// FoxNose: > 0 pins the speaker count and skips estimation entirely.
    pub num_speakers: i32,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            method: DiarizeMethod::VadTurns,
            pyannote_model_path: None,
            n_threads: 4,
            slice_t0: 0.0,
            foxnose_embedder_path: None,
            min_speakers: 0,
            max_speakers: 0,
            num_speakers: 0,
        }
    }
}

/// A speaker turn the diarizer derived from the AUDIO, independent of the
/// caller's segment grid (#395).
///
/// Only [`DiarizeMethod::FoxNose`] produces these; the other methods label
/// the caller's segments directly and yield none. `t0` / `t1` are seconds on
/// the same absolute timeline as [`DiarizeSegment`] (i.e.
/// [`DiarizeOptions::slice_t0`] is already accounted for), so a turn and a
/// segment compare directly. `speaker` is dense and zero-based — a turn is
/// always labelled, unlike a segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiarizeTurn {
    pub t0: f64,
    pub t1: f64,
    pub speaker: i32,
}

/// Assign a speaker index to each of `segs`, mutating each
/// [`DiarizeSegment::speaker`] in place.
///
/// Five methods — see [`DiarizeMethod`]. `left` is mono PCM for
/// mono-only methods, otherwise the left channel of a stereo pair.
/// When `is_stereo` is true, `right` must be `Some`. All PCM is 16 kHz
/// float32.
///
/// Returns `Ok(())` on success. Only the model-backed methods
/// ([`DiarizeMethod::Pyannote`], [`DiarizeMethod::FoxNose`]) can fail
/// (model load failure).
///
/// Labelling alone can never resolve finer than the grid you pass in: a
/// segment that straddles a speaker change is silently awarded to whoever
/// holds the majority of it. Use [`diarize_segments_with_turns`] when you
/// have word timings and want to split such a segment.
pub fn diarize_segments(
    segs: &mut [DiarizeSegment],
    left: &[f32],
    right: Option<&[f32]>,
    is_stereo: bool,
    opts: &DiarizeOptions,
) -> Result<(), String> {
    diarize_inner(segs, left, right, is_stereo, opts, false).map(|_| ())
}

/// Same as [`diarize_segments`], and additionally returns the speaker turns
/// the method derived from the audio (#395).
///
/// The turns are what let a caller resolve a speaker change INSIDE one of its
/// own segments — split a merged run wherever the turn id changes, instead of
/// accepting the majority label for the whole span. Only
/// [`DiarizeMethod::FoxNose`] derives turns; every other method returns an
/// empty `Vec`, which is not an error.
///
/// The segments are labelled exactly as [`diarize_segments`] would label
/// them; asking for turns changes nothing about the labels.
pub fn diarize_segments_with_turns(
    segs: &mut [DiarizeSegment],
    left: &[f32],
    right: Option<&[f32]>,
    is_stereo: bool,
    opts: &DiarizeOptions,
) -> Result<Vec<DiarizeTurn>, String> {
    diarize_inner(segs, left, right, is_stereo, opts, true)
}

fn diarize_inner(
    segs: &mut [DiarizeSegment],
    left: &[f32],
    right: Option<&[f32]>,
    is_stereo: bool,
    opts: &DiarizeOptions,
    want_turns: bool,
) -> Result<Vec<DiarizeTurn>, String> {
    if segs.is_empty() || left.is_empty() {
        return Ok(Vec::new());
    }

    let path_c = match (&opts.pyannote_model_path, opts.method) {
        (Some(p), DiarizeMethod::Pyannote) => Some(
            CString::new(p.as_str())
                .map_err(|e| format!("pyannote_model_path contains NUL: {e}"))?,
        ),
        _ => None,
    };
    let foxnose_c = match (&opts.foxnose_embedder_path, opts.method) {
        (Some(p), DiarizeMethod::FoxNose) => Some(
            CString::new(p.as_str())
                .map_err(|e| format!("foxnose_embedder_path contains NUL: {e}"))?,
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
        foxnose_embedder_path: foxnose_c
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null()),
        min_speakers: opts.min_speakers,
        max_speakers: opts.max_speakers,
        num_speakers: opts.num_speakers,
        _pad2: 0,
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

    if !want_turns {
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
        return finish(rc, segs, &abi_segs, &[], 0).map(|_| Vec::new());
    }

    // The ABI keeps no state between calls, so a buffer that turns out to be
    // too small costs a SECOND full diarization pass. Size the first attempt
    // so that can't happen: turns are merged embedding windows, the hop is
    // 0.6 s, and no turn is shorter than one hop — so one slot per 0.5 s of
    // audio (plus one per segment, plus slack) is comfortably above the
    // ceiling. The retry below is the belt to that pair of braces.
    let mut cap = (left.len() as f64 / 16_000.0 / 0.5).ceil() as usize + segs.len() + 16;
    let mut turns: Vec<crispasr_sys::CrispasrDiarizeTurnAbi> = Vec::new();
    for attempt in 0..2 {
        turns.clear();
        turns.resize(
            cap,
            crispasr_sys::CrispasrDiarizeTurnAbi {
                t0_cs: 0,
                t1_cs: 0,
                speaker: -1,
                _pad: 0,
            },
        );
        let mut n_turns: i32 = 0;
        let rc = unsafe {
            crispasr_sys::crispasr_diarize_segments_turns_abi(
                left.as_ptr(),
                right_ptr,
                left.len() as i32,
                if is_stereo { 1 } else { 0 },
                abi_segs.as_mut_ptr(),
                abi_segs.len() as i32,
                &abi_opts,
                turns.as_mut_ptr(),
                cap as i32,
                &mut n_turns,
            )
        };
        // rc 2 = the buffer was short; n_turns is the required capacity.
        if rc == 2 && attempt == 0 {
            cap = n_turns.max(1) as usize;
            continue;
        }
        return finish(rc, segs, &abi_segs, &turns, n_turns);
    }
    unreachable!("the retry loop returns on its second pass")
}

/// Shared tail of both diarize entry points: map the ABI return code, copy the
/// labels back, and convert the turns out of centiseconds.
fn finish(
    rc: i32,
    segs: &mut [DiarizeSegment],
    abi_segs: &[crispasr_sys::CrispasrDiarizeSegAbi],
    abi_turns: &[crispasr_sys::CrispasrDiarizeTurnAbi],
    n_turns: i32,
) -> Result<Vec<DiarizeTurn>, String> {
    match rc {
        0 => {
            for (i, s) in segs.iter_mut().enumerate() {
                s.speaker = abi_segs[i].speaker;
            }
            let n = (n_turns.max(0) as usize).min(abi_turns.len());
            Ok(abi_turns[..n]
                .iter()
                .map(|t| DiarizeTurn {
                    t0: t.t0_cs as f64 / 100.0,
                    t1: t.t1_cs as f64 / 100.0,
                    speaker: t.speaker,
                })
                .collect())
        }
        1 => Err("diarize model load failed (pyannote / foxnose embedder)".to_string()),
        -1 => Err("invalid arguments to crispasr_diarize_segments_abi".to_string()),
        2 => Err(format!(
            "diarize turn buffer too small — {n_turns} turns needed; this is a \
             crispasr bug, please report it with the audio length and segment count"
        )),
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
        let why = match n {
            -1 => " (bad arguments)",
            -2 => " (allocation failed)",
            -3 => " (the VAD model could not be loaded)",
            _ => "",
        };
        return Err(format!("crispasr_vad_slices failed (rc={n}){why}"));
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

// =========================================================================
// Chat / LLM — safe wrapper over include/crispasr_chat.h
// =========================================================================
//
// EU AI Act note: this surface generates synthetic TEXT, which the runtime
// does NOT mark for you (unlike every audio path, which watermarks). See
// [`ChatSession::ai_disclosure_text`] and docs/eu-ai-act.md §6.6 before
// shipping a product on top of it.

use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

/// Why a chat call failed.
///
/// The chat ABI promises exactly one error code as a contract —
/// `CRISPASR_CHAT_ERR_ABORTED` — because a caller running its own
/// cancellation has to tell a cancel apart from a decode fault. That
/// distinction is the whole reason this is an enum and not a `String`:
/// [`ChatError::Aborted`] means "you cancelled it", everything else is
/// [`ChatError::Failed`] carrying the C diagnostic verbatim. Do not switch
/// on numbers — no other code is stable.
///
/// `From<ChatError> for String` is implemented, so `?` still works inside
/// the `Result<_, String>` functions that make up the rest of this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatError {
    /// A registered abort callback stopped the generation (see
    /// [`ChatSession::with_abort_callback`]). Whatever text was already
    /// delivered through the token callback is valid, and the session has
    /// been flushed back to its just-opened state — reuse it directly, no
    /// [`ChatSession::reset`] needed.
    Aborted(String),
    /// Any other failure. The string is the C side's diagnostic.
    Failed(String),
}

impl ChatError {
    /// True when a registered abort callback stopped the generation, as
    /// opposed to anything going wrong.
    pub fn is_aborted(&self) -> bool {
        matches!(self, ChatError::Aborted(_))
    }

    /// The underlying diagnostic, without the variant.
    pub fn message(&self) -> &str {
        match self {
            ChatError::Aborted(m) | ChatError::Failed(m) => m,
        }
    }

    fn from_raw(err: &crispasr_sys::CrispasrChatError, code_hint: i32, fallback: &str) -> Self {
        // The one-shot path signals failure by returning null, so `err` is
        // the only carrier there; the streaming path also returns the code.
        let code = if err.code != 0 { err.code } else { code_hint };
        let bytes: Vec<u8> = err
            .message
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        let mut msg = String::from_utf8_lossy(&bytes).into_owned();
        if msg.is_empty() {
            msg = fallback.to_string();
        }
        if code == crispasr_sys::CRISPASR_CHAT_ERR_ABORTED {
            ChatError::Aborted(msg)
        } else {
            ChatError::Failed(msg)
        }
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Aborted(m) => write!(f, "aborted: {m}"),
            ChatError::Failed(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ChatError {}

impl From<ChatError> for String {
    fn from(e: ChatError) -> String {
        e.to_string()
    }
}

/// One turn in a conversation.
///
/// `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"` — the
/// OpenAI chat schema; the model's chat template maps those onto whatever
/// it actually expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// A message with an arbitrary role.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// A `"system"` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// A `"user"` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// An `"assistant"` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Per-session open options for [`ChatSession::open_with_options`].
///
/// Every field is `None` by default, meaning "keep the ABI's own default"
/// (read from `crispasr_chat_open_params_default`, so it cannot drift from
/// the C side). Set only what you need:
///
/// * `n_ctx` — context window in tokens; `None` uses the model's own.
///   Size it against [`ChatSession::count_tokens`] plus the tokens you
///   expect back.
/// * `n_batch` / `n_ubatch` — logical and physical prompt-batch sizes
///   (ABI default 512 each). A long prompt is prefilled in `n_batch`-sized
///   pieces, and the abort hook is checked between them, so `n_batch` also
///   sets the coarse bound on cancel latency.
/// * `n_gpu_layers` — `-1` offloads every layer (the ABI default), `0` is
///   CPU only, a positive value is a partial offload.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub n_ctx: Option<i32>,
    pub n_batch: Option<i32>,
    pub n_ubatch: Option<i32>,
    pub n_gpu_layers: Option<i32>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    /// Overrides the chat template baked into the GGUF. `None` reads
    /// `tokenizer.chat_template` from the model, falling back to "chatml".
    pub chat_template: Option<String>,
}

/// Keeps the C strings a [`crispasr_sys::CrispasrChatOpenParams`] points at
/// alive for as long as the params are in use.
struct RawOpenParams {
    _template: Option<CString>,
    params: crispasr_sys::CrispasrChatOpenParams,
}

impl ChatOptions {
    fn to_raw(&self) -> Result<RawOpenParams, ChatError> {
        let mut params = crispasr_sys::CrispasrChatOpenParams::default();
        // Start from the ABI's defaults rather than our own copy of them.
        unsafe { crispasr_sys::crispasr_chat_open_params_default(&mut params) };
        if let Some(v) = self.n_threads {
            params.n_threads = v;
        }
        if let Some(v) = self.n_threads_batch {
            params.n_threads_batch = v;
        }
        if let Some(v) = self.n_ctx {
            params.n_ctx = v;
        }
        if let Some(v) = self.n_batch {
            params.n_batch = v;
        }
        if let Some(v) = self.n_ubatch {
            params.n_ubatch = v;
        }
        if let Some(v) = self.n_gpu_layers {
            params.n_gpu_layers = v;
        }
        if let Some(v) = self.use_mmap {
            params.use_mmap = v;
        }
        if let Some(v) = self.use_mlock {
            params.use_mlock = v;
        }
        let template = match &self.chat_template {
            Some(t) => {
                let c = CString::new(t.as_str())
                    .map_err(|e| ChatError::Failed(format!("chat_template NUL: {e}")))?;
                params.chat_template = c.as_ptr();
                Some(c)
            }
            None => None,
        };
        Ok(RawOpenParams {
            _template: template,
            params,
        })
    }
}

/// Per-call sampler options for the `*_with_options` generate entry points.
///
/// As with [`ChatOptions`], `None` keeps the ABI default. Note that
/// `temperature: Some(0.0)` selects greedy decoding, which short-circuits
/// the rest of the sampler chain — `top_k`, `top_p`, `min_p` and `seed`
/// have no effect there — and `repeat_penalty: Some(1.0)` drops the
/// penalties sampler, which makes `repeat_last_n` inert too.
#[derive(Debug, Clone, Default)]
pub struct ChatGenerateOptions {
    /// Hard cap on generated tokens. `Some(0)` does NOT mean "generate
    /// nothing": the ABI reads any non-positive value as "unset" and applies
    /// its own default of 256. Use `prefill_only` to suppress generation.
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub seed: Option<u32>,
    /// Generation halts the first time any of these appears in the decoded
    /// output; the output is truncated before the match.
    pub stop: Vec<String>,
    /// Prefill the prompt but suppress assistant generation — for
    /// measuring prompt cost.
    pub prefill_only: bool,
}

/// Keeps the stop-string array a
/// [`crispasr_sys::CrispasrChatGenerateParams`] points at alive.
struct RawGenerateParams {
    _stop_owned: Vec<CString>,
    _stop_ptrs: Vec<*const c_char>,
    params: crispasr_sys::CrispasrChatGenerateParams,
}

impl ChatGenerateOptions {
    fn to_raw(&self) -> Result<RawGenerateParams, ChatError> {
        let mut params = crispasr_sys::CrispasrChatGenerateParams::default();
        unsafe { crispasr_sys::crispasr_chat_generate_params_default(&mut params) };
        if let Some(v) = self.max_tokens {
            params.max_tokens = v;
        }
        if let Some(v) = self.temperature {
            params.temperature = v;
        }
        if let Some(v) = self.top_k {
            params.top_k = v;
        }
        if let Some(v) = self.top_p {
            params.top_p = v;
        }
        if let Some(v) = self.min_p {
            params.min_p = v;
        }
        if let Some(v) = self.repeat_penalty {
            params.repeat_penalty = v;
        }
        if let Some(v) = self.repeat_last_n {
            params.repeat_last_n = v;
        }
        if let Some(v) = self.seed {
            params.seed = v;
        }
        params.prefill_only = self.prefill_only;

        let mut stop_owned = Vec::with_capacity(self.stop.len());
        for s in &self.stop {
            stop_owned.push(
                CString::new(s.as_str())
                    .map_err(|e| ChatError::Failed(format!("stop sequence NUL: {e}")))?,
            );
        }
        let stop_ptrs: Vec<*const c_char> = stop_owned.iter().map(|s| s.as_ptr()).collect();
        if stop_ptrs.is_empty() {
            params.stop = std::ptr::null();
            params.n_stop = 0;
        } else {
            params.stop = stop_ptrs.as_ptr();
            params.n_stop = stop_ptrs.len();
        }
        Ok(RawGenerateParams {
            _stop_owned: stop_owned,
            _stop_ptrs: stop_ptrs,
            params,
        })
    }
}

/// Keeps the C strings a message array points at alive for the call.
struct RawMessages {
    _owned: Vec<(CString, CString)>,
    raw: Vec<crispasr_sys::CrispasrChatMessage>,
}

fn raw_messages(messages: &[ChatMessage]) -> Result<RawMessages, ChatError> {
    let mut owned = Vec::with_capacity(messages.len());
    for m in messages {
        let role = CString::new(m.role.as_str())
            .map_err(|e| ChatError::Failed(format!("role NUL: {e}")))?;
        let content = CString::new(m.content.as_str())
            .map_err(|e| ChatError::Failed(format!("content NUL: {e}")))?;
        owned.push((role, content));
    }
    // CString owns a heap buffer, so these pointers survive `owned` moving.
    let raw = owned
        .iter()
        .map(|(r, c)| crispasr_sys::CrispasrChatMessage {
            role: r.as_ptr(),
            content: c.as_ptr(),
        })
        .collect();
    Ok(RawMessages { _owned: owned, raw })
}

/// What the token trampoline needs: the caller's closure, plus a slot for
/// a panic it raised. A panic must not unwind through C, so it is caught
/// here and resumed once the native call has returned.
///
/// `failed` is the session's shared flag, which the abort trampoline reads:
/// nobody is left reading the output once the token closure has panicked, so
/// a registered abort hook answers "stop" from then on.
///
/// `pending` holds the tail of a character the C side delivered in pieces. A
/// model that falls back to byte tokens emits one chunk per BYTE, so a chunk
/// can end part-way through a character; decoding each chunk on its own would
/// turn every such character into replacement characters for good.
struct TokenState<'a> {
    on_token: &'a mut dyn FnMut(&str),
    failed: &'a Cell<bool>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    pending: Vec<u8>,
}

impl TokenState<'_> {
    /// Hand `text` to the caller's closure, catching a panic rather than
    /// letting it unwind through C. Empty text is not delivered.
    fn deliver(&mut self, text: &str) {
        if self.panic.is_some() || text.is_empty() {
            return;
        }
        let on_token = &mut *self.on_token;
        let outcome = catch_unwind(AssertUnwindSafe(|| on_token(text)));
        if let Err(p) = outcome {
            self.panic = Some(p);
            self.failed.set(true);
        }
    }

    /// Deliver everything in `bytes` that completes a character, keeping any
    /// trailing incomplete sequence for the next chunk.
    fn deliver_utf8(&mut self, bytes: &[u8]) {
        let text = take_complete_utf8(&mut self.pending, bytes);
        self.deliver(&text);
    }

    /// Deliver whatever is still buffered when the generation ends. Those
    /// bytes are a character the generation stopped in the middle of, so they
    /// are malformed on their own — hand them over with replacement rather
    /// than drop output silently.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        self.deliver(&tail);
    }
}

/// Append `bytes` to `pending` and split off the longest prefix that is
/// valid UTF-8, leaving a trailing incomplete sequence in `pending`.
///
/// A sequence that is merely unfinished (`Utf8Error::error_len() == None`) is
/// held back — its remaining bytes arrive in a later chunk. One that is
/// genuinely invalid (`Some(n)`) never becomes valid however much follows, so
/// it becomes one replacement character and decoding continues after it.
fn take_complete_utf8(pending: &mut Vec<u8>, bytes: &[u8]) -> String {
    pending.extend_from_slice(bytes);
    let mut out = String::new();
    let mut consumed = 0;
    loop {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(rest) => {
                out.push_str(rest);
                consumed = pending.len();
                break;
            }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(&String::from_utf8_lossy(
                    &pending[consumed..consumed + good],
                ));
                match e.error_len() {
                    Some(bad) => {
                        out.push('\u{fffd}');
                        consumed += good + bad;
                    }
                    None => {
                        consumed += good;
                        break;
                    }
                }
            }
        }
    }
    pending.drain(..consumed);
    out
}

extern "C" fn token_trampoline(chunk: *const c_char, user: *mut c_void) {
    if user.is_null() || chunk.is_null() {
        return;
    }
    // SAFETY: `user` is the `&mut TokenState` registered for this one call.
    // The C side only invokes this synchronously from inside that call, on
    // the same thread, so the reference is live and unaliased here.
    let st = unsafe { &mut *(user as *mut TokenState) };
    if st.panic.is_some() {
        // Already unwound once — don't call the closure again.
        return;
    }
    // SAFETY: valid for the duration of the callback, per the ABI.
    let bytes = unsafe { CStr::from_ptr(chunk) }.to_bytes();
    st.deliver_utf8(bytes);
}

/// The abort hook's counterpart to [`TokenState`].
struct AbortState<'a> {
    should_continue: &'a mut dyn FnMut() -> bool,
    token_failed: &'a Cell<bool>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

extern "C" fn abort_trampoline(user: *mut c_void) -> bool {
    if user.is_null() {
        return true; // nothing registered — let the generation continue
    }
    // SAFETY: `user` is the `&mut AbortState` registered by
    // `with_abort_callback`, which restores the previous registration before
    // the state goes out of scope (on the unwind path too).
    let st = unsafe { &mut *(user as *mut AbortState) };
    if st.panic.is_some() {
        return false; // already unwound once — stop the generation
    }
    if st.token_failed.get() {
        // The token closure panicked: nothing is reading the output any
        // more, so stop without asking the caller's predicate again.
        return false;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| (st.should_continue)()));
    match outcome {
        // Same question, same polarity as the C ABI: `true` continues.
        Ok(keep_going) => keep_going,
        Err(p) => {
            // A hook that panicked cannot be trusted to answer again, and
            // finishing the generation would waste the work anyway.
            st.panic = Some(p);
            false
        }
    }
}

/// A chat / LLM session over a GGUF model.
///
/// One call at a time per session — the C side serialises its context with
/// a mutex, so a second concurrent call blocks rather than racing. The KV
/// cache persists across [`ChatSession::generate`] calls, and reuse depends
/// on passing the WHOLE conversation every time: the session compares the
/// templated prompt against the tokens it already holds and decodes only
/// what is new. Passing just the latest turn is not wrong, it simply
/// shares no prefix.
///
/// ```no_run
/// use crispasr::{ChatMessage, ChatOptions, ChatSession};
///
/// let msgs = vec![ChatMessage::user("Say hello.")];
/// let opts = ChatOptions {
///     n_ctx: Some(4096),
///     ..Default::default()
/// };
/// let chat = ChatSession::open_with_options("gemma-3-1b-it-Q4_K_M.gguf", &opts).unwrap();
/// assert!(chat.count_tokens(&msgs).unwrap() < chat.n_ctx());
/// println!("{}", chat.generate(&msgs).unwrap());
/// ```
pub struct ChatSession {
    handle: *mut crispasr_sys::CrispasrChatSession,
    /// Set by the token trampoline when the caller's closure panicked, read
    /// by the abort trampoline so a registered hook answers "stop".
    token_failed: Cell<bool>,
    /// The `AbortState` currently registered with the C session, or null.
    /// [`ChatSession::with_abort_callback`] puts back what it found here so a
    /// nested scope does not disarm the enclosing one.
    abort_user: Cell<*mut c_void>,
}

// Not `Sync` — one call at a time, and the abort registration is
// session-global state.
unsafe impl Send for ChatSession {}

impl ChatSession {
    /// Open a GGUF chat model with the ABI's default parameters.
    pub fn open(model_path: &str) -> Result<Self, ChatError> {
        Self::open_with_options(model_path, &ChatOptions::default())
    }

    /// Open a GGUF chat model, overriding the parameters named in
    /// `options` (context window, batch sizes, GPU offload, threads,
    /// chat template).
    pub fn open_with_options(model_path: &str, options: &ChatOptions) -> Result<Self, ChatError> {
        let path = CString::new(model_path)
            .map_err(|e| ChatError::Failed(format!("invalid path: {e}")))?;
        let raw = options.to_raw()?;
        let mut err = crispasr_sys::CrispasrChatError::default();
        let handle =
            unsafe { crispasr_sys::crispasr_chat_open(path.as_ptr(), &raw.params, &mut err) };
        if handle.is_null() {
            return Err(ChatError::from_raw(
                &err,
                0,
                &format!("failed to open chat model {model_path:?}"),
            ));
        }
        Ok(Self {
            handle,
            token_failed: Cell::new(false),
            abort_user: Cell::new(std::ptr::null_mut()),
        })
    }

    /// Conservative working set in bytes (weights + KV cache + activations)
    /// for a model on disk, reading its metadata but never its tensor data —
    /// a pre-flight guard for low-memory devices. `options` matters mostly
    /// for `n_ctx`, which sizes the KV term linearly; leave it unset and the
    /// model's own trained context is used.
    ///
    /// The number is deliberately high, not approximate. The KV term bills
    /// both the K and the V cache at the full attention width `n_embd`, but
    /// a grouped-query model gives each layer a K/V width that is a fraction
    /// of that: on Gemma 3 1B the KV term comes out 4.50× llama.cpp's real
    /// cache (117.00 MiB against 26.00 MiB at `n_ctx` 1024), which is 1.33×
    /// on the whole estimate at `n_ctx` 4096. Over-reporting is the safe
    /// direction for a "will this fit?" guard: it can turn away a model that
    /// would just have fitted, and never admits one that would not.
    pub fn memory_estimate(model_path: &str, options: &ChatOptions) -> Result<usize, ChatError> {
        let path = CString::new(model_path)
            .map_err(|e| ChatError::Failed(format!("invalid path: {e}")))?;
        let raw = options.to_raw()?;
        let mut err = crispasr_sys::CrispasrChatError::default();
        let bytes = unsafe {
            crispasr_sys::crispasr_chat_memory_estimate(path.as_ptr(), &raw.params, &mut err)
        };
        if bytes == 0 {
            return Err(ChatError::from_raw(
                &err,
                0,
                "could not estimate chat model memory",
            ));
        }
        Ok(bytes)
    }

    /// The canonical "you are talking to an AI" wording (EU AI Act
    /// Art. 50(1)). Show it at or before the first turn of any
    /// conversational product, and show it visibly — Art. 50(5) requires
    /// disclosures to meet accessibility requirements. Nothing in this
    /// crate marks generated text for you.
    pub fn ai_disclosure_text() -> &'static str {
        let p = unsafe { crispasr_sys::crispasr_chat_ai_disclosure_text() };
        if p.is_null() {
            return "";
        }
        // SAFETY: a static string owned by the library; never freed.
        unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
    }

    /// The context window in tokens. Compare against
    /// [`Self::count_tokens`] when sizing a prompt.
    pub fn n_ctx(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_chat_n_ctx(self.handle) }
    }

    /// The chat template the session resolved against — e.g. "chatml",
    /// "llama3", "gemma". Empty only if the handle has none.
    pub fn template_name(&self) -> String {
        let p = unsafe { crispasr_sys::crispasr_chat_template_name(self.handle) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: owned by the session, valid until it is dropped.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Clear the KV cache and history so the next generate re-prefills from
    /// scratch. Call it when starting a new conversation in a reused
    /// session — not after an abort, which already resets the session.
    pub fn reset(&self) -> Result<(), ChatError> {
        let mut err = crispasr_sys::CrispasrChatError::default();
        let rc = unsafe { crispasr_sys::crispasr_chat_reset(self.handle, &mut err) };
        if rc != 0 {
            return Err(ChatError::from_raw(&err, rc, "crispasr_chat_reset failed"));
        }
        Ok(())
    }

    /// Tokens the model's own tokenizer produces for `messages` once the
    /// chat template has been applied — the prompt a FRESH session
    /// prefills, so it can be compared straight against [`Self::n_ctx`].
    /// It covers the template's control tokens, the leading BOS, and the
    /// trailing generation prompt.
    ///
    /// An empty `messages` counts the template's own opening, which is
    /// whatever that template emits for no messages — template-dependent,
    /// and possibly nothing at all: several chat templates write only from
    /// inside their loop over the messages, and those return `0`. Do not
    /// read a positive overhead into it.
    ///
    /// A pure query: it touches neither the KV cache nor the history, so
    /// it can be called freely between generations. For a session part-way
    /// through a conversation the number is an upper bound, since the
    /// history already holds part of the prompt.
    pub fn count_tokens(&self, messages: &[ChatMessage]) -> Result<i32, ChatError> {
        let msgs = raw_messages(messages)?;
        let mut err = crispasr_sys::CrispasrChatError::default();
        let n = unsafe {
            crispasr_sys::crispasr_chat_count_tokens(
                self.handle,
                msgs.raw.as_ptr(),
                msgs.raw.len(),
                &mut err,
            )
        };
        if n < 0 {
            return Err(ChatError::from_raw(
                &err,
                0,
                "crispasr_chat_count_tokens failed",
            ));
        }
        Ok(n)
    }

    /// Generate a reply with the ABI's default sampler settings.
    pub fn generate(&self, messages: &[ChatMessage]) -> Result<String, ChatError> {
        self.generate_with_options(messages, &ChatGenerateOptions::default())
    }

    /// Generate a reply, overriding the sampler settings named in `params`.
    pub fn generate_with_options(
        &self,
        messages: &[ChatMessage],
        params: &ChatGenerateOptions,
    ) -> Result<String, ChatError> {
        let msgs = raw_messages(messages)?;
        let gp = params.to_raw()?;
        // The flag describes the generation about to run, not the last one.
        self.token_failed.set(false);
        let mut err = crispasr_sys::CrispasrChatError::default();
        let out = unsafe {
            crispasr_sys::crispasr_chat_generate(
                self.handle,
                msgs.raw.as_ptr(),
                msgs.raw.len(),
                &gp.params,
                &mut err,
            )
        };
        if out.is_null() {
            return Err(ChatError::from_raw(
                &err,
                0,
                "crispasr_chat_generate failed",
            ));
        }
        // SAFETY: a malloc'd NUL-terminated buffer we now own.
        let text = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        unsafe { crispasr_sys::crispasr_chat_string_free(out) };
        Ok(text)
    }

    /// Stream a reply with the ABI's default sampler settings, calling
    /// `on_token` once per detokenised chunk (usually one per token — see
    /// [`Self::generate_stream_with_options`] for when it is fewer).
    pub fn generate_stream<F: FnMut(&str)>(
        &self,
        messages: &[ChatMessage],
        on_token: F,
    ) -> Result<(), ChatError> {
        self.generate_stream_with_options(messages, &ChatGenerateOptions::default(), on_token)
    }

    /// Stream a reply, overriding the sampler settings named in `params`.
    ///
    /// `on_token` fires on the calling thread for the duration of this call
    /// only, so it need not be `Send` or `'static`. Concatenating the
    /// chunks yields exactly what [`Self::generate_with_options`] would
    /// return for the same messages and params — except when a stop sequence
    /// ends the generation. The C side hands each piece to the callback
    /// before it scans for a match, so the chunk the match lands in has
    /// already been delivered, while the one-shot return value is truncated
    /// before the match. With `stop` set, the streamed text is therefore the
    /// one-shot text plus that last chunk, and a caller who wants the
    /// truncated form has to cut it back themselves.
    ///
    /// Every chunk is whole characters. A model that spells a character the
    /// tokeniser does not hold emits it one byte per token, so those bytes
    /// are buffered here and delivered when the character is complete: such a
    /// run of tokens produces ONE call rather than one call per token, and
    /// the number of calls is therefore at most the number of tokens, not
    /// equal to it. If the generation stops part-way through a character the
    /// leftover bytes are delivered as replacement characters once the call
    /// finishes, rather than dropped.
    ///
    /// `on_token` runs with the session mutex held, so it must not call back
    /// into this session — re-entering deadlocks, the same way it does from
    /// the abort hook.
    ///
    /// If `on_token` panics the panic is caught at the FFI boundary,
    /// further chunks are dropped, and the panic is resumed once the native
    /// call has returned — it never unwinds through C. An abort callback
    /// registered with [`Self::with_abort_callback`] then answers "stop"
    /// from the next check onwards, without consulting your predicate again,
    /// so the generation is cancelled rather than decoding output nobody
    /// reads. With no abort callback registered there is no way to ask the
    /// ABI to stop, so the generation runs to completion first.
    pub fn generate_stream_with_options<F: FnMut(&str)>(
        &self,
        messages: &[ChatMessage],
        params: &ChatGenerateOptions,
        mut on_token: F,
    ) -> Result<(), ChatError> {
        let msgs = raw_messages(messages)?;
        let gp = params.to_raw()?;
        self.token_failed.set(false);
        let mut state = TokenState {
            on_token: &mut on_token,
            failed: &self.token_failed,
            panic: None,
            pending: Vec::new(),
        };
        let mut err = crispasr_sys::CrispasrChatError::default();
        let rc = unsafe {
            crispasr_sys::crispasr_chat_generate_stream(
                self.handle,
                msgs.raw.as_ptr(),
                msgs.raw.len(),
                &gp.params,
                Some(token_trampoline),
                &mut state as *mut TokenState as *mut c_void,
                &mut err,
            )
        };
        // Whatever is still buffered belongs to the caller, aborted run or
        // not; `flush` is a no-op once the closure has panicked.
        state.flush();
        // A panic the callback raised outranks whatever the call returned.
        if let Some(p) = state.panic.take() {
            resume_unwind(p);
        }
        if rc != 0 {
            return Err(ChatError::from_raw(
                &err,
                rc,
                "crispasr_chat_generate_stream failed",
            ));
        }
        Ok(())
    }

    /// Run `body` with `should_continue` registered as this session's abort
    /// hook, then clear it.
    ///
    /// **Polarity: `should_continue` returns `true` to LET THE GENERATION
    /// CONTINUE** and `false` to abort it — a "may I keep going?" predicate.
    /// That is the polarity of `crispasr_chat_abort_callback` in
    /// `crispasr_chat.h`, which in turn matches the encoder-begin callback on
    /// the ASR surface; this crate forwards your answer to C unchanged.
    ///
    /// The hook is called on the generating thread before each prompt batch
    /// during prefill and before each sampled token, and additionally from
    /// inside a running compute graph on the CPU backend — so it must be
    /// cheap and non-blocking, and it must not call back into this session
    /// (the session mutex is held for the whole generation; re-entering
    /// deadlocks). Cancel latency is therefore one prompt batch or one
    /// token on Metal / CUDA, and finer on CPU.
    ///
    /// On abort the generate call fails with [`ChatError::Aborted`], having
    /// already delivered its partial text through the token callback, and
    /// the session is flushed back to its just-opened state — reuse it
    /// without a [`Self::reset`].
    ///
    /// A panic inside `should_continue` is caught at the FFI boundary, aborts
    /// the generation, and is resumed after `body` returns. A panic inside the
    /// token callback of [`Self::generate_stream_with_options`] also aborts
    /// the generation, through this hook: from the next check onwards it
    /// answers "stop" without consulting `should_continue` again.
    ///
    /// The registration is scoped to this call because the hook is a
    /// pointer to a closure on your stack: it is taken back before
    /// `should_continue` can go out of scope, including on the unwind path.
    /// Nesting works — `body` is handed the session, so it may open a scope
    /// of its own, and the inner scope puts the outer hook back when it ends
    /// rather than leaving the session with none.
    ///
    /// ```no_run
    /// use std::cell::Cell;
    /// use crispasr::{ChatMessage, ChatSession};
    ///
    /// let chat = ChatSession::open("model.gguf").unwrap();
    /// let msgs = vec![ChatMessage::user("Write an essay.")];
    /// let seen = Cell::new(0usize);
    /// let res = chat.with_abort_callback(
    ///     || seen.get() < 20,
    ///     |c| {
    ///         c.generate_stream(&msgs, |chunk| {
    ///             seen.set(seen.get() + 1);
    ///             print!("{chunk}");
    ///         })
    ///     },
    /// );
    /// assert!(res.is_err_and(|e| e.is_aborted()));
    /// ```
    pub fn with_abort_callback<A, B, R>(&self, mut should_continue: A, body: B) -> R
    where
        A: FnMut() -> bool,
        B: FnOnce(&ChatSession) -> R,
    {
        let mut state = AbortState {
            should_continue: &mut should_continue,
            token_failed: &self.token_failed,
            panic: None,
        };

        /// Puts back whatever was registered before, on every exit path,
        /// unwinding included — the session must not hold a pointer to
        /// `state` past this frame, and an enclosing scope's hook must
        /// survive this one ending.
        struct Restore<'s> {
            session: &'s ChatSession,
            previous: *mut c_void,
        }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                unsafe {
                    if self.previous.is_null() {
                        crispasr_sys::crispasr_chat_set_abort_callback(
                            self.session.handle,
                            None,
                            std::ptr::null_mut(),
                        );
                    } else {
                        // SAFETY: the enclosing scope's `AbortState` is still
                        // on the stack — this frame is nested inside its body.
                        crispasr_sys::crispasr_chat_set_abort_callback(
                            self.session.handle,
                            Some(abort_trampoline),
                            self.previous,
                        );
                    }
                }
                self.session.abort_user.set(self.previous);
            }
        }

        let previous = self.abort_user.get();
        let user = &mut state as *mut AbortState as *mut c_void;
        unsafe {
            crispasr_sys::crispasr_chat_set_abort_callback(
                self.handle,
                Some(abort_trampoline),
                user,
            )
        };
        self.abort_user.set(user);
        let guard = Restore {
            session: self,
            previous,
        };
        let out = body(self);
        drop(guard);

        if let Some(p) = state.panic.take() {
            resume_unwind(p);
        }
        out
    }
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_chat_close(self.handle) }
    }
}

#[cfg(test)]
mod chat_stream_tests {
    use super::take_complete_utf8;

    /// One byte at a time is what a byte-fallback token stream looks like.
    #[test]
    fn a_character_split_over_several_chunks_is_delivered_once() {
        let mut pending = Vec::new();
        let mut out = String::new();
        for b in "🪿".as_bytes() {
            out.push_str(&take_complete_utf8(&mut pending, &[*b]));
        }
        assert_eq!(out, "🪿");
        assert!(pending.is_empty(), "nothing left over: {pending:?}");
    }

    #[test]
    fn a_complete_chunk_passes_straight_through() {
        let mut pending = Vec::new();
        assert_eq!(take_complete_utf8(&mut pending, b"hello"), "hello");
        assert!(pending.is_empty());
    }

    #[test]
    fn a_partial_tail_is_held_back_and_nothing_before_it_is_lost() {
        let mut pending = Vec::new();
        // "ab" plus the first two bytes of a three-byte character.
        assert_eq!(take_complete_utf8(&mut pending, b"ab\xe2\x82"), "ab");
        assert_eq!(pending, b"\xe2\x82");
        assert_eq!(take_complete_utf8(&mut pending, b"\xacc"), "€c");
        assert!(pending.is_empty());
    }

    /// `error_len() == Some(n)`: no continuation byte can rescue these, so
    /// they become one replacement character and decoding carries on.
    #[test]
    fn a_genuinely_invalid_sequence_is_replaced_not_buffered() {
        let mut pending = Vec::new();
        assert_eq!(take_complete_utf8(&mut pending, b"a\xffb"), "a\u{fffd}b");
        assert!(
            pending.is_empty(),
            "invalid bytes must not be held: {pending:?}"
        );
    }

    /// A truncated sequence followed by an unrelated character is invalid
    /// once the next byte arrives, not merely unfinished.
    #[test]
    fn a_truncated_sequence_is_replaced_once_the_next_character_arrives() {
        let mut pending = Vec::new();
        assert_eq!(take_complete_utf8(&mut pending, b"\xe2\x82"), "");
        assert_eq!(take_complete_utf8(&mut pending, b"x"), "\u{fffd}x");
        assert!(pending.is_empty());
    }

    #[test]
    fn an_empty_chunk_delivers_nothing_and_keeps_the_buffer() {
        let mut pending = Vec::new();
        assert_eq!(take_complete_utf8(&mut pending, b"\xf0\x9f"), "");
        assert_eq!(take_complete_utf8(&mut pending, b""), "");
        assert_eq!(pending, b"\xf0\x9f");
    }
}
