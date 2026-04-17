//! Safe Rust wrapper for CrispASR speech recognition.
//!
//! # Quick start
//!
//! ```no_run
//! use crispasr::CrispASR;
//!
//! let model = CrispASR::new("ggml-base.en.bin").unwrap();
//! let segments = model.transcribe_pcm(&pcm_f32).unwrap();
//! for seg in &segments {
//!     println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
//! }
//! ```

use std::ffi::{CStr, CString};

/// A transcription segment with timing information.
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start: f64, // seconds
    pub end: f64,   // seconds
    pub no_speech_prob: f32,
}

/// A loaded CrispASR model.
///
/// Not `Sync` — do not share between threads.
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
        let params = unsafe {
            crispasr_sys::whisper_full_default_params_by_ref(strategy)
        };

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
