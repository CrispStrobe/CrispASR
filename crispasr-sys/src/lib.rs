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
}
