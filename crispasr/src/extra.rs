//! Extra session operations — music, conversion, TTS extras, audio helpers.
//! This file is included from lib.rs to keep the main file manageable.
//! All returned native views are copied before the next native call can replace session-owned storage.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};

use crate::Session;

// --- Music types ---

#[derive(Debug, Clone)]
pub struct PianoNote {
    pub onset_ms: f32,
    pub offset_ms: f32,
    pub midi_note: i32,
    pub velocity: i32,
}

#[derive(Debug, Clone)]
pub struct PianoResult {
    pub notes: Vec<PianoNote>,
    pub programs: Vec<i32>,
    pub sample_rate: i32,
}

#[derive(Debug, Clone)]
pub struct ChordSpan {
    pub start_ms: f32,
    pub end_ms: f32,
    pub label: i32,
    pub confidence: f32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ChordResult {
    pub spans: Vec<ChordSpan>,
    pub vocab_size: i32,
}

#[derive(Debug, Clone)]
pub struct BeatEvent {
    pub time_s: f32,
    pub is_downbeat: bool,
}

#[derive(Debug, Clone)]
pub struct BeatResult {
    pub events: Vec<BeatEvent>,
    pub tempo_bpm: f32,
    pub sample_rate: i32,
}

#[derive(Debug, Clone)]
pub struct TabResult {
    pub emissions: Vec<f32>,
    pub frames: i32,
    pub strings: i32,
    pub classes: i32,
    pub silent_class: i32,
    pub frame_period: f32,
    pub open_midi: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct PitchFrame {
    pub time_ms: f32,
    pub f0_hz: f32,
    pub voiced_prob: f32,
}

impl Session {
    /// Transcribe mono PCM at piano_sample_rate(); empty notes are valid (not an error).
    pub fn piano(&self, pcm: &[f32]) -> Result<PianoResult, String> {
        if pcm.is_empty() {
            return Err("piano: empty PCM".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_piano(self.handle, pcm.as_ptr(), pcm.len() as c_int)
        };
        if n < 0 {
            return Err(format!("piano failed (rc={})", n));
        }
        if n == 0 {
            return Ok(PianoResult {
                notes: Vec::new(),
                programs: Vec::new(),
                sample_rate: self.piano_sample_rate(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_piano_notes(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("piano: negative note count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("piano: notes pointer is null".to_string());
            }
            // count==0 with null ptr is valid empty result
        }
        let notes = if count == 0 {
            Vec::new()
        } else {
            let len = count
                .checked_mul(4)
                .ok_or("piano: count overflow".to_string())?;
            let flat = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                v.push(PianoNote {
                    onset_ms: flat[i * 4 + 0],
                    offset_ms: flat[i * 4 + 1],
                    midi_note: flat[i * 4 + 2] as i32,
                    velocity: flat[i * 4 + 3] as i32,
                });
            }
            v
        };
        let mut out_n2 = 0;
        let ptr2 =
            unsafe { crispasr_sys::crispasr_session_piano_note_programs(self.handle, &mut out_n2) };
        if out_n2 < 0 {
            return Err(format!("piano: negative programs count {}", out_n2));
        }
        if out_n2 as usize != count {
            return Err(format!(
                "piano notes/programs length mismatch {} vs {}",
                count, out_n2
            ));
        }
        if ptr2.is_null() && count != 0 {
            return Err("piano: programs pointer is null".to_string());
        }
        let progs = if count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(ptr2, count).to_vec() }
        };
        Ok(PianoResult {
            notes,
            programs: progs,
            sample_rate: self.piano_sample_rate(),
        })
    }

    pub fn piano_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_piano_sample_rate(self.handle) as i32 }
    }

    pub fn piano_n_notes(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_piano_n_notes(self.handle) as i32 }
    }

    pub fn chords(&self, pcm: &[f32], sample_rate: i32) -> Result<ChordResult, String> {
        if pcm.is_empty() {
            return Err("chords: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("chords: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_chords(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!("chords failed (rc={})", n));
        }
        if n == 0 {
            return Ok(ChordResult {
                spans: Vec::new(),
                vocab_size: self.chords_vocab_size(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_chords_spans(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("chords: negative span count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() && count != 0 {
            return Err("chords: spans pointer is null".to_string());
        }
        let flat = if count == 0 {
            Vec::new()
        } else {
            let len = count
                .checked_mul(4)
                .ok_or("chords: count overflow".to_string())?;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };
        let mut spans = Vec::with_capacity(count);
        for i in 0..count {
            let name_ptr =
                unsafe { crispasr_sys::crispasr_session_chords_span_name(self.handle, i as c_int) };
            let name = if name_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() }
            };
            spans.push(ChordSpan {
                start_ms: flat[i * 4 + 0],
                end_ms: flat[i * 4 + 1],
                label: flat[i * 4 + 2] as i32,
                confidence: flat[i * 4 + 3],
                name,
            });
        }
        Ok(ChordResult {
            spans,
            vocab_size: self.chords_vocab_size(),
        })
    }

    pub fn chords_vocab_size(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_chords_vocab_size(self.handle) as i32 }
    }

    pub fn beats(&self, pcm: &[f32], sample_rate: i32) -> Result<BeatResult, String> {
        if pcm.is_empty() {
            return Err("beats: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("beats: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_beats(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!(
                "beats failed (rc={}) — sample rate mismatch or no beat arm",
                n
            ));
        }
        if n == 0 {
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_beats_events(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("beats: negative event count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("beats: events pointer is null".to_string());
            }
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        if count == 0 {
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        let len = count
            .checked_mul(2)
            .ok_or("beats: count overflow".to_string())?;
        let flat = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        let mut events = Vec::with_capacity(count);
        for i in 0..count {
            events.push(BeatEvent {
                time_s: flat[i * 2 + 0],
                is_downbeat: flat[i * 2 + 1] != 0.0,
            });
        }
        Ok(BeatResult {
            events,
            tempo_bpm: self.beats_tempo_bpm(),
            sample_rate: self.beats_sample_rate(),
        })
    }

    pub fn beats_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_beats_sample_rate(self.handle) as i32 }
    }

    pub fn beats_tempo_bpm(&self) -> f32 {
        unsafe { crispasr_sys::crispasr_session_beats_tempo_bpm(self.handle) }
    }

    pub fn tab(&self, pcm: &[f32], sample_rate: i32) -> Result<TabResult, String> {
        if pcm.is_empty() {
            return Err("tab: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("tab: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_tab(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!("tab failed (rc={})", n));
        }
        if n == 0 {
            return Ok(TabResult {
                emissions: Vec::new(),
                frames: 0,
                strings: 0,
                classes: 0,
                silent_class: self.tab_silent_class(),
                frame_period: self.tab_frame_period(),
                open_midi: Vec::new(),
            });
        }
        let mut out_frames = 0;
        let mut out_strings = 0;
        let mut out_classes = 0;
        let ptr = unsafe {
            crispasr_sys::crispasr_session_tab_emissions(
                self.handle,
                &mut out_frames,
                &mut out_strings,
                &mut out_classes,
            )
        };
        if out_frames < 0 || out_strings < 0 || out_classes < 0 {
            return Err(format!(
                "tab: negative dimensions {}/{}/{}",
                out_frames, out_strings, out_classes
            ));
        }
        if ptr.is_null() {
            return Err("tab: emissions pointer is null".to_string());
        }
        let frames = out_frames as usize;
        let strs = out_strings as usize;
        let cls = out_classes as usize;
        let total = frames
            .checked_mul(strs)
            .and_then(|v| v.checked_mul(cls))
            .ok_or("tab: dimensions overflow".to_string())?;
        if total == 0 {
            // Empty emissions but non-zero dims should have been n==0 early; total==0 means empty
            // but we already handled n==0 above, so this would be unexpected zero product
            // Still safe to return empty vec without calling from_raw_parts
            return Ok(TabResult {
                emissions: Vec::new(),
                frames: out_frames as i32,
                strings: out_strings as i32,
                classes: out_classes as i32,
                silent_class: self.tab_silent_class(),
                frame_period: self.tab_frame_period(),
                open_midi: Vec::new(),
            });
        }
        let emissions = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };
        let mut open_midi = Vec::with_capacity(strs);
        for i in 0..strs {
            open_midi.push(unsafe {
                crispasr_sys::crispasr_session_tab_string_open_midi(self.handle, i as c_int) as i32
            });
        }
        Ok(TabResult {
            emissions,
            frames: out_frames as i32,
            strings: out_strings as i32,
            classes: out_classes as i32,
            silent_class: self.tab_silent_class(),
            frame_period: self.tab_frame_period(),
            open_midi,
        })
    }

    pub fn tab_silent_class(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_tab_silent_class(self.handle) as i32 }
    }

    pub fn tab_frame_period(&self) -> f32 {
        unsafe { crispasr_sys::crispasr_session_tab_frame_period(self.handle) }
    }

    pub fn pitch(&self, pcm: &[f32], hop_ms: f32) -> Result<Vec<PitchFrame>, String> {
        if pcm.is_empty() {
            return Err("pitch: empty PCM".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_pitch(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                hop_ms,
            )
        };
        if n < 0 {
            return Err(format!("pitch failed (rc={})", n));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_pitch_frames(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("pitch: negative frame count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("pitch: frames pointer is null".to_string());
            }
            return Ok(Vec::new());
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let len = count
            .checked_mul(3)
            .ok_or("pitch: count overflow".to_string())?;
        let flat = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        let mut frames = Vec::with_capacity(count);
        for i in 0..count {
            frames.push(PitchFrame {
                time_ms: flat[i * 3 + 0],
                f0_hz: flat[i * 3 + 1],
                voiced_prob: flat[i * 3 + 2],
            });
        }
        Ok(frames)
    }

    pub fn pitch_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_pitch_sample_rate(self.handle) as i32 }
    }

    // --- Conversion (RVC) ---

    pub fn convert_content_dim(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_content_dim(self.handle) as i32 }
    }

    pub fn convert_n_speakers(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_n_speakers(self.handle) as i32 }
    }

    pub fn convert_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_sample_rate(self.handle) as i32 }
    }

    pub fn convert_inter_channels(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_inter_channels(self.handle) as i32 }
    }

    pub fn convert_upsample_product(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_upsample_product(self.handle) as i32 }
    }

    /// Convert ContentVec features to mono PCM. Validates replay noise lengths using model dimension getters.
    pub fn convert(
        &self,
        content: &[f32],
        n_frames: i32,
        f0_hz: &[f32],
        speaker_id: i32,
        noise_zp: Option<&[f32]>,
        noise_sine: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if n_frames <= 0 {
            return Err("convert: n_frames must be positive".to_string());
        }
        let dim = self.convert_content_dim();
        if dim <= 0 {
            return Err(
                "convert: could not query content dim (is this a conversion model?)".to_string(),
            );
        }
        let dim_usize = dim as usize;
        let n_frames_usize = n_frames as usize;
        let exp_content = dim_usize
            .checked_mul(n_frames_usize)
            .ok_or("convert: content dimensions overflow".to_string())?;
        if content.len() != exp_content {
            return Err(format!(
                "convert: content length {} != n_frames {} * content_dim {}",
                content.len(),
                n_frames,
                dim
            ));
        }
        if f0_hz.len() != n_frames_usize {
            return Err(format!(
                "convert: f0 length {} != n_frames {}",
                f0_hz.len(),
                n_frames
            ));
        }
        let n_speakers = self.convert_n_speakers();
        if speaker_id < 0 || speaker_id >= n_speakers {
            return Err(format!(
                "convert: speaker_id {} out of range [0,{})",
                speaker_id, n_speakers
            ));
        }
        let inter_ch = self.convert_inter_channels();
        let upsample = self.convert_upsample_product();
        if inter_ch < 0 || upsample < 0 {
            return Err("convert: negative inter_channels or upsample_product".to_string());
        }
        if let Some(zp) = noise_zp {
            let exp_zp = (inter_ch as usize)
                .checked_mul(n_frames_usize)
                .ok_or("convert: noise_zp dimensions overflow".to_string())?;
            if zp.len() != exp_zp {
                return Err(format!(
                    "convert: noise_zp length {} != inter_channels {} * n_frames {}",
                    zp.len(),
                    inter_ch,
                    n_frames
                ));
            }
        }
        if let Some(sine) = noise_sine {
            let exp_sine = n_frames_usize
                .checked_mul(upsample as usize)
                .ok_or("convert: noise_sine dimensions overflow".to_string())?;
            if sine.len() != exp_sine {
                return Err(format!(
                    "convert: noise_sine length {} != n_frames {} * upsample_product {}",
                    sine.len(),
                    n_frames,
                    upsample
                ));
            }
        }
        let zp_ptr = noise_zp.map_or(std::ptr::null(), |v| v.as_ptr());
        let sine_ptr = noise_sine.map_or(std::ptr::null(), |v| v.as_ptr());
        let n_out = unsafe {
            crispasr_sys::crispasr_session_convert(
                self.handle,
                content.as_ptr(),
                n_frames as c_int,
                f0_hz.as_ptr(),
                speaker_id as c_int,
                zp_ptr,
                sine_ptr,
            )
        };
        if n_out <= 0 {
            return Err(format!("convert failed (rc={})", n_out));
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_convert_audio(self.handle, &mut out_n) };
        if out_n <= 0 {
            return Err(format!("convert: invalid audio length {}", out_n));
        }
        if ptr.is_null() {
            return Err("convert: audio pointer is null".to_string());
        }
        let audio = unsafe { std::slice::from_raw_parts(ptr, out_n as usize).to_vec() };
        Ok(audio)
    }

    // --- TTS extras ---

    pub fn set_tts_cfg_scale(&self, scale: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_tts_cfg_scale(self.handle, scale) };
        if rc == -2 {
            return Err(format!(
                "set_tts_cfg_scale not supported by backend (rc={})",
                rc
            ));
        }
        if rc != 0 {
            return Err(format!("set_tts_cfg_scale failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn tada_set_makeref_models(
        &self,
        encoder: Option<&str>,
        aligner: Option<&str>,
    ) -> Result<(), String> {
        let enc_c = encoder.map(|s| CString::new(s).unwrap());
        let ali_c = aligner.map(|s| CString::new(s).unwrap());
        let enc_ptr = enc_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ali_ptr = ali_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let rc = unsafe {
            crispasr_sys::crispasr_session_tada_set_makeref_models(self.handle, enc_ptr, ali_ptr)
        };
        if rc != 0 {
            return Err(format!("tada_set_makeref_models failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn disclaimer_text() -> String {
        let ptr = unsafe { crispasr_sys::crispasr_session_disclaimer_text() };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    pub fn get_disclaimer_pcm(&self) -> Result<Vec<f32>, String> {
        let mut n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_get_disclaimer_pcm(self.handle, &mut n) };
        if n <= 0 {
            // Ensure no leak if native returned allocated buffer with n<=0
            if !ptr.is_null() {
                unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
            }
            return Err("get_disclaimer_pcm: no audio (call before SetVoice?)".to_string());
        }
        if ptr.is_null() {
            return Err("get_disclaimer_pcm: null pointer with positive length".to_string());
        }
        let audio =
            unsafe { std::slice::from_raw_parts(ptr as *const c_float, n as usize).to_vec() };
        unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
        Ok(audio)
    }

    pub fn last_synth_error(&self) -> String {
        let ptr = unsafe { crispasr_sys::crispasr_session_last_synth_error(self.handle) };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Synchronous sentence streaming TTS. Fires cb(pcm, is_final) per sentence chunk.
    /// PCM is copied before callback returns; panics are caught and returned as errors.
    pub fn synthesize_streaming<F>(&mut self, text: &str, mut cb: F) -> Result<(), String>
    where
        F: FnMut(&[f32], bool),
    {
        let ctext = CString::new(text).map_err(|e| e.to_string())?;
        // Wrap callback to handle panic and copy
        struct Ctx<F> {
            cb: F,
            err: Option<String>,
        }
        unsafe extern "C" fn trampoline<F: FnMut(&[f32], bool)>(
            pcm: *const c_float,
            n: c_int,
            is_final: c_int,
            user_data: *mut std::os::raw::c_void,
        ) {
            if user_data.is_null() {
                return;
            }
            let ctx = &mut *(user_data as *mut Ctx<F>);
            if ctx.err.is_some() {
                return;
            }
            if pcm.is_null() || n <= 0 {
                ctx.err = Some("synthesize_streaming: empty chunk from native".to_string());
                return;
            }
            // Copy PCM before callback returns
            let src = std::slice::from_raw_parts(pcm, n as usize);
            let cp = src.to_vec();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (ctx.cb)(&cp, is_final != 0);
            }));
            if result.is_err() {
                ctx.err = Some("synthesize_streaming callback panicked".to_string());
            }
        }
        let mut ctx = Ctx { cb, err: None };
        let rc = unsafe {
            crispasr_sys::crispasr_session_synthesize_streaming(
                self.handle,
                ctext.as_ptr(),
                Some(trampoline::<F>),
                &mut ctx as *mut _ as *mut std::os::raw::c_void,
            )
        };
        if let Some(e) = ctx.err {
            return Err(e);
        }
        if rc != 0 {
            return Err(format!("synthesize_streaming failed (rc={})", rc));
        }
        Ok(())
    }

    // --- Audio helpers (standalone, not session-bound) ---

    // Session-bound audio rate helpers already exist as input_sample_rate etc.

    pub fn set_pcm_sample_rate(&self, rate: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_pcm_sample_rate(self.handle, rate as c_int)
        };
        if rc != 0 {
            return Err(format!("set_pcm_sample_rate failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_parakeet_att_context(&self, left: i32, right: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_parakeet_att_context(
                self.handle,
                left as c_int,
                right as c_int,
            )
        };
        if rc != 0 {
            return Err(format!("set_parakeet_att_context failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_audio_ctx(&self, audio_ctx: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_audio_ctx(self.handle, audio_ctx as c_int)
        };
        if rc != 0 {
            return Err(format!("set_audio_ctx failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_hikari_refresh_stride(&self, stride: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_hikari_refresh_stride(self.handle, stride as c_int)
        };
        if rc != 0 {
            return Err(format!("set_hikari_refresh_stride failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_grammar_strict(&self, strict: bool) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_grammar_strict(
                self.handle,
                if strict { 1 } else { 0 },
            )
        };
        if rc != 0 {
            return Err(format!("set_grammar_strict failed (rc={})", rc));
        }
        Ok(())
    }
}

// Standalone functions

// Native audio buffers are owned immediately, including partial error outputs.
struct OwnedNativeAudio(*mut c_float);
impl Drop for OwnedNativeAudio {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { crispasr_sys::crispasr_audio_free(self.0) }
        }
    }
}
fn copy_native_audio(p: *const c_float, n: c_int) -> Result<Vec<f32>, String> {
    if n < 0 || (n > 0 && p.is_null()) {
        return Err("invalid native PCM buffer".into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n as usize).to_vec() })
}
pub fn audio_load(path: &str, target_rate: Option<i32>) -> Result<(Vec<f32>, i32), String> {
    let path = CString::new(path).map_err(|e| e.to_string())?;
    if target_rate.is_some_and(|r| r <= 0) {
        return Err("positive target rate required".into());
    }
    let (mut p, mut n, mut rate) = (std::ptr::null_mut(), 0, 0);
    let rc = unsafe {
        if let Some(target) = target_rate {
            crispasr_sys::crispasr_audio_load_at_rate(
                path.as_ptr(),
                target,
                &mut p,
                &mut n,
                &mut rate,
            )
        } else {
            crispasr_sys::crispasr_audio_load(path.as_ptr(), &mut p, &mut n, &mut rate)
        }
    };
    let _owned = OwnedNativeAudio(p);
    if rc != 0 {
        return Err(format!("audio_load failed (rc={rc})"));
    }
    Ok((copy_native_audio(p, n)?, rate))
}
pub fn audio_load_stereo(path: &str) -> Result<(Vec<f32>, Vec<f32>, i32, i32), String> {
    let path = CString::new(path).map_err(|e| e.to_string())?;
    let (mut l, mut r, mut n, mut rate, mut channels) =
        (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0, 0);
    let rc = unsafe {
        crispasr_sys::crispasr_audio_load_stereo(
            path.as_ptr(),
            &mut l,
            &mut r,
            &mut n,
            &mut rate,
            &mut channels,
        )
    };
    let _left = OwnedNativeAudio(l);
    let _right = OwnedNativeAudio(if l == r { std::ptr::null_mut() } else { r });
    if rc != 0 {
        return Err(format!("audio_load_stereo failed (rc={rc})"));
    }
    Ok((
        copy_native_audio(l, n)?,
        copy_native_audio(r, n)?,
        rate,
        channels,
    ))
}
pub fn audio_resample(pcm: &[f32], source_rate: i32, target_rate: i32) -> Result<Vec<f32>, String> {
    if source_rate <= 0 || target_rate <= 0 {
        return Err("positive sample rates required".into());
    }
    if pcm.is_empty() {
        return Ok(Vec::new());
    }
    let count = c_int::try_from(pcm.len()).map_err(|_| "PCM exceeds native count range")?;
    let (mut p, mut n) = (std::ptr::null_mut(), 0);
    let rc = unsafe {
        crispasr_sys::crispasr_audio_resample(
            pcm.as_ptr(),
            count,
            source_rate,
            target_rate,
            &mut p,
            &mut n,
        )
    };
    let _owned = OwnedNativeAudio(p);
    if rc != 0 {
        return Err(format!("audio_resample failed (rc={rc})"));
    }
    copy_native_audio(p, n)
}

struct OwnedNativeContainer(*mut u8);
impl Drop for OwnedNativeContainer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { crispasr_sys::crispasr_c2pa_free(self.0) };
        }
    }
}
fn copy_native_container(ptr: *mut u8, count: usize) -> Result<Vec<u8>, String> {
    let owned = OwnedNativeContainer(ptr);
    if owned.0.is_null() || count == 0 || count > isize::MAX as usize {
        return Err("Invalid native container pointer or length".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(owned.0, count) }.to_vec())
}

pub fn pcm_to_wav(pcm: &[f32], sample_rate: i32) -> Result<Vec<u8>, String> {
    if pcm.is_empty() {
        return Err("pcm_to_wav: empty PCM".to_string());
    }
    if sample_rate <= 0 {
        return Err("pcm_to_wav: invalid sample_rate".to_string());
    }
    let mut out_len: usize = 0;
    let ptr = unsafe {
        crispasr_sys::crispasr_pcm_to_wav(
            pcm.as_ptr(),
            pcm.len() as c_int,
            sample_rate as c_int,
            &mut out_len,
        )
    };
    copy_native_container(ptr, out_len)
}

pub fn c2pa_sign(
    data: &[u8],
    format: &str,
    cert_path: Option<&str>,
    key_path: Option<&str>,
) -> Result<Vec<u8>, String> {
    if data.is_empty() {
        return Err("c2pa_sign: empty data".to_string());
    }
    let cformat = CString::new(format).map_err(|e| e.to_string())?;
    if cert_path.is_some() != key_path.is_some() || format.is_empty() {
        return Err("Certificate/key must be paired and format nonempty".into());
    }
    let ccert = cert_path
        .map(CString::new)
        .transpose()
        .map_err(|e| e.to_string())?;
    let ckey = key_path
        .map(CString::new)
        .transpose()
        .map_err(|e| e.to_string())?;
    let cert_ptr = ccert.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let key_ptr = ckey.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut out_len: usize = 0;
    let ptr = unsafe {
        crispasr_sys::crispasr_c2pa_sign(
            data.as_ptr(),
            data.len(),
            cformat.as_ptr(),
            cert_ptr,
            key_ptr,
            &mut out_len,
        )
    };
    copy_native_container(ptr, out_len)
}

pub fn watermark_load_model(gguf_path: &str) -> Result<(), String> {
    let cpath = CString::new(gguf_path).map_err(|e| e.to_string())?;
    let rc = unsafe { crispasr_sys::crispasr_watermark_load_model(cpath.as_ptr()) };
    if rc != 0 {
        return Err(format!("watermark_load_model failed (rc={})", rc));
    }
    Ok(())
}

pub fn free_aligner_cache() {
    unsafe { crispasr_sys::crispasr_free_aligner_cache() }
}

pub fn set_gpu_backend(name: Option<&str>) {
    let cname = name.map(|s| CString::new(s).unwrap());
    let ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    unsafe { crispasr_sys::crispasr_set_gpu_backend(ptr) }
}

pub fn detect_backends_from_gguf(path: &str) -> Result<Vec<String>, String> {
    let cpath = CString::new(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0i8; 512];
    let n = unsafe {
        crispasr_sys::crispasr_detect_backends_from_gguf(
            cpath.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        )
    };
    if n <= 0 {
        return Err(format!("detect_backends_from_gguf failed (rc={})", n));
    }
    let s = unsafe {
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    };
    Ok(s.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect())
}

pub fn c_api_version() -> String {
    let ptr = unsafe { crispasr_sys::crispasr_c_api_version() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

pub fn dart_helpers_version() -> String {
    let ptr = unsafe { crispasr_sys::crispasr_dart_helpers_version() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Direct TitaNet speaker embedding context; embeddings are 192-dimensional.
/// Requires external synchronization when moving it between threads.
pub struct TitaNet {
    handle: *mut c_void,
}
unsafe impl Send for TitaNet {}
impl TitaNet {
    pub fn new(model: &str, threads: i32) -> Result<Self, String> {
        let model = CString::new(model).map_err(|e| e.to_string())?;
        let handle = unsafe { crispasr_sys::crispasr_titanet_init(model.as_ptr(), threads) };
        if handle.is_null() {
            return Err("TitaNet model could not open".into());
        }
        Ok(Self { handle })
    }
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>, String> {
        if pcm.is_empty() {
            return Err("Nonempty PCM required".into());
        }
        let n = c_int::try_from(pcm.len()).map_err(|_| "PCM exceeds native count range")?;
        let mut output = vec![0.0; 192];
        let dim = unsafe {
            crispasr_sys::crispasr_titanet_embed(self.handle, pcm.as_ptr(), n, output.as_mut_ptr())
        };
        if dim != 192 {
            return Err(format!("TitaNet must return 192 floats (got {dim})"));
        }
        Ok(output)
    }
    /// Dot product for embeddings normalized to unit L2 length by TitaNet.
    pub fn cosine_sim(a: &[f32], b: &[f32]) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err("Embedding dimensions must match".into());
        }
        let dim =
            c_int::try_from(a.len()).map_err(|_| "Embedding dimension exceeds native range")?;
        Ok(unsafe { crispasr_sys::crispasr_titanet_cosine_sim(a.as_ptr(), b.as_ptr(), dim) })
    }
}
impl Drop for TitaNet {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_titanet_free(self.handle) };
    }
}

/// Explicit native v2 load options. Defaults match the library's GPU/flash defaults.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub n_threads: i32,
    pub use_gpu: bool,
    pub verbosity: i32,
    pub flash_attention: bool,
    /// -1 uses the maximum; zero keeps LLM layers on CPU.
    pub gpu_layers: i32,
}
impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            n_threads: 4,
            use_gpu: true,
            verbosity: 0,
            flash_attention: true,
            gpu_layers: -1,
        }
    }
}
#[repr(C)]
struct NativeSessionOptions {
    abi_version: c_int,
    n_threads: c_int,
    use_gpu: c_int,
    verbosity: c_int,
    flash_attn: c_int,
    n_gpu_layers: c_int,
    reserved: [c_int; 6],
}
impl Session {
    pub fn open_with_options(
        model: &str,
        backend: Option<&str>,
        options: &SessionOptions,
    ) -> Result<Self, String> {
        if options.n_threads < 0 || options.verbosity < 0 || options.gpu_layers < -1 {
            return Err("Invalid session load options".into());
        }
        let threads = if options.n_threads == 0 {
            4
        } else {
            options.n_threads
        };
        let params = NativeSessionOptions {
            abi_version: 2,
            n_threads: threads,
            use_gpu: options.use_gpu.into(),
            verbosity: options.verbosity,
            flash_attn: options.flash_attention.into(),
            n_gpu_layers: options.gpu_layers,
            reserved: [0; 6],
        };
        let path = CString::new(model).map_err(|e| e.to_string())?;
        let backend = backend
            .map(CString::new)
            .transpose()
            .map_err(|e| e.to_string())?;
        let handle = unsafe {
            crispasr_sys::crispasr_session_open_with_params(
                path.as_ptr(),
                backend.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                &params as *const _ as *const c_void,
            )
        };
        if handle.is_null() {
            return Err("session_open_with_params failed".into());
        }
        Ok(Self::from_native(handle, threads))
    }
    /// Teacher-forced candidate scores; requires a supporting backend.
    pub fn score_texts(
        &self,
        pcm: &[f32],
        language: Option<&str>,
        prompt: Option<&str>,
        texts: &[&str],
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        if pcm.is_empty() || texts.is_empty() {
            return Err("Nonempty PCM and candidates required".into());
        }
        let samples = c_int::try_from(pcm.len()).map_err(|_| "PCM exceeds native range")?;
        let count =
            c_int::try_from(texts.len()).map_err(|_| "Candidate count exceeds native range")?;
        let language = language
            .map(CString::new)
            .transpose()
            .map_err(|e| e.to_string())?;
        let prompt = prompt
            .map(CString::new)
            .transpose()
            .map_err(|e| e.to_string())?;
        let strings: Vec<CString> = texts
            .iter()
            .map(|s| CString::new(*s))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let pointers: Vec<_> = strings.iter().map(|s| s.as_ptr()).collect();
        let mut scores = vec![0.0; texts.len()];
        let mut counts = vec![0; texts.len()];
        let rc = unsafe {
            crispasr_sys::crispasr_session_score_texts(
                self.handle,
                pcm.as_ptr(),
                samples,
                language.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                prompt.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                pointers.as_ptr(),
                count,
                scores.as_mut_ptr(),
                counts.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("score_texts failed (rc={rc})"));
        }
        Ok((scores, counts))
    }
}

// Each model owns a context; only &mut processing is allowed, and Drop frees it.
macro_rules! standalone_text_model {
    ($name:ident, $init:path, $process:path, $free_text:path, $free:path) => {
        pub struct $name {
            handle: *mut c_void,
        }
        unsafe impl Send for $name {}
        impl $name {
            pub fn open(model: &str) -> Result<Self, String> {
                let model = CString::new(model).map_err(|e| e.to_string())?;
                let handle = unsafe { $init(model.as_ptr()) };
                if handle.is_null() {
                    return Err("Text processor could not open".into());
                }
                Ok(Self { handle })
            }
            /// Copy allocated UTF-8 before releasing it; null is a failure, empty is valid.
            pub fn process(&mut self, text: &str) -> Result<String, String> {
                let text = CString::new(text).map_err(|e| e.to_string())?;
                let output = unsafe { $process(self.handle, text.as_ptr()) };
                if output.is_null() {
                    return Err("Text processing failed".into());
                }
                let owned = unsafe { CStr::from_ptr(output) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { $free_text(output) };
                Ok(owned)
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { $free(self.handle) };
            }
        }
    };
}
standalone_text_model!(
    TruecaseModel,
    crispasr_sys::crispasr_truecase_init,
    crispasr_sys::crispasr_truecase_process,
    crispasr_sys::crispasr_truecase_free_text,
    crispasr_sys::crispasr_truecase_free
);
standalone_text_model!(
    PCSModel,
    crispasr_sys::crispasr_pcs_init,
    crispasr_sys::crispasr_pcs_process,
    crispasr_sys::crispasr_pcs_free_text,
    crispasr_sys::crispasr_pcs_free
);

/// Global polling selects the session which most recently used default streaming callbacks.
pub fn get_progress() -> i32 {
    unsafe { crispasr_sys::crispasr_get_progress() }
}
pub fn reset_progress() {
    unsafe { crispasr_sys::crispasr_reset_progress() };
}
pub fn get_streamed_segment_count() -> i32 {
    unsafe { crispasr_sys::crispasr_get_streamed_segment_count() }
}
pub fn reset_streamed_segments() {
    unsafe { crispasr_sys::crispasr_reset_streamed_segments() };
}
pub fn get_streamed_token_count() -> i32 {
    unsafe { crispasr_sys::crispasr_get_streamed_token_count() }
}
pub fn reset_streamed_tokens() {
    unsafe { crispasr_sys::crispasr_reset_streamed_tokens() };
}
/// Drain an owned result, preserving the same metadata as ordinary transcription.
pub fn drain_streamed_segments() -> Result<Vec<crate::SessionSegment>, String> {
    let result = unsafe { crispasr_sys::crispasr_drain_streamed_segments() };
    if result.is_null() {
        return Ok(Vec::new());
    }
    Session::copy_owned_session_result(result).map(|(segments, _)| segments)
}
/// Copy NUL-separated raw token fragments from the native borrowed polling buffer.
///
/// # Safety
/// Serialize this call with every native drain, inference and close for the
/// globally selected session. Another thread must not invalidate its borrowed
/// buffer between the native return and completion of this copy.
pub unsafe fn drain_streamed_token_bytes() -> Result<Vec<Vec<u8>>, String> {
    let mut count = 0;
    let mut ptr = unsafe { crispasr_sys::crispasr_drain_streamed_tokens(&mut count) };
    if count < 0 || (count > 0 && ptr.is_null()) {
        return Err("Invalid native token buffer".into());
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let token = unsafe { CStr::from_ptr(ptr) }.to_bytes();
        out.push(token.to_vec());
        ptr = unsafe { ptr.add(token.len() + 1) };
    }
    Ok(out)
}
/// Display text from polled tokens; use raw bytes to concatenate split UTF-8 first.
///
/// # Safety
/// The synchronization requirements of [`drain_streamed_token_bytes`] apply.
pub unsafe fn drain_streamed_tokens() -> Result<Vec<String>, String> {
    unsafe { drain_streamed_token_bytes() }.map(|tokens| {
        tokens
            .into_iter()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .collect()
    })
}
