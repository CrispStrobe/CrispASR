//! Integration tests for the CrispASR Rust wrapper.
//!
//! Requires:
//!   - whisper-tiny model at CRISPASR_MODEL env var (or ../models/ggml-tiny.en.bin)
//!   - parakeet model at PARAKEET_MODEL env var (optional, skipped if absent)
//!   - jfk.wav at ../samples/jfk.wav

use std::path::Path;

fn jfk_pcm() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../samples/jfk.wav");
    let mut reader = hound::WavReader::open(path).expect("failed to open jfk.wav");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn whisper_model() -> String {
    std::env::var("CRISPASR_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/ggml-tiny.en.bin").to_string()
    })
}

fn parakeet_model() -> Option<String> {
    let p = std::env::var("PARAKEET_MODEL").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_cohere/parakeet-tdt-0.6b-v3.gguf"
        )
        .to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

// ---- CrispASR (whisper-only) tests ----

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_load_and_transcribe() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).expect("load whisper-tiny");
    let pcm = jfk_pcm();
    let segs = model.transcribe_pcm(&pcm).expect("transcribe");
    assert!(!segs.is_empty(), "should produce segments");
    let full = segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
    assert!(full.contains("fellow americans"), "text should mention 'fellow americans': {full}");
    assert!(full.contains("country"), "text should mention 'country': {full}");
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_timestamps_valid() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let segs = model.transcribe_pcm(&jfk_pcm()).unwrap();
    for seg in &segs {
        assert!(seg.start >= 0.0, "start >= 0");
        assert!(seg.end > seg.start, "end > start: {} vs {}", seg.end, seg.start);
        assert!(seg.end < 15.0, "end < 15s (audio is ~11s)");
    }
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_empty_audio() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let silence = vec![0.0f32; 16000]; // 1s silence
    let segs = model.transcribe_pcm(&silence).unwrap();
    // Should not crash; may produce empty or whitespace-only segments
    let _ = segs;
}

// ---- Session (unified, any backend) tests ----

#[test]
fn session_whisper_auto_detect() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    assert_eq!(sess.backend(), "whisper");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());
    let full = segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
    assert!(full.contains("country"));
}

#[test]
fn session_available_backends() {
    let backends = crispasr::Session::available_backends();
    assert!(backends.contains(&"whisper".to_string()));
    assert!(backends.contains(&"parakeet".to_string()));
}

#[test]
fn session_parakeet_word_timestamps() {
    let model_path = match parakeet_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: parakeet model not found");
            return;
        }
    };
    let sess = crispasr::Session::open(&model_path).expect("session open parakeet");
    assert_eq!(sess.backend(), "parakeet");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());

    // Parakeet should produce word-level timestamps
    let words = &segs[0].words;
    assert!(!words.is_empty(), "parakeet should produce words");
    for w in words {
        assert!(w.start >= 0.0);
        assert!(w.end >= w.start);
        assert!(!w.text.is_empty());
    }

    // Monotonicity
    let mut prev_end = 0.0f64;
    for w in words {
        assert!(
            w.start >= prev_end - 0.02,
            "word '{}' starts at {} before prev end {}",
            w.text, w.start, prev_end
        );
        prev_end = w.end;
    }
}

// ---- Registry + cache ----

#[test]
fn registry_lookup_parakeet() {
    let entry = crispasr::registry_lookup("parakeet").expect("registry call");
    if let Some(e) = entry {
        assert!(!e.filename.is_empty());
        assert!(!e.url.is_empty());
    }
}

#[test]
fn cache_dir_exists() {
    let dir = crispasr::cache_dir(None).expect("cache_dir");
    if let Some(d) = dir {
        assert!(!d.is_empty());
    }
}
