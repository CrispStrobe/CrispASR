// whisper_params.h — command-line parameter struct shared between the
// historical whisper CLI surface and the CrispASR backend dispatch layer.
//
// Keep the `whisper_params` name for CLI/source compatibility. This is a
// frontend params struct, not a signal that the whole project is still named
// whisper.

#pragma once

#include "crispasr.h"
#include "grammar-parser.h"

#include <algorithm>
#include <cfloat>
#include <climits> // INT_MIN (att_context_* sentinels) — issue #257
#include <string>
#include <thread>
#include <vector>

struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    bool split_on_punct = false;
    int32_t best_of = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size = -1; // -1 = greedy; beam search only when explicitly set via -bs N
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float no_speech_thold = 0.6f;
    float grammar_penalty = 100.0f;
    bool grammar_strict = false;
    float temperature = 0.0f;
    float temperature_inc = 0.2f;
    uint64_t seed = 0; // RNG seed for sampling (0 = non-deterministic)

    bool debug_mode = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool output_jsn = false;
    bool output_jsn_full = false;
    bool output_lrc = false;
    bool no_prints = false;
    bool verbose = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_confidence = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool log_score = false;
    bool use_gpu = true;
    bool flash_attn = true;
    int32_t gpu_device = 0;
    std::string gpu_backend;
    bool suppress_nst = false;
    bool carry_initial_prompt = false;

    std::string language = "auto";
    std::string prompt;
    std::string ask;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "auto";
    std::string model_quant;
    std::string grammar;
    std::string grammar_rule;

    std::string tdrz_speaker_turn = " [SPEAKER_TURN]";
    std::string suppress_regex;
    std::string openvino_encode_device = "CPU";
    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;

    bool vad = false;
    std::string vad_model = "";
    float vad_threshold = 0.5f;
    // Issue #83 follow-up: tracks whether the user passed `-vt` /
    // `--vad-threshold` explicitly. Backends with non-Silero probability
    // calibration (whisper-vad-encdec) auto-lower the threshold when this
    // is false so the out-of-the-box default doesn't drop most speech.
    bool vad_threshold_explicit = false;
    int vad_min_speech_duration_ms = 250;
    int vad_min_silence_duration_ms = 100;
    float vad_max_speech_duration_s = FLT_MAX;
    int vad_speech_pad_ms = 30;
    float vad_samples_overlap = 0.1f;
    bool vad_stitch = false;
    // #227: a --vad-import file whose chunk length differs from this run WARNS
    // by default and is used anyway -- refusing would break existing scripts
    // and clients that upgraded into the check. Opt in to a hard failure with
    // --vad-import-strict (CLI) or the vad_import_strict form field (server);
    // both surfaces share this field and must stay in agreement.
    bool vad_import_strict = false;
    // #227: --vad-export-raw exports RAW VAD speech segments instead of chunk
    // boundaries. These are model- AND chunk-length-independent, so one export
    // can be imported by any run and is re-chunked per run. Non-breaking:
    // --vad-export keeps its chunk-boundary behaviour.
    bool vad_export_raw = false;
    // Issue #227: reuse VAD segment boundaries across ASR runs. --vad-export
    // writes the computed slices to a JSON file; --vad-import reads slices from
    // one instead of running VAD (skips the VAD model entirely).
    std::string vad_export_file = "";
    std::string vad_import_file = "";
    // Server-side (#227) equivalents. The HTTP API can't take file paths — that
    // would be an arbitrary read/write on the server host — so boundaries are
    // returned inline in the response (`vad_export=true`) and supplied inline as
    // the serialized JSON (`vad_import=<json>`). Same wire format as the CLI's
    // files; both use crispasr_{serialize,parse}_vad_slices.
    bool vad_export_inline = false;
    std::string vad_import_json = "";

    // §248 source separation. --separate routes to the separation dispatcher
    // (crispasr_separate_cli) BEFORE any transcribe backend is built; the model
    // arch (mel-band-roformer / htdemucs) is auto-detected from the GGUF.
    bool separate = false;
    std::string stems;          // comma-separated stem subset; empty/"all" = every stem
    std::string sep_output_dir; // output dir for stems; empty = alongside input

    // Pitch (F0) estimation. --pitch routes to the pitch dispatcher
    // (crispasr_pitch_cli) BEFORE any transcribe backend is built; the model
    // arch (crepe) is auto-detected from the GGUF.
    // Chord recognition. --chords routes to the chord dispatcher
    // (crispasr_chords_cli) BEFORE any transcribe backend is built, mirroring
    // --pitch. The BTC weights are CC-BY-NC-SA, so this path needs
    // --accept-license cc-by-nc-sa-4.0 to auto-download.
    bool chords = false;
    std::string chords_format; // "text" (default) or "json"

    // Piano transcription. --piano routes to the piano dispatcher
    // (crispasr_piano_cli) BEFORE any transcribe backend is built, for the same
    // reason as --chords: the output is note EVENTS (onset/offset/midi/
    // velocity), not text segments. Reaching it through transcribe() renders
    // each note as text like "C4 v=80", which is lossy.
    bool piano = false;
    std::string piano_format; // "text" (default) or "json"

    // Beat / downbeat tracking. --beats routes to the beat dispatcher
    // (crispasr_beats_cli) BEFORE any transcribe backend is built, for the
    // same reason as --chords: the output is a beat grid, not text segments.
    bool beats = false;
    std::string beats_format; // "text" (default) or "json"

    // Guitar tablature. --tab routes to the tab dispatcher (crispasr_tab_cli)
    // BEFORE any transcribe backend is built, for the same reason as --chords:
    // the output is a per-frame per-string grid of fret SCORES, not text
    // segments. The CLI display argmaxes it; real consumers take the
    // log-probabilities through the C ABI and run their own constrained decoder.
    bool tab = false;
    std::string tab_format; // "text" (default) or "json"

    bool pitch = false;
    std::string pitch_format;   // "text" (default) or "json"
    float pitch_hop_ms = 10.0f; // CREPE reference hop

    std::string backend;
    std::string source_lang;
    std::string target_lang;
    bool punctuation = true;
    std::string punc_model;
    std::string truecase_model;
    int flush_after = 0;
    bool show_alternatives = false;
    int32_t n_alternatives = 3;
    bool return_logits = false;
    std::string aligner_model;
    // PLAN issue #62: when true, the CTC forced aligner runs even on
    // backends that already produce native timestamps — replacing
    // their words rather than skipping. Default false keeps the existing
    // semantics ("aligner only if the native path didn't produce
    // words"). Useful when the user trusts the aligner's accuracy more
    // than the backend's native timing (whisper, parakeet, canary,
    // cohere, kyutai-stt).
    bool force_aligner = false;
    // SubtitleEdit #10775: opt out of the canary auto-aligner default.
    // For --backend canary, when the user requests word-level output
    // (srt/vtt/json-full/wts/max-len/split-on-punct/print-colors) and
    // hasn't passed --aligner-model explicitly, crispasr now defaults
    // to `-am auto --force-aligner` because canary's native cross-attn
    // DTW timing is ~5× looser than canary-ctc-aligner (~414 ms vs
    // ~78 ms MAE on word boundaries — see canary-ctc-aligner-GGUF
    // README). `--no-auto-aligner` reverts to the pre-default native
    // DTW path (no second forward pass, no ~442 MB download).
    bool no_auto_aligner = false;
    // Issue #311: machine-reliable strict failure semantics for explicitly
    // requested auxiliary pipeline stages. By default CrispASR degrades
    // gracefully (a VAD/aligner/punc model that fails to load is skipped with
    // a stderr warning and a 0 exit code). Integrations that treat these stages
    // as *required task properties* need a zero exit to actually mean "every
    // requested stage succeeded". These opt-in flags make a failed-to-load /
    // failed-to-produce required stage return a NON-ZERO exit instead.
    //
    //  * `strict_pipeline` (--strict-pipeline): require every stage that was
    //    explicitly requested on this command line (VAD if --vad/-vm, word
    //    timestamps if -am/--force-aligner, punctuation if --punc-model).
    //  * the per-stage flags force one specific requirement regardless.
    //
    // A stage that RAN successfully but legitimately produced nothing (VAD ran
    // and detected no speech) is NOT a failure. Direct local model paths are
    // used as-is and never fall back to auto-download.
    bool strict_pipeline = false;
    bool require_vad = false;
    bool require_word_timestamps = false;
    bool require_punctuation = false;
    int32_t max_new_tokens = 512;
    // Whether the user passed --max-new-tokens. Backends whose sensible default
    // differs from 512 (e.g. moss-diarize wants 1024) forward the CLI value only
    // when this is set, so an unset default never SHRINKS a backend below its own
    // (#292). Mirrors chunk_seconds_explicit.
    bool max_new_tokens_explicit = false;
    float frequency_penalty = 0.0f;
    int32_t chunk_seconds = 30;
    bool chunk_seconds_explicit = false; // true when user passed --chunk-seconds
    float chunk_overlap_seconds = 3.0f;  // overlap context on each side of chunk boundary
    // Issue #89 / #114 follow-up: NeMo-style LCS hypothesis stitching at
    // chunk boundaries. "auto" fires when chunking with overlap is active,
    // matching upstream BatchedFrameASRTDT. "on" forces it (e.g. for
    // bindings testing); "off" disables for A/B comparison or debugging.
    std::string lcs_dedup = "auto"; // {auto|on|off}
    // Lower bound on LCS length to act on. NeMo's default is 1; raise to
    // 3-4 if your audio has long-silence regions where blank tokens
    // dominate the boundary token run (avoids over-slicing).
    int lcs_min_length = 1;
    bool warmup = false;    // run a short dummy transcribe after init to amortize first-call overhead (PLAN #80e)
    bool no_warmup = false; // --no-warmup: skip the always-on server warmup (e.g. crashes on some Vulkan drivers, #165)
    std::string parakeet_decoder; // "tdt" (default), "ctc" — selects parakeet decode head
    // Issue #257: parakeet/canary local-attention window (encoder frames, 1 = ~80 ms)
    // — NeMo change_attention_model("rel_pos_local_attn", [L,R]); bounds long-audio
    // encoder memory to O(T·window). INT_MIN = unset (use the model default);
    // negative = full attention. Set via --att-context "L,R".
    int att_context_left = INT_MIN;
    int att_context_right = INT_MIN;
    std::string hotwords;        // comma-separated hotword list (PLAN #98)
    float hotwords_boost = 2.0f; // per-token log-prob boost for hotword prefix matches
    // Free-form hotword/context text injected into the vibevoice-asr prompt
    // (only backend that reads this so far). Matches the `context_info` param
    // in microsoft/VibeVoice's vibevoice_asr_processor.py.
    std::string context;
    // #205: granite-speech incremental decoding — seed the assistant turn with a
    // previously-decoded transcript so the model continues from it instead of
    // re-decoding (model-card `prefix_text`). Output is the continuation only.
    std::string prefix_text;
    std::string lid_backend;
    std::string lid_model;
    // Post-ASR text LID: when set, after transcription completes, run
    // a fastText-supervised classifier (GlotLID / LID-176) on the
    // assembled transcript and emit `lang=<code>\tconf=<score>` to
    // stderr. Path points at a lid_fasttext.* GGUF.
    std::string lid_on_transcript;
    std::string diarize_method;
    std::string sherpa_bin;
    std::string sherpa_segment_model;
    std::string sherpa_embedding_model;
    int sherpa_num_clusters = 0;

    // Speaker identification via TitaNet embeddings + profile DB.
    std::string speaker_db;         // path to speaker profile directory
    std::string enroll_speaker;     // enrollment mode: save embedding as this name
    std::string titanet_model;      // TitaNet GGUF path or "auto"
    float speaker_threshold = 0.7f; // cosine similarity threshold for matching
    // Closed roster for identification (issue #266): comma-separated names
    // of enrolled participants the deployer asserts are present in THIS
    // recording. Matching runs per global speaker cluster and only against
    // these profiles; there is no open "identify anyone in the db" mode
    // (that would be remote biometric identification under the EU AI Act,
    // Annex III 1(a)). REQUIRED for --speaker-db matching.
    std::string expect_speakers;
    // The named-profile DB (--enroll-speaker / --speaker-db) persists
    // voiceprints linked to real names — biometric special-category data
    // under GDPR Art. 9. It is OFF by default: the deployer must affirm a
    // lawful basis + explicit consent via --speaker-db-consent before
    // enrollment or matching will run.
    bool speaker_db_consent = false;

    // Embedding-based diarization clustering (issue #107 P3). When set,
    // after --diarize-method pyannote runs, each speech segment is
    // embedded by this model and clustered on cosine similarity to
    // produce globally stable speaker IDs (independent of pyannote's
    // local track indices). Path, "auto", or empty for no clustering.
    // Currently dispatches to the TitaNet adapter; pluggable.
    std::string diarize_embedder;           // model path or "auto"
    float diarize_cluster_threshold = 0.5f; // cosine merge threshold
    int diarize_max_speakers = 8;           // upper bound for cluster count
    int diarize_num_speakers = 0;           // >0 pins the count (foxnose)
    // Set when the user passed --diarize-max-speakers explicitly, so a
    // per-method default never SHRINKS or GROWS a value they chose. Same
    // contract as max_new_tokens_explicit (#292).
    bool diarize_max_speakers_explicit = false;
    // Set when the user passed --diarize-cluster-threshold explicitly. The
    // embedder path otherwise estimates the speaker count with spectral
    // clustering (#326) and the cosine threshold is not consulted at all;
    // honouring it only when asked for keeps that knob working for anyone who
    // tuned it, without letting its default decide the answer.
    bool diarize_cluster_threshold_explicit = false;

    // Set by the unified runner: foxnose diarization happens in one global
    // pass after transcription, so the per-slice path must stand down.
    bool diarize_foxnose_global = false;

    // #324: `--diarize-method foxnose` consumes --diarize-embedder itself, as
    // the WeSpeaker model for its own spectral clustering. The generic
    // TitaNet remap must therefore NOT also try to load that path — it is a
    // different architecture and fails with a confusing
    // "block_repeats/kernels array size mismatch". There are three call sites
    // (cli.cpp, crispasr_run.cpp, crispasr_server.cpp); they all ask here so a
    // fourth cannot silently drift.
    bool diarize_embedder_is_foxnose() const {
        return diarize_method == "foxnose" || diarize_method == "foxnose-diarize";
    }
    // #466: NVIDIA Nemotron-3-Diarization (streaming Sortformer). Like foxnose
    // it diarizes in one global pass (speakers are numbered by first arrival,
    // so per-slice runs would renumber them). Model from --diarize-model.
    bool diarize_is_sortformer() const {
        return diarize_method == "sortformer" || diarize_method == "nemotron3-diar" || diarize_method == "nemotron3";
    }
    // Methods that label speakers in one pass over the whole recording.
    bool diarize_is_global_method() const { return diarize_embedder_is_foxnose() || diarize_is_sortformer(); }
    // The legacy whisper path labels speakers in a post-step after decoding
    // (crispasr_apply_diarize). Only the stereo methods can be estimated live
    // in the per-segment print callback; for every other method the live line
    // would carry whisper's stereo-energy guess, "(speaker ?)" on mono input.
    bool diarize_labels_after_decode() const {
        return diarize && !diarize_method.empty() && diarize_method != "energy" && diarize_method != "xcorr";
    }
    // GGUF for --diarize-method sortformer ("auto" / empty: NVIDIA's q8_0 GGUF).
    std::string diarize_model;
    // #466: sortformer chunk schedule, "offline" (default) or a streaming preset.
    std::string sortformer_mode;
    bool stream = false;
    bool mic = false;
    bool stream_continuous = false;
    bool stream_monitor = false;
    bool server = false;
    std::string server_host = "127.0.0.1";
    int32_t server_port = 8080;
    std::string server_api_keys;
    // --server-workers N (>1): load N independent backend instances so
    // "pure-ASR" requests (explicit language, no aligner, no punctuation/
    // truecaser) run concurrently instead of serializing on one model. Costs
    // N× model memory. 1 = single shared instance (default, unchanged). The
    // CRISPASR_SERVER_WORKERS env var, when set, overrides this. See
    // docs/concurrency.md.
    int32_t server_workers = 1;
    // --ws-port: real-time WebSocket ASR streaming on a second port.
    //   -1 = disabled (default), 0 = server_port + 1, N = port N.
    int32_t server_ws_port = -1;
    // --wyoming-port: Wyoming protocol (Home Assistant Assist) TCP server.
    //   -1 = disabled (default), N = port N.
    int32_t wyoming_port = -1;
    int32_t stream_step_ms = 3000;
    int32_t stream_length_ms = 10000;
    int32_t stream_keep_ms = 200;
    // Issue #84: opt-in JSON-Lines streaming output for wrappers
    // (browser bridges, translators) that need machine-readable
    // partial/final/silence events instead of plain stdout text.
    // When true, the streaming loop emits one JSON object per event
    // on stdout: {"type":"partial"|"final"|"silence", "utterance_id":N,
    // "text":"…", "t0":sec, "t1":sec}. Plain-text stdout is suppressed.
    bool stream_json = false;
    // Trailing-silence threshold (ms) for promoting the open partial
    // to final and incrementing utterance_id. 0 = never auto-finalize
    // (only an open partial is emitted). Default 800 ms is a sensible
    // pause length for live captions / translation handoff and matches
    // the example in the issue. Only relevant with --stream-json.
    int32_t stream_final_silence_ms = 800;
    // JSON streaming + VAD only: merge adjacent VAD slices across tiny
    // detector jitter gaps, but never across --stream-final-on-silence-ms.
    // 0 disables streaming-specific post-merge.
    int32_t stream_vad_merge_gap_ms = 250;
    // JSON streaming + VAD only: minimum interval between live partial ASR
    // decodes. 0 = decode partials every --stream-step, preserving the
    // previous behavior. VAD timing/finalization still runs every step.
    int32_t stream_partial_decode_ms = 0;
    // JSON streaming + VAD only (#404): bound the audio each live partial
    // decode covers to the last N seconds of the open utterance. Text already
    // decoded ahead of the moving anchor is kept as a committed prefix, so
    // partial.text still covers the whole utterance; `final.text` is
    // unaffected (redecode mode re-decodes the full utterance regardless).
    // Cuts land on the quietest 100 ms (energy-min, same policy as the
    // long-audio chunker). 0 = off (decode the full open slice, previous
    // behavior). Minimum useful value ~4 s: values below the 2 s encoder
    // floor + slack behave like 4.
    int32_t stream_partial_tail_sec = 0;
    // JSON streaming + VAD only: control FireRedPunc placement when
    // --punc-model is loaded. "final" avoids the high-frequency partial
    // punc path; "partial" preserves the older partial+final behavior.
    std::string stream_punc = "final"; // off|final|partial
    // Issue #84 round 2 (CKwasd retest): how to compute `final.text`
    // when an utterance closes. The round-1 design just echoed the
    // last rolling-window partial — wrong because the rolling window
    // evicts old audio, so for utterances longer than `--stream-length`
    // the tail of the latest partial does not cover the full
    // utterance. Two modes:
    //   "redecode" (default) — buffer the utterance's PCM (capped at
    //     stream_utterance_max_sec) and run one extra `transcribe()`
    //     on the whole buffer at finalize time. Best quality, costs
    //     one encoder pass per utterance + ~4 MB / utterance memory.
    //   "prefix" — keep the round-1 cost (no extra transcribe) but
    //     accumulate stable text via longest-common-prefix matching
    //     across consecutive partials. Lower quality for utterances
    //     longer than the rolling window, useful when re-decoding
    //     would blow latency or compute budget. The latest partial
    //     beyond the committed prefix is concatenated at finalize.
    std::string stream_final_mode = "redecode";
    // Hard cap on how much PCM we buffer per utterance in redecode
    // mode. Force-finalize when exceeded so memory stays bounded on
    // monologues. 60 s × 16 kHz × 4 bytes ≈ 3.84 MB.
    int32_t stream_utterance_max_sec = 60;
    // Issue #84: gate the noisy `firered_vad: N frames, max_prob=…`
    // and `fbank[0,:3]=…` stderr dumps in src/firered_vad.cpp behind
    // an explicit opt-in. Set by --firered-vad-debug (or the
    // CRISPASR_FIRERED_VAD_DEBUG env var directly).
    bool firered_vad_debug = false;
    bool auto_download = false;
    // Attests acceptance of a RESTRICTED model licence (cc-by-nc-*, gemma,
    // llama*, lfm1.0, other): the exact SPDX-ish tag, or "all". Separate from
    // --i-have-rights, which attests SPEAKER CONSENT for voice cloning — one
    // flag must not grant two unrelated permissions.
    std::string accept_license;
    bool dry_run_resolve = false;
    bool dry_run_ignore_cache = false;
    std::string cache_dir;
    // Issue #128 — llama-server-style convenience flags. `--hf-repo
    // OWNER/REPO[:FILE]` synthesises the HF resolve URL and fetches the
    // model into the auto-download cache, then runs the rest of the
    // pipeline as if the user had passed `-m <cached-path>`. `--hf-file
    // FILE` overrides the FILE part when the user prefers two flags.
    // Empty = not requested; the existing -m / --auto-download paths
    // run unchanged.
    std::string hf_repo;
    std::string hf_file;
    std::string tts_text;
    std::string tts_output;
    bool s2s = false;       // speech-to-speech mode: audio in → audio out
    std::string s2s_output; // S2S output WAV path (default: s2s_output.wav)
    std::string tts_voice;
    int tts_steps = 20;
    std::string tts_codec_model;
    std::string tts_codec_quant;
    std::string tts_ref_text;
    std::string tts_ref_asr;  // ASR backend for auto-transcribing ref audio (default: whisper)
    std::string tts_instruct; // VoiceDesign: natural-language voice description

    // #316: bypass the G2P and feed these phonemes to the TTS backend verbatim.
    // The phoneme string is the boundary between text processing and the
    // acoustic model, so being able to drive it directly is what lets you tell a
    // G2P bug from a model bug — and it is how you reproduce another
    // implementation's pronunciation exactly. Honoured by kokoro; backends
    // without a phonemes-in entry point report it rather than silently
    // synthesizing the text.
    // CLI: --tts-phonemes "<IPA>"
    std::string tts_phonemes;
    bool tts_trim_silence = false;
    int tts_pad_silence_ms = 0;

    // --make-ref: create a TADA voice reference GGUF from --voice <audio.wav>
    // + --ref-text "transcript". Requires tada-encoder.gguf + tada-aligner-*.gguf.
    bool make_ref = false;
    std::string make_ref_output;  // output path (default: tada-ref-custom.gguf)
    std::string make_ref_aligner; // aligner GGUF path (auto-discovered if empty)
    std::string make_ref_encoder; // encoder GGUF path (auto-discovered if empty)

    // Forced alignment / word timestamps via the TADA aligner (reuses the
    // make-ref encoder+aligner). Input: --voice <audio> + --ref-text "<text>".
    bool align = false;
    std::string align_output;         // output path (default: stdout)
    std::string align_format = "srt"; // srt | json | plain

    // Issue #217: standalone CTC forced alignment mode.
    // Input: audio file + text (--ref-text or --text-file).
    // Runs the CTC aligner (canary-ctc / wav2vec2 / qwen3-fa) without
    // requiring an ASR model or transcription step.
    bool align_only = false;
    std::string text_file; // path to .txt or .srt (text extracted, timestamps stripped)
    // Output granularity: "word" = one entry per aligned word; "segment" =
    // one entry per input SRT cue (or per non-empty .txt line), re-timed
    // from the word alignment; "auto" = segment for .srt input, word otherwise.
    std::string align_granularity = "auto"; // auto | word | segment

    // AudioSeal neural watermark model (optional upgrade from spread-spectrum).
    // When set, loads the GGUF and uses it for watermark embed/detect
    // instead of the built-in spread-spectrum watermark.
    std::string watermark_model;

    // --detect-watermark PATH: standalone watermark detection mode.
    // When set, reads the WAV file, runs watermark detection, prints
    // the result, and exits. Exposes the detection API for end users.
    std::string detect_watermark_file;

    // --print-speaker-identity PATH: standalone verb. Resolves whose voice a
    // model or voice pack produces and prints it, then exits. One source of
    // truth for the answer — a script that restated the verdicts would be a
    // third copy to drift from the C++ table.
    std::string print_speaker_identity_file;

    // C2PA (Content Credentials) signing — compile-time gated on
    // CRISPASR_HAVE_C2PA. Paths to self-signed or CA-issued X.509 cert
    // and key. Generate with: scripts/generate-c2pa-cert.sh
    std::string c2pa_cert;
    std::string c2pa_key;

    // Voice-cloning consent gate (EU AI Act / deepfake disclosure).
    // CLI: --i-have-rights sets this to true. Without it, voice cloning
    // (--voice <file.wav>) is refused.
    // Server: the request body must include a non-empty
    // `consent_attestation` string when voice ends in .wav.
    bool tts_voice_clone_consent = false;
    std::string tts_consent_attestation;

    // --consent-log <path> / CRISPASR_CONSENT_LOG. When set, every [CONSENT]
    // record is ALSO appended as one JSON object per line. Default off: turning
    // on a persistent record of who attested what is the operator's decision,
    // and it is their artefact to retain and erase, not ours. Tamper-resistance
    // comes from where they put the file (append-only perms, WORM, a SIEM) —
    // see crispasr_consent_record.h.
    std::string consent_log;

    // Set when THIS run baked `tts_voice` from a user-supplied recording (the
    // TADA inline-clone path bakes a .wav into a temp .gguf and rewrites
    // tts_voice to point at it). Without this the rewrite erased the only
    // evidence the voice was a clone, and the consent + spoken-disclosure gates
    // — which classified by filename suffix — both scored the most explicit
    // cloning command in the CLI as "not a clone".
    // See crispasr_voice_clone_policy.h.
    bool tts_voice_baked_from_wav = false;
    // The RECORDING that bake started from, before tts_voice was rewritten to
    // point at the baked pack. Consent was given for this file, so this is what
    // the audit record must hash — the pack is a derived artefact, and hashing
    // it would bind the record to something the speaker never saw.
    std::string tts_voice_source_recording;

    // Operator override for whose voice a PRESET voice is: "real_person",
    // "synthetic" or "unknown"/empty. Outranks the pack's declaration and the
    // backend's default (crispasr_speaker_identity.h).
    //
    // real_person turns on the Art. 50(4) spoken disclosure for a non-cloned
    // voice, and deliberately does NOT turn on the --i-have-rights gate: the
    // donor's consent to the model being trained is a licensing matter settled
    // upstream that this operator cannot attest to.
    // CLI: --speaker-identity   Server: "speaker_identity"
    std::string tts_speaker_identity;

    // Skip the spoken AI-disclosure prefix on voice-cloned output.
    // Machine-readable provenance (watermark + C2PA) is always retained.
    // CLI: --no-spoken-disclaimer   Server: "spoken_disclaimer": false
    bool tts_no_spoken_disclaimer = false;

    // Turn off the imperceptible AI-content watermark on TTS output.
    // Equivalent to the CRISPASR_NO_WATERMARK env var; both emit a one-time
    // warning that AI-usage marking responsibility then rests with the operator.
    // On by default — see docs/issue-260/PLAN.md for the regulatory background.
    // CLI: --no-watermark
    bool tts_no_watermark = false;

    // Explicit attestation that the operator accepts AI-content marking/disclosure
    // responsibility. REQUIRED to honor ANY provenance opt-out (--no-watermark or
    // --no-spoken-disclaimer); without it those opt-outs are hard-refused. Mirrors
    // the voice-clone --i-have-rights consent gate.
    // CLI: --accept-marking-responsibility
    // Server: request field "marking_attestation": "<text>" (per-request spoken
    //   disclaimer opt-out), or the launch flag for a server-level --no-watermark.
    bool tts_marking_responsibility_accepted = false;
    std::string tts_marking_attestation;

    // Disable C2PA Content Credentials signing on synthesized output. A provenance
    // opt-out like --no-watermark: REQUIRES tts_marking_responsibility_accepted,
    // else refused. On the CLI the watertight rule still forces the audio watermark
    // on when C2PA is off, so output is never fully unmarked; on the server the
    // operator takes on the marking duty for every response.
    // CLI/server launch: --no-c2pa
    bool tts_no_c2pa = false;

    // Server mode: directory containing voice profiles for /v1/audio/speech.
    // Each profile is a sibling pair: <name>.wav + <name>.txt (the WAV is
    // the reference audio, the TXT is its transcription used by Qwen3-TTS
    // ICL prefill), or a baked voice pack <name>.gguf. The server's
    // POST /v1/audio/speech `voice` field maps to a stem in this directory.
    std::string tts_voice_dir;

    // Server mode: hard ceiling on /v1/audio/speech `input` length. Default
    // matches OpenAI's documented limit (4096 chars). Set to 0 to disable
    // the cap entirely (useful when long-form chunking is enabled and the
    // caller explicitly wants the chunker to handle whatever they send).
    // Above the cap, the route returns 400 with code="input_too_long".
    int tts_max_input_chars = 4096;

    // Server mode: speed multiplier (0.25 .. 4.0, OpenAI range). Applied
    // by the route handler as a post-synth linear resampler — backends
    // produce audio at their native rate, the server resamples before the
    // WAV/PCM/f32 dispatch. Default 1.0 = no resample. The CLI ignores
    // this; only the server route reads it.
    float tts_speed = 1.0f;

    // 75c-opt-2: per-request TTS backend knobs exposed via /v1/audio/speech.
    // Negative sentinel = "use backend default". The server route parses
    // these from JSON and each backend adapter applies them via native
    // setter calls when non-sentinel.
    float tts_top_p = -1.0f;
    float tts_min_p = -1.0f;
    int tts_top_k = -1;
    float tts_repetition_penalty = -1.0f;
    int tts_do_sample = -1;         // tada talker sampling: -1 unset, 0 greedy, 1 sample
    int tts_num_candidates = -1;    // tada per-token flow-matching candidates (quality/speed)
    float tts_cfg_scale = -1.0f;    // chatterbox cfg_weight, f5 cfg_strength, tada acoustic_cfg
    int tts_num_steps = -1;         // chatterbox cfm_steps, f5 ode_steps, tada num_flow_matching_steps
    float tts_noise_scale = -1.0f;  // piper VITS variance
    float tts_noise_w = -1.0f;      // piper stochastic duration predictor
    float tts_noise_temp = -1.0f;   // tada FM noise temperature (InferenceOptions.noise_temp)
    float tts_exaggeration = -1.0f; // chatterbox expressiveness
    int tts_speaker_id = -1;        // piper multi-speaker model
    int tts_max_speech_tokens = -1; // chatterbox max AR tokens
    int tts_min_speech_tokens = -1; // moss-tts min AR audio frames (exact-window fill)

    // CLI: --tts-play plays synthesised output on the local default speaker.
    // --tts-play-device N selects a non-default device by index (-1 = default).
    // Playback uses the same watermarked PCM that is written to --tts-output.
    bool tts_play = false;
    int tts_play_device = -1;

    // --tts-stream: stream synthesized audio to stdout as raw signed-16-bit
    // little-endian mono PCM (at the backend's sample rate), emitting each
    // sentence chunk as it is produced instead of writing one WAV at the end.
    // For piping into a player, e.g. `… --tts-stream | ffplay -f s16le -ar 48000 -`.
    bool tts_stream = false;

    // G2P phonemizer dictionary source:
    //   ""           → auto (OLaPh MIT preferred, then open-dict-data CC-BY-SA)
    //   "olaph"      → OLaPh MIT dicts from cstr/g2p-dicts HuggingFace
    //   "open-dict"  → open-dict-data CC-BY-SA from Wiktionary
    //   path         → custom dict file path
    std::string g2p_dict;

    // Server mode: when non-empty, every server response gets the
    // Access-Control-Allow-* headers set so browser clients can call us
    // cross-origin. Default empty = no CORS headers (server stays
    // default-locked; a deployed instance opts in explicitly with
    // --cors-origin '*' or a specific origin).
    std::string server_cors_origin;

    // Text-to-text translation input (m2m100 + future translate-only
    // backends). When `--text` is set on a backend that declares
    // CAP_TRANSLATE and has no input audio, crispasr_run dispatches to
    // backend->translate_text() instead of transcribe.
    //
    // Language handling has TWO independent pairs to support 2-stage
    // pipelines (e.g., ASR that only does EN→EN-text, then m2m100
    // translates the EN-text to Tamil — those two stages have different
    // source/target conventions):
    //   - source_lang / target_lang : primary backend (canary AST etc.)
    //   - translate_source_lang / translate_target_lang : second-stage
    //     translator (m2m100). Empty falls back to source_lang /
    //     target_lang. So standalone `--backend m2m100 -sl en -tl de`
    //     just works without learning new flags; the dedicated `-trsl`
    //     / `-trtl` flags only matter when the primary backend's
    //     `-sl`/`-tl` mean something else (e.g., 2-stage piping).
    std::string text_input;
    // 0 = let the backend apply its own documented default (#439). Both
    // translate runtimes already default to 200 when passed <= 0, matching
    // m2m100/wmt21's config.json max_length; madlad declares none and 200 is
    // the runtime's own figure. This was 256 — a number from nowhere that
    // SILENTLY OVERRODE both, because `translate_max_tokens > 0` is true for
    // the default, so every "fall back to the backend" branch was dead code.
    int translate_max_tokens = 0;
    std::string translate_source_lang; // overrides source_lang for the translator stage
    std::string translate_target_lang; // overrides target_lang for the translator stage

    // Text-LLM chat (server-mode /v1/chat/completions). Independent from
    // the audio backend's `model` path so a server can serve ASR + chat
    // off two different GGUFs. When empty, /v1/chat/completions returns
    // 503 with `chat_disabled` rather than 404. See docs/prompts/chat-abi.md
    // for the underlying ABI.
    std::string chat_model; // path to a GGUF chat model
    int32_t chat_n_ctx = 4096;
    int32_t chat_n_gpu_layers = -1;

    // Source separation (server-mode POST /v1/audio/separation). Independent
    // from the primary ASR backend — a secondary GGUF loaded at startup.
    // GGUF arch auto-detected (htdemucs / mel-band-roformer). When empty,
    // /v1/audio/separation returns 503 with `separation_disabled`.
    std::string separate_model;
};
