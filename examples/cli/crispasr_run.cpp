// crispasr_run.cpp — top-level dispatch for non-whisper backends.
//
// Called from cli.cpp main() when params.backend is a non-whisper backend.
// Drives the pipeline: resolve model -> detect backend -> load audio ->
// segment via VAD (or fixed chunks) -> transcribe -> print + write outputs.
//
// The whisper code path in cli.cpp is left completely untouched so the
// historical crispasr behaviour is bit-identical.

#include "crispasr_backend.h"
#include "crispasr_cache.h"
#include <cassert>
#include "crispasr_gap_fill.h"
#include "crispasr_split_pipeline.h"
#include "tada_encoder.h"

#include <sys/stat.h>
#include "crispasr_chunk_context_gate.h"
#include "crispasr_lcs_dedup.h"
#include "crispasr_long_audio_fallback.h"
#include "crispasr_mic_cli.h"
#include "crispasr_speaker.h"
#include "crispasr_popen.h"
#include "crispasr_beats_cli.h"
#include "crispasr_chords_cli.h"
#include "crispasr_tab_cli.h"
#include "crispasr_piano_cli.h"
#include "crispasr_pitch_cli.h"
#include "crispasr_separate_cli.h"
#include "crispasr_vad_cli.h"
#include "crispasr_output.h"
#include "crispasr_strict.h"          // #311: shared strict-pipeline reqs (also used by the server)
#include "crispasr_phonemes_policy.h" // #316: who can be driven by phonemes
#include "crispasr_punctuation_policy.h"
#include "crispasr_punc_loader.h"
#include "crispasr_truecase_loader.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_aligner_cli.h"
#include "crispasr_aligner.h"
#include "crispasr_lid_cli.h"
#include "crispasr_lid.h" // crispasr_lid_free_cache()
#include "crispasr_diarize_cli.h"
#include "crispasr_speaker_embedder.h"
#include "tiron_link.h"
#include "crispasr_mem.h"
#include "crispasr_stream_finalize.h"
#include "crispasr_stream_partial_decode.h"
#include "crispasr_stream_punc.h"
#include "whisper_params.h"
#include "fireredpunc.h"
#include "truecaser.h"
#include "truecaser_crf.h"
#include "truecaser_lstm.h"
#include "pcs.h"
#include "titanet.h"
#include "speaker_db.h"

#include "core/audio_window.h"
#include "core/crispasr_c2pa.h"
#include "crispasr_tts_chunking.h"
#include "crispasr_tts_disclaimer.h"
#include "crispasr_consent_record.h"
#include "crispasr_voice_clone_policy.h"
#include "crispasr_voice_provenance.h"
#include "core/crispasr_watermark.h"
#include "crispasr_watermark_dispatch.h"
#include "crispasr_watermark_stats.h"
#include "core/crispasr_wav_writer.h"
#include "core/segment_hygiene.h" // PLAN.md §W2/§W5/§W6 opt-in segment cleanup
#include "crispasr_mp3_writer.h"  // MP3 output via in-tree glint encoder
#include "crispasr_aac_writer.h"  // AAC-LC (ADTS) output via in-tree glint encoder
#include "crispasr_mp4_writer.h"  // AAC/Opus-in-MP4 muxer (C2PA-capable container)
#include "crispasr_opus_writer.h" // Ogg Opus output via in-tree glint encoder
#include "common-crispasr.h"      // read_audio_data

#include <algorithm>
#include <regex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Resolve the --watermark-model argument to a GGUF path (#260). "auto"/"default"
// pulls the AudioSeal neural watermark (MIT, opt-in SOTA) from the registry,
// auto-downloading when enabled; any other value is passed through as a literal
// path/name; empty stays empty (→ the always-on built-in spread-spectrum
// watermark). A failed AudioSeal resolve falls back to spread-spectrum via the
// dispatcher, which treats an unloadable model as "no neural watermark".
static std::string crispasr_resolve_watermark_model(const whisper_params& params) {
    if (params.watermark_model.empty())
        return "";
    if (params.watermark_model == "auto" || params.watermark_model == "default")
        return crispasr_resolve_model_cli("auto", "audioseal", params.no_prints, params.cache_dir, params.auto_download,
                                          "");
    return params.watermark_model;
}

// True if the container implied by `out_path` will carry a C2PA manifest under
// the current build and CRISPASR_NO_C2PA_REMUX setting. WAV/MP3/M4A/MP4 always
// can; raw ADTS .aac / Ogg .opus can only when remux to MP4 is enabled (the
// default). When C2PA is compiled out (CRISPASR_NO_C2PA_NATIVE and no c2pa-rs)
// nothing carries it. Used to keep the CLI watertight: when this is false the
// audio watermark is the only robust AI mark, so --no-watermark must not strip
// it (see crispasr_wm_dispatch::set_forced). Must mirror the container decision
// in crispasr_write_synth_audio below.
static bool crispasr_output_carries_c2pa(const std::string& out_path, bool no_c2pa = false) {
    if (no_c2pa)
        return false; // C2PA signing disabled (--no-c2pa) ⇒ watermark is the only floor
    if (out_path.empty())
        return false; // no container (e.g. raw PCM --tts-stream) ⇒ no manifest
#if defined(CRISPASR_HAVE_C2PA) || !defined(CRISPASR_NO_C2PA_NATIVE)
    auto ends_ci = [&](const char* suf) {
        const size_t n = std::strlen(suf);
        if (out_path.size() < n)
            return false;
        for (size_t i = 0; i < n; ++i)
            if (std::tolower((unsigned char)out_path[out_path.size() - n + i]) != std::tolower((unsigned char)suf[i]))
                return false;
        return true;
    };
    const bool is_aac = ends_ci(".aac");
    const bool is_opus = ends_ci(".opus") || ends_ci(".ogg");
    if (is_aac || is_opus)
        return std::getenv("CRISPASR_NO_C2PA_REMUX") == nullptr; // raw container ⇒ no manifest
    return true; // wav/mp3/m4a/mp4 (and the wav default) all carry a manifest
#else
    (void)out_path;
    return false;
#endif
}

// Watertight-CLI guarantee: no CLI output path may ever emit a fully unmarked
// AI file/stream. If `out_path` can't carry a C2PA manifest, force the audio
// watermark on (overriding --no-watermark / CRISPASR_NO_WATERMARK) so at least
// one robust machine-readable mark remains. Call once, after set_disabled(), and
// before the watermark embed for that output. Pass an empty path for --tts-stream.
static void crispasr_enforce_cli_watermark_floor(const std::string& out_path, const whisper_params& params) {
    const bool carries = crispasr_output_carries_c2pa(out_path, params.tts_no_c2pa);
    crispasr_wm_dispatch::set_forced(!carries);
    if (!carries && (params.tts_no_watermark || std::getenv("CRISPASR_NO_WATERMARK") != nullptr)) {
        fprintf(stderr,
                "crispasr: note: '%s' can't carry a C2PA manifest, so --no-watermark is "
                "overridden — the audio watermark is kept so the output stays marked as "
                "AI-generated. Use a C2PA-capable container (WAV/MP3/M4A/MP4, the default) "
                "to allow --no-watermark.\n",
                out_path.empty() ? "<pcm-stream>" : out_path.c_str());
    }
}

// Enforce the marking-responsibility attestation (hard-refuse policy, #294
// follow-up). Any provenance opt-out (--no-watermark / --no-spoken-disclaimer)
// requires an explicit --accept-marking-responsibility, mirroring the voice-clone
// --i-have-rights gate. Returns 0 if OK, or an exit code to hard-refuse. Emits a
// [MARKING] audit line (parallel to [CONSENT]) when an opt-out is honored.
static int crispasr_check_marking_attestation(const whisper_params& params) {
    const char* which = params.tts_no_watermark           ? "--no-watermark"
                        : params.tts_no_spoken_disclaimer ? "--no-spoken-disclaimer"
                        : params.tts_no_c2pa              ? "--no-c2pa"
                                                          : nullptr;
    if (!which)
        return 0; // no opt-out requested → nothing to attest
    if (!params.tts_marking_responsibility_accepted) {
        fprintf(stderr,
                "crispasr: error: %s requires --accept-marking-responsibility.\n"
                "  Disabling AI-content provenance marking shifts the marking/disclosure\n"
                "  duty to you, the operator. By passing --accept-marking-responsibility\n"
                "  you affirm you accept that responsibility for this output.\n"
                "  Usage: crispasr --tts \"text\" %s --accept-marking-responsibility\n",
                which, which);
        return 12;
    }
    std::time_t t = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
    fprintf(stderr, "[MARKING] ts=%s no_watermark=%s no_spoken_disclaimer=%s no_c2pa=%s attestation=\"%s\"\n", ts,
            params.tts_no_watermark ? "yes" : "no", params.tts_no_spoken_disclaimer ? "yes" : "no",
            params.tts_no_c2pa ? "yes" : "no", params.tts_marking_attestation.c_str());
    return 0;
}

// Serialize synthesized (TTS/S2S) float32 PCM to `out_path` — WAV by
// default, MP3 or AAC-LC/ADTS (in-tree glint encoder) when the path
// ends in .mp3 / .aac. All carry AI-provenance metadata (WAV LIST/INFO
// chunk, MP3/AAC ID3v2 TXXX tag). C2PA Content Credentials signing is
// WAV-only; requesting it with a lossy output warns instead of
// silently dropping the manifest. Returns 0 on success, 16 on failure
// (caller's exit code).
static int crispasr_write_synth_audio(const std::string& out_path, const float* pcm, int n_samples, int sample_rate,
                                      const std::string& c2pa_cert, const std::string& c2pa_key,
                                      const std::string& cache_dir = "", bool sign_c2pa = true) {
    auto has_ext = [&](const char* lo, const char* up) {
        return out_path.size() >= 4 &&
               (out_path.compare(out_path.size() - 4, 4, lo) == 0 || out_path.compare(out_path.size() - 4, 4, up) == 0);
    };
    // Case-insensitive suffix test (handles extensions of any length, e.g. the
    // 5-char ".opus" that has_ext's fixed 4-char compare can't).
    auto ends_with_ci = [&](const char* suf) {
        const size_t n = std::strlen(suf);
        if (out_path.size() < n)
            return false;
        for (size_t i = 0; i < n; ++i)
            if (std::tolower((unsigned char)out_path[out_path.size() - n + i]) != std::tolower((unsigned char)suf[i]))
                return false;
        return true;
    };
    const bool is_mp3 = has_ext(".mp3", ".MP3");
    const bool is_aac = has_ext(".aac", ".AAC");
    const bool is_m4a = has_ext(".m4a", ".M4A");
    const bool is_mp4 = has_ext(".mp4", ".MP4");
    const bool is_opus = ends_with_ci(".opus") || ends_with_ci(".ogg");

    // Native C2PA embeds in ISO-BMFF (MP4) but NOT raw ADTS AAC / Ogg Opus. So
    // when C2PA is active we mux AAC/Opus into an MP4 container (.m4a / .mp4) so
    // the output carries a real manifest, not just the watermark. Explicit
    // .m4a/.mp4 output is always AAC-in-MP4. Set CRISPASR_NO_C2PA_REMUX=1 to keep
    // the raw .aac/.opus container (watermark + metadata provenance only).
#if defined(CRISPASR_HAVE_C2PA) || !defined(CRISPASR_NO_C2PA_NATIVE)
    const bool c2pa_active = std::getenv("CRISPASR_NO_C2PA_REMUX") == nullptr;
#else
    const bool c2pa_active = false;
#endif
    const bool opus_mp4 = is_opus && c2pa_active;
    const bool aac_mp4 = is_m4a || is_mp4 || (is_aac && c2pa_active);

    std::string path = out_path; // may change extension when upgrading a raw container
    std::string blob;
    const char* c2pa_fmt = "audio/wav";
    if (aac_mp4 || opus_mp4) {
        blob = opus_mp4 ? crispasr_mp4::make_opus_mp4(pcm, n_samples, sample_rate)
                        : crispasr_mp4::make_aac_mp4(pcm, n_samples, sample_rate);
        if (blob.empty()) {
            fprintf(stderr, "crispasr: error: MP4 encoding failed for '%s'\n", path.c_str());
            return 16;
        }
        c2pa_fmt = "audio/mp4";
        if (is_aac) {
            path.erase(path.size() - 4); // drop ".aac"
            path += ".m4a";
            fprintf(stderr,
                    "crispasr: note: emitting '%s' (AAC-in-MP4) so C2PA can embed a manifest; "
                    "set CRISPASR_NO_C2PA_REMUX=1 for raw .aac\n",
                    path.c_str());
        } else if (is_opus) {
            if (const size_t dot = path.find_last_of('.'); dot != std::string::npos)
                path.erase(dot); // drop the extension
            path += ".mp4";
            fprintf(stderr,
                    "crispasr: note: emitting '%s' (Opus-in-MP4) so C2PA can embed a manifest; "
                    "set CRISPASR_NO_C2PA_REMUX=1 for raw .opus\n",
                    path.c_str());
        }
    } else if (is_mp3) {
        blob = crispasr_make_mp3(pcm, n_samples, sample_rate);
        c2pa_fmt = "audio/mpeg";
        if (blob.empty()) {
            fprintf(stderr, "crispasr: error: MP3 encoding failed for '%s'\n", path.c_str());
            return 16;
        }
    } else if (is_aac || is_opus) {
        // C2PA remux opted out — raw ADTS/Ogg, watermark + tag only.
        blob =
            is_aac ? crispasr_make_aac(pcm, n_samples, sample_rate) : crispasr_make_opus(pcm, n_samples, sample_rate);
        c2pa_fmt = "";
        if (blob.empty()) {
            fprintf(stderr, "crispasr: error: %s encoding failed for '%s'\n", is_aac ? "AAC" : "Opus", path.c_str());
            return 16;
        }
    } else {
        blob = crispasr_make_wav_int16(pcm, n_samples, sample_rate);
    }

    // C2PA Content Credentials signing. Effective signer creds are the
    // user-provided --c2pa-cert/--c2pa-key, or (on by default when C2PA is
    // compiled in) an auto-provisioned per-install self-signed cert. Signing is
    // best-effort provenance: any failure or an unembeddable container leaves
    // the watermark + metadata tag as the provenance signal.
    if (!sign_c2pa) {
        // --no-c2pa: attested provenance opt-out. On the CLI the audio watermark
        // is forced on (crispasr_enforce_cli_watermark_floor) so output is still
        // marked; here we simply skip embedding the manifest.
        fprintf(stderr,
                "crispasr: note: C2PA signing disabled (--no-c2pa); '%s' written without a manifest "
                "(audio watermark still applied)\n",
                path.c_str());
    } else if (c2pa_fmt && *c2pa_fmt) {
        crispasr_c2pa_sign_auto(blob, c2pa_fmt, c2pa_cert, c2pa_key, cache_dir);
    } else if (!c2pa_cert.empty() || !c2pa_key.empty()) {
        fprintf(stderr,
                "crispasr: note: C2PA cannot embed a manifest in this container; "
                "'%s' written unsigned (watermark + metadata provenance still applied)\n",
                path.c_str());
    }
    FILE* fout = fopen(path.c_str(), "wb");
    if (!fout) {
        fprintf(stderr, "crispasr: error: cannot write '%s'\n", path.c_str());
        return 16;
    }
    fwrite(blob.data(), 1, blob.size(), fout);
    fclose(fout);
    return 0;
}

// Apply FireRedPunc punctuation restoration to all segments.
static void apply_punc_model(fireredpunc_context* punc_ctx, std::vector<crispasr_segment>& segs) {
    if (!punc_ctx)
        return;
    for (auto& seg : segs) {
        char* result = fireredpunc_process(punc_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

static std::string apply_punc_text(fireredpunc_context* punc_ctx, const std::string& text) {
    if (!punc_ctx || text.empty())
        return text;
    char* result = fireredpunc_process(punc_ctx, text.c_str());
    if (!result)
        return text;
    std::string out = result;
    free(result);
    return out;
}

static bool stream_punc_partials_enabled(const whisper_params& params) {
    return crispasr_stream_punc_partials_enabled(params.stream_punc);
}

static bool stream_punc_finals_enabled(const whisper_params& params) {
    return crispasr_stream_punc_finals_enabled(params.stream_punc);
}



// Apply PCS (punctuation + capitalization + segmentation) to all segments.
static void apply_pcs_model(pcs_context* pcs_ctx, std::vector<crispasr_segment>& segs) {
    if (!pcs_ctx)
        return;
    for (auto& seg : segs) {
        char* result = pcs_process(pcs_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

// Apply statistical truecaser to all segments.
static void apply_truecase_model(truecaser_context* tc_ctx, std::vector<crispasr_segment>& segs) {
    if (!tc_ctx)
        return;
    for (auto& seg : segs) {
        char* result = truecaser_process(tc_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

// Apply CRF truecaser to all segments.
static void apply_truecase_crf_model(truecaser_crf_context* tc_crf_ctx, std::vector<crispasr_segment>& segs) {
    if (!tc_crf_ctx)
        return;
    for (auto& seg : segs) {
        char* result = truecaser_crf_process(tc_crf_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

// Apply BiLSTM truecaser to all segments.
static void apply_truecase_lstm_model(truecaser_lstm_context* tc_lstm_ctx, std::vector<crispasr_segment>& segs) {
    if (!tc_lstm_ctx)
        return;
    for (auto& seg : segs) {
        char* result = truecaser_lstm_process(tc_lstm_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

// Capability-vs-request check. For each requested feature, warn on stderr
// when the backend doesn't support it. Not fatal — the feature is silently
// ignored. Returns the number of warnings emitted.
int warn_unsupported(const CrispasrBackend& backend, const whisper_params& p) {
    const uint32_t caps = backend.capabilities();
    int warns = 0;

    auto warn = [&](const char* feature) {
        fprintf(stderr, "crispasr: warning: backend '%s' does not support %s — ignoring\n", backend.name(), feature);
        warns++;
    };

    // Diarize is now handled at the dispatcher level via the generic
    // crispasr_apply_diarize() post-step (energy / xcorr / future
    // pyannote / ecapa), so no warning even when the backend itself
    // doesn't claim CAP_DIARIZE — the dispatcher will label the
    // segments after transcribe() returns. Tinydiarize still requires
    // backend support (whisper-only).
    if (p.tinydiarize && !(caps & CAP_DIARIZE))
        warn("--tinydiarize");
    if (p.translate && !(caps & CAP_TRANSLATE))
        warn("--translate");
    if (!p.grammar.empty() && !(caps & CAP_GRAMMAR))
        warn("--grammar");
    if (p.temperature != 0.0f && !(caps & CAP_TEMPERATURE))
        warn("--temperature");
    if (!p.punctuation && !(caps & CAP_PUNCTUATION_TOGGLE))
        warn("--no-punctuation");
    if (!p.source_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--source-lang");
    if (!p.target_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--target-lang");
    if (p.n_processors > 1 && !(caps & CAP_PARALLEL_PROCESSORS))
        warn("--processors > 1");

    return warns;
}

// Merge individual-slice results into a flat list preserving time order.
//
// PLAN.md §W2/§W5/§W6 hygiene runs HERE rather than at the four
// `merge_segments(...)` call sites, because this is the structural chokepoint
// they all pass through — a hand-patched list of call sites is the exact shape
// of bug the copies-in-sync guard was built for (it covered 1 of 14 files for
// months). Every stage is opt-in via env; with none set this is byte-for-byte
// the old function.
//
// It also has to be here rather than per-slice: §W5 collapses runs of
// near-identical segments, and a loop that straddles a slice boundary is only
// visible once the slices are flat.
std::vector<crispasr_segment> merge_segments(std::vector<std::vector<crispasr_segment>>&& per_slice,
                                             const std::vector<crispasr_audio_slice>& /*slices*/) {
    std::vector<crispasr_segment> out;
    size_t total = 0;
    for (auto& v : per_slice)
        total += v.size();
    out.reserve(total);
    for (auto& v : per_slice) {
        for (auto& s : v)
            out.push_back(std::move(s));
    }

    // Issue #356: the structural chokepoint every slice result passes through
    // is also the last place a producer's ordering mistake can be caught before
    // it reaches the writers. Diagnostic only — this reports, it never reorders,
    // because a reorder here would paper over the producing bug.
    const auto hy = core_seg_hygiene::config_from_env();
    if (!core_seg_hygiene::any_enabled(hy)) {
        crispasr_warn_if_segments_backward(out, "slice merge");
        return out;
    }

    std::vector<core_seg_hygiene::Seg> view;
    view.reserve(out.size());
    for (const auto& s : out)
        view.push_back({s.text, s.t0, s.t1, 0.0f, false});

    int dropped = 0;
    const auto kept = core_seg_hygiene::apply_all(view, hy, &dropped);

    // Map each surviving view back onto its original, so every field the view
    // does not carry (speaker, words, tokens, chunk_id) is preserved rather
    // than reconstructed. apply_all only removes segments and rewrites
    // text/t1 — it never reorders — so a forward scan on t0 is exact.
    //
    // Resolve the whole mapping BEFORE touching `out`: the bail-out below
    // returns `out` unchanged, which is only true if nothing has been moved
    // out of it yet. Matching first, mutating second, keeps that promise.
    std::vector<size_t> pick;
    pick.reserve(kept.size());
    size_t oi = 0;
    for (const auto& k : kept) {
        while (oi < out.size() && out[oi].t0 != k.t0)
            oi++;
        if (oi >= out.size())
            break;
        pick.push_back(oi++);
    }
    if (pick.size() != kept.size()) { // unmatched view: never silently lose content
        crispasr_warn_if_segments_backward(out, "slice merge");
        return out;
    }

    std::vector<crispasr_segment> res;
    res.reserve(pick.size());
    for (size_t i = 0; i < pick.size(); i++) {
        crispasr_segment seg = std::move(out[pick[i]]);
        seg.text = kept[i].text;
        seg.t1 = kept[i].t1; // a merged run spans to the end of its last member
        res.push_back(std::move(seg));
    }
    if (dropped > 0 || res.size() != out.size())
        fprintf(stderr, "crispasr[hygiene]: %zu -> %zu segments (%d dropped)\n", out.size(), res.size(), dropped);
    crispasr_warn_if_segments_backward(res, "slice merge + hygiene");
    return res;
}

bool crispasr_words_have_positive_span(const std::vector<crispasr_word>& words) {
    return !words.empty() && words.back().t1 > words.front().t0;
}

// True if any segment carries a non-whitespace character (i.e. real text).
// Bytes <= 0x20 are ASCII whitespace/control; UTF-8 continuation/lead bytes
// are >= 0x80, so this also counts non-Latin scripts as text.
static bool crispasr_segs_have_text(const std::vector<crispasr_segment>& segs) {
    for (const auto& s : segs) {
        for (unsigned char c : s.text) {
            if (c > 0x20)
                return true;
        }
    }
    return false;
}

// Silent-failure guard (issue #240). A degenerate / over-quantized model can
// emit an empty transcript for clearly non-silent audio while still reporting
// a successful run; in scripted / embedding contexts (e.g. SubtitleEdit) that
// is indistinguishable from success. Emit a stderr warning so the empty output
// is at least visible. Gated by !no_prints (same as the timing line), and by a
// peak-amplitude silence gate so genuinely silent input never warns.
static void crispasr_warn_if_empty_transcript(bool have_text, const std::vector<float>& samples, double audio_s,
                                              const whisper_params& params) {
    if (params.no_prints || have_text || audio_s < 0.5)
        return;
    float peak = 0.0f;
    for (float v : samples)
        peak = std::fmax(peak, std::fabs(v));
    if (peak < 0.01f) // ~ -40 dBFS: treat as silence, no speech expected
        return;
    fprintf(stderr,
            "crispasr: WARNING: no text produced for %.1fs of non-silent audio (peak %.2f). "
            "Possible causes: the audio has no speech, an unsupported language, or an "
            "over-quantized model (try a q8_0/f16 build).\n",
            audio_s, (double)peak);
}

// Stdout serialization mutex. Used by the parallel-processors path to
// keep stdout transcript lines from interleaving across worker threads.
// The single-threaded path acquires it too — no measurable cost since
// it's an uncontended lock when n_processors == 1.
std::mutex g_stdout_mutex;

// Global post-merge speaker stages (issue #266). Runs on the FULL merged
// segment list so anonymous clustering and named identification share one
// foundation and one execution order:
//   global embedding clustering  ->  per-cluster speaker-db matching
// Identification is a closed-roster confirmation: the db is narrowed to
// params.expect_speakers before any match, each cluster is matched
// independently against the claimed profiles, unmatched clusters keep
// their anonymous "(speaker N) " labels, and nothing runs after this
// stage that could overwrite a matched name.
// Tiron (#295): thin CLI wrapper over the library-hoisted linker
// (crispasr_tiron_link_transcript) so the CLI, session C-ABI, and server all
// share ONE implementation. Returns true if the segments were tiron output.
static bool crispasr_apply_tiron_linking(std::vector<crispasr_segment>& segs, const std::vector<float>& samples,
                                         const whisper_params& params) {
    // Cross-window linking is opt-in (auto-downloads a speaker embedder); without
    // a diarization flag keep the model's window-local <|speakerN|> markers.
    const bool want_link = params.diarize || !params.diarize_embedder.empty();
    const std::string spec = want_link ? (!params.diarize_embedder.empty() ? params.diarize_embedder : "auto") : "";

    std::vector<TironTranscriptSeg> ts(segs.size());
    for (size_t i = 0; i < segs.size(); i++) {
        ts[i].text = segs[i].text;
        ts[i].t0_cs = segs[i].t0;
        ts[i].t1_cs = segs[i].t1;
        ts[i].chunk_id = segs[i].chunk_id;
    }
    const int n_spk = crispasr_tiron_link_transcript(ts, samples.data(), (int)samples.size(), spec.c_str(),
                                                     params.n_threads, params.cache_dir.c_str());
    if (n_spk < 0)
        return false; // not tiron

    // Copy the (stripped) text + labels back; drop bare-marker segments.
    std::vector<crispasr_segment> kept;
    kept.reserve(segs.size());
    for (size_t i = 0; i < segs.size(); i++) {
        if (ts[i].drop)
            continue;
        segs[i].text = ts[i].text;
        if (!ts[i].speaker.empty())
            segs[i].speaker = ts[i].speaker;
        kept.push_back(std::move(segs[i]));
    }
    if (n_spk > 0)
        segs = std::move(kept);
    if (n_spk > 0 && !params.no_prints) {
        fprintf(stderr, "crispasr[tiron]: linked across windows -> %d meeting-level speakers\n", n_spk);
    }
    return true;
}

static void crispasr_apply_global_speaker_stages(std::vector<crispasr_segment>& all_segs,
                                                 const std::vector<float>& samples, const whisper_params& params) {
    // Tiron output uses the model's own local speaker markers; link them into
    // global SPEAKER_NN and skip the generic pyannote/embedding diarizer.
    if (crispasr_apply_tiron_linking(all_segs, samples, params))
        return;

    // #324: foxnose runs here, once, over the whole audio — this is what makes
    // its speaker numbering consistent across slices.
    if (crispasr_apply_foxnose_global(all_segs, samples, params))
        return; // foxnose owns the labels; the TitaNet remap must not re-run

    const bool want_cluster =
        params.diarize && !params.diarize_embedder.empty() && !params.diarize_embedder_is_foxnose();
    const bool want_ident = !params.speaker_db.empty() && params.speaker_db_consent && !params.expect_speakers.empty();
    if ((!want_cluster && !want_ident) || all_segs.empty() || samples.empty())
        return;

    // One embedder for both stages. Identification without --diarize uses
    // the enrollment-side model (--titanet-model / auto) so dimensions
    // match the enrolled profiles.
    const std::string spec = !params.diarize_embedder.empty()
                                 ? params.diarize_embedder
                                 : (!params.titanet_model.empty() ? params.titanet_model : std::string("auto"));
    auto embedder = crispasr_make_speaker_embedder(spec, params.n_threads, params.cache_dir);
    if (!embedder)
        return;

    CrispasrClusterEmbeddings clusters;
    if (want_cluster)
        crispasr_remap_speakers_via_embeddings(all_segs, samples.data(), (int)samples.size(), embedder.get(), params,
                                               want_ident ? &clusters : nullptr);

    if (!want_ident)
        return;

    speaker_db* db = speaker_db_load(params.speaker_db.c_str());
    if (!db)
        return;
    const int claimed = speaker_db_retain(db, params.expect_speakers.c_str());
    if (claimed <= 0) {
        fprintf(stderr,
                "crispasr: speaker-db: none of the claimed speakers (--expect-speakers '%s') are\n"
                "  enrolled in '%s' — all labels stay anonymous\n",
                params.expect_speakers.c_str(), params.speaker_db.c_str());
    } else if (want_cluster) {
        crispasr_identify_speaker_clusters(all_segs, clusters, db, params.speaker_threshold, params.no_prints);
    } else {
        crispasr_identify_single_speaker(all_segs, samples.data(), (int)samples.size(), embedder.get(), db,
                                         params.speaker_threshold, params.no_prints);
    }
    speaker_db_free(db);
}

// ── Issue #311: strict pipeline requirements ──────────────────────────────
// Distinct non-zero exit codes so integrations can tell which required stage
// failed without parsing stderr.
enum crispasr_strict_rc {
    CRISPASR_STRICT_RC_VAD = 30,   // required VAD model failed to load / run
    CRISPASR_STRICT_RC_WORDS = 31, // required word timestamps missing on a non-empty segment
    CRISPASR_STRICT_RC_PUNC = 32,  // required punctuation model failed to load
};

// `crispasr_strict_reqs` + `crispasr_compute_strict_reqs` + the missing-word-ts
// counter are shared with the HTTP server via crispasr_strict.h (included above)
// so the two front-ends can't drift.

// Post-hoc word-timestamp gate: when required, (a) an explicitly requested
// aligner must have loaded (case 4 — caught even if native word timestamps
// would otherwise mask the failure), and (b) every segment with non-empty text
// must carry word timestamps, native or aligned (case 5). Returns 0 when
// satisfied / not required, else CRISPASR_STRICT_RC_WORDS after printing a
// machine-stable error. Empty/no-speech transcripts pass vacuously.
static int crispasr_strict_check_words(const std::vector<crispasr_segment>& segs, bool required,
                                       const std::string& fname_inp, bool aligner_load_failed = false) {
    if (!required)
        return 0;
    if (aligner_load_failed) {
        fprintf(stderr,
                "crispasr: error: the explicitly requested forced aligner failed to load for '%s' "
                "(--require-word-timestamps/--strict-pipeline).\n",
                fname_inp.c_str());
        return CRISPASR_STRICT_RC_WORDS;
    }
    int missing = crispasr_count_missing_word_ts(segs);
    if (missing > 0) {
        fprintf(stderr,
                "crispasr: error: word timestamps required (--require-word-timestamps/--strict-pipeline) but %d "
                "non-empty segment(s) in '%s' have none — the aligner failed to load/produce words, or the backend "
                "emitted no native word timing.\n",
                missing, fname_inp.c_str());
        return CRISPASR_STRICT_RC_WORDS;
    }
    return 0;
}

// Process a single input file end-to-end with the given backend instance.
// Pulled out of the main loop so the parallel-processors path can call
// it from worker threads. Each call holds its own audio buffers + segment
// state, so multiple workers can run concurrently against pre-loaded
// per-thread backend instances. Returns 0 on success, non-zero on
// failure.
int process_one_input(CrispasrBackend& backend, const std::string& fname_inp, const std::string& fname_out,
                      whisper_params params, fireredpunc_context* punc_ctx = nullptr,
                      truecaser_context* tc_ctx = nullptr, pcs_context* pcs_ctx = nullptr,
                      truecaser_crf_context* tc_crf_ctx = nullptr, truecaser_lstm_context* tc_lstm_ctx = nullptr) {
    // #311: OR-accumulated across every forced-aligner call in this file so the
    // strict word-timestamp gate can fail an explicitly-requested aligner that
    // could not load, even when the backend's native word timing masks it.
    bool aligner_load_failed = false;
    // Resolve the output path base for this input. -of FNAME (passed via
    // `fname_out`) wins; otherwise we strip the audio extension off the
    // input path and append the format extension. Mirrors the whisper
    // path's `fout_factory` resolution at cli.cpp:1586-1587.
    auto out_path = [&](const char* ext) -> std::string {
        if (!fname_out.empty())
            return fname_out + ext;
        return crispasr_make_out_path(fname_inp, ext);
    };
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    const bool want_stereo = params.diarize;
    // Load audio at the backend's native rate (e.g. 24 kHz for kyutai/vibevoice)
    // to avoid the lossy 16k→Nk double-resample path (issue #263).
    const int native_rate = backend.input_sample_rate();
    if (!read_audio_data(fname_inp, samples, stereo, want_stereo, native_rate)) {
        fprintf(stderr, "crispasr: error: failed to read audio '%s'\n", fname_inp.c_str());
        return 20;
    }

    // #91: --offset-t MS / --duration MS — restrict processing to a time
    // window of the input. The whisper backend honours these via its own
    // seek (cli.cpp); this dispatcher (all other backends) had no window
    // support, so apply it to the raw decoded PCM here — VAD, chunking and
    // transcription then all operate on the window, and reported timestamps
    // are shifted back by the offset at emit time (process_slice below) so
    // they stay in original-audio time.
    {
        const auto win =
            core_audio_window::compute((int64_t)samples.size(), params.offset_t_ms, params.duration_ms, native_rate);
        if (win.active && win.past_end) {
            fprintf(stderr, "crispasr: error: --offset-t %d ms is past the end of '%s' (%.1f s)\n", params.offset_t_ms,
                    fname_inp.c_str(), (double)samples.size() / native_rate);
            return 0;
        }
        if (win.active) {
            core_audio_window::trim(samples, win);
            for (auto& ch : stereo)
                core_audio_window::trim(ch, win);
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: processing window [%.2f s, %.2f s) of '%s'\n",
                        (double)win.start / native_rate, (double)(win.start + win.len) / native_rate,
                        fname_inp.c_str());
            }
        }
    }

    // When --verbose (-v) or CRISPASR_VERBOSE=1 is set, activate ALL
    // backend-specific debug/bench/verbose env vars. Only sets if not
    // already set, so explicit per-backend vars still take precedence.
    if (params.verbose || (getenv("CRISPASR_VERBOSE") && getenv("CRISPASR_VERBOSE")[0])) {
        params.verbose = true;
        auto setenv_safe = [](const char* name, const char* val) {
#ifdef _WIN32
            if (!getenv(name)) {
                _putenv_s(name, val);
            }
#else
            setenv(name, val, 0);
#endif
        };
        setenv_safe("WAV2VEC2_VERBOSE", "1");
        setenv_safe("WAV2VEC2_BENCH", "1");
        setenv_safe("VIBEVOICE_BENCH", "1");
        setenv_safe("VIBEVOICE_DEBUG", "1");
        setenv_safe("FIRERED_BENCH", "1");
        setenv_safe("COHERE_DEBUG", "1");
        setenv_safe("COHERE_BENCH", "1");
        setenv_safe("OMNIASR_BENCH", "1");
        setenv_safe("QWEN3_TTS_BENCH", "1");
        setenv_safe("QWEN3_TTS_DEBUG", "1");
        setenv_safe("FIREREDPUNC_DEBUG", "1");
    }

    crispasr_log_mem(params.verbose, "after audio decode");
    if (params.verbose) {
        double dur = (double)samples.size() / (double)native_rate;
        double est = crispasr_estimate_mem_mb(dur, backend.name());
        fprintf(stderr, "crispasr[verbose]: audio %.1fs (%zu samples, %.1f MB PCM), est encoder mem ~%.0f MB\n", dur,
                samples.size(), samples.size() * 4.0 / 1e6, est);
    }
    bool have_stereo = want_stereo && stereo.size() == 2 && !stereo[0].empty() && stereo[0].size() == stereo[1].size();
    if (have_stereo) {
        const size_t n = stereo[0].size();
        const size_t check = std::min<size_t>(n, 4096);
        bool channels_equal = true;
        for (size_t i = 0; i < check; i++) {
            if (stereo[0][i] != stereo[1][i]) {
                channels_equal = false;
                break;
            }
        }
        if (channels_equal)
            have_stereo = false;
    }

    const int SR = native_rate;
    if (!params.no_prints) {
        fprintf(stderr, "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n", (int)samples.size(),
                (double)samples.size() / SR, SR, params.n_threads);
    }

    // Speaker enrollment mode: extract TitaNet embedding, save to DB, exit.
    if (!params.enroll_speaker.empty()) {
        // Biometric consent gate. Enrollment persists a voiceprint linked
        // to a real name — special-category data under GDPR Art. 9. Refuse
        // unless the deployer has affirmed a lawful basis + explicit consent.
        if (!params.speaker_db_consent) {
            fprintf(stderr, "crispasr: error: --enroll-speaker requires --speaker-db-consent.\n"
                            "  Enrollment stores a voiceprint linked to a real name (biometric data,\n"
                            "  GDPR Art. 9 special category). Pass --speaker-db-consent only if you have\n"
                            "  explicit consent from this person and a lawful basis to store it.\n"
                            "  For privacy-clean stable speaker labels that identify no one, use\n"
                            "  --diarize-speakers instead (no database, no names).\n");
            return 25;
        }
        std::string tmodel = params.titanet_model;
        if (tmodel.empty() || tmodel == "auto") {
            tmodel =
                crispasr_resolve_model("auto", "titanet", params.no_prints, params.cache_dir, params.auto_download, "");
        }
        if (tmodel.empty()) {
            fprintf(stderr, "crispasr: error: cannot resolve TitaNet model for enrollment\n");
            return 21;
        }
        auto* tctx = titanet_init(tmodel.c_str(), params.n_threads);
        if (!tctx) {
            fprintf(stderr, "crispasr: error: failed to load TitaNet model '%s'\n", tmodel.c_str());
            return 22;
        }
        float emb[192];
        int dim = titanet_embed(tctx, samples.data(), (int)samples.size(), emb);
        titanet_free(tctx);
        if (dim <= 0) {
            fprintf(stderr, "crispasr: error: TitaNet embedding extraction failed\n");
            return 23;
        }
        std::string db_dir = params.speaker_db;
        if (db_dir.empty())
            db_dir = params.cache_dir.empty()
                         ? std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.cache/crispasr/speakers"
                         : params.cache_dir + "/speakers";
        if (!speaker_db_enroll(db_dir.c_str(), params.enroll_speaker.c_str(), emb, dim,
                               /*consent_attested=*/params.speaker_db_consent)) {
            fprintf(stderr, "crispasr: error: failed to enroll speaker '%s'\n", params.enroll_speaker.c_str());
            return 24;
        }
        return 0;
    }

    // Optional language-identification pre-step.
    bool want_auto_lang = params.detect_language || params.language == "auto";
    const bool has_native_lid = (backend.capabilities() & CAP_LANGUAGE_DETECT) != 0;
    const bool lid_disabled = params.lid_backend == "off" || params.lid_backend == "none";
    crispasr_lid_info lid_info; // stored for JSON output

    // #227: a monolingual backend (e.g. moonshine, English-only) can only emit
    // one language, so on a plain `-l auto` transcription resolve it directly and
    // skip external LID — no point downloading/running whisper-tiny to "detect"
    // a language the backend can't change. An explicit --detect-language still
    // runs LID: the user is asking for the audio's actual language, which the
    // backend's sole output language may not reflect.
    if (const char* sole_lang = backend.sole_language();
        sole_lang && params.language == "auto" && !params.detect_language) {
        params.language = sole_lang;
        if (params.source_lang.empty())
            params.source_lang = sole_lang;
        want_auto_lang = false;
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s is %s-only — skipping language detection\n", backend.name(), sole_lang);
    }

    // Some backends can identify the language with the model already loaded
    // (cohere probes its own supported set). Prefer that over an external
    // detector when the user did not name one explicitly: it downloads
    // nothing, and — the real reason — it can only return a language the
    // backend actually supports, whereas whisper-tiny LID knows 99 and will
    // happily hand back one the model was never trained on.
    bool probed_ok = false;
    if (want_auto_lang && !has_native_lid && !lid_disabled) {
        crispasr_lid_result probe;
        if (crispasr_backend_probe_language(backend, samples.data(), (int)samples.size(), params, probe)) {
            lid_info.lang_code = probe.lang_code;
            lid_info.confidence = probe.confidence;
            lid_info.source = probe.source;
            params.language = probe.lang_code;
            if (params.source_lang.empty())
                params.source_lang = probe.lang_code;
            probed_ok = true;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: LID -> language = '%s' (%s, p=%.3f)\n", probe.lang_code.c_str(),
                        probe.source.c_str(), probe.confidence);
            }
        }
    }

    if (want_auto_lang && !has_native_lid && !lid_disabled && !probed_ok) {
        crispasr_lid_result lid;
        if (crispasr_detect_language_cli(samples.data(), (int)samples.size(), params, lid)) {
            lid_info.lang_code = lid.lang_code;
            lid_info.confidence = lid.confidence;
            lid_info.source = lid.source;
            params.language = lid.lang_code;
            if (params.source_lang.empty()) {
                params.source_lang = lid.lang_code;
            }
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: LID -> language = '%s' (%s, p=%.3f)\n", lid.lang_code.c_str(),
                        lid.source.c_str(), lid.confidence);
            }
        } else {
            // LID couldn't run or returned nothing. Leaving params.language="auto"
            // here propagates a literal "auto" string to backends that expect a
            // real language code (e.g. canary embeds "<|en|>" directly), causing
            // empty/garbage transcripts. Fall back to "en" — the safest default
            // when LID is unavailable. The user can override with `-l <code>`.
            if (params.language == "auto") {
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: LID failed and no -l was set — "
                                    "defaulting to 'en'. Pass `-l <code>` to override.\n");
                }
                params.language = "en";
                if (params.source_lang.empty())
                    params.source_lang = "en";
            } else if (!params.no_prints) {
                fprintf(stderr, "crispasr: LID failed, falling back to params.language='%s'\n",
                        params.language.c_str());
            }
        }
    }

    // Free the cached whisper LID context to release GPU VRAM before
    // the ASR backend allocates its own model + KV cache (#35 OOM fix).
    crispasr_lid_free_cache();

    // Issue #89: backends with CAP_UNBOUNDED_INPUT (parakeet, canary,
    // wav2vec2, firered-asr, fastconformer-ctc, granite-nar) use
    // non-autoregressive encoders (FastConformer, CTC) that handle
    // arbitrary-length audio without chunking. Fixed 30 s chunk
    // boundaries cause text loss at chunk starts because bidirectional
    // encoders lose context at the cut points.
    //
    // Backends WITHOUT CAP_UNBOUNDED_INPUT (whisper, cohere, moonshine,
    // voxtral, granite, qwen3, glm-asr, kyutai-stt, mimo-asr, gemma4,
    // omniasr) use either fixed-window encoders (whisper) or
    // autoregressive decoders with KV cache that grow with input
    // length — these need chunking to avoid OOM on long audio.
    //
    // When the user didn't explicitly pass --chunk-seconds, disable
    // chunking for unbounded-input backends so the full audio is
    // processed in one encoder pass.
    // Issue #89: CAP_UNBOUNDED_INPUT backends (parakeet, canary, wav2vec2,
    // firered-asr, fastconformer-ctc, granite-nar) use bidirectional
    // encoders that produce inferior features when chunked (7-9% text
    // loss). Default to full-audio encoding for best quality.
    //
    // When the user explicitly passes --chunk-seconds, honor it — they
    // need chunking to avoid OOM on very long audio. Overlap-save
    // context (--chunk-overlap) mitigates boundary artifacts but the
    // encoder quality loss from reduced context is inherent to the
    // architecture.
    int effective_chunk_seconds = params.chunk_seconds;
    if (!params.chunk_seconds_explicit && (backend.capabilities() & (CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING))) {
        // A backend that streams unbounded input OR chunks internally (tiron's
        // fixed 30 s windows) gets the whole clip; slicing it here would
        // double-chunk / duplicate overlap.
        effective_chunk_seconds = 0;
    }
    // Issue #257: a backend that chunks internally (parakeet / canary — full-
    // attention FastConformer) is corrupted by the dispatcher's per-slice
    // transcribe + overlap-save trim + LCS merge. When the user forces
    // --chunk-seconds on such a backend, hand it the whole clip and let its
    // internal chunker honour the requested size (see the header for the gate).
    if (crispasr_chunk_context::backend_self_chunks_on_explicit((backend.capabilities() & CAP_INTERNAL_CHUNKING) != 0,
                                                                params.chunk_seconds_explicit, params.chunk_seconds)) {
        effective_chunk_seconds = 0;
    }

    // Issue #89: CAP_UNBOUNDED_INPUT backends are mathematically able to take
    // arbitrarily long audio, but in practice the FastConformer encoder + TDT
    // decoder break down past ~30 s in a single pass — per-feature z-norm
    // stats drift from the training distribution (model trained on ~10-15 s
    // utterances), the position encodings exit the trained range, and the TDT
    // decoder starts emitting blanks.  On the reporter's 300 s YouTube clip
    // with the original 60 s fallback, only 4 words survived in the first
    // 60 s on Vulkan/AMD hardware due to z-norm drift at that length.
    //
    // Fallback to 30 s chunking when the user provided no VAD and no explicit
    // --chunk-seconds and the audio is longer than the safe single-pass
    // window.  30 s is short enough for stable z-norm but long enough to
    // avoid excessive chunk boundaries.  The overlap-save gate in
    // `use_chunk_context` (issue #114) still applies, so chunk boundaries
    // get the ± chunk_overlap_seconds context they need.
    //
    // `--chunk-seconds 0` explicitly requests full-audio / library-internal
    // streaming (no dispatcher slicing). Honouring that intent requires
    // suppressing the auto-fallback: should_auto_chunk_long(0, …) would
    // otherwise fire because effective_chunk_seconds == 0 satisfies its
    // "not explicitly chunked" path. Guard with chunk_seconds_explicit.
    constexpr int kLongAudioFallbackChunkSeconds = 30;
    // Issue #89: backends that degenerate on arbitrary-length chunks
    // (e.g. parakeet-ja) auto-enable VAD for long audio so the model
    // gets silence-bounded segments matching its training distribution.
    const bool is_long_audio = (int)samples.size() > kLongAudioFallbackChunkSeconds * SR;
    // A backend that declares its OWN safe single-pass window (vad_slice_cap_seconds
    // — 12 s for parakeet-ja, #89) has to be protected at that window, not at the
    // unrelated 30 s global long-audio constant. Otherwise every JA clip in the
    // 12–30 s gap is "not long audio", skips the auto-VAD safeguard, and runs one
    // full pass past the encoder's trained range: the 14 s JA regression fixture
    // degraded to a hallucinated leading + trailing sentence and misread digits
    // (`6対3` → `0失点`). With VAD it reproduces the pinned reference core exactly.
    const int backend_window_s = backend.vad_slice_cap_seconds();
    const bool exceeds_backend_window = backend_window_s > 0 && (int)samples.size() > backend_window_s * SR;
    if (backend.prefers_vad() && (is_long_audio || exceeds_backend_window) && !params.vad && params.vad_model.empty() &&
        !params.chunk_seconds_explicit) {
        params.vad = true;
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: %s backend auto-enabling --vad on %.1fs audio "
                    "(this model needs utterance-bounded segments for clean output; "
                    "pass --chunk-seconds N to override)\n",
                    backend.name(), (double)samples.size() / SR);
        }
    }
    const bool wants_vad = params.vad || !params.vad_model.empty();
    const bool long_audio_no_vad =
        !params.chunk_seconds_explicit &&
        crispasr_long_audio::should_auto_chunk_long(effective_chunk_seconds, wants_vad, backend.capabilities(),
                                                    (int)samples.size(), SR, kLongAudioFallbackChunkSeconds);
    if (long_audio_no_vad) {
        effective_chunk_seconds = kLongAudioFallbackChunkSeconds;
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: %s backend on %.1fs audio without --vad or --chunk-seconds — "
                    "auto-chunking at %d s to keep encoder in its safe window "
                    "(pass --vad for finer slicing, or --chunk-seconds 0 for library-internal streaming)\n",
                    backend.name(), (double)samples.size() / SR, kLongAudioFallbackChunkSeconds);
        }
    } else if (!params.no_prints) {
        if (effective_chunk_seconds == 0 && (backend.capabilities() & CAP_UNBOUNDED_INPUT)) {
            fprintf(stderr,
                    "crispasr: %s backend — full-audio / library-internal streaming "
                    "(use --chunk-seconds N if OOM; --vad to skip silence)\n",
                    backend.name());
        } else if (params.chunk_seconds_explicit && params.chunk_seconds > 0 &&
                   (backend.capabilities() & CAP_UNBOUNDED_INPUT) && (int)samples.size() > params.chunk_seconds * SR) {
            fprintf(stderr,
                    "crispasr: %s backend — chunking at %ds may reduce quality; "
                    "consider --chunk-seconds 0 or --vad for better results\n",
                    backend.name(), params.chunk_seconds);
        }
    }

    // Issue #89: backends with a bounded safe decode window (parakeet-ja,
    // ~12 s) need VAD slices re-split at energy minima down to that cap —
    // continuous speech merges into 40 s+ slices that decode sparse. An
    // explicit --chunk-seconds keeps the user in charge of the split size.
    int slice_chunk_seconds = effective_chunk_seconds;
    const int vad_cap = backend.vad_slice_cap_seconds();
    if (wants_vad && !params.chunk_seconds_explicit && vad_cap > 0 &&
        (slice_chunk_seconds == 0 || slice_chunk_seconds > vad_cap)) {
        slice_chunk_seconds = vad_cap;
    }
    // Issue #227: when --vad-import is given, read the segment boundaries from
    // the JSON file instead of running VAD (skips the VAD model entirely). This
    // lets the same audio be transcribed by several backends while paying the
    // VAD cost only once (--vad-export writes the boundaries on the first run).
    std::vector<crispasr_audio_slice> slices;
    if (!params.vad_import_file.empty()) {
        std::ifstream in(params.vad_import_file, std::ios::binary);
        if (!in) {
            fprintf(stderr, "crispasr: error: cannot open --vad-import file '%s'\n", params.vad_import_file.c_str());
            return 1;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        int imported_sr = 0;
        float imported_chunk = 0.0f;
        bool imported_raw = false;
        if (!crispasr_parse_vad_slices(ss.str(), slices, &imported_sr, &imported_chunk, &imported_raw)) {
            fprintf(stderr, "crispasr: error: malformed --vad-import file '%s'\n", params.vad_import_file.c_str());
            return 1;
        }
        // Issue #227: the slices are CHUNK boundaries, so they are only valid
        // for the chunk length that produced them. Reusing a 30 s export under
        // --chunk-seconds 5 would silently transcribe with the wrong chunking,
        // which looks like a model regression rather than a stale file.
        // Compare the REQUESTED chunk length on both sides, not the effective
        // one. --vad-export runs before backend init (it needs no ASR model),
        // so it cannot know the effective value -- that depends on the
        // backend's CAP_UNBOUNDED_INPUT and vad_slice_cap_seconds. Comparing
        // export-requested against import-EFFECTIVE is apples to oranges: with
        // whisper + --vad the effective value collapses to 0, so a correct
        // reuse was rejected with the advice "run with --chunk-seconds 0.00".
        const float requested_chunk = params.chunk_seconds > 0 ? (float)params.chunk_seconds : 30.0f;
        if (!imported_raw && crispasr_vad_chunk_mismatch(imported_chunk, requested_chunk)) {
            // WARN, do not fail, unless asked. The boundaries are still usable
            // -- they are just chunked differently than this run requested --
            // and turning a working --vad-import script into rc=1 on upgrade is
            // a worse outcome than a wrong chunk size the user can see.
            fprintf(stderr,
                    "crispasr: %s: --vad-import file was exported at --chunk-seconds %.2f but this run requests "
                    "%.2f.\n"
                    "       The exported boundaries are chunk boundaries, not raw speech segments, so the "
                    "chunking will not match this run.\n"
                    "       Re-export at %.2f, or pass --chunk-seconds %.2f, to make them agree.\n",
                    params.vad_import_strict ? "error" : "warning", imported_chunk, requested_chunk, requested_chunk,
                    imported_chunk);
            if (params.vad_import_strict)
                return 1;
        }

        // Boundaries are sample indices at the rate they were computed. If that
        // differs from this run's rate, rescale to keep them aligned in time
        // (t0_cs/t1_cs are rate-independent centiseconds — left as-is).
        if (imported_sr > 0 && imported_sr != SR) {
            fprintf(stderr, "crispasr: --vad-import boundaries were computed at %d Hz, this run is %d Hz — rescaling\n",
                    imported_sr, SR);
            for (auto& s : slices) {
                s.start = (int)((int64_t)s.start * SR / imported_sr);
                s.end = (int)((int64_t)s.end * SR / imported_sr);
            }
        }
        // Clamp to the current buffer and drop empty/invalid slices so a stale
        // or hand-edited file can't index out of bounds.
        const int n_samp = (int)samples.size();
        std::vector<crispasr_audio_slice> clean;
        clean.reserve(slices.size());
        for (auto& s : slices) {
            if (s.start < 0)
                s.start = 0;
            if (s.end > n_samp)
                s.end = n_samp;
            if (s.end > s.start)
                clean.push_back(s);
        }
        const size_t dropped = slices.size() - clean.size();
        slices = std::move(clean);

        // A raw-segment export carries no chunking, so re-chunk it for THIS run
        // exactly as a fresh VAD pass would (issue #227). This is what makes a
        // raw export reusable across runs with different chunk lengths without
        // ever tripping the mismatch gate: the segments are the model output,
        // the chunking is re-derived here.
        if (imported_raw && slice_chunk_seconds > 0) {
            slices = crispasr_rechunk_slices(slices, samples.data(), (int)samples.size(), SR, slice_chunk_seconds);
        }
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: imported %zu %s from '%s'%s\n", slices.size(),
                    imported_raw ? "VAD speech segment(s)" : "chunk boundary/boundaries",
                    params.vad_import_file.c_str(),
                    dropped ? (" (" + std::to_string(dropped) + " out-of-range dropped)").c_str() : "");
        }
    } else {
        bool vad_load_failed = false;
        slices = crispasr_compute_audio_slices(samples.data(), (int)samples.size(), SR, slice_chunk_seconds, params,
                                               &vad_load_failed);
        // #311: a required VAD model that failed to load must fail the run
        // *before* the slices.empty() no-speech path below silently succeeds
        // (and before any energy-chunk fallback is used for transcription).
        if (vad_load_failed && crispasr_compute_strict_reqs(params).vad) {
            fprintf(stderr,
                    "crispasr: error: required VAD model '%s' failed to load for '%s' "
                    "(--require-vad/--strict-pipeline) — refusing to fall back to fixed chunking.\n",
                    params.vad_model.c_str(), fname_inp.c_str());
            return CRISPASR_STRICT_RC_VAD;
        }
    }

    // NOTE: --vad-export is now handled before backend init
    // (crispasr_run_backend, issue #227). This site is no longer reached
    // when vad_export_file is set.

    if (slices.empty()) {
        // A loaded VAD that found no speech is a valid, successful outcome
        // (#311 acceptance case 2) — distinct from the load failure above.
        fprintf(stderr, "crispasr: warning: no speech detected in '%s'\n", fname_inp.c_str());
        return 0;
    }

    if (!params.no_prints && slices.size() > 1) {
        fprintf(stderr, "crispasr: processing %zu slice(s)\n", slices.size());
    }

    auto t_start = std::chrono::steady_clock::now();

    // --------------- VAD stitching path (crispasr-style) ---------------
    // When VAD produces multiple slices, stitch them into one contiguous
    // buffer (with 0.1s silence gaps) and process as a single transcribe()
    // call. This preserves cross-segment context and avoids boundary
    // artifacts. Timestamps are remapped from stitched-buffer positions
    // back to original-audio positions.
    //
    // Skip stitching for whisper backend (it has its own internal VAD+seek)
    // and when there's only one slice (no benefit).
    // Stitching concatenates all VAD segments into one buffer for a single
    // transcribe() call. This preserves cross-segment context but collapses
    // the output into one big segment — breaking SRT/VTT subtitle output.
    // Default: use per-slice path (each VAD segment → separate transcript
    // segment with correct timestamps). Users can opt in to stitching with
    // --vad-stitch if they want cross-segment context at the cost of
    // single-segment output.
    const bool use_stitching = slices.size() > 1 && params.vad && params.backend != "whisper" && params.vad_stitch;

    if (use_stitching) {
        auto stitched = crispasr_stitch_vad_slices(samples.data(), (int)samples.size(), SR, slices);
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: stitched %zu VAD segments → %.1fs (from %.1fs original)\n", slices.size(),
                    (double)stitched.total_duration_cs / 100.0, (double)samples.size() / SR);
        }

        // Transcribe the stitched buffer as one call.
        auto segs = backend.transcribe(stitched.samples.data(), (int)stitched.samples.size(), 0, params);

        // Remap timestamps from stitched-buffer space to original-audio space.
        for (auto& seg : segs) {
            seg.t0 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t0);
            seg.t1 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t1);
            for (auto& w : seg.words) {
                w.t0 = crispasr_vad_remap_timestamp(stitched.mapping, w.t0);
                w.t1 = crispasr_vad_remap_timestamp(stitched.mapping, w.t1);
            }
        }

        // Optional CTC alignment (on original audio, not stitched).
        // Issue #62: --force-aligner bypasses both the CAP gate and the
        // "skip already-aligned" guard, letting the user prefer the
        // aligner's timestamps over native ones (whisper / parakeet /
        // canary / cohere / kyutai-stt all produce native timing).
        const bool want_align =
            !params.aligner_model.empty() && ((backend.capabilities() & CAP_TIMESTAMPS_CTC) || params.force_aligner);
        if (params.verbose) {
            fprintf(stderr, "crispasr[verbose]: align[stitched]: aligner='%s' caps_ctc=%d force=%d -> want=%d\n",
                    params.aligner_model.c_str(), !!(backend.capabilities() & CAP_TIMESTAMPS_CTC),
                    params.force_aligner ? 1 : 0, want_align ? 1 : 0);
        }
        if (want_align) {
            for (auto& seg : segs) {
                if (!seg.words.empty() && !params.force_aligner)
                    continue;
                // Find the original audio region for this segment.
                const int s = (int)((double)seg.t0 / 100.0 * SR);
                const int e = std::min((int)samples.size(), (int)((double)seg.t1 / 100.0 * SR));
                if (e > s) {
                    bool load_failed = false;
                    auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + s, e - s, seg.t0,
                                                    params.n_threads, &load_failed);
                    aligner_load_failed = aligner_load_failed || load_failed;
                    if (crispasr_words_have_positive_span(words)) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }
        }

        // Issue #267: diarize the stitched result AFTER alignment so
        // word timestamps are available for speaker-turn splitting.
        // The stitched path transcribes all VAD segments as one buffer,
        // so diarize operates on the whole result with original-audio
        // timestamps already remapped.
        if (params.diarize && !segs.empty()) {
            // Build caches over the full original audio (same logic as
            // the per-slice path below, but scoped to the stitched block
            // since it returns before the per-slice declarations).
            CrispasrSherpaCache stitch_sherpa_cache;
            if (params.diarize_method == "sherpa" || params.diarize_method == "sherpa-onnx" ||
                params.diarize_method == "ecapa") {
                const float* full = samples.data();
                std::vector<float> mono_buf;
                if (have_stereo && !stereo[0].empty() && !stereo[1].empty()) {
                    const size_t n = std::min(stereo[0].size(), stereo[1].size());
                    mono_buf.resize(n);
                    for (size_t mi = 0; mi < n; mi++)
                        mono_buf[mi] = 0.5f * (stereo[0][mi] + stereo[1][mi]);
                    full = mono_buf.data();
                }
                if (!crispasr_compute_sherpa_cache(full, (int)samples.size(), params, stitch_sherpa_cache))
                    stitch_sherpa_cache = {};
            }
            CrispasrPyannoteCache stitch_pyannote_cache;
            if (params.diarize_method == "pyannote") {
                const float* full = samples.data();
                std::vector<float> mono_buf;
                if (have_stereo && !stereo[0].empty() && !stereo[1].empty()) {
                    const size_t n = std::min(stereo[0].size(), stereo[1].size());
                    mono_buf.resize(n);
                    for (size_t mi = 0; mi < n; mi++)
                        mono_buf[mi] = 0.5f * (stereo[0][mi] + stereo[1][mi]);
                    full = mono_buf.data();
                }
                if (!crispasr_compute_pyannote_cache(full, (int)samples.size(), params, stitch_pyannote_cache))
                    stitch_pyannote_cache = {};
            }
            const CrispasrPyannoteCache* pya_ptr = stitch_pyannote_cache.valid() ? &stitch_pyannote_cache : nullptr;
            const CrispasrSherpaCache* shp_ptr = stitch_sherpa_cache.valid() ? &stitch_sherpa_cache : nullptr;
            if (have_stereo) {
                crispasr_apply_diarize(stereo[0], stereo[1], /*is_stereo=*/true, 0, segs, params, pya_ptr, shp_ptr);
            } else {
                crispasr_apply_diarize(samples, samples, /*is_stereo=*/false, 0, segs, params, pya_ptr, shp_ptr);
            }
        }

        // Fall through to the shared output path below by wrapping
        // the stitched result into per_slice / all_segs.
        std::vector<std::vector<crispasr_segment>> stitched_per_slice(1);
        stitched_per_slice[0] = std::move(segs);
        auto all_segs = merge_segments(std::move(stitched_per_slice), slices);

        // Issue #267: global speaker stages for the stitched path too.
        crispasr_apply_global_speaker_stages(all_segs, samples, params);

        apply_punc_model(punc_ctx, all_segs);
        apply_truecase_model(tc_ctx, all_segs);
        apply_truecase_crf_model(tc_crf_ctx, all_segs);
        apply_truecase_lstm_model(tc_lstm_ctx, all_segs);

        apply_pcs_model(pcs_ctx, all_segs);
        if (!params.punctuation) {
            for (auto& seg : all_segs)
                crispasr_strip_punctuation(seg);
        }

        const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);
        const bool show_timestamps =
            !params.no_timestamps &&
            (params.output_srt || params.output_vtt || params.max_len > 0 || params.print_colors || params.diarize);
        {
            auto t_end = std::chrono::steady_clock::now();
            double t_total = std::chrono::duration<double>(t_end - t_start).count();
            double audio_s = (double)samples.size() / SR;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                        audio_s / std::max(t_total, 0.001));
            }
            crispasr_warn_if_empty_transcript(crispasr_segs_have_text(all_segs), samples, audio_s, params);
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            crispasr_print_stdout(disp, show_timestamps);
            if (params.show_alternatives)
                crispasr_print_alternatives(all_segs, params.n_alternatives);
            else if (params.print_confidence)
                crispasr_print_confidence(all_segs);
        }
        // #311: fail before writing output files if required word timestamps are missing.
        if (int rc = crispasr_strict_check_words(all_segs, crispasr_compute_strict_reqs(params).words, fname_inp,
                                                 aligner_load_failed))
            return rc;
        if (params.output_txt)
            crispasr_write_txt(out_path(".txt"), disp);
        if (params.output_srt)
            crispasr_write_srt(out_path(".srt"), disp);
        if (params.output_vtt)
            crispasr_write_vtt(out_path(".vtt"), disp);
        if (params.output_csv)
            crispasr_write_csv(out_path(".csv"), disp);
        if (params.output_lrc)
            crispasr_write_lrc(out_path(".lrc"), disp);
        if (params.output_jsn)
            crispasr_write_json(out_path(".json"), all_segs, backend.name(), params.model, params.language,
                                params.output_jsn_full, lid_info.lang_code.empty() ? nullptr : &lid_info);
        if (params.return_logits) {
            if (const auto* logits = backend.last_ctc_logits()) {
                crispasr_write_ctc_logits_json(out_path(".ctc-logits.json"), *logits, backend.name());
            } else if (!params.no_prints) {
                fprintf(stderr, "crispasr: warning: backend '%s' did not produce CTC logits\n", backend.name());
            }
        }
        return 0;
    }

    // --------------- Per-slice path (non-VAD or single slice) ---------------
    // Process VAD slices — parallel when multiple slices AND n_processors > 1
    std::vector<std::vector<crispasr_segment>> per_slice(slices.size());
    std::vector<crispasr_ctc_logits> per_slice_logits(slices.size());

    // Pyannote cross-slice fix (issue #107): pre-compute the
    // segmentation posteriors once over the FULL mono audio, then have
    // each per-slice diarize call score against the cached buffer. Per-
    // slice pyannote runs would reset local track indices (spk0/1/2
    // mean different physical speakers in each forward pass), which is
    // why the bug-report podcast saw speakers swapping across slices.
    //
    // Only allocated when the user actually picked --diarize-method
    // pyannote — otherwise we incur no extra cost. Stereo input is
    // downmixed to mono for pyannote (matches what apply_pyannote does
    // when called per-slice today).
    // Issue #110: global sherpa pre-compute. Run sherpa once over the
    // full audio (instead of per-slice) so speaker IDs are globally stable.
    CrispasrSherpaCache sherpa_cache;
    if (params.diarize &&
        (params.diarize_method == "sherpa" || params.diarize_method == "sherpa-onnx" ||
         params.diarize_method == "ecapa") &&
        !samples.empty()) {
        const float* full = samples.data();
        std::vector<float> mono_buf;
        if (have_stereo && !stereo[0].empty() && !stereo[1].empty()) {
            const size_t n = std::min(stereo[0].size(), stereo[1].size());
            mono_buf.resize(n);
            for (size_t i = 0; i < n; i++)
                mono_buf[i] = 0.5f * (stereo[0][i] + stereo[1][i]);
            full = mono_buf.data();
        }
        const int n_samples_full = (int)samples.size();
        if (!crispasr_compute_sherpa_cache(full, n_samples_full, params, sherpa_cache)) {
            // Cache build failed — fall back to per-slice sherpa.
            sherpa_cache = {};
        }
    }

    // #324: foxnose diarizes in ONE global pass after transcription so speaker
    // identities are consistent across slices; the per-slice path stands down.
    if (params.diarize && params.diarize_embedder_is_foxnose())
        const_cast<whisper_params&>(params).diarize_foxnose_global = true;

    CrispasrPyannoteCache pyannote_cache;
    if (params.diarize && params.diarize_method == "pyannote" && !samples.empty()) {
        const float* full = samples.data();
        std::vector<float> mono_buf;
        if (have_stereo && !stereo[0].empty() && !stereo[1].empty()) {
            const size_t n = std::min(stereo[0].size(), stereo[1].size());
            mono_buf.resize(n);
            for (size_t i = 0; i < n; i++)
                mono_buf[i] = 0.5f * (stereo[0][i] + stereo[1][i]);
            full = mono_buf.data();
        }
        const int n_samples_full = (int)samples.size();
        if (!crispasr_compute_pyannote_cache(full, n_samples_full, params, pyannote_cache)) {
            // Cache build failed (model missing, etc.). Fall back to
            // per-slice apply_pyannote — same code path as before P2a.
            // crispasr_apply_diarize handles the missing-cache case by
            // running the model per slice.
            pyannote_cache = {};
        }
    }

    // Issue #89 overlap-save chunking: when slicing is active, extend
    // each chunk by context_s seconds on each side so the bidirectional
    // encoder has left/right context at chunk boundaries. Only the
    // center region (the original slice range) is committed; words in
    // the extension zones are discarded via word-level filtering.
    //
    // Earlier attempt (617cd02) failed because (a) TDT emission frames
    // shift ±1-2 frames between contexts, dropping boundary words when
    // strict t0 filtering is used, and (b) segment text rebuild
    // inserted spaces before every word, breaking Japanese. Fixes:
    // (a) use a tolerance margin of 200 ms at boundaries so shifted
    // frames aren't lost; (b) concatenate word texts without inserting
    // spaces (the tokenizer already includes leading spaces where
    // appropriate).
    const float kChunkContextS = params.chunk_overlap_seconds;
    constexpr int64_t kBoundaryToleranceCs = 20; // 200 ms tolerance for TDT frame shift
    // Issue #114 — gate lives in crispasr_chunk_context_gate.h so the
    // unit test in tests/test-issue-114-chunk-context-gate.cpp can pin
    // it without spinning up a model. See the header for the rationale.
    // A handful of backends do their own internal chunking near the 30 s
    // boundary. Wrapping their fallback chunks in extra acoustic context
    // pushes the per-call input over that boundary, with backend-specific
    // bad outcomes (truncation, LLM retry loops). The opt-out list lives
    // in crispasr_chunk_context_gate.h; tools/check-overlap-save-bug.sh
    // is the A/B sweep that surfaces new offenders.
    const bool backend_ok = crispasr_chunk_context::backend_allows_chunk_context(backend.name());
    const bool use_chunk_context = crispasr_chunk_context::should_use_chunk_context(
        effective_chunk_seconds, slices.size(), kChunkContextS, wants_vad, backend_ok);

    // Per-slice progress counter for --print-progress on unified backends.
    // Atomic so parallel workers can safely increment it.
    std::atomic<size_t> slices_done{0};

    // Slice i's encoder input range (slice ± optional acoustic context).
    auto slice_ext_range = [&](size_t i, int& ext_start, int& ext_end, int64_t& ext_t0_cs) {
        const auto& sl = slices[i];
        ext_start = sl.start;
        ext_end = sl.end;
        if (use_chunk_context) {
            const int ctx_samples = (int)(kChunkContextS * SR);
            ext_start = std::max(0, sl.start - ctx_samples);
            ext_end = std::min((int)samples.size(), sl.end + ctx_samples);
        }
        ext_t0_cs = (int64_t)((double)ext_start / SR * 100.0);
    };

    // Everything after the model call: logits capture, context trimming,
    // storage, progress. Shared by the sequential, worker-pool and pipelined
    // paths so all three produce identical per_slice contents.
    // Per-slice progress, split out of finish_slice so the pipelined path can
    // report as each slice DECODES while still storing segments after the join.
    auto tick_slice_progress = [&]() {
        if (params.print_progress && slices.size() > 1) {
            const size_t done = slices_done.fetch_add(1) + 1;
            const int pct = (int)(done * 100 / slices.size());
            fprintf(stderr, "crispasr: progress = %3d%% (%zu/%zu slices)\n", pct, done, slices.size());
        }
    };

    auto finish_slice = [&](size_t i, std::vector<crispasr_segment> segs, CrispasrBackend& be,
                            bool report_progress = true) {
        const auto& sl = slices[i];
        if (params.return_logits) {
            if (const auto* logits = be.last_ctc_logits())
                per_slice_logits[i] = *logits;
            else
                per_slice_logits[i] = {};
        }

        // Trim back to the original slice range when context was added.
        if (use_chunk_context && !segs.empty()) {
            const bool is_first = (i == 0);
            const bool is_last = (i == slices.size() - 1);
            // Left boundary: first slice keeps everything, others trim.
            const int64_t left_cs = is_first ? 0 : (sl.t0_cs - kBoundaryToleranceCs);
            // Right boundary: last slice keeps everything, others trim.
            const int64_t right_cs = is_last ? INT64_MAX : (sl.t1_cs + kBoundaryToleranceCs);

            for (auto& seg : segs) {
                if (seg.words.empty()) {
                    // No word-level data — filter at segment level.
                    // Keep the segment if its center is in range.
                    const int64_t mid = (seg.t0 + seg.t1) / 2;
                    if (mid < left_cs || mid >= right_cs)
                        seg.text.clear();
                    continue;
                }
                // Word-level filtering: keep words whose t0 is in range.
                std::vector<crispasr_word> kept;
                for (auto& w : seg.words) {
                    if (w.t0 >= left_cs && w.t0 < right_cs)
                        kept.push_back(std::move(w));
                }
                // Rebuild segment text from the surviving words. Two word
                // conventions coexist: whisper/parakeet carry a leading space
                // in word.text (" on"), while granite's [T:N]-parsed words do
                // not ("on"). Insert a separating space only when the current
                // word does not already start with one AND the boundary is not
                // CJK (#205: granite long-audio text was concatenated to
                // "previouslyonmccloud's" because the words lack leading
                // spaces; the original no-space concat fixed JA kana-spacing,
                // 617cd02, which the CJK guard preserves).
                std::string rebuilt;
                for (const auto& w : kept) {
                    if (w.text.empty())
                        continue;
                    if (!rebuilt.empty()) {
                        const unsigned char prev_last = (unsigned char)rebuilt.back();
                        const unsigned char cur_first = (unsigned char)w.text[0];
                        const bool already_spaced = (cur_first == ' ');
                        // 3-byte+ UTF-8 lead bytes (>= 0xE0) cover CJK / kana /
                        // hangul, which are written without inter-word spaces.
                        const bool cjk_boundary = (prev_last >= 0xE0) || (cur_first >= 0xE0);
                        if (!already_spaced && !cjk_boundary)
                            rebuilt += ' ';
                    }
                    rebuilt += w.text;
                }
                // Strip leading space if present (first word of segment
                // may have a leading space from BPE convention).
                if (!rebuilt.empty() && rebuilt[0] == ' ')
                    rebuilt = rebuilt.substr(1);
                seg.text = std::move(rebuilt);
                seg.words = std::move(kept);
                if (!seg.words.empty()) {
                    seg.t0 = seg.words.front().t0;
                    seg.t1 = seg.words.back().t1;
                }
            }
            // Remove empty segments.
            segs.erase(
                std::remove_if(segs.begin(), segs.end(), [](const crispasr_segment& s) { return s.text.empty(); }),
                segs.end());
        }

        // Issue #89 gap-fill second pass (bounded-window backends only).
        if (be.vad_slice_cap_seconds() > 0) {
            const char* gf = getenv("CRISPASR_GAP_FILL");
            if (!gf || atoi(gf) != 0)
                crispasr_gap_fill_slice(be, params, samples.data(), (int)samples.size(), SR, sl, segs);
        }

        // Issue #267: run external CTC alignment BEFORE diarization so
        // that word timestamps are available when the diarize step splits
        // segments at speaker-turn boundaries. Previously diarize ran
        // first and could only assign a dominant speaker to whole ASR
        // segments; now it can split at word boundaries.
        //
        // Issue #62: --force-aligner bypasses CAP gate + already-aligned skip.
        const bool want_align =
            !params.aligner_model.empty() && ((backend.capabilities() & CAP_TIMESTAMPS_CTC) || params.force_aligner);
        if (params.verbose) {
            fprintf(stderr, "crispasr[verbose]: align[slice]: aligner='%s' caps_ctc=%d force=%d -> want=%d\n",
                    params.aligner_model.c_str(), !!(backend.capabilities() & CAP_TIMESTAMPS_CTC),
                    params.force_aligner ? 1 : 0, want_align ? 1 : 0);
        }
        if (want_align) {
            for (auto& seg : segs) {
                if (!seg.words.empty() && !params.force_aligner)
                    continue;
                bool load_failed = false;
                auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + sl.start,
                                                sl.end - sl.start, sl.t0_cs, params.n_threads, &load_failed);
                aligner_load_failed = aligner_load_failed || load_failed;
                if (crispasr_words_have_positive_span(words)) {
                    seg.t0 = words.front().t0;
                    seg.t1 = words.back().t1;
                    seg.words = std::move(words);
                }
            }
        }

        // Issue #267: diarize AFTER alignment so word timestamps (native
        // or externally aligned) are available for speaker-turn splitting.
        // When no words are present (no aligner, no native timestamps),
        // the diarize code falls back to segment-level dominant-speaker
        // assignment — the same behaviour as before.
        if (params.diarize && !segs.empty()) {
            const CrispasrPyannoteCache* pya_ptr = pyannote_cache.valid() ? &pyannote_cache : nullptr;
            const CrispasrSherpaCache* shp_ptr = sherpa_cache.valid() ? &sherpa_cache : nullptr;
            if (have_stereo) {
                std::vector<float> sl_l(stereo[0].begin() + sl.start, stereo[0].begin() + sl.end);
                std::vector<float> sl_r(stereo[1].begin() + sl.start, stereo[1].begin() + sl.end);
                crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true, sl.t0_cs, segs, params, pya_ptr, shp_ptr);
            } else {
                std::vector<float> mono_slice(samples.begin() + sl.start, samples.begin() + sl.end);
                crispasr_apply_diarize(mono_slice, mono_slice,
                                       /*is_stereo=*/false, sl.t0_cs, segs, params, pya_ptr, shp_ptr);
            }
        }

        // NOTE (issue #266): speaker-db identification no longer runs here.
        // One embedding per dispatcher slice assigned a single identity to
        // every segment in the slice — wrong for mixed-speaker slices — and
        // global clustering later overwrote the names anyway. Identification
        // now runs once, post-merge, per global speaker cluster: see
        // crispasr_apply_global_speaker_stages().

        // #91: shift reported timestamps back into original-audio time when
        // a --offset-t window trimmed the buffer (segs are in window-relative
        // centiseconds at this point). This is the single choke point every
        // output surface reads, and it re-runs cleanly on the file-output
        // redo pass since each process_slice call rebuilds segs from scratch.
        if (params.offset_t_ms > 0) {
            const int64_t off_cs = (int64_t)params.offset_t_ms / 10;
            for (auto& seg : segs) {
                seg.t0 += off_cs;
                seg.t1 += off_cs;
                for (auto& w : seg.words) {
                    w.t0 += off_cs;
                    w.t1 += off_cs;
                }
                for (auto& tok : seg.tokens) {
                    if (tok.t0 >= 0)
                        tok.t0 += off_cs;
                    if (tok.t1 >= 0)
                        tok.t1 += off_cs;
                }
            }
        }

        // #292: stamp the chunk index so a consumer can tell that "(speaker N)"
        // labels are chunk-local and restart per chunk. Only when there is more
        // than one chunk — a single-pass run leaves chunk_id at its -1 default.
        if (slices.size() > 1)
            for (auto& seg : segs)
                seg.chunk_id = (int)i;

        per_slice[i] = std::move(segs);

        // Per-slice progress for unified backends (whisper uses its own
        // encoder-level callback). Print to stderr so it doesn't mix
        // with transcript output. The pipelined path already ticked it at
        // decode time, so it passes report_progress=false here.
        if (report_progress)
            tick_slice_progress();
    };

    auto process_slice = [&](size_t i, CrispasrBackend& be) {
        int ext_start = 0, ext_end = 0;
        int64_t ext_t0_cs = 0;
        slice_ext_range(i, ext_start, ext_end, ext_t0_cs);
        finish_slice(i, be.transcribe(samples.data() + ext_start, ext_end - ext_start, ext_t0_cs, params), be);
    };

    const int n_workers = params.return_logits ? 1 : std::min(params.n_processors, (int32_t)slices.size());

    auto merged_ctc_logits = [&]() {
        crispasr_ctc_logits merged;
        for (const auto& lg : per_slice_logits) {
            if (lg.data.empty() || lg.n_frames <= 0 || lg.n_vocab <= 0)
                continue;
            if (merged.n_vocab == 0) {
                merged.n_vocab = lg.n_vocab;
                merged.normalization = lg.normalization;
                merged.vocab = lg.vocab;
            }
            if (merged.n_vocab != lg.n_vocab)
                continue;
            merged.data.insert(merged.data.end(), lg.data.begin(), lg.data.end());
            merged.n_frames += lg.n_frames;
        }
        return merged;
    };

    // Encode ∥ decode pipelining over slices. Backends that expose a split
    // transcribe run the encoder (GPU) for slice N+1 on a worker thread while
    // this thread runs the decoder (CPU) for slice N. Same shape of win as the
    // worker pool below, but with ONE model resident instead of N — so it is
    // preferred whenever the user has not explicitly asked for -p workers.
    //
    // Skipped when return_logits is set: last_ctc_logits() is per-backend state
    // read right after the model call, and only the serial path can attribute it
    // to the right slice.
    //
    // Not every slice necessarily qualifies (a slice long enough to take a
    // backend's multi-window route does not), and the pipeline must never fall
    // back to transcribe() while the producer is running — that would encode on
    // two threads at once. So the slice list is walked in RUNS of consecutive
    // qualifying slices: each run gets its own producer thread that is joined
    // before anything else touches the model, and non-qualifying slices are
    // processed sequentially in between.
    //
    // The conditions live in crispasr_split_pipeline.h so there is one place to
    // add one and a unit test per condition. Two were missing here: the env
    // override dropped return_logits / worker-pool, and NOTHING covered the #89
    // gap-fill that finish_slice runs when vad_slice_cap_seconds() > 0 — that
    // one re-enters be.transcribe() on the consumer thread and aborts the
    // process on a ggml assert when the producer is mid-encode.
    crispasr_split::Inputs split_in;
    split_in.multiple_slices = slices.size() > 1;
    split_in.backend_supports_split = backend.supports_split_transcribe();
    split_in.worker_pool_requested = n_workers > 1;
    split_in.return_logits = params.return_logits;
    {
        const char* gf = getenv("CRISPASR_GAP_FILL");
        const bool gap_fill_on = !gf || atoi(gf) != 0;
        split_in.post_pass_reenters_model = backend.vad_slice_cap_seconds() > 0 && gap_fill_on;
    }
    const bool use_split_pipeline = crispasr_split::enabled(split_in, getenv("CRISPASR_SLICE_PIPELINE"));

    // Run slices [lo, hi) through the encode ∥ decode pipeline.
    auto pipeline_run = [&](size_t lo, size_t hi) {
        struct pending {
            CrispasrBackend::encoded_slice enc;
            int64_t t0_cs = 0;
            bool ok = false;
        };
        std::deque<pending> q;
        std::mutex m;
        std::condition_variable cv_full, cv_empty;
        std::vector<size_t> failed_slices;
        const size_t kCap = 2;

        std::thread producer([&] {
            for (size_t i = lo; i < hi; i++) {
                int ext_start = 0, ext_end = 0;
                int64_t ext_t0_cs = 0;
                slice_ext_range(i, ext_start, ext_end, ext_t0_cs);
                pending p;
                p.t0_cs = ext_t0_cs;
                p.enc = backend.encode_slice(samples.data() + ext_start, ext_end - ext_start, params);
                p.ok = (p.enc.h != nullptr);
                std::unique_lock<std::mutex> lk(m);
                cv_full.wait(lk, [&] { return q.size() < kCap; });
                q.push_back(p);
                cv_empty.notify_one();
            }
        });

        // Join on every exit path. Without this an exception escaping the loop
        // below reaches ~std::thread with the producer still joinable, which is
        // std::terminate.
        struct join_guard {
            std::thread& t;
            ~join_guard() {
                if (t.joinable())
                    t.join();
            }
        } jg{producer};

        // Decoded segments are held until the producer has joined: the repair
        // pass below re-encodes, so it cannot run while the producer is
        // encoding — and it has to run BEFORE finish_slice trims and stores,
        // which is where transcribe() applies it. Progress is still ticked here
        // so a pipelined run reports as it goes rather than all at the end.
        std::vector<std::vector<crispasr_segment>> decoded(hi - lo);
        for (size_t i = lo; i < hi; i++) {
            pending p;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_empty.wait(lk, [&] { return !q.empty(); });
                p = q.front();
                q.pop_front();
                cv_full.notify_one();
            }
            if (!p.ok) {
                // Unexpected encode failure (e.g. VRAM OOM) on a slice that did
                // qualify. Defer to AFTER the join: the fallback re-encodes, and
                // that must not overlap the producer.
                failed_slices.push_back(i);
                continue;
            }
            decoded[i - lo] = backend.decode_slice(p.enc, p.t0_cs, params);
            tick_slice_progress();
        }

        producer.join();

        // Single-threaded again: repair, then store; then retry anything the
        // encoder dropped (process_slice re-encodes too, hence also here).
        for (size_t i = lo; i < hi; i++) {
            if (std::find(failed_slices.begin(), failed_slices.end(), i) != failed_slices.end())
                continue;
            int ext_start = 0, ext_end = 0;
            int64_t ext_t0_cs = 0;
            slice_ext_range(i, ext_start, ext_end, ext_t0_cs);
            backend.repair_slice(samples.data() + ext_start, ext_end - ext_start, ext_t0_cs, decoded[i - lo], params);
            finish_slice(i, std::move(decoded[i - lo]), backend, /*report_progress=*/false);
        }
        for (size_t i : failed_slices)
            process_slice(i, backend);
    };

    if (use_split_pipeline) {
        // Per-call settings that transcribe() would apply on every call —
        // sampling, beam, attention context, hotwords. The split pair cannot do
        // it (encode_slice runs on the producer thread), so it happens once
        // here, on this thread, before any producer starts. Without it
        // `--beam-size 4` decoded greedily.
        backend.begin_split_run(params);
        std::vector<char> qualifies(slices.size(), 0);
        size_t n_ok = 0;
        for (size_t i = 0; i < slices.size(); i++) {
            int es = 0, ee = 0;
            int64_t et = 0;
            slice_ext_range(i, es, ee, et);
            qualifies[i] = backend.can_split_slice(ee - es, params) ? 1 : 0;
            n_ok += qualifies[i];
        }
        if (!params.no_prints && params.verbose)
            fprintf(stderr, "crispasr: pipelining encode/decode over %zu/%zu slices\n", n_ok, slices.size());

        size_t i = 0;
        while (i < slices.size()) {
            if (!qualifies[i]) {
                process_slice(i, backend);
                i++;
                continue;
            }
            size_t j = i;
            while (j < slices.size() && qualifies[j])
                j++;
            if (j - i >= 2)
                pipeline_run(i, j);
            else
                process_slice(i, backend); // lone qualifying slice: nothing to overlap
            i = j;
        }
    } else if (n_workers > 1 && slices.size() > 1) {
        // Parallel slice processing with separate backend instances
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: parallel processing %zu slices with %d workers\n", slices.size(), n_workers);
        }

        // Create extra backend instances for worker threads
        std::vector<std::unique_ptr<CrispasrBackend>> workers;
        workers.reserve(n_workers - 1);
        bool pool_ok = true;
        for (int w = 1; w < n_workers; w++) {
            auto wb = crispasr_create_backend(params.backend);
            if (!wb || !wb->init(params)) {
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: warning: failed to create worker %d, reducing parallelism\n", w);
                pool_ok = false;
                break;
            }
            workers.push_back(std::move(wb));
        }

        if (pool_ok && !workers.empty()) {
            // Dispatch slices round-robin across workers
            std::vector<std::thread> threads;
            std::atomic<size_t> next_slice{0};

            auto worker_fn = [&](CrispasrBackend& be) {
                while (true) {
                    size_t idx = next_slice.fetch_add(1);
                    if (idx >= slices.size())
                        break;
                    process_slice(idx, be);
                }
            };

            // Launch worker threads (workers[0..N-2] + main thread uses backend)
            for (auto& w : workers) {
                threads.emplace_back(worker_fn, std::ref(*w));
            }
            // Main thread also processes slices
            worker_fn(backend);

            for (auto& t : threads)
                t.join();
        } else {
            // Fallback to sequential
            for (size_t i = 0; i < slices.size(); i++)
                process_slice(i, backend);
        }
    } else if (params.flush_after > 0 && slices.size() > 1) {
        // Progressive mode: process slices sequentially, flush output after each.
        // This gives media players SRT entries as soon as each VAD segment is done.
        int srt_index = 1;                 // running SRT entry counter
        bool progressive_any_text = false; // issue #240 silent-failure guard
        const bool show_ts = !params.no_timestamps && (params.output_srt || params.output_vtt || params.max_len > 0 ||
                                                       params.print_colors || params.diarize);
        for (size_t i = 0; i < slices.size(); i++) {
            process_slice(i, backend);

            // Post-process this slice immediately
            auto slice_segs = std::move(per_slice[i]);
            apply_punc_model(punc_ctx, slice_segs);
            apply_truecase_model(tc_ctx, slice_segs);
            apply_truecase_crf_model(tc_crf_ctx, slice_segs);
            apply_truecase_lstm_model(tc_lstm_ctx, slice_segs);

            apply_pcs_model(pcs_ctx, slice_segs);
            if (!params.punctuation) {
                for (auto& seg : slice_segs)
                    crispasr_strip_punctuation(seg);
            }

            auto disp = crispasr_make_disp_segments(slice_segs, params.max_len, params.split_on_punct);

            // Print SRT entries progressively to stdout
            for (const auto& d : disp) {
                for (unsigned char c : d.text) {
                    if (c > 0x20) {
                        progressive_any_text = true;
                        break;
                    }
                }
                if (params.output_srt) {
                    int t0_ms = (int)(d.t0 * 10);
                    int t1_ms = (int)(d.t1 * 10);
                    printf("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n\n", srt_index++, t0_ms / 3600000,
                           (t0_ms / 60000) % 60, (t0_ms / 1000) % 60, t0_ms % 1000, t1_ms / 3600000,
                           (t1_ms / 60000) % 60, (t1_ms / 1000) % 60, t1_ms % 1000, d.text.c_str());
                } else {
                    if (show_ts) {
                        int s0 = (int)(d.t0 * 10), s1 = (int)(d.t1 * 10);
                        printf("[%02d:%02d:%02d.%03d --> %02d:%02d:%02d.%03d]  %s\n", s0 / 3600000, (s0 / 60000) % 60,
                               (s0 / 1000) % 60, s0 % 1000, s1 / 3600000, (s1 / 60000) % 60, (s1 / 1000) % 60,
                               s1 % 1000, d.text.c_str());
                    } else {
                        printf("%s", d.text.c_str());
                    }
                }
            }
            fflush(stdout);
        }

        // Timing
        {
            auto t_end = std::chrono::steady_clock::now();
            double t_total = std::chrono::duration<double>(t_end - t_start).count();
            double audio_s = (double)samples.size() / SR;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                        audio_s / t_total);
            }
            crispasr_warn_if_empty_transcript(progressive_any_text, samples, audio_s, params);
        }

        // Write output files (full set, from all slices combined)
        // Re-collect all per_slice segments for file output
        // (stdout already got progressive output above)
        if (params.output_txt || params.output_vtt || params.output_csv || params.output_lrc || params.output_jsn) {
            // Re-run all slices to collect for file output
            std::vector<std::vector<crispasr_segment>> per_slice_redo(slices.size());
            for (size_t i = 0; i < slices.size(); i++) {
                process_slice(i, backend);
                per_slice_redo[i] = std::move(per_slice[i]);
            }
            auto all_segs = merge_segments(std::move(per_slice_redo), slices);
            // Mirror the global speaker stages from the sequential path
            // below so file outputs in the parallel/output-redo path get
            // globally stable speaker IDs (#107 P3) and cluster-level
            // named identification (#266) too.
            crispasr_apply_global_speaker_stages(all_segs, samples, params);
            apply_punc_model(punc_ctx, all_segs);
            apply_truecase_model(tc_ctx, all_segs);
            apply_truecase_crf_model(tc_crf_ctx, all_segs);
            apply_truecase_lstm_model(tc_lstm_ctx, all_segs);

            apply_pcs_model(pcs_ctx, all_segs);
            if (!params.punctuation)
                for (auto& seg : all_segs)
                    crispasr_strip_punctuation(seg);
            auto disp_all = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

            // #311: fail before writing output files if required word timestamps are missing.
            if (int rc = crispasr_strict_check_words(all_segs, crispasr_compute_strict_reqs(params).words, fname_inp,
                                                     aligner_load_failed))
                return rc;
            if (params.output_txt)
                crispasr_write_txt(out_path(".txt"), disp_all);
            if (params.output_srt)
                crispasr_write_srt(out_path(".srt"), disp_all);
            if (params.output_vtt)
                crispasr_write_vtt(out_path(".vtt"), disp_all);
            if (params.output_csv)
                crispasr_write_csv(out_path(".csv"), disp_all);
            if (params.output_lrc)
                crispasr_write_lrc(out_path(".lrc"), disp_all);
            if (params.output_jsn)
                crispasr_write_json(out_path(".json"), all_segs, backend.name(), params.model, params.language,
                                    params.output_jsn_full, lid_info.lang_code.empty() ? nullptr : &lid_info);
            if (params.return_logits) {
                auto logits = merged_ctc_logits();
                if (!logits.data.empty())
                    crispasr_write_ctc_logits_json(out_path(".ctc-logits.json"), logits, backend.name());
            }
        }
        return 0;
    } else {
        // Sequential (single slice or n_processors == 1)
        for (size_t i = 0; i < slices.size(); i++)
            process_slice(i, backend);
    }

    // LCS-based dedup across adjacent chunks (issue #89 / #114 follow-up,
    // matching upstream NeMo's BatchedFrameASRTDT hypothesis stitching).
    // Only active when the overlap-save context window was added in the
    // first place — for VAD-derived slices there is no overlap region, so
    // no duplicates to remove. `delay_tokens` is sized to the chunk
    // overlap measured in encoder frames; parakeet's frame_dur is 80 ms
    // and each frame can emit one TDT token, so `chunk_overlap_seconds *
    // 1000 / 80` is a safe upper bound on how far back in the previous
    // chunk we need to look.
    // CLI knob (--lcs-dedup): "auto" (default) follows use_chunk_context,
    // "on" forces it even on bindings/test paths, "off" disables for A/B.
    // --lcs-min-length raises the floor on what counts as a match — useful
    // when audio has long-silence regions where blank tokens dominate the
    // boundary run.
    const bool lcs_default = use_chunk_context && per_slice.size() > 1;
    const bool lcs_active =
        (params.lcs_dedup == "on") ? (per_slice.size() > 1) : (params.lcs_dedup == "off" ? false : lcs_default);
    if (lcs_active) {
        const int delay_tokens = (int)(params.chunk_overlap_seconds * 1000.0f / 80.0f + 0.5f);
        crispasr_lcs::apply_lcs_chunk_dedup(per_slice, delay_tokens > 0 ? delay_tokens : 1, params.lcs_min_length);
    }

    auto all_segs = merge_segments(std::move(per_slice), slices);

    // Global speaker stages (#107 P3 clustering + #266 identification):
    // embedding-based clustering anchors speaker IDs globally across all
    // slices, then the optional speaker-db stage matches each CLUSTER
    // against the claimed roster. Runs after merge so both label writers
    // share one foundation and matched names can't be overwritten.
    crispasr_apply_global_speaker_stages(all_segs, samples, params);

    apply_punc_model(punc_ctx, all_segs);
    apply_truecase_model(tc_ctx, all_segs);
    apply_truecase_crf_model(tc_crf_ctx, all_segs);
    apply_truecase_lstm_model(tc_lstm_ctx, all_segs);

    apply_pcs_model(pcs_ctx, all_segs);
    if (!params.punctuation) {
        for (auto& seg : all_segs) {
            crispasr_strip_punctuation(seg);
        }
    }

    const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

    const bool show_timestamps = !params.no_timestamps && (params.output_srt || params.output_vtt ||
                                                           params.max_len > 0 || params.print_colors || params.diarize);
    {
        auto t_end = std::chrono::steady_clock::now();
        double t_total = std::chrono::duration<double>(t_end - t_start).count();
        double audio_s = (double)samples.size() / SR;
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                    audio_s / std::max(t_total, 0.001));
        }
        crispasr_warn_if_empty_transcript(crispasr_segs_have_text(all_segs), samples, audio_s, params);

        // Serialize stdout across parallel workers so multi-file
        // transcripts don't interleave line-by-line.
        std::lock_guard<std::mutex> lock(g_stdout_mutex);
        crispasr_print_stdout(disp, show_timestamps);
        if (params.show_alternatives) {
            crispasr_print_alternatives(all_segs, params.n_alternatives);
        } else if (params.print_confidence) {
            crispasr_print_confidence(all_segs);
        }
    }

    // #311: fail before writing output files if required word timestamps are missing.
    if (int rc = crispasr_strict_check_words(all_segs, crispasr_compute_strict_reqs(params).words, fname_inp,
                                             aligner_load_failed))
        return rc;
    if (params.output_txt)
        crispasr_write_txt(out_path(".txt"), disp);
    if (params.output_srt)
        crispasr_write_srt(out_path(".srt"), disp);
    if (params.output_vtt)
        crispasr_write_vtt(out_path(".vtt"), disp);
    if (params.output_csv)
        crispasr_write_csv(out_path(".csv"), disp);
    if (params.output_lrc)
        crispasr_write_lrc(out_path(".lrc"), disp);
    if (params.output_jsn)
        crispasr_write_json(out_path(".json"), all_segs, backend.name(), params.model, params.language,
                            params.output_jsn_full, lid_info.lang_code.empty() ? nullptr : &lid_info);
    if (params.return_logits) {
        auto logits = merged_ctc_logits();
        if (!logits.data.empty())
            crispasr_write_ctc_logits_json(out_path(".ctc-logits.json"), logits, backend.name());
        else if (!params.no_prints)
            fprintf(stderr, "crispasr: warning: backend '%s' did not produce CTC logits\n", backend.name());
    }

    return 0;
}

} // namespace

// Resolve the TADA encoder + aligner GGUFs (make-ref-encoder/aligner flags, else
// model dir → shared cache → auto-download from the model's HF repo, honouring
// --language), load the voice audio at 24 kHz, and run the aligner+encoder
// pipeline into `result`. Shared by --make-ref, --align, and query-time inline
// .wav voice cloning. Returns 0 on success, non-zero (with a stderr message
// prefixed by `label`) on failure.
static int tada_run_aligner_pipeline(const whisper_params& params, const std::string& audio_path,
                                     const std::string& transcript, const char* label, tada_encoder_result& result,
                                     double* out_audio_seconds = nullptr) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? "." : p.substr(0, sep);
    };
    const std::string aux_base = (params.backend == "tada-1b" || params.backend == "tada-tts-1b")
                                     ? "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/"
                                     : "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/";
    auto resolve_aux = [&](const std::string& name) -> std::string {
        std::string local = dir_of(params.model) + "/" + name;
        struct stat st;
        if (stat(local.c_str(), &st) == 0)
            return local;
        std::string cached = crispasr_cache::dir(params.cache_dir) + "/" + name;
        if (crispasr_cache::file_present(cached))
            return cached;
        if (params.auto_download)
            return crispasr_cache::ensure_cached_file(name, aux_base + name, params.no_prints, "crispasr",
                                                      params.cache_dir);
        return std::string();
    };
    std::string encoder_path = params.make_ref_encoder;
    std::string aligner_path = params.make_ref_aligner;
    if (encoder_path.empty())
        encoder_path = resolve_aux("tada-encoder-f16.gguf");
    if (aligner_path.empty()) {
        const std::string lang = (params.language.empty() || params.language == "auto") ? "en" : params.language;
        aligner_path = resolve_aux("tada-aligner-" + lang + ".gguf");
        if (aligner_path.empty() && lang != "en")
            aligner_path = resolve_aux("tada-aligner-en.gguf");
    }
    if (encoder_path.empty()) {
        fprintf(stderr,
                "crispasr[%s]: cannot find tada-encoder GGUF. Add --auto-download to fetch it, pass "
                "--make-ref-encoder <path>, or place tada-encoder-f16.gguf next to the model.\n",
                label);
        return 20;
    }
    if (aligner_path.empty()) {
        fprintf(stderr,
                "crispasr[%s]: cannot find tada-aligner GGUF. Add --auto-download to fetch it, pass "
                "--make-ref-aligner <path>, or place tada-aligner-en.gguf next to the model.\n",
                label);
        return 20;
    }
    fprintf(stderr, "crispasr[%s]: encoder=%s\n", label, encoder_path.c_str());
    fprintf(stderr, "crispasr[%s]: aligner=%s\n", label, aligner_path.c_str());
    fprintf(stderr, "crispasr[%s]: voice=%s text='%s'\n", label, audio_path.c_str(), transcript.c_str());

    std::vector<float> ref_audio;
    std::vector<std::vector<float>> stereo_dummy;
    if (!read_audio_data(audio_path, ref_audio, stereo_dummy, false)) {
        fprintf(stderr, "crispasr[%s]: failed to load audio '%s'\n", label, audio_path.c_str());
        return 20;
    }
    // read_audio_data returns 16 kHz — resample to 24 kHz for the encoder.
    int n_16k = (int)ref_audio.size();
    int n_24k = (int)((int64_t)n_16k * 24000 / 16000);
    std::vector<float> audio_24k(n_24k);
    for (int i = 0; i < n_24k; i++) {
        float src = (float)i * 16000.0f / 24000.0f;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < n_16k)
            audio_24k[i] = ref_audio[idx] * (1.0f - frac) + ref_audio[idx + 1] * frac;
        else if (idx < n_16k)
            audio_24k[i] = ref_audio[idx];
    }
    fprintf(stderr, "crispasr[%s]: audio %.2fs @ 24kHz (%d samples)\n", label, n_24k / 24000.0f, n_24k);
    if (out_audio_seconds)
        *out_audio_seconds = (double)n_24k / 24000.0;

    tada_encoder_params ep = tada_encoder_default_params();
    ep.n_threads = params.n_threads;
    ep.seed = params.seed;
    ep.verbosity = params.no_prints ? 0 : 1;
    tada_encoder_context* ectx = tada_encoder_init(encoder_path.c_str(), ep);
    if (!ectx) {
        fprintf(stderr, "crispasr[%s]: failed to load encoder '%s'\n", label, encoder_path.c_str());
        return 20;
    }
    int rc = tada_encoder_encode(ectx, aligner_path.c_str(), audio_24k.data(), n_24k, transcript.c_str(), result);
    tada_encoder_free(ectx);
    if (rc != 0)
        fprintf(stderr, "crispasr[%s]: encode failed (rc=%d)\n", label, rc);
    return rc;
}

// Append a streamed segment's transcript text, prefixing its native
// diarization label when the backend produced one. `seg.speaker` is empty
// for non-diarizing backends (so this is a no-op there) and carries the
// "(Speaker N) " form for native diarizers (moss-diarize, vibevoice) —
// matching the file-mode `prefix_speaker()` convention in crispasr_output.cpp.
// NOTE: like all streamed diarize labels, the speaker ordinals are
// window/utterance-local — "Speaker 1" in one step is not guaranteed to be
// the same physical voice as "Speaker 1" in a later step (no cross-window
// clustering runs in streaming mode; see docs/streaming.md).
static inline void crispasr_stream_append_seg(std::string& out, const crispasr_segment& s) {
    out += s.speaker;
    out += s.text;
}

// Distinct non-empty speaker label shared by every segment, or "" when the
// segments carry no label or disagree (a mid-utterance speaker turn). Used to
// attach a structured "speaker" field to a single-speaker `final` JSON event
// without inlining labels into `text` (the JSON convention keeps `text` clean).
static std::string crispasr_stream_common_speaker(const std::vector<crispasr_segment>& segs) {
    std::string spk;
    for (const auto& s : segs) {
        if (s.speaker.empty())
            continue;
        if (spk.empty())
            spk = s.speaker;
        else if (spk != s.speaker)
            return "";
    }
    // Trim a trailing space carried by the "(Speaker N) " form.
    while (!spk.empty() && (spk.back() == ' ' || spk.back() == '\t'))
        spk.pop_back();
    return spk;
}

int crispasr_run_backend(const whisper_params& params_in) {
    whisper_params params = params_in;

    // #311: validate strict-pipeline flag combinations up front. A per-stage
    // --require-* whose stage was never requested is a configuration error
    // (usage exit 2), distinct from a stage that ran and failed (exit 30-32).
    // --require-word-timestamps has no precondition — it is a property of the
    // output (native word timing OR the forced aligner satisfy it).
    // #316: --tts-phonemes needs a backend with a phonemes-in entry point (see
    // crispasr_phonemes_policy.h). Checked HERE, before any model is loaded,
    // because it depends only on the requested backend name — failing after a
    // multi-second load would be rude, and it made the refusal untestable
    // without that backend's weights on disk.
    if (!params.tts_phonemes.empty() && !crispasr_phonemes_policy::backend_supports(params.backend)) {
        fprintf(stderr, "crispasr: error: %s\n", crispasr_phonemes_policy::unsupported_message(params.backend).c_str());
        return 2;
    }

    if (params.require_vad && !(params.vad || !params.vad_model.empty())) {
        fprintf(stderr, "crispasr: error: --require-vad needs VAD to be requested (pass --vad or --vad-model/-vm).\n");
        return 2;
    }
    if (params.require_punctuation && params.punc_model.empty()) {
        fprintf(stderr, "crispasr: error: --require-punctuation needs a punctuation model (pass --punc-model).\n");
        return 2;
    }

    // §248: source separation is its own task (audio out, not transcripts).
    // Route to the separation dispatcher before any transcribe backend is built.
    if (params.separate)
        return crispasr_run_separate(params);

    // Chord recognition is its own task (a chord timeline out, not
    // transcripts). Same early-dispatch rule as --separate.
    if (params.chords)
        return crispasr_run_chords(params);

    // Guitar tablature is its own task (a per-frame string/fret score grid out,
    // not transcripts). Same early-dispatch rule as --chords.
    if (params.tab)
        return crispasr_run_tab(params);

    // Piano transcription is its own task (note EVENTS out, not transcripts).
    // Same early-dispatch rule as --chords.
    if (params.piano)
        return crispasr_run_piano(params);

    // Beat tracking is its own task (a beat/downbeat grid out, not
    // transcripts). Same early-dispatch rule as --chords.
    if (params.beats)
        return crispasr_run_beats(params);

    // Pitch (F0) is its own task too (pitch frames out, not transcripts).
    // Same early-dispatch rule as --separate.
    if (params.pitch)
        return crispasr_run_pitch(params);

    // ── Speaker-db policy gates (issue #266) ──────────────────────────────
    // Named identification is recorded-file only, consent-gated, and a
    // closed-roster confirmation. All three are hard invariants:
    //  * never in streaming/live mode (a real-time identification path is
    //    exactly what the EU AI Act's RBI regime restricts — Art. 5(1)(h));
    //  * no consent affirmation, no biometric processing (GDPR Art. 9);
    //  * no open 1:N scan — the deployer must claim WHO is present via
    //    --expect-speakers; clusters match only against those profiles.
    if (params.stream && (!params.speaker_db.empty() || !params.enroll_speaker.empty())) {
        fprintf(stderr, "crispasr: error: --speaker-db/--enroll-speaker are not available in streaming mode.\n"
                        "  Named speaker identification is restricted to recorded files (post-processing);\n"
                        "  real-time identification is deliberately unsupported.\n");
        return 26;
    }
    if (!params.speaker_db.empty() && params.enroll_speaker.empty()) {
        if (!params.speaker_db_consent) {
            fprintf(stderr, "crispasr: --speaker-db ignored: matching named voiceprints is biometric\n"
                            "  identification (GDPR Art. 9). Re-run with --speaker-db-consent to affirm\n"
                            "  consent + a lawful basis. For privacy-clean stable speaker labels that\n"
                            "  identify no one, use --diarize-speakers instead.\n");
            params.speaker_db.clear();
        } else if (params.expect_speakers.empty()) {
            fprintf(stderr, "crispasr: error: --speaker-db requires --expect-speakers \"NameA,NameB\".\n"
                            "  Identification is a closed-roster confirmation of participants you assert are\n"
                            "  present (and who consented at enrollment). An open \"who is this voice\" scan of\n"
                            "  the whole database is deliberately unsupported: that would be 1:N remote\n"
                            "  biometric identification (EU AI Act, Annex III 1(a)). Unmatched clusters keep\n"
                            "  anonymous (speaker N) labels.\n");
            return 27;
        } else if (params.diarize && params.diarize_embedder.empty()) {
            // Identification is defined per global speaker cluster, so
            // --speaker-db with diarization implies the clustering
            // embedder (same default as --diarize-speakers).
            params.diarize_embedder = "auto";
            if (!params.no_prints)
                fprintf(stderr, "crispasr: --speaker-db with --diarize enables global speaker clustering "
                                "(--diarize-embedder auto)\n");
        }
    }

    // ── --print-speaker-identity: whose voice does this file produce? ─────
    // Standalone verb, resolved with the SAME code the disclosure gate uses:
    // the stamp inside the file first, then the researched legacy table. Runs
    // before any backend/model resolution — it inspects a file, not a session.
    //
    // Exists so a script never has to restate a verdict. Anything that needs to
    // know (the stamping driver, a packaging step, an operator asking "will
    // this disclose?") asks the binary and gets the answer the runtime will
    // actually act on, instead of keeping a third copy of the table that drifts.
    //
    // Prints one of real_person / synthetic / unknown on stdout. Exit 0 when
    // the answer is established, 3 when it is unknown — so a shell driver can
    // skip unknowns with `if crispasr --print-speaker-identity f; then ...`.
    if (!params.print_speaker_identity_file.empty()) {
        const std::string& path = params.print_speaker_identity_file;
        // The stamp is authoritative wherever it exists; the tables are the
        // legacy fallback for files published before it. Same strongest-duty
        // combination the synthesis path applies, so this cannot report an
        // answer weaker than the one that will be enforced.
        const crispasr_voice::SpeakerIdentity stamped = crispasr_voice::read_model_speaker_identity(path);
        const crispasr_voice::PackProvenance pack = crispasr_voice::read_pack_provenance(path);
        // Both tables: this file may be a voice PACK or a MODEL, and the verb
        // has no session to ask which. The model table keys on a backend name,
        // so derive it from what the file declares about itself.
        crispasr_voice::SpeakerIdentity table = crispasr_voice::identity_for_voice_pack(path);
        if (table == crispasr_voice::SpeakerIdentity::Unknown) {
            const std::string arch = crispasr_voice::read_gguf_architecture(path);
            if (!arch.empty())
                table = crispasr_voice::identity_for_model(crispasr_voice::backend_for_architecture(arch), path);
        }
        const crispasr_voice::SpeakerIdentity id = crispasr_voice::resolve_speaker_identity(
            /*override=*/crispasr_voice::parse_speaker_identity(params.tts_speaker_identity),
            /*pack=*/pack.identity == crispasr_voice::SpeakerIdentity::Unknown ? table : pack.identity,
            /*backend=*/crispasr_voice::SpeakerIdentity::Unknown, stamped);
        printf("%s\n", crispasr_voice::to_string(id));
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: '%s' -> speaker_identity=%s (stamp=%s, table=%s)\n", path.c_str(),
                    crispasr_voice::to_string(id), crispasr_voice::to_string(stamped),
                    crispasr_voice::to_string(table));
        }
        return id == crispasr_voice::SpeakerIdentity::Unknown ? 3 : 0;
    }

    // ── --detect-watermark: standalone watermark detection verb ───────────
    // Reads a WAV file, runs watermark detection, prints the result, exits.
    // This is handled before any backend/model resolution so no GGUF is
    // needed (unless --watermark-model is also given for AudioSeal).
    if (!params.detect_watermark_file.empty()) {
        const std::string& wav_path = params.detect_watermark_file;
        FILE* fin = fopen(wav_path.c_str(), "rb");
        if (!fin) {
            fprintf(stderr, "crispasr: error: cannot open '%s'\n", wav_path.c_str());
            return 1;
        }

        // Read WAV: skip 44-byte header, read int16 samples, convert to float32.
        fseek(fin, 0, SEEK_END);
        long file_size = ftell(fin);
        if (file_size < 44) {
            fprintf(stderr, "crispasr: error: '%s' is too small to be a valid WAV\n", wav_path.c_str());
            fclose(fin);
            return 1;
        }

        // Parse sample rate from WAV header (bytes 24-27, little-endian uint32)
        fseek(fin, 24, SEEK_SET);
        uint32_t wav_sr = 0;
        if (fread(&wav_sr, 4, 1, fin) != 1) {
            fprintf(stderr, "crispasr: error: cannot read sample rate from '%s'\n", wav_path.c_str());
            fclose(fin);
            return 1;
        }

        fseek(fin, 44, SEEK_SET);
        long pcm_bytes = file_size - 44;
        int n_samples = (int)(pcm_bytes / sizeof(int16_t));
        std::vector<int16_t> pcm_i16(n_samples);
        if ((int)fread(pcm_i16.data(), sizeof(int16_t), n_samples, fin) != n_samples) {
            fprintf(stderr, "crispasr: error: short read from '%s'\n", wav_path.c_str());
            fclose(fin);
            return 1;
        }
        fclose(fin);

        // Convert int16 to float32
        std::vector<float> pcm(n_samples);
        for (int i = 0; i < n_samples; i++) {
            pcm[i] = (float)pcm_i16[i] / 32768.0f;
        }

        // Initialize watermark dispatcher (AudioSeal if --watermark-model given;
        // "auto" pulls the AudioSeal GGUF from the registry — #260)
        crispasr_wm_dispatch::init(crispasr_resolve_watermark_model(params));

        float confidence = crispasr_wm_dispatch::detect(pcm.data(), n_samples, (int)wav_sr);

        const bool neural = crispasr_wm_dispatch::get_ctx() != nullptr;
        const double dur_s = wav_sr > 0 ? (double)n_samples / (double)wav_sr : 0.0;

        fprintf(stdout, "File: %s\n", wav_path.c_str());
        fprintf(stdout, "Detector: %s\n", neural ? "AudioSeal (neural)" : "spread-spectrum (built-in)");
        fprintf(stdout, "Analysed: %.2f s\n", dur_s);
        fprintf(stdout, "Watermark confidence: %.4f\n", confidence);

        if (neural) {
            // AudioSeal returns a probability, not a bin-agreement fraction —
            // the binomial null in crispasr_watermark_stats.h does not apply.
            fprintf(stdout, "Result: %s\n",
                    confidence > 0.5f ? "AI-GENERATED WATERMARK DETECTED"
                                      : "No watermark detected (this does NOT mean the audio is human-made)");
        } else {
            // Spread-spectrum: the score is the fraction of CRISPASR_WATERMARK_NBINS
            // pseudo-random bins agreeing with the embedder's sign pattern, so
            // unwatermarked audio scores 0.5 on average — NOT 0. Report the exact
            // probability of reaching this score by chance instead of a bare
            // threshold: the old `> 0.65` bar is p = 0.055, i.e. it called roughly
            // one in eighteen clean files watermarked, in the past tense.
            if (crispasr_watermark_detect_uses_frames()) {
                // Per-frame t + decoy specificity. The score is a calibrated
                // confidence, not a bin count, so the binomial null does not
                // apply and no p-value is printed — quoting one would invent an
                // n that was never scored.
                const auto verdict = crispasr_wm_stats::classify_frames(confidence);
                fprintf(stdout, "Statistic: per-frame t + decoy specificity\n");
                fprintf(stdout, "Result: %s\n", crispasr_wm_stats::verdict_line(verdict));
                if (verdict != crispasr_wm_stats::Verdict::Detected && dur_s < 1.0) {
                    fprintf(stdout,
                            "Note: %.2f s is below the %d-frame minimum this statistic needs; it returns 0\n"
                            "      rather than guess. A negative is not evidence the audio is human-made.\n",
                            dur_s, crispasr_wm::kDetectMinFrames);
                }
            } else {
                const double p = crispasr_wm_stats::p_value(confidence, CRISPASR_WATERMARK_NBINS);
                const auto verdict = crispasr_wm_stats::classify(confidence, CRISPASR_WATERMARK_NBINS);
                fprintf(stdout, "Statistic: bin-sign agreement (legacy)\n");
                fprintf(stdout, "Chance of this score without a watermark: %.2g\n", p);
                fprintf(stdout, "Result: %s\n", crispasr_wm_stats::verdict_line(verdict));
                if (verdict != crispasr_wm_stats::Verdict::Detected && dur_s < 10.0) {
                    fprintf(stdout,
                            "Note: %.1f s is short for this detector — it averages spectra across frames, so\n"
                            "      confidence grows with duration (measured: 69%% of 1 s clips vs 100%% of 10 s\n"
                            "      clips clear the 0.65 bar). Unset CRISPASR_WATERMARK_DETECT to use the\n"
                            "      per-frame statistic, which reads 97%% at 1 s with fewer false positives.\n",
                            dur_s);
                }
            }
        }

        crispasr_wm_dispatch::shutdown();
        return 0;
    }

    // ── --align-only: standalone CTC forced alignment (issue #217) ─────────
    // Runs the CTC aligner on user-provided text + audio without ASR.
    // Accepts text from --ref-text or --text-file (.txt or .srt).
    if (params.align_only) {
        // Resolve aligner model.
        std::string am = params.aligner_model;
        if (am.empty() || am == "auto" || am == "default") {
            am = crispasr_resolve_model_cli(am.empty() ? "auto" : am, "canary-ctc-aligner", params.no_prints,
                                            params.cache_dir, params.auto_download);
        } else {
            const std::string resolved =
                crispasr_resolve_model_cli(am, "", params.no_prints, params.cache_dir, params.auto_download);
            if (!resolved.empty())
                am = resolved;
        }
        if (am.empty()) {
            fprintf(stderr, "crispasr[align-only]: no aligner model. Pass -am <path.gguf> "
                            "or -am auto --auto-download.\n");
            return 10;
        }

        // Validate output granularity.
        const std::string& gran = params.align_granularity;
        if (gran != "auto" && gran != "word" && gran != "segment") {
            fprintf(stderr, "crispasr[align-only]: invalid --align-granularity '%s' (auto|word|segment)\n",
                    gran.c_str());
            return 10;
        }

        // Read transcript text as segments: SRT cues, non-empty .txt lines,
        // or the --ref-text string as a single segment. The flat transcript
        // fed to the aligner is the segments joined with spaces, so aligned
        // words can be re-grouped per segment afterwards.
        std::vector<std::string> segment_texts;
        bool is_srt_input = false;
        bool is_json_input = false;
        auto trim = [](std::string s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
                s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
                s.pop_back();
            return s;
        };
        if (!params.text_file.empty()) {
            std::string raw;
            // "-" means stdin (#317). An embedder like Subtitle Edit already
            // holds the transcript in memory and would otherwise have to spill
            // it to a temp file purely to hand it over — a temp file it then
            // owns, has to name uniquely, and has to clean up. Piping is one
            // less thing to get wrong, and costs nothing when unused.
            if (params.text_file == "-") {
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0)
                    raw.append(buf, n);
                if (ferror(stdin)) {
                    fprintf(stderr, "crispasr[align-only]: error reading transcript from stdin\n");
                    return 10;
                }
                if (raw.empty()) {
                    fprintf(stderr, "crispasr[align-only]: --text-file - was given but stdin was empty.\n");
                    return 10;
                }
            } else {
                FILE* tf = fopen(params.text_file.c_str(), "rb");
                if (!tf) {
                    fprintf(stderr, "crispasr[align-only]: cannot open text file '%s'\n", params.text_file.c_str());
                    return 10;
                }
                fseek(tf, 0, SEEK_END);
                long sz = ftell(tf);
                fseek(tf, 0, SEEK_SET);
                raw.assign((size_t)sz, '\0');
                if ((long)fread(&raw[0], 1, sz, tf) != sz) {
                    fprintf(stderr, "crispasr[align-only]: short read from '%s'\n", params.text_file.c_str());
                    fclose(tf);
                    return 10;
                }
                fclose(tf);
            }

            // Detect format by extension; stdin has no extension so sniff
            // content instead. Order: .srt, .json, then plain .txt fallback.
            const std::string& p = params.text_file;
            is_srt_input = (p.size() >= 4 && (p.substr(p.size() - 4) == ".srt" || p.substr(p.size() - 4) == ".SRT"));
            if (p == "-")
                is_srt_input = raw.find(" --> ") != std::string::npos;

            // #317: detect JSON — by extension or by content sniff for stdin.
            // Accepts CrispASR --output-json transcription and the align-only
            // JSON format, extracting the "text" field from each segment.
            is_json_input = (p.size() >= 5 && (p.substr(p.size() - 5) == ".json" || p.substr(p.size() - 5) == ".JSON"));
            if (p == "-" && !is_srt_input) {
                // Sniff: JSON starts with '{' or '[' (ignoring whitespace).
                for (size_t c = 0; c < raw.size(); c++) {
                    if (raw[c] == ' ' || raw[c] == '\t' || raw[c] == '\n' || raw[c] == '\r')
                        continue;
                    is_json_input = (raw[c] == '{' || raw[c] == '[');
                    break;
                }
            }

            if (is_srt_input) {
                for (auto& cue : crispasr_parse_srt_cues(raw))
                    segment_texts.push_back(trim(std::move(cue)));
            } else if (is_json_input) {
                segment_texts = crispasr_parse_json_segments(raw);
                if (segment_texts.empty()) {
                    fprintf(stderr, "crispasr[align-only]: JSON input has no 'text' fields. "
                                    "Expected CrispASR --output-json format with a \"transcription\" array, "
                                    "or an array of {\"text\": \"...\"}.\n");
                    return 10;
                }
                if (!params.no_prints)
                    fprintf(stderr, "crispasr[align-only]: parsed %zu segment(s) from JSON input\n",
                            segment_texts.size());
            } else {
                // Plain .txt: each non-empty line is one segment.
                size_t i = 0;
                while (i < raw.size()) {
                    size_t nl = raw.find('\n', i);
                    if (nl == std::string::npos)
                        nl = raw.size();
                    std::string line = raw.substr(i, nl - i);
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    i = nl + 1;
                    line = trim(std::move(line));
                    if (!line.empty())
                        segment_texts.push_back(std::move(line));
                }
            }
        } else if (!params.tts_ref_text.empty()) {
            std::string t = trim(params.tts_ref_text);
            if (!t.empty())
                segment_texts.push_back(std::move(t));
        } else {
            // #317: two people hit this and read it as the aligner failing. It
            // is not — alignment needs a transcript to align, and only the
            // caller has one. Say what to pass, since the answer is short.
            fprintf(stderr, "crispasr[align-only]: no transcript given — alignment needs the text to align.\n"
                            "  --text-file <file.txt>   one segment per line\n"
                            "  --text-file <file.srt>   re-time existing cues (text kept, timings discarded)\n"
                            "  --text-file <file.json>  CrispASR JSON output (--output-json); extracts segment texts\n"
                            "  --text-file -            read from stdin (auto-detects SRT/JSON/plain text)\n"
                            "  --ref-text \"...\"         a single segment inline\n"
                            "Only -am (the aligner model) is needed besides; -m/--backend, --vad and\n"
                            "--max-len belong to transcription and are unused here.\n");
            return 10;
        }

        std::string transcript;
        for (const auto& s : segment_texts) {
            if (!transcript.empty())
                transcript += ' ';
            transcript += s;
        }

        if (transcript.empty()) {
            fprintf(stderr, "crispasr[align-only]: transcript is empty.\n");
            return 10;
        }

        // segment output: explicit, or auto for structured input (SRT cues / JSON segments).
        const bool segment_mode = gran == "segment" || (gran == "auto" && (is_srt_input || is_json_input));

        // Load audio.
        if (params.fname_inp.empty()) {
            fprintf(stderr, "crispasr[align-only]: requires an audio file (-f <audio.wav>).\n");
            return 10;
        }
        std::vector<float> samples;
        std::vector<std::vector<float>> stereo_dummy;
        if (!read_audio_data(params.fname_inp[0], samples, stereo_dummy, false)) {
            fprintf(stderr, "crispasr[align-only]: failed to load audio '%s'\n", params.fname_inp[0].c_str());
            return 10;
        }
        if (!params.no_prints) {
            fprintf(stderr, "crispasr[align-only]: aligner=%s\n", am.c_str());
            fprintf(stderr, "crispasr[align-only]: audio=%.2fs (%d samples @ 16kHz)\n",
                    (float)samples.size() / 16000.0f, (int)samples.size());
            fprintf(stderr, "crispasr[align-only]: transcript='%.80s%s'\n", transcript.c_str(),
                    transcript.size() > 80 ? "…" : "");
        }

        // Run alignment.
        auto aligned = crispasr_align_words(am, transcript, samples.data(), (int)samples.size(), /*t_offset_cs=*/0,
                                            params.n_threads);
        if (aligned.empty()) {
            fprintf(stderr, "crispasr[align-only]: alignment failed or produced no words.\n");
            return 10;
        }

        // Group words back into the input segments when requested.
        std::vector<CrispasrAlignedSegment> segments;
        if (segment_mode) {
            segments = crispasr_group_aligned_segments(segment_texts, aligned);
            if (segments.empty()) {
                fprintf(stderr, "crispasr[align-only]: segment grouping produced no segments.\n");
                return 10;
            }
        }

        // Format output.
        const double clip_end = (double)samples.size() / 16000.0;
        auto ts = [](double s, bool comma) -> std::string {
            if (s < 0)
                s = 0;
            int h = (int)(s / 3600);
            int m = (int)((s - h * 3600) / 60);
            int sec = (int)(s - h * 3600 - m * 60);
            int ms = (int)((s - (int)s) * 1000 + 0.5);
            char buf[32];
            snprintf(buf, sizeof(buf), "%02d:%02d:%02d%c%03d", h, m, sec, comma ? ',' : '.', ms);
            return buf;
        };
        auto json_esc = [](const std::string& s) {
            std::string esc;
            for (char c : s) {
                if (c == '"' || c == '\\')
                    esc += '\\';
                esc += c;
            }
            return esc;
        };
        std::string out;
        const std::string& fmt = params.align_format;
        if (fmt == "json") {
            char num[96];
            out = "[\n";
            if (segment_mode) {
                for (size_t i = 0; i < segments.size(); i++) {
                    const auto& seg = segments[i];
                    snprintf(num, sizeof(num), "\"start\": %.3f, \"end\": %.3f, \"words\": [", seg.t0_cs / 100.0,
                             seg.t1_cs / 100.0);
                    out += "  {\"text\": \"" + json_esc(seg.text) + "\", " + num;
                    for (size_t w = seg.word_begin; w < seg.word_end; w++) {
                        snprintf(num, sizeof(num), "\"start\": %.3f, \"end\": %.3f}", aligned[w].t0_cs / 100.0,
                                 aligned[w].t1_cs / 100.0);
                        out += std::string(w > seg.word_begin ? ", " : "") + "{\"word\": \"" +
                               json_esc(aligned[w].text) + "\", " + num;
                    }
                    out += std::string("]}") + (i + 1 < segments.size() ? "," : "") + "\n";
                }
            } else {
                for (size_t i = 0; i < aligned.size(); i++) {
                    snprintf(num, sizeof(num), "\"start\": %.3f, \"end\": %.3f}%s\n", aligned[i].t0_cs / 100.0,
                             aligned[i].t1_cs / 100.0, i + 1 < aligned.size() ? "," : "");
                    out += "  {\"word\": \"" + json_esc(aligned[i].text) + "\", " + num;
                }
            }
            out += "]\n";
        } else if (fmt == "plain") {
            if (segment_mode) {
                for (auto& seg : segments)
                    out += ts(seg.t0_cs / 100.0, false) + "\t" + seg.text + "\n";
            } else {
                for (auto& w : aligned)
                    out += ts(w.t0_cs / 100.0, false) + "\t" + w.text + "\n";
            }
        } else { // srt (default)
            if (segment_mode) {
                for (size_t i = 0; i < segments.size(); i++) {
                    double t0 = segments[i].t0_cs / 100.0;
                    double t1 = segments[i].t1_cs / 100.0;
                    if (t1 <= t0)
                        t1 = (i + 1 < segments.size()) ? segments[i + 1].t0_cs / 100.0 : clip_end;
                    out += std::to_string(i + 1) + "\n" + ts(t0, true) + " --> " + ts(t1, true) + "\n" +
                           segments[i].text + "\n\n";
                }
            } else {
                for (size_t i = 0; i < aligned.size(); i++) {
                    double t0 = aligned[i].t0_cs / 100.0;
                    double t1 = aligned[i].t1_cs / 100.0;
                    if (t1 <= t0)
                        t1 = (i + 1 < aligned.size()) ? aligned[i + 1].t0_cs / 100.0 : clip_end;
                    out += std::to_string(i + 1) + "\n" + ts(t0, true) + " --> " + ts(t1, true) + "\n" +
                           aligned[i].text + "\n\n";
                }
            }
        }

        if (params.align_output.empty()) {
            fputs(out.c_str(), stdout);
        } else {
            FILE* f = fopen(params.align_output.c_str(), "wb");
            if (!f) {
                fprintf(stderr, "crispasr[align-only]: cannot write '%s'\n", params.align_output.c_str());
                return 10;
            }
            fwrite(out.data(), 1, out.size(), f);
            fclose(f);
            if (!params.no_prints) {
                if (segment_mode)
                    fprintf(stderr, "crispasr[align-only]: %zu words in %zu segments → %s\n", aligned.size(),
                            segments.size(), params.align_output.c_str());
                else
                    fprintf(stderr, "crispasr[align-only]: %zu words → %s\n", aligned.size(),
                            params.align_output.c_str());
            }
        }
        crispasr_aligner_free_cache();
        return 0;
    }

    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: model arg          = '%s'\n", params.model.c_str());
        fprintf(stderr, "crispasr[verbose]: backend arg        = '%s'\n",
                params.backend.empty() ? "auto" : params.backend.c_str());
        fprintf(stderr, "crispasr[verbose]: use_gpu            = %s\n", params.use_gpu ? "true" : "false");
        fprintf(stderr, "crispasr[verbose]: gpu_backend        = '%s'\n",
                params.gpu_backend.empty() ? "auto" : params.gpu_backend.c_str());
        fprintf(stderr, "crispasr[verbose]: gpu_device         = %d\n", params.gpu_device);
        fprintf(stderr, "crispasr[verbose]: cache_dir override = '%s'\n",
                params.cache_dir.empty() ? "(default)" : params.cache_dir.c_str());
        fprintf(stderr, "crispasr[verbose]: auto_download      = %s\n", params.auto_download ? "true" : "false");
        fprintf(stderr, "crispasr[verbose]: n_threads          = %d\n", params.n_threads);
        fprintf(stderr, "crispasr[verbose]: flash_attn         = %s\n", params.flash_attn ? "true" : "false");
    }

    // Resolve backend name: explicit --backend takes priority; otherwise
    // auto-detect from the GGUF file. Defaults are handled in cli.cpp.
    std::string backend_name = params.backend;
    const bool model_is_auto = params.model == "auto" || params.model == "default";
    if (backend_name.empty() || backend_name == "auto") {
        if (model_is_auto) {
            // `-m auto` with no --backend. Before defaulting to
            // whisper-download, scan the cache for any already-downloaded
            // registered model (whisper > parakeet > canary > …). Users
            // who already have, say, a parakeet GGUF from a previous
            // session shouldn't trigger a fresh 147 MB whisper download.
            CrispasrRegistryEntry cached;
            if (crispasr_find_cached_model(cached, params.cache_dir, params.model_quant)) {
                backend_name = cached.backend;
                params.model = crispasr_cache::dir(params.cache_dir) + "/" + cached.filename;
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: -m auto — using cached %s model (%s)\n", backend_name.c_str(),
                            cached.filename.c_str());
                }
            } else {
                backend_name = "whisper";
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: -m auto with no cached model — defaulting to whisper\n");
                }
            }
        } else {
            backend_name = crispasr_detect_backend_from_gguf(params.model);
            if (backend_name.empty()) {
                fprintf(stderr,
                        "crispasr: error: could not auto-detect backend from '%s'. "
                        "Use --backend NAME to force one.\n",
                        params.model.c_str());
                return 10;
            }
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: detected backend '%s' from GGUF metadata\n", backend_name.c_str());
            }
        }
    }

    // PLAN #74a — chatterbox-family auto-route by --language. Pure
    // English variants ("chatterbox" / "chatterbox-turbo" / aliases)
    // get swapped to the language-matching sibling when the user passes
    // `-l de` (kartoffelbox-turbo) or `-l ar` (lahgtna-chatterbox), but
    // only when -m auto is in effect — if the user passed an explicit
    // model path they've already picked the variant. Mirrors the kokoro
    // `-l de` German-backbone routing convention. No-op when the user
    // already named a language-specific chatterbox variant.
    if (model_is_auto && !params.language.empty() && params.language != "auto") {
        auto is_en_chatterbox = [](const std::string& n) {
            return n == "chatterbox" || n == "chatterbox-tts" || n == "chatterbox-base" || n == "chatterbox-turbo" ||
                   n == "chatterbox_turbo";
        };
        if (is_en_chatterbox(backend_name)) {
            std::string routed;
            if (params.language == "de") {
                routed = "kartoffelbox-turbo";
            } else if (params.language == "ar") {
                routed = "lahgtna-chatterbox";
            }
            if (!routed.empty()) {
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: -l %s with --backend %s — auto-routing to %s\n", params.language.c_str(),
                            backend_name.c_str(), routed.c_str());
                }
                backend_name = routed;
                params.backend = routed;
            }
        }
    }

    // #231 — "cohere-ar" is the Arabic shorthand for the cohere backend
    // (routes to the same runtime; the registry resolves the recommended
    // Arabic imatrix GGUF for `-m auto`). Default the language to "ar" so
    // `--backend cohere-ar audio.wav` works without also requiring `-l ar`.
    // Only fires when the user hasn't already picked a language (matches
    // the chatterbox block's "auto" == unset convention above); an explicit
    // `-l <lang>` always wins, e.g. for LID experiments against the model.
    if (backend_name == "cohere-ar" && (params.language.empty() || params.language == "auto")) {
        params.language = "ar";
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: --backend cohere-ar — defaulting language to 'ar' (pass -l to override)\n");
        }
    }

    // MUST STAY ABOVE crispasr_resolve_model_cli(): that call downloads the
    // model, so leaving this below it made --vad-export fetch ggml-base.bin
    // before exiting -- the standalone verb still required a network round trip
    // and a model the run never used (issue #227, reported by AppleSheeple, who
    // worked around it with `-m /dev/null`).
    // Issue #227: VAD-export-only short circuit. --vad-export computes
    // speech boundaries and writes them to a JSON file — no ASR model
    // needed. The user can import the result on a second run with
    // --vad-import. Loading the audio and running Silero VAD is cheap;
    // loading an ASR backend is not, so we return before backend init.
    if (!params.vad_export_file.empty()) {
        int vad_rc = 0;
        for (size_t fi = 0; fi < params.fname_inp.size(); fi++) {
            const auto& fname = params.fname_inp[fi];
            std::vector<float> samples;
            std::vector<std::vector<float>> stereo_dummy;
            if (!read_audio_data(fname, samples, stereo_dummy, false)) {
                fprintf(stderr, "crispasr: error: cannot read audio '%s'\n", fname.c_str());
                vad_rc = 20;
                continue;
            }
            constexpr int SR = 16000;
            // Raw export computes with chunk 0 so no post-split runs -- the
            // result is merged VAD speech segments, independent of chunk length.
            const float slice_chunk = params.vad_export_raw         ? 0.0f
                                      : params.chunk_seconds > 0.0f ? params.chunk_seconds
                                                                    : 30.0f;
            bool export_vad_load_failed = false;
            auto slices = crispasr_compute_audio_slices(samples.data(), (int)samples.size(), SR, (int)slice_chunk,
                                                        params, &export_vad_load_failed);
            // #311 follow-up: --vad-export is a separate verb from the
            // transcribe path, and only that path carried the strict check. A
            // VAD model that failed to load here still wrote a file full of
            // fixed chunk boundaries labelled as if VAD had produced them —
            // and, with --strict-pipeline, still exited 0.
            if (export_vad_load_failed && crispasr_compute_strict_reqs(params).vad) {
                fprintf(stderr,
                        "crispasr: error: required VAD model '%s' failed to load for '%s' "
                        "(--require-vad/--strict-pipeline) — refusing to export fixed chunks as VAD output.\n",
                        params.vad_model.c_str(), fname.c_str());
                return CRISPASR_STRICT_RC_VAD;
            }
            // Multi-file: each input gets its own export path derived
            // from the input name. Single-file: use the explicit path.
            std::string export_path = params.vad_export_file;
            if (params.fname_inp.size() > 1) {
                export_path = crispasr_make_out_path(fname, ".vad.json");
            }
            std::ofstream out(export_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                fprintf(stderr, "crispasr: warning: cannot write --vad-export file '%s'\n", export_path.c_str());
                vad_rc = 1;
            } else {
                out << crispasr_serialize_vad_slices(slices, SR, slice_chunk, params.vad_export_raw);
                if (!params.no_prints) {
                    // Say which kind: they are NOT the same thing, and calling a
                    // 30 s chunk a "VAD segment" is what made this confusing in
                    // the first place (issue #227).
                    fprintf(stderr, "crispasr: exported %zu %s to '%s'\n", slices.size(),
                            params.vad_export_raw ? "VAD speech segment(s)" : "chunk boundary/boundaries",
                            export_path.c_str());
                }
            }
        }
        return vad_rc;
    }

    // Resolve "-m auto" via the model registry + curl/wget download.
    const std::string resolved = crispasr_resolve_model_cli(params.model, backend_name, params.no_prints,
                                                            params.cache_dir, params.auto_download, params.model_quant);
    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: resolved model     = '%s'\n", resolved.c_str());
    }
    if (resolved.empty()) {
        return 11;
    }
    params.model = resolved;

    // A pure-CTC FastConformer model (parakeet-ctc-*, stt_*_fastconformer_ctc:
    // encoder + CTC head, no RNN-T decoder/joint) cannot run on the parakeet
    // (transducer) backend. Autodetection already routes such GGUFs to
    // fastconformer-ctc by arch ("canary-ctc") and filename, but an explicit
    // `--backend parakeet` bypasses that and dead-ends at the parakeet guard.
    // Reroute here so the transducer-only backend never gets a CTC model.
    if (backend_name == "parakeet" && crispasr_gguf_is_pure_ctc(params.model)) {
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: '%s' is a pure-CTC model (no RNN-T decoder) — the parakeet\n"
                    "crispasr: backend is transducer-only; auto-routing to --backend fastconformer-ctc\n",
                    params.model.c_str());
        }
        backend_name = "fastconformer-ctc";
        params.backend = "fastconformer-ctc";
    }

    // Issue #125 follow-up: when the LM has a companion file in the
    // registry (e.g. mimo-tokenizer-q4_k.gguf for mimo-asr), fetch it now
    // so `--auto-download` produces a fully-functional setup. Previously
    // companion-fetch was wired only into TTS backends (chatterbox /
    // orpheus / indextts / qwen3-tts), so ASR backends with a hard
    // companion dependency (mimo-asr) hit "not found" errors even with
    // --auto-download set. Doing it here in the dispatcher covers every
    // current and future backend uniformly. Companion lands in the same
    // cache_dir as the LM so the backend's local `discover_*` finds it.
    //
    // Fix for #146 / #148: skip the companion pre-download when the user
    // already told us where the codec is (--codec-model), or when the
    // companion already sits next to the model file or in the cache dir
    // (the backend's discover_* will find it without a download prompt).
    if (!backend_name.empty() && params.tts_codec_model.empty()) {
        CrispasrRegistryEntry entry;
        if (crispasr_registry_lookup(backend_name, entry, params.model_quant) && !entry.companion_filename.empty()) {
            // Check whether the companion already exists locally before
            // triggering the resolve → download-prompt path:
            //   1. next to the model file (sibling directory)
            //   2. in the cache dir / well-known search dirs
            bool companion_found = false;
            {
                const auto sep = params.model.find_last_of("/\\");
                if (sep != std::string::npos) {
                    const std::string sibling = params.model.substr(0, sep + 1) + entry.companion_filename;
                    FILE* f = fopen(sibling.c_str(), "rb");
                    if (f) {
                        fclose(f);
                        companion_found = true;
                    }
                }
            }
            if (!companion_found) {
                const std::string cached =
                    crispasr_cache::probe_cached_file(entry.companion_filename, params.cache_dir);
                if (!cached.empty())
                    companion_found = true;
            }

            if (!companion_found) {
                const std::string resolved_companion = crispasr_resolve_model_cli(
                    entry.companion_filename, backend_name, params.no_prints, params.cache_dir, params.auto_download,
                    params.tts_codec_quant.empty() ? params.model_quant : params.tts_codec_quant);
                if (params.verbose) {
                    fprintf(stderr, "crispasr[verbose]: resolved companion = '%s'\n", resolved_companion.c_str());
                }
            } else if (params.verbose) {
                fprintf(stderr, "crispasr[verbose]: companion '%s' found locally, skipping download\n",
                        entry.companion_filename.c_str());
            }
            // Soft-fail: backend init prints its own actionable error if
            // the companion is genuinely required and didn't resolve.
        }
    }

    // SubtitleEdit #10775: implicit `-am auto --force-aligner` for
    // canary when the user requests word-level output but didn't
    // pass an aligner. Canary's native timing is cross-attention DTW
    // on the encoder–decoder, MAE ~414 ms on word boundaries
    // (canary.cpp:1377-1390 / canary-ctc-aligner-GGUF README).
    // The official NeMo Forced Aligner companion model
    // (canary-ctc-aligner-q4_k, ~442 MB, separate FastConformer+CTC
    // head) gives ~78 ms MAE — 5.3× tighter — and is the path NVIDIA
    // recommends. Users who want the legacy DTW timing can opt out
    // with --no-auto-aligner.
    //
    // Gates:
    //   - backend is canary (the only backend with a documented MAE
    //     gap that big AND a curated aligner sibling in the registry)
    //   - --aligner-model not set (we don't override an explicit one)
    //   - --no-auto-aligner not set
    //   - output type that benefits from word ts (srt/vtt/json-full/
    //     wts/max-len/split-on-punct/print-colors). Plain transcript
    //     stdout doesn't pay the second-forward-pass cost.
    //   - not stream/mic/server/text-only/TTS-only mode
    if (backend_name == "canary" && params.aligner_model.empty() && !params.no_auto_aligner && !params.stream &&
        !params.mic && !params.server && params.text_input.empty() && params.tts_text.empty()) {
        const bool wants_word_ts = params.output_srt || params.output_vtt || params.output_jsn_full ||
                                   params.output_wts || params.split_on_punct || params.max_len > 0 ||
                                   params.print_colors;
        if (wants_word_ts) {
            params.aligner_model = "auto";
            params.force_aligner = true;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: canary auto-aligner: enabling `-am auto --force-aligner` "
                                "(canary-ctc-aligner, ~442 MB; ~78 ms MAE vs ~414 ms for native DTW). "
                                "Pass --no-auto-aligner to disable.\n");
            }
            if (params.verbose) {
                fprintf(stderr, "crispasr[verbose]: auto-aligner: backend=canary wants_word_ts=1 "
                                "explicit_aligner=0 -> aligner_model='auto' force_aligner=1\n");
            }
        }
    }

    // Issue #62: `-am auto` resolves to the registered CTC aligner
    // (canary-ctc-aligner-q4_k, ~442 MB). Same registry / cache /
    // download path as -m auto. Lets users add force-alignment without
    // hunting for an aligner GGUF first.
    if (params.aligner_model == "auto" || params.aligner_model == "default") {
        const std::string resolved_aligner = crispasr_resolve_model_cli(
            params.aligner_model, "canary-ctc-aligner", params.no_prints, params.cache_dir, params.auto_download);
        if (resolved_aligner.empty()) {
            fprintf(stderr, "crispasr: error: failed to resolve `-am auto` (canary-ctc-aligner)\n");
            return 19;
        }
        if (params.verbose) {
            fprintf(stderr, "crispasr[verbose]: resolved aligner   = '%s'\n", resolved_aligner.c_str());
        }
        params.aligner_model = resolved_aligner;
    } else if (!params.aligner_model.empty()) {
        const std::string resolved_aligner = crispasr_resolve_model_cli(params.aligner_model, "", params.no_prints,
                                                                        params.cache_dir, params.auto_download);
        if (params.verbose && resolved_aligner != params.aligner_model) {
            fprintf(stderr, "crispasr[verbose]: resolved aligner   = '%s'\n", resolved_aligner.c_str());
        }
        params.aligner_model = resolved_aligner;
    }

    // Query-time inline voice cloning for TADA: if the user asks to synthesize
    // (--tts) with a .wav voice reference, bake the reference GGUF in-memory here
    // (before the backend loads) and rewrite --voice to it, so the two-step
    // "--make-ref then --voice ref.gguf" collapses to one command. Requires
    // --ref-text (the transcript drives the alignment). #201 follow-up.
    {
        const bool is_tada = backend_name == "tada" || backend_name == "tada-tts" || backend_name == "tada-1b" ||
                             backend_name == "tada-tts-1b" || backend_name == "tada-3b" || backend_name == "tada-3b-ml";
        const std::string& v = params.tts_voice;
        const bool voice_is_wav =
            v.size() >= 4 && (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".WAV");
        if (is_tada && !params.tts_text.empty() && voice_is_wav && !params.make_ref && !params.align) {
            if (params.tts_ref_text.empty()) {
                fprintf(stderr, "crispasr[tada]: cloning from a .wav needs --ref-text \"exact transcript of %s\".\n",
                        v.c_str());
                return 20;
            }
            tada_encoder_result result;
            if (tada_run_aligner_pipeline(params, v, params.tts_ref_text, "tada-clone", result) != 0)
                return 20;
            // Bake to a temp ref GGUF in the cache dir, keyed so repeat runs reuse it.
            std::string tmp = crispasr_cache::dir(params.cache_dir) + "/tada-inline-voice.gguf";
            if (tada_encoder_write_ref_gguf(tmp.c_str(), result, params.tts_ref_text.c_str(),
                                            params.language.empty() ? nullptr : params.language.c_str(),
                                            /*cloned_from_recording=*/true,
                                            params.tts_consent_attestation.c_str()) != 0) {
                fprintf(stderr, "crispasr[tada-clone]: failed to write reference GGUF\n");
                return 20;
            }
            // Guard against a silent write no-op (e.g. an unwritable/dangling cache
            // dir): if the ref didn't actually land, fail loudly rather than
            // synthesizing in the default voice.
            struct stat rst;
            if (stat(tmp.c_str(), &rst) != 0 || rst.st_size == 0) {
                fprintf(stderr,
                        "crispasr[tada-clone]: baked ref did not land at '%s' (cache dir unwritable?). "
                        "Pass --cache-dir <writable-dir>.\n",
                        tmp.c_str());
                return 20;
            }
            if (!params.no_prints)
                fprintf(stderr, "crispasr[tada-clone]: baked voice ref (%d tokens) → %s\n", result.n_tokens,
                        tmp.c_str());
            // Remember the recording BEFORE the rewrite: consent was given for
            // that file, so it is what the audit record hashes. The baked pack
            // is derived and not byte-stable across runs.
            params.tts_voice_source_recording = params.tts_voice;
            params.tts_voice = tmp; // backend init() now sees a .gguf reference
            // The rewrite above erases the ONLY evidence this voice is a clone:
            // the consent + spoken-disclosure gates below classify by suffix, so
            // `--voice victim.wav` silently became a .gguf and stopped being a
            // clone — no --i-have-rights demanded, no [CONSENT] line, no audible
            // AI disclosure, on the flow docs/tts.md documents as the one-command
            // clone. Remember it explicitly instead.
            params.tts_voice_baked_from_wav = true;
        }
    }

    // Create and init the backend.
    std::unique_ptr<CrispasrBackend> backend = crispasr_create_backend(backend_name);
    if (!backend) {
        fprintf(stderr, "crispasr: error: backend '%s' is not available in this build\n", backend_name.c_str());
        return 12;
    }

    warn_unsupported(*backend, params);

    if (!backend->init(params)) {
        fprintf(stderr, "crispasr: error: failed to initialise backend '%s'\n", backend_name.c_str());
        return 13;
    }
    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: backend '%s' initialised OK\n", backend_name.c_str());
    }

    // #80e: optional warmup — transcribe a short silence buffer to
    // amortize first-call overhead (graph alloc, GPU kernel compile).
    // Enabled via --warmup, CRISPASR_WARMUP=1, or automatically in
    // JSON+VAD streaming mode (--stream-json) to eliminate startup
    // latency spikes from GPU kernel compilation.
    if (params.stream_json || params.warmup || getenv("CRISPASR_WARMUP")) {
        auto t_warmup_start = std::chrono::steady_clock::now();
        backend->warmup();
        if (params.verbose || !params.no_prints) {
            auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warmup_start).count();
            fprintf(stderr, "crispasr: warmup completed in %.0f ms\n", dt * 1000.0);
        }
    }

    // SubtitleEdit #10775: when a user explicitly passes --aligner-model
    // for a backend that already produces native word timestamps, the
    // alignment dispatch loop (crispasr_run.cpp:305 / :398 / :1252)
    // skips the CTC pass per-segment because seg.words is already
    // populated. Result: aligner is loaded into memory and silently
    // becomes a no-op; the user sees the backend's native timing and
    // assumes "alignment doesn't work for <backend>." Make the
    // requirement explicit at startup so the next user doesn't have
    // to read the dispatch code to figure it out.
    if (!params.aligner_model.empty() && !params.force_aligner && (backend->capabilities() & CAP_WORD_TIMESTAMPS) &&
        !params.no_prints) {
        fprintf(stderr,
                "crispasr: warning: --aligner-model is set, but backend '%s' already produces "
                "native word timestamps. The aligner will be loaded but skipped per-segment "
                "(CTC alignment only runs when no native words are available). Add "
                "--force-aligner / -falign to override the native words with the CTC aligner's "
                "output.\n",
                backend_name.c_str());
    }

    // ---- Text-to-text translation mode: m2m100 standalone ----
    // Triggered by `--text "..."` on a backend declaring CAP_TRANSLATE.
    // Source / target languages: use the dedicated --tr-sl / --tr-tl
    // when set (for 2-stage pipelines), otherwise fall back to -sl /
    // -tl. Result goes to stdout. See PLAN #74 / m2m100 wiring notes.
    if (!params.text_input.empty()) {
        if (!(backend->capabilities() & CAP_TRANSLATE)) {
            fprintf(stderr,
                    "crispasr: error: backend '%s' does not support text-to-text translation "
                    "(missing CAP_TRANSLATE)\n",
                    backend_name.c_str());
            return 16;
        }
        const std::string& src =
            !params.translate_source_lang.empty() ? params.translate_source_lang : params.source_lang;
        const std::string& tgt =
            !params.translate_target_lang.empty() ? params.translate_target_lang : params.target_lang;
        if (src.empty() || tgt.empty()) {
            fprintf(stderr, "crispasr: error: --text requires source + target language. Pass `-sl <code> "
                            "-tl <code>` (or `--tr-sl` / `--tr-tl` for 2-stage pipes).\n");
            return 17;
        }
        std::string out = backend->translate_text(params.text_input, src, tgt, params);
        if (out.empty()) {
            fprintf(stderr, "crispasr: error: translation failed\n");
            return 18;
        }
        printf("%s\n", out.c_str());
        return 0;
    }

    // ---- make-ref / align mode: TADA aligner pipeline ----
    // --make-ref writes a voice reference GGUF; --align emits forced-alignment
    // word timestamps. Both share the encoder+aligner resolution, audio load,
    // and encode; they differ only in what they do with the result.
    if (params.make_ref || params.align) {
        const char* verb = params.make_ref ? "make-ref" : "align";
        if (params.tts_voice.empty()) {
            fprintf(stderr, "crispasr[%s]: requires --voice <audio.wav>\n", verb);
            return 20;
        }
        if (params.tts_ref_text.empty()) {
            fprintf(stderr, "crispasr[%s]: requires --ref-text \"transcript of the audio\"\n", verb);
            return 20;
        }
        // --make-ref extracts a reusable voiceprint from a person's recording —
        // the clone itself, one step ahead of synthesis. It sat before the TTS
        // block's consent gate and returned early, so it was the one way to build
        // a clone with no attestation demanded anywhere. --align only emits word
        // timestamps and stays ungated.
        if (params.make_ref && !params.tts_voice_clone_consent) {
            fprintf(stderr,
                    "crispasr[make-ref]: building a voice reference requires the --i-have-rights flag.\n"
                    "\n"
                    "  --make-ref extracts a reusable voiceprint from '%s'. By passing\n"
                    "  --i-have-rights you attest:\n"
                    "  \"I have the consent of the speaker whose voice this clones,\n"
                    "   or it is my own voice.\"\n",
                    params.tts_voice.c_str());
            return 17;
        }

        const std::string out_path = params.make_ref_output.empty() ? "tada-ref-custom.gguf" : params.make_ref_output;

        tada_encoder_result result;
        double audio_seconds = 0.0;
        if (tada_run_aligner_pipeline(params, params.tts_voice, params.tts_ref_text, verb, result, &audio_seconds) != 0)
            return 20;
        int rc = 0;

        // --align: emit forced-alignment word timestamps and exit.
        if (params.align) {
            const double fps = result.frame_rate > 0 ? (double)result.frame_rate : 50.0;
            // Group BPE tokens into words: a token whose decoded text starts with
            // a space (GPT-2 space marker) begins a new word.
            struct Word {
                std::string text;
                double start;
            };
            std::vector<Word> words;
            for (int i = 0; i < result.n_tokens; i++) {
                const std::string& tk = i < (int)result.token_texts.size() ? result.token_texts[i] : std::string();
                double t = (i < (int)result.token_positions.size() ? result.token_positions[i] : 0.0f) / fps;
                bool new_word = words.empty() || (!tk.empty() && tk[0] == ' ');
                if (new_word) {
                    std::string w = tk;
                    if (!w.empty() && w[0] == ' ')
                        w.erase(0, 1);
                    words.push_back({w, t});
                } else {
                    words.back().text += tk;
                }
            }
            const double clip_end = audio_seconds;
            auto ts = [](double s, bool comma) -> std::string {
                if (s < 0)
                    s = 0;
                int h = (int)(s / 3600);
                int m = (int)((s - h * 3600) / 60);
                int sec = (int)(s - h * 3600 - m * 60);
                int ms = (int)((s - (int)s) * 1000 + 0.5);
                char buf[32];
                snprintf(buf, sizeof(buf), "%02d:%02d:%02d%c%03d", h, m, sec, comma ? ',' : '.', ms);
                return buf;
            };
            std::string out;
            const std::string& fmt = params.align_format;
            if (fmt == "json") {
                out = "[\n";
                for (size_t i = 0; i < words.size(); i++) {
                    // cppcheck-suppress containerOutOfBounds
                    double end = (i + 1 < words.size()) ? words[i + 1].start : clip_end;
                    std::string esc;
                    for (char c : words[i].text) {
                        if (c == '"' || c == '\\')
                            esc += '\\';
                        esc += c;
                    }
                    char line[256];
                    snprintf(line, sizeof(line), "  {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f}%s\n",
                             esc.c_str(), words[i].start, end, i + 1 < words.size() ? "," : "");
                    out += line;
                }
                out += "]\n";
            } else if (fmt == "plain") {
                for (auto& w : words)
                    out += ts(w.start, false) + "\t" + w.text + "\n";
            } else { // srt (default)
                for (size_t i = 0; i < words.size(); i++) {
                    // cppcheck-suppress containerOutOfBounds
                    double end = (i + 1 < words.size()) ? words[i + 1].start : clip_end;
                    out += std::to_string(i + 1) + "\n" + ts(words[i].start, true) + " --> " + ts(end, true) + "\n" +
                           words[i].text + "\n\n";
                }
            }
            if (params.align_output.empty()) {
                fputs(out.c_str(), stdout);
            } else {
                FILE* f = fopen(params.align_output.c_str(), "wb");
                if (!f) {
                    fprintf(stderr, "crispasr[align]: cannot write '%s'\n", params.align_output.c_str());
                    return 20;
                }
                fwrite(out.data(), 1, out.size(), f);
                fclose(f);
                fprintf(stderr, "crispasr[align]: %d tokens → %zu words → %s\n", result.n_tokens, words.size(),
                        params.align_output.c_str());
            }
            return 0;
        }

        // --make-ref: write the voice reference GGUF.
        fprintf(stderr, "crispasr[make-ref]: %d tokens × %d-d → %s\n", result.n_tokens, result.embed_dim,
                out_path.c_str());
        // Stamped as a clone: this pack came from a real recording, and the
        // synthesis-time gates read the stamp back rather than guessing from the
        // .gguf suffix (which is how --make-ref output used to reach a backend
        // with no attestation demanded and no audible disclosure attached).
        rc = tada_encoder_write_ref_gguf(out_path.c_str(), result, params.tts_ref_text.c_str(),
                                         params.language.empty() ? nullptr : params.language.c_str(),
                                         /*cloned_from_recording=*/true, params.tts_consent_attestation.c_str());
        if (rc != 0) {
            fprintf(stderr, "crispasr[make-ref]: failed to write GGUF (rc=%d)\n", rc);
            return 20;
        }
        fprintf(stderr, "crispasr[make-ref]: saved %s\n", out_path.c_str());
        return 0;
    }

    // ---- TTS mode: synthesize speech from text ----
    if (!params.tts_text.empty()) {
        if (!(backend->capabilities() & CAP_TTS)) {
            fprintf(stderr, "crispasr: error: backend '%s' does not support TTS\n", backend_name.c_str());
            return 14;
        }

        // Initialize AudioSeal neural watermark if --watermark-model is set
        if (!params.watermark_model.empty()) {
            crispasr_wm_dispatch::init(crispasr_resolve_watermark_model(params));
        }
        // Any provenance opt-out requires the explicit marking attestation
        // (hard-refuse without it), before we honor --no-watermark.
        if (int rc = crispasr_check_marking_attestation(params))
            return rc;
        // Honor the --no-watermark opt-out (equivalent to CRISPASR_NO_WATERMARK).
        crispasr_wm_dispatch::set_disabled(params.tts_no_watermark);

        // Consent-record sink: --consent-log set this at parse time; the env
        // var covers wrappers that cannot add a flag. Explicit flag wins.
        crispasr_consent::init_log_path_from_env();

        // Voice-cloning consent gate. A clone is a .wav reference, a voice baked
        // from one during this run, or a pack that declares it was derived from a
        // real recording — NOT merely "the path ends in .wav", which missed the
        // inline-bake rewrite above and every .gguf-only cloning backend
        // (chatterbox has no .wav path at all). See crispasr_voice_clone_policy.h.
        const crispasr_voice::CloneDecision clone_decision = crispasr_voice::classify_voice(
            params.tts_voice, params.tts_voice_dir, params.tts_voice_baked_from_wav, backend->voice_bank_path());
        const bool is_voice_clone = clone_decision.is_clone;
        if (is_voice_clone && !params.tts_voice_clone_consent) {
            fprintf(stderr,
                    "crispasr: error: voice cloning requires the --i-have-rights flag.\n"
                    "\n"
                    "  By passing --i-have-rights you attest:\n"
                    "  \"I have the consent of the speaker whose voice this clones,\n"
                    "   or it is my own voice.\"\n"
                    "\n"
                    "  Usage: crispasr --tts \"text\" --voice speaker.wav --i-have-rights\n"
                    "  (this voice is a clone: %s)\n",
                    clone_decision.reason);
            return 17;
        }
        if (is_voice_clone) {
            // Log consent attestation with timestamp for audit trail.
            // `ref_sha256` binds the record to the BYTES that were cloned, not
            // just the name they were cloned under — a name is not evidence,
            // because the file can be swapped afterwards and the line still
            // reads true. Hash whatever the backend will actually open (the
            // --voice-dir resolution), so a bare name records the same thing a
            // full path would. See crispasr_consent_record.h.
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            // For an inline bake, hash the ORIGINAL recording rather than the
            // temp pack it was baked into — that is the file consent covers.
            const std::string ref_target =
                !params.tts_voice_source_recording.empty()
                    ? params.tts_voice_source_recording
                    : crispasr_voice::resolve_voice_path(params.tts_voice, params.tts_voice_dir);
            std::string ref = crispasr_consent::file_sha256(ref_target);
            crispasr_consent::emit(
                "CONSENT", ts,
                {{"voice", params.tts_voice},
                 {"clone_reason", clone_decision.reason},
                 {"attestation", params.tts_consent_attestation, /*quoted=*/true},
                 {"ref_sha256", ref.empty() ? "none" : ref},
                 {"ref_is", params.tts_voice_source_recording.empty() ? "resolved-voice" : "source-recording"}});
        }

        // Art. 50(4) applies to a PRESET voice too when that voice is an
        // identifiable person: the audience cannot tell which pipeline made the
        // audio, and Art. 3(60) does not ask them to. Resolve whose voice this
        // is — operator override, then the pack's declaration, then the
        // backend's — and disclose on either ground. Notably this does NOT feed
        // the consent gate above; see crispasr_speaker_identity.h.
        bool identity_recognised = true;
        const crispasr_voice::SpeakerIdentity identity_override =
            crispasr_voice::parse_speaker_identity(params.tts_speaker_identity, &identity_recognised);
        if (!identity_recognised) {
            // Report the typo rather than silently treating it as Unknown: the
            // operator meant to answer this question, and "real-person" landing
            // as "unknown" would drop the disclosure they asked for.
            fprintf(stderr,
                    "crispasr: warning: unrecognised --speaker-identity '%s'; treating as 'unknown'. "
                    "Expected real_person, synthetic or unknown.\n",
                    params.tts_speaker_identity.c_str());
        }
        const crispasr_voice::SpeakerIdentity speaker_identity = crispasr_voice::resolve_speaker_identity(
            identity_override, clone_decision.pack_identity, backend->declared_speaker_identity(params.model),
            crispasr_voice::read_model_speaker_identity(params.model));
        const bool needs_spoken_disclosure =
            crispasr_voice::requires_spoken_disclosure(is_voice_clone, speaker_identity);
        if (crispasr_voice::should_warn_unknown_identity(is_voice_clone, speaker_identity) &&
            crispasr_voice::claim_unknown_identity_warning(backend_name)) {
            fprintf(stderr, "%s\n", crispasr_voice::unknown_identity_warning(backend_name).c_str());
        }

        // Sample rate of the synthesized PCM — backend-declared. Most TTS
        // backends emit 24 kHz; voxcpm2-tts emits 48 kHz. Hard-coding 24 kHz
        // here is why voxcpm2 output played at half-speed before this fix.
        const int sr_in = backend->tts_sample_rate();

        // §218 (#182): sentence-chunk long input before synthesis — every TTS
        // talker has a finite positional/training horizon (chatterbox base T3
        // hard-caps at 2050 text positions; longer text was truncated). Split on
        // sentence boundaries, synthesize each chunk within the model's healthy
        // horizon, and concatenate with a 200 ms pause between chunks. The
        // server `/v1/audio/speech` path already does this (#66); this brings the
        // CLI `--tts` path to parity. Single-sentence input is a 1-element vector
        // (one std::vector move of overhead). The policy wrapper keeps VibeVoice
        // voice cloning single-shot (chunking breaks its continuous-prompt ICL).
        // --tts-stream: emit each sentence chunk to stdout as raw s16le mono
        // PCM as soon as it's synthesized (progressive playback), instead of
        // concatenating into one WAV. Watermark is embedded per chunk; the
        // spoken disclaimer (if voice-cloned) is emitted first. All logs stay
        // on stderr so stdout is a clean PCM stream.
        if (params.tts_stream) {
            if (!params.no_prints)
                fprintf(stderr, "crispasr: streaming TTS as s16le mono @ %d Hz to stdout\n", sr_in);
            // Raw PCM stream carries no container ⇒ no C2PA floor. Keep the audio
            // watermark on regardless of --no-watermark so the stream stays marked.
            crispasr_enforce_cli_watermark_floor("", params);
            auto emit = [&](std::vector<float>& pcm) {
                if (pcm.empty())
                    return;
                crispasr_wm_dispatch::embed(pcm.data(), (int)pcm.size(), sr_in);
                std::vector<int16_t> s16(pcm.size());
                for (size_t i = 0; i < pcm.size(); i++) {
                    float v = pcm[i] * 32767.0f;
                    s16[i] = (int16_t)(v < -32768.0f ? -32768.0f : (v > 32767.0f ? 32767.0f : v));
                }
                fwrite(s16.data(), sizeof(int16_t), s16.size(), stdout);
                fflush(stdout);
            };
            if (needs_spoken_disclosure && !params.tts_no_spoken_disclaimer) {
                const auto& disc = crispasr_tts_get_disclaimer(backend.get(), params);
                if (!disc.empty()) {
                    std::vector<float> d(disc.begin(), disc.end());
                    emit(d);
                    std::vector<float> gap((size_t)sr_in / 5, 0.0f); // 200 ms
                    emit(gap);
                }
            }
            const std::vector<std::string> stream_chunks =
                crispasr_tts_plan_chunks_for_backend(params.tts_text, backend->name());
            bool any = false;
            for (size_t ci = 0; ci < stream_chunks.size(); ci++) {
                std::vector<float> c = backend->synthesize(stream_chunks[ci], params);
                if (c.empty())
                    continue;
                if (any) {
                    std::vector<float> gap((size_t)sr_in / 5, 0.0f); // 200 ms between chunks
                    emit(gap);
                }
                emit(c);
                any = true;
            }
            fflush(stdout);
            crispasr_wm_dispatch::shutdown();
            if (!any) {
                fprintf(stderr, "crispasr: error: TTS synthesis failed\n");
                return 15;
            }
            if (!params.no_prints)
                fprintf(stderr, "crispasr: TTS stream complete\n");
            return 0;
        }

        std::vector<float> audio;
        {
            const std::vector<std::string> chunks_txt =
                crispasr_tts_plan_chunks_for_backend(params.tts_text, backend->name());
            std::vector<std::vector<float>> chunk_pcm;
            chunk_pcm.reserve(chunks_txt.size());
            for (size_t ci = 0; ci < chunks_txt.size(); ci++) {
                if (params.verbose && chunks_txt.size() > 1)
                    fprintf(stderr, "crispasr[tts]: chunk %zu/%zu (%zu chars)\n", ci + 1, chunks_txt.size(),
                            chunks_txt[ci].size());
                std::vector<float> c = backend->synthesize(chunks_txt[ci], params);
                if (!c.empty())
                    chunk_pcm.push_back(std::move(c));
            }
            audio = crispasr_tts_concat_with_silence(chunk_pcm, sr_in / 5);
        }
        if (audio.empty()) {
            fprintf(stderr, "crispasr: error: TTS synthesis failed\n");
            return 15;
        }

        // Prepend the spoken AI-disclosure — for a voice clone, and for a preset
        // voice that belongs to an identifiable person (Art. 50(4) attaches to
        // the audio, not to the pipeline). The disclaimer is synthesized with
        // the neutral/default voice (not the cloned voice) and cached. 300ms
        // silence gap. Skipped when --no-spoken-disclaimer is set; watermark +
        // C2PA provenance remain regardless.
        if (needs_spoken_disclosure && !params.tts_no_spoken_disclaimer) {
            crispasr_tts_prepend_disclaimer(audio, backend.get(), params);
        }

        // Optional leading-silence trim. RMS gate over a 20 ms window;
        // drop frames below -50 dBFS (≈ 0.0032 RMS) until the gate
        // opens, then back off 50 ms so we don't clip the consonant onset.
        if (params.tts_trim_silence) {
            const int win = sr_in / 50;       // 20 ms
            const int headroom = sr_in / 20;  // 50 ms
            const float rms_thresh = 0.0032f; // ≈ -50 dBFS
            size_t cut = 0;
            for (size_t i = 0; i + (size_t)win < audio.size(); i += (size_t)win) {
                double e = 0.0;
                for (int k = 0; k < win; k++)
                    e += (double)audio[i + (size_t)k] * (double)audio[i + (size_t)k];
                float rms = (float)std::sqrt(e / (double)win);
                if (rms >= rms_thresh) {
                    cut = i > (size_t)headroom ? i - (size_t)headroom : 0;
                    break;
                }
            }
            if (cut > 0) {
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: trimmed %.2fs of leading silence\n", (double)cut / (double)sr_in);
                audio.erase(audio.begin(), audio.begin() + (std::ptrdiff_t)cut);
            }
        }

        // Resolve the output path first so we can enforce the watertight floor
        // BEFORE embedding: if this container can't carry C2PA, --no-watermark is
        // overridden so the file is never fully unmarked.
        std::string out_path = params.tts_output.empty() ? "tts_output.wav" : params.tts_output;
        crispasr_enforce_cli_watermark_floor(out_path, params);

        // Embed watermark (AudioSeal if loaded, otherwise spread-spectrum)
        crispasr_wm_dispatch::embed(audio.data(), (int)audio.size(), sr_in);

        // Write output audio (backend-native sample rate, mono) — WAV by
        // default, MP3/AAC when --tts-output ends in .mp3/.aac.
        if (int rc = crispasr_write_synth_audio(out_path, audio.data(), (int)audio.size(), sr_in, params.c2pa_cert,
                                                params.c2pa_key, params.cache_dir, !params.tts_no_c2pa))
            return rc;

        // Post-embed watermark verification: re-detect on the in-memory
        // PCM (which has already been watermarked) and warn if confidence
        // is too low. This catches edge cases where the embed silently
        // failed or the audio is too short / silent to hold a watermark.
        // Only meaningful for the spread-spectrum detector, and only on audio
        // long enough to score: the detector averages spectra across frames, so
        // a short clip scores near chance (0.5) even when the embed worked
        // perfectly. The old bare `< 0.6` bar therefore warned on most clips
        // under a couple of seconds — a warning that fires on healthy output
        // teaches operators to ignore it. Gate on duration and say what the
        // number means. See crispasr_watermark_stats.h.
        if (!crispasr_wm_dispatch::is_disabled() && crispasr_wm_dispatch::get_ctx() == nullptr) {
            const double dur_s = sr_in > 0 ? (double)audio.size() / (double)sr_in : 0.0;
            const float conf = crispasr_wm_dispatch::detect(audio.data(), (int)audio.size(), sr_in);
            const bool frames = crispasr_watermark_detect_uses_frames();
            const auto verdict = frames ? crispasr_wm_stats::classify_frames(conf)
                                        : crispasr_wm_stats::classify(conf, CRISPASR_WATERMARK_NBINS);
            // The per-frame statistic finds a fresh embed on ~100% of clips at
            // 2.5 s and up, so the 5 s guard it needed is only about the legacy
            // path. Keep it there; the self-check is a diagnostic either way and
            // must never be read as the marking gate (embedding is unconditional).
            const double min_dur = frames ? 2.5 : 5.0;
            if (dur_s >= min_dur && verdict == crispasr_wm_stats::Verdict::NotDetected) {
                if (frames) {
                    fprintf(stderr,
                            "crispasr: warning: watermark self-check did not find the mark it just embedded "
                            "(score=%.3f over %.1fs). The file is still marked if C2PA signing succeeded; "
                            "re-run --detect-watermark to confirm.\n",
                            conf, dur_s);
                } else {
                    fprintf(stderr,
                            "crispasr: warning: watermark self-check did not find the mark it just embedded "
                            "(score=%.3f over %.1fs, p=%.2g). The file is still marked if C2PA signing "
                            "succeeded; re-run --detect-watermark to confirm.\n",
                            conf, dur_s, crispasr_wm_stats::p_value(conf, CRISPASR_WATERMARK_NBINS));
                }
            }
        }

        // Close the loop on the consent record: the same run_id appears on the
        // [CONSENT] line above, so a disputed clip can be walked back to the
        // attestation that authorised it. Emitted for CLONES only — a preset
        // needs no attestation, so there would be nothing to correlate to.
        // out_sha256 is of the file as WRITTEN (watermarked, C2PA-signed), which
        // is the artefact that leaves the machine.
        if (is_voice_clone) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            std::string out_hash = crispasr_consent::file_sha256(out_path);
            crispasr_consent::emit("CONSENT-OUTPUT", ts,
                                   {{"output", out_path},
                                    {"out_sha256", out_hash.empty() ? "none" : out_hash},
                                    {"seconds", std::to_string((double)audio.size() / (double)sr_in)}});
        }

        if (!params.no_prints)
            fprintf(stderr, "crispasr: TTS output written to '%s' (%zu samples @ %d Hz, %.2f sec)\n", out_path.c_str(),
                    audio.size(), sr_in, (double)audio.size() / (double)sr_in);

        // --tts-play: play the watermarked PCM on the local speaker.
        // Uses the same audio[] buffer (already watermarked at line above).
        if (params.tts_play) {
            crispasr_speaker* spk = crispasr_speaker_open(sr_in, 1, params.tts_play_device);
            if (!spk) {
                fprintf(stderr, "crispasr: warning: --tts-play: could not open playback device\n");
            } else {
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: playing on '%s'\n", crispasr_speaker_default_device_name());
                if (crispasr_speaker_play(spk, audio.data(), (int)audio.size(), sr_in, 1) == 0)
                    crispasr_speaker_wait(spk);
                else
                    fprintf(stderr, "crispasr: warning: --tts-play: playback failed\n");
                crispasr_speaker_close(spk);
            }
        }

        crispasr_wm_dispatch::shutdown();
        return 0;
    }

    // ---- S2S mode: speech-to-speech (audio in → audio out) ----
    if (params.s2s) {
        if (!(backend->capabilities() & CAP_S2S)) {
            fprintf(stderr, "crispasr: error: backend '%s' does not support S2S\n", backend_name.c_str());
            return 14;
        }
        if (params.fname_inp.empty() || params.fname_inp[0].empty()) {
            fprintf(stderr, "crispasr: error: S2S requires audio input (-f <file>)\n");
            return 3;
        }

        // Load input audio (16 kHz mono PCM)
        std::vector<float> s2s_samples;
        std::vector<std::vector<float>> s2s_stereo_unused;
        if (!read_audio_data(params.fname_inp[0], s2s_samples, s2s_stereo_unused, false)) {
            fprintf(stderr, "crispasr: error: failed to read audio '%s'\n", params.fname_inp[0].c_str());
            return 20;
        }

        if (!params.watermark_model.empty()) {
            crispasr_wm_dispatch::init(crispasr_resolve_watermark_model(params));
        }
        // Any provenance opt-out requires the explicit marking attestation
        // (hard-refuse without it), before we honor --no-watermark.
        if (int rc = crispasr_check_marking_attestation(params))
            return rc;
        // Honor the --no-watermark opt-out (equivalent to CRISPASR_NO_WATERMARK).
        crispasr_wm_dispatch::set_disabled(params.tts_no_watermark);

        std::string transcript;
        auto audio = backend->speech_to_speech(s2s_samples.data(), (int)s2s_samples.size(), &transcript, params);
        if (audio.empty()) {
            fprintf(stderr, "crispasr: error: S2S synthesis failed\n");
            return 15;
        }

        const int sr_out = backend->tts_sample_rate();

        // Print transcript if available
        if (!transcript.empty()) {
            if (!params.no_prints)
                fprintf(stderr, "crispasr: S2S transcript: %s\n", transcript.c_str());
            printf("%s\n", transcript.c_str());
        }

        // Resolve output path + enforce the watertight floor before embedding.
        std::string out_path = params.s2s_output.empty() ? "s2s_output.wav" : params.s2s_output;
        crispasr_enforce_cli_watermark_floor(out_path, params);

        // Embed watermark
        crispasr_wm_dispatch::embed(audio.data(), (int)audio.size(), sr_out);

        // Write output audio — WAV by default, MP3/AAC when --s2s-output
        // ends in .mp3/.aac.
        if (int rc = crispasr_write_synth_audio(out_path, audio.data(), (int)audio.size(), sr_out, params.c2pa_cert,
                                                params.c2pa_key, params.cache_dir, !params.tts_no_c2pa))
            return rc;

        if (!params.no_prints)
            fprintf(stderr, "crispasr: S2S output written to '%s' (%zu samples @ %d Hz, %.2f sec)\n", out_path.c_str(),
                    audio.size(), sr_out, (double)audio.size() / (double)sr_out);
        crispasr_wm_dispatch::shutdown();
        return 0;
    }

    // Auto-punctuation for CTC backends: when the user hasn't set --punc-model
    // and the backend doesn't natively toggle punctuation, auto-enable
    // FireRedPunc. This gives CTC backends (fc-ctc, wav2vec2, firered-asr,
    // omniasr-ctc) punctuated output by default. Users can suppress with
    // --no-punctuation or --punc-model none.
    if (crispasr_should_auto_enable_punctuation(backend->capabilities(), params)) {
        params.punc_model = "auto";
        if (!params.no_prints)
            fprintf(stderr, "crispasr: auto-enabling punctuation restoration for backend '%s'\n", backend->name());
    }

    // Optional punctuation restoration post-processor.
    // `--punc-model auto` or `--punc-model firered` → auto-download Q4_K (~50 MB).
    // The alias → model mapping is shared with the HTTP server via the resolver
    // in crispasr_punc_loader.h, so the two front-ends can't drift apart.
    const crispasr_punc_spec punc_spec = crispasr_resolve_punc_model(params.punc_model);
    std::unique_ptr<fireredpunc_context, decltype(&fireredpunc_free)> punc_ctx(nullptr, fireredpunc_free);
    if (punc_spec.kind == crispasr_punc_kind::fireredpunc) {
        std::string punc_path = punc_spec.direct_path;
        if (punc_path.empty() && !punc_spec.cache_filename.empty())
            punc_path = crispasr_cache::ensure_cached_file(punc_spec.cache_filename, punc_spec.url, params.no_prints,
                                                           "crispasr[punc]", params.cache_dir);
        if (!punc_path.empty()) {
            punc_ctx.reset(fireredpunc_init(punc_path.c_str()));
            if (!punc_ctx) {
                fprintf(stderr, "crispasr: warning: failed to load punc model '%s' — continuing without\n",
                        punc_path.c_str());
            } else if (!params.no_prints) {
                fprintf(stderr, "crispasr: loaded punctuation model '%s'\n", punc_path.c_str());
            }
        }
    }

    // PCS model (punctuation + capitalization + segmentation in one model).
    // `--punc-model pcs` loads the 1-800-BAD-CODE XLM-RoBERTa model which
    // handles punc, truecasing, and SBD together. When PCS is active, it
    // replaces both fireredpunc and the statistical truecaser.
    std::unique_ptr<pcs_context, decltype(&pcs_free)> pcs_ctx(nullptr, pcs_free);
    if (punc_spec.kind == crispasr_punc_kind::pcs) {
        std::string pcs_path = punc_spec.direct_path;
        if (pcs_path.empty() && !punc_spec.cache_filename.empty())
            pcs_path = crispasr_cache::ensure_cached_file(punc_spec.cache_filename, punc_spec.url, params.no_prints,
                                                          "crispasr[pcs]", params.cache_dir);
        if (!pcs_path.empty()) {
            pcs_ctx.reset(pcs_init(pcs_path.c_str()));
            if (pcs_ctx) {
                punc_ctx.reset(); // PCS replaces fireredpunc
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: loaded PCS model '%s'\n", pcs_path.c_str());
            }
        }
    }

    // #311: strict punctuation — the two loadable kinds are fireredpunc and pcs.
    // If punctuation is required but neither context loaded (missing/corrupt/
    // unresolvable model), fail with a non-zero exit instead of the default
    // "continuing without" warning.
    if (crispasr_compute_strict_reqs(params).punc && !punc_ctx && !pcs_ctx) {
        fprintf(stderr,
                "crispasr: error: required punctuation model '%s' failed to load "
                "(--require-punctuation/--strict-pipeline).\n",
                params.punc_model.c_str());
        return CRISPASR_STRICT_RC_PUNC;
    }

    // Optional truecaser post-processor.
    // `--truecase-model auto` / `de` → statistical German truecaser (~9 MB).
    // `--truecase-model crf`  → CRF German truecaser with context (~25 MB).
    // `--truecase-model lstm` → BiLSTM character-level truecaser (~3 MB, best quality).
    std::unique_ptr<truecaser_context, decltype(&truecaser_free)> tc_ctx(nullptr, truecaser_free);
    std::unique_ptr<truecaser_crf_context, decltype(&truecaser_crf_free)> tc_crf_ctx(nullptr, truecaser_crf_free);
    std::unique_ptr<truecaser_lstm_context, decltype(&truecaser_lstm_free)> tc_lstm_ctx(nullptr, truecaser_lstm_free);
    crispasr_load_truecase(params.truecase_model, params.no_prints, params.cache_dir, tc_ctx, tc_crf_ctx, tc_lstm_ctx,
                           "crispasr[tc]");

    // ---- Streaming mode: read raw PCM from stdin, transcribe chunks ----
    if (params.stream) {
        const int SR = 16000;
        const int step_samples = (params.stream_step_ms * SR) / 1000;
        const int length_samples = (params.stream_length_ms * SR) / 1000;
        const int keep_samples = (params.stream_keep_ms * SR) / 1000;
        const std::string stream_vad_path = crispasr_resolve_vad_model(params);

        crispasr_vad_options stream_vad_opts;
        stream_vad_opts.threshold = params.vad_threshold;
        stream_vad_opts.threshold_explicit = params.vad_threshold_explicit;
        stream_vad_opts.min_speech_duration_ms = params.vad_min_speech_duration_ms;
        stream_vad_opts.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        stream_vad_opts.speech_pad_ms = params.vad_speech_pad_ms;
        // Streaming windows are already bounded by --stream-length.
        stream_vad_opts.chunk_seconds = 0;
        stream_vad_opts.n_threads = params.n_threads;
        if (params.stream_json) {
            stream_vad_opts.post_merge_policy = crispasr_vad_post_merge_policy::streaming_json;
            stream_vad_opts.stream_close_gap_ms = params.stream_vad_merge_gap_ms;
            stream_vad_opts.stream_final_silence_ms = params.stream_final_silence_ms;
        }

        // If --mic, spawn a subprocess to capture audio from the default mic
        FILE* mic_pipe = nullptr;
        if (params.mic) {
            fprintf(stderr, "crispasr[mic]: capturing from default microphone...\n");
            fprintf(stderr, "crispasr[mic]: press Ctrl+C to stop\n\n");
            // Try platform-specific mic capture commands
#if defined(__APPLE__)
            // macOS: use sox (most reliable), ffmpeg fallback
            mic_pipe = popen("rec -q -t s16 -r 16000 -c 1 - 2>/dev/null || "
                             "ffmpeg -f avfoundation -i ':default' -f s16le -ar 16000 -ac 1 - 2>/dev/null",
                             "r");
#elif defined(_WIN32)
            {
                // Resolve the default capture device via miniaudio. On
                // localized Windows installs this comes back as a UTF-8
                // string with non-ASCII characters (e.g. zh `麥克風 (...)`
                // or de `Mikrofon (...)`); the ffmpeg dshow argument
                // therefore needs UTF-8-safe propagation through the CRT
                // — see issue #70 follow-up. crispasr_popen widens to
                // wchar_t and calls _wpopen so the device name survives.
                const char* dev = crispasr_mic_default_device_name();
                std::string dshow_arg = crispasr_windows_dshow_audio_arg_from_name(dev);
                std::string cmd = "ffmpeg -f dshow -i " + dshow_arg + " -f s16le -ar 16000 -ac 1 -";
                if (params.stream_monitor || !params.no_prints) {
                    fprintf(stderr, "crispasr[mic]: device=%s\n", dev && *dev ? dev : "(default)");
                    fprintf(stderr, "crispasr[mic]: ffmpeg cmd: %s\n", cmd.c_str());
                }
                // Suppress ffmpeg stderr on the wire but leave it user-
                // discoverable (we just printed the command above).
                cmd += " 2>NUL";
                mic_pipe = crispasr::crispasr_popen(cmd, "rb");
            }
#else
            // Linux: try arecord first, then ffmpeg with pulseaudio
            mic_pipe = popen("arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null || "
                             "ffmpeg -f pulse -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null || "
                             "ffmpeg -f alsa -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null",
                             "r");
#endif
            if (!mic_pipe) {
                fprintf(stderr, "crispasr[mic]: failed to open microphone. Install sox, ffmpeg, or arecord.\n");
                return 20;
            }
        } else {
            fprintf(stderr, "crispasr[stream]: reading raw s16le 16kHz mono PCM from stdin\n");
            fprintf(stderr, "crispasr[stream]: step=%dms length=%dms keep=%dms\n", params.stream_step_ms,
                    params.stream_length_ms, params.stream_keep_ms);
            fprintf(stderr, "crispasr[stream]: pipe audio in, e.g.:\n");
            fprintf(stderr, "  ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 - | crispasr --stream -m model.gguf\n\n");
        }

        FILE* audio_src = mic_pipe ? mic_pipe : stdin;

#if defined(_WIN32)
        if (!mic_pipe)
            _setmode(_fileno(stdin), _O_BINARY);
#endif

        // Issue #84: true rolling buffer — accumulate audio up to
        // length_samples, never collapse back to keep+step. The old
        // code reallocated `pcm_window` to `keep_samples + n_new`
        // every step, capping the steady-state decode buffer at
        // ~3.4 s for the default 200 ms / 3 s settings even though
        // `--stream-length` was advertised as the context window.
        // Start empty (no leading-zero padding) and let the buffer
        // grow naturally up to `length_samples`, then drop the
        // oldest samples from the front to maintain the cap.
        std::vector<float> pcm_window;
        pcm_window.reserve(length_samples);
        size_t pcm_head = 0; // ponytail: ring head to avoid O(N) erase on every step
        auto pcm_window_size = [&]() -> size_t { return pcm_window.size() - pcm_head; };
        auto pcm_window_data = [&]() -> const float* { return pcm_window.data() + pcm_head; };
        std::vector<int16_t> read_buf(step_samples);
        std::string prev_text;
        // Issue #84 round 2 (CKwasd retest): the JSON streaming state
        // machine is utterance-centric, not chunk-centric. Each
        // utterance is a continuous speech region (delimited by VAD
        // trailing-silence ≥ `--stream-final-on-silence-ms`, falling
        // back to "model decoded nothing for N steps" when VAD is off);
        // `final.text` covers the **whole** utterance, not just the
        // last rolling-window hypothesis the way round 1 emitted.
        //
        // `utterance_pcm` accumulates the speech-region PCM (capped at
        // `--stream-utterance-max-sec`) so finalize can re-decode the
        // whole region in one shot — the rolling `pcm_window` evicts
        // old audio and would lose the start of long utterances. The
        // `prefix_*` strings drive the alternative `--stream-final-mode
        // prefix` accumulator (longest-common-prefix across consecutive
        // partials) for callers that don't want the extra encoder pass.
        // `last_speech_end_sample` is the stream-timeline sample where
        // VAD last saw speech; `--stream-final-on-silence-ms` is checked
        // against `now - last_speech_end_sample`, not "all-empty steps".
        //
        // Round 3 (CKwasd retest after round 2) addresses four remaining
        // semantic issues in the JSON stream:
        //   1. After finalize, the rolling `pcm_window` still contains
        //      the finished utterance's speech for up to `--stream-length`,
        //      and VAD re-discovers it every step. `finalized_until_sample`
        //      bookmarks the upper bound of the finalized region so
        //      re-discovered slices can't seed a new utterance with the
        //      previous text.
        //   2. The trailing-silence check used to be gated on
        //      `step_slice_text.empty()`, which a re-discovered old slice
        //      can keep non-empty for a full window length. Finalization
        //      now fires purely on `now - last_speech_end_sample`.
        //   3. Multiple VAD slices in one step each called
        //      `on_partial_text`, producing competing partials for the
        //      same `utterance_id`. Round 3 coalesces all slice texts
        //      that belong to the open utterance into a single partial
        //      per utterance per step.
        //   4. `utterance_pcm` accumulated every step's new samples
        //      (including post-`last_speech_end` silence). Finalize now
        //      trims it to `[utterance_start_sample, last_speech_end_sample]`
        //      so the redecode input matches the [t0..t1] interval the
        //      final event advertises.
        int64_t utterance_id = 0;
        bool have_open_utterance = false;
        int64_t utterance_start_sample = 0;
        int64_t last_speech_end_sample = 0;
        int64_t finalized_until_sample = 0;
        std::vector<float> utterance_pcm;
        std::string prefix_committed;
        std::string last_partial_text; // dedupe key + prefix-mode tail
        int64_t utterance_pcm_size_at_save = 0;
        int64_t last_partial_decode_sample = -1;
        int64_t cumulative_samples = 0;
        // ponytail: VAD throttle cache — outside while loop so throttled steps can shift previous slices
        std::vector<crispasr_audio_slice> cached_vad_slices;
        int64_t cached_vad_window_start = 0;
        bool cached_vad_valid = false;
        const int64_t utterance_max_samples = (int64_t)params.stream_utterance_max_sec * SR;
        const int64_t partial_decode_interval_samples =
            crispasr_stream_partial_decode_interval_samples(params.stream_partial_decode_ms, params.stream_step_ms, SR);
        // Track whether the audio source ever produced any samples; if
        // it goes EOF without a single one, the subprocess most likely
        // failed before delivering PCM (e.g. ffmpeg couldn't open the
        // dshow device because the name got mangled). Surface a hint
        // so users don't see the silent exit reported in issue #70.
        bool any_samples_read = false;

        while (true) {
            // Read one step of raw s16le samples from audio source
            size_t n_read = fread(read_buf.data(), sizeof(int16_t), step_samples, audio_src);
            if (n_read == 0) {
                if (mic_pipe && !any_samples_read) {
                    fprintf(stderr, "\ncrispasr[mic]: pipe ended before any PCM was read.\n"
                                    "  Most likely the capture subprocess (ffmpeg/sox/arecord) failed\n"
                                    "  to open the requested device. Re-run the printed command above\n"
                                    "  without `2>NUL` / `2>/dev/null` to see its stderr, or list\n"
                                    "  available devices: `ffmpeg -list_devices true -f dshow -i dummy`\n"
                                    "  (Windows) / `arecord -l` (Linux).\n");
                }
                break; // EOF
            }
            any_samples_read = true;

            // Convert s16le to float (ring head avoids O(N) erase)
            const size_t n_new = n_read;
            // compact head when it grows too large (avoid unbounded vector growth)
            if (pcm_head > (size_t)length_samples && pcm_head > pcm_window.size() / 2) {
                pcm_window.erase(pcm_window.begin(), pcm_window.begin() + pcm_head);
                pcm_head = 0;
            }
            const size_t write_pos = pcm_window.size();
            pcm_window.resize(write_pos + n_new);
            for (size_t i = 0; i < n_new; i++)
                pcm_window[write_pos + i] = read_buf[i] / 32768.0f;

            if ((int)pcm_window_size() > length_samples) {
                const size_t drop = pcm_window_size() - (size_t)length_samples;
                pcm_head += drop;
                // occasional compact to keep capacity bounded
                if (pcm_head > (size_t)length_samples) {
                    pcm_window.erase(pcm_window.begin(), pcm_window.begin() + pcm_head);
                    pcm_head = 0;
                }
            }
            cumulative_samples += (int64_t)n_new;
            (void)keep_samples; // legacy, intentionally unused

            // Monitor: show progress during processing
            if (params.stream_monitor) {
                fprintf(stderr, "\xE2\x96\xB6"); // ▶ = processing chunk
                fflush(stderr);
            }

            std::vector<crispasr_segment> segs;
            // Per-slice text for the JSON state machine. Each entry is
            // `(slice, text)`. When `--stream-partial-decode-ms` skips this
            // step's ASR partial decode, the text is empty but the slice timing
            // still drives utterance boundaries and finalization.
            std::vector<std::pair<crispasr_audio_slice, std::string>> step_slice_text;
            bool decoded_segments_this_step = false;
            if (!stream_vad_path.empty()) {
                // ponytail: small-window bypass — delta has no gain when overlap <=50% vs guard overhead
                const bool use_delta = length_samples >= 4000 && length_samples >= step_samples * 3;
                if (backend->capabilities() & CAP_STREAM_DELTA) {
                    if (use_delta) backend->set_stream_delta((int)n_new);
                    else backend->set_stream_delta(0);
                }
                // Hard guard: continue_decode + VAD merge would crash -1073741819 (cross-KV invalid
                // after window shift). No backend supports it today (crispasr_backend.h:239
                // supports_continue_with_vad()==false); fail fast if a future backend claims support.
                assert(!backend->supports_continue_with_vad() && "continue_decode+VAD merge unsupported: cross-KV invalid after window shift");
                const int64_t window_start_sample_now = cumulative_samples - (int64_t)pcm_window_size();
                std::vector<crispasr_audio_slice> slices;
                const bool final_silence_due_early =
                    params.stream_json && params.stream_final_silence_ms > 0 && have_open_utterance &&
                    last_speech_end_sample > 0 &&
                    (cumulative_samples - last_speech_end_sample) * 1000 / SR >= params.stream_final_silence_ms;
                const bool allow_partial_decode_early = crispasr_stream_partial_decode_allow(
                    params.stream_json, last_partial_decode_sample, final_silence_due_early, cumulative_samples,
                    partial_decode_interval_samples);
                // ponytail: throttling adds 500ms boundary delay — disable for small windows already on fallback path
                const bool vad_throttle_on = length_samples >= 4000;
                const bool need_fresh_vad = (vad_throttle_on ? allow_partial_decode_early : true) || !params.stream_json || !cached_vad_valid;
                if (need_fresh_vad) {
                    slices = crispasr_compute_vad_slices(pcm_window_data(), (int)pcm_window_size(), SR,
                                                                stream_vad_path.c_str(), stream_vad_opts);
                    cached_vad_slices = slices;
                    cached_vad_window_start = window_start_sample_now;
                    cached_vad_valid = true;
                } else {
                    // shift cached slices by window advance; drop those that fell out
                    // ponytail: must also shift t0/t1_cs (window-relative) or mid assignment drifts — fix throttled VAD bug
                    const int64_t shift = window_start_sample_now - cached_vad_window_start;
                    const int64_t shift_cs = shift * 100 / SR;
                    slices.reserve(cached_vad_slices.size());
                    for (auto sl : cached_vad_slices) {
                        sl.start -= (int)shift; sl.end -= (int)shift;
                        sl.t0_cs -= shift_cs; sl.t1_cs -= shift_cs;
                        if (sl.end <= 0 || sl.start >= (int)pcm_window_size()) continue;
                        sl.start = std::max(0, sl.start); sl.end = std::min((int)pcm_window_size(), sl.end);
                        if (sl.t1_cs <= 0) continue;
                        sl.t0_cs = std::max<int64_t>(0, sl.t0_cs);
                        if (sl.end > sl.start) slices.push_back(sl);
                    }
                }
                const bool final_silence_due = final_silence_due_early;
                const bool allow_partial_decode = allow_partial_decode_early;
                bool partial_decode_attempted_this_step = false;
                constexpr int kStraddleMinSamples = 32000;
                if (params.stream_json) {
                    // ponytail: single full-window transcribe so delta survives; midpoint assigns segments to slices
                    if (!allow_partial_decode) {
                        for (auto &sl : slices) {
                            if (window_start_sample_now + sl.end <= finalized_until_sample) continue;
                            step_slice_text.emplace_back(sl, std::string());
                        }
                    } else if (!slices.empty()) {
                        const int64_t window_cs = window_start_sample_now * 100 / SR;
                        auto full_segs = backend->transcribe(pcm_window_data(), (int)pcm_window_size(), window_cs, params);
                        if (!full_segs.empty()) decoded_segments_this_step = true;
                        if (stream_punc_partials_enabled(params)) apply_punc_model(punc_ctx.get(), full_segs);
                        apply_truecase_model(tc_ctx.get(), full_segs);
                        apply_truecase_crf_model(tc_crf_ctx.get(), full_segs);
                        apply_truecase_lstm_model(tc_lstm_ctx.get(), full_segs);
                        apply_pcs_model(pcs_ctx.get(), full_segs);
                        if (!params.punctuation) for (auto &s : full_segs) crispasr_strip_punctuation(s);
                        // filter finalized region for assignment
                        std::vector<std::string> slice_text(slices.size());
                        for (auto &seg : full_segs) {
                            if (seg.t1 <= finalized_until_sample * 100 / SR) continue;
                            const int64_t mid = (seg.t0 + seg.t1) / 2;
                            int best = -1;
                            for (size_t i = 0; i < slices.size(); i++) {
                                const int64_t s0 = window_start_sample_now * 100 / SR + slices[i].t0_cs;
                                const int64_t s1 = window_start_sample_now * 100 / SR + slices[i].t1_cs;
                                if (mid >= s0 && mid < s1) { best = (int)i; break; }
                            }
                            if (best < 0) {
                                // fallback: nearest slice by midpoint distance
                                int64_t best_d = INT64_MAX;
                                for (size_t i = 0; i < slices.size(); i++) {
                                    const int64_t s0 = window_start_sample_now * 100 / SR + slices[i].t0_cs;
                                    const int64_t s1 = window_start_sample_now * 100 / SR + slices[i].t1_cs;
                                    const int64_t mid_s = (s0 + s1) / 2;
                                    int64_t d = llabs(mid - mid_s);
                                    if (d < best_d) { best_d = d; best = (int)i; }
                                }
                            }
                            if (best >= 0) {
                                // straddling guard: skip if slice fully finalized or subrange too short
                                const int64_t s_start_abs = window_start_sample_now + slices[best].start;
                                const int64_t s_end_abs = window_start_sample_now + slices[best].end;
                                if (s_end_abs <= finalized_until_sample) continue;
                                if (s_start_abs < finalized_until_sample) {
                                    const int sub_len = (int)(s_end_abs - finalized_until_sample);
                                    if (sub_len < kStraddleMinSamples) continue;
                                }
                                slice_text[best] += seg.text;
                            }
                        }
                        for (size_t i = 0; i < slices.size(); i++) {
                            if (window_start_sample_now + slices[i].end <= finalized_until_sample) continue;
                            if (slices[i].start < finalized_until_sample - window_start_sample_now) {
                                const int sub_len = (int)(slices[i].end - (finalized_until_sample - window_start_sample_now));
                                if (sub_len < kStraddleMinSamples) continue;
                            }
                            step_slice_text.emplace_back(slices[i], std::move(slice_text[i]));
                        }
                        partial_decode_attempted_this_step = true;
                    }
                } else {
                    // non-JSON VAD: single full-window transcribe, aggregate
                    const int64_t window_cs = window_start_sample_now * 100 / SR;
                    auto full_segs = backend->transcribe(pcm_window_data(), (int)pcm_window_size(), window_cs, params);
                    if (!full_segs.empty()) decoded_segments_this_step = true;
                    segs.insert(segs.end(), std::make_move_iterator(full_segs.begin()), std::make_move_iterator(full_segs.end()));
                }
                if (partial_decode_attempted_this_step) last_partial_decode_sample = cumulative_samples;
                // ponytail: utterance-level delta — save utterance portion of cross-KV for later finalize reuse
                if (params.stream_json && have_open_utterance && decoded_segments_this_step && (backend->capabilities() & CAP_STREAM_UTTERANCE)) {
                    const int64_t ue = last_speech_end_sample > 0 ? last_speech_end_sample : utterance_start_sample + 1;
                    backend->save_utterance_cross_kv(utterance_start_sample, ue, window_start_sample_now, SR, utterance_id);
                    utterance_pcm_size_at_save = (int64_t)utterance_pcm.size();
                }
            } else {
                // No-VAD path: transcribe the full window.
                // Also set streaming delta hint for backends that support it.
                if (backend->capabilities() & CAP_STREAM_DELTA) {
                    backend->set_stream_delta((int)n_new);
                }
                const int64_t no_vad_window_start_cs = (cumulative_samples - (int64_t)pcm_window_size()) * 100 / SR;
                segs = backend->transcribe(pcm_window_data(), (int)pcm_window_size(), no_vad_window_start_cs, params);
                if (!segs.empty())
                    decoded_segments_this_step = true;
            }

            const bool json_vad_path = params.stream_json && !stream_vad_path.empty();
            if (!json_vad_path) {
                apply_punc_model(punc_ctx.get(), segs);
                apply_truecase_model(tc_ctx.get(), segs);
                apply_truecase_crf_model(tc_crf_ctx.get(), segs);
                apply_truecase_lstm_model(tc_lstm_ctx.get(), segs);
                apply_pcs_model(pcs_ctx.get(), segs);
                if (!params.punctuation) {
                    for (auto& seg : segs) {
                        crispasr_strip_punctuation(seg);
                    }
                }
            }
            // No-VAD JSON path: synthesize a single "slice" covering
            // the whole window so the same state machine handles both
            // VAD-on and VAD-off cases.
            if (params.stream_json && stream_vad_path.empty() && !segs.empty()) {
                std::string all_text;
                for (const auto& s : segs)
                    all_text += s.text;
                crispasr_audio_slice fake_sl{0, (int)pcm_window_size(), 0, 0};
                step_slice_text.emplace_back(fake_sl, std::move(all_text));
            }

            if (params.stream_monitor && !decoded_segments_this_step) {
                fprintf(stderr, "\xC2\xB7"); // · = silence
                fflush(stderr);
            }

            // Issue #84 round 2: utterance-centric JSON state machine.
            // The round-1 design echoed the last rolling-window partial
            // as `final.text`, which dropped the start of utterances
            // longer than `--stream-length`. The fix is to (a) buffer
            // the utterance's PCM in `utterance_pcm`, (b) drive
            // finalization off VAD-detected trailing silence (when VAD
            // is on) instead of "the whole window decoded to nothing,"
            // and (c) re-decode the buffered PCM at finalize time so
            // `final.text` covers the full utterance. The cheaper
            // `--stream-final-mode prefix` path keeps round-1's cost
            // by accumulating a longest-common-prefix across consecutive
            // partials instead of re-decoding.
            if (params.stream_json) {
                const int64_t now_sample = cumulative_samples;
                const int64_t window_start_sample = cumulative_samples - (int64_t)pcm_window_size();
                // Track whether this step produced a `partial` or `final`
                // event so the silence heartbeat at the bottom only fires
                // when nothing else did. With the round-3 issue-1 skip,
                // `step_slice_text.empty()` is no longer a reliable proxy
                // (slices can be present but all filtered as old).
                bool emitted_event_this_step = false;

                auto finalize_utterance = [&]() {
                    if (!have_open_utterance)
                        return;
                    // Round 3 (CKwasd #4): trim utterance_pcm to the
                    // VAD-determined speech region [t0..t1] before the
                    // redecode pass. The buffer unconditionally appends
                    // every step's new samples (including post-speech
                    // silence) so finalize sees a buffer that can extend
                    // past last_speech_end_sample. Resizing to the actual
                    // speech-region length makes the redecode input
                    // match the final event's advertised interval.
                    if (last_speech_end_sample > utterance_start_sample) {
                        const int64_t want = last_speech_end_sample - utterance_start_sample;
                        if ((int64_t)utterance_pcm.size() > want)
                            utterance_pcm.resize((size_t)want);
                    }
                    // Build final_text. In `redecode` mode the primary path
                    // re-runs the backend on the VAD-trimmed utterance PCM;
                    // when that's skipped (sub-2-s buffer below the encoder
                    // min — see crispasr::kStreamRedecodeMinSamples) or
                    // returns empty, fall back to the prefix-mode stitcher
                    // so a non-empty partial visible to a UI never gets
                    // replaced by an empty final. (Round 4 of #84: CKwasd
                    // report 2026-05-11 "empty finals on sub-2-s utterances".)
                    std::string final_text;
                    // Native diarization label for this utterance, when the
                    // redecode produced a single-speaker segment set. Emitted
                    // as a structured "speaker" field so JSON consumers don't
                    // parse inline labels; text stays clean. Window/utterance-
                    // local (no cross-utterance clustering) — see docs.
                    std::string final_speaker;
                    bool final_text_from_redecode = false;
                    if (params.stream_final_mode == "redecode") {
                        if ((int)utterance_pcm.size() >= crispasr::kStreamRedecodeMinSamples) {
                            // Disable nested VAD: utterance_pcm is already
                            // the speech region we identified, no need to
                            // re-segment it (and the nested VAD path would
                            // discover the same slices we're collapsing here).
                            whisper_params decode_params = params;
                            decode_params.vad = false;
                            decode_params.vad_model.clear();
                            if ((backend->capabilities() & CAP_STREAM_UTTERANCE)) {
                                const int n_new_since_save = (int)utterance_pcm.size() - (int)utterance_pcm_size_at_save;
                                backend->restore_utterance_cross_kv(n_new_since_save, utterance_id);
                            }
                            auto utt_segs =
                                backend->transcribe(utterance_pcm.data(), (int)utterance_pcm.size(), 0, decode_params);
                            if (stream_punc_finals_enabled(params))
                                apply_punc_model(punc_ctx.get(), utt_segs);
                            apply_truecase_model(tc_ctx.get(), utt_segs);
                            apply_truecase_crf_model(tc_crf_ctx.get(), utt_segs);
                            apply_truecase_lstm_model(tc_lstm_ctx.get(), utt_segs);
                            apply_pcs_model(pcs_ctx.get(), utt_segs);
                            if (!params.punctuation) {
                                for (auto& seg : utt_segs)
                                    crispasr_strip_punctuation(seg);
                            }
                            for (const auto& s : utt_segs)
                                final_text += s.text;
                            final_speaker = crispasr_stream_common_speaker(utt_segs);
                            final_text_from_redecode = !final_text.empty();
                        }
                        if (final_text.empty())
                            final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
                    } else {
                        // prefix mode: stitch committed prefix + last partial tail
                        final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
                    }
                    if (stream_punc_finals_enabled(params) && !final_text_from_redecode)
                        final_text = apply_punc_text(punc_ctx.get(), final_text);
                    const double t0 = (double)utterance_start_sample / (double)SR;
                    const double t1 = (double)last_speech_end_sample / (double)SR;
                    std::string spk_field;
                    if (!final_speaker.empty())
                        spk_field = ",\"speaker\":\"" + crispasr_json_escape(final_speaker) + "\"";
                    fprintf(stdout,
                            "{\"type\":\"final\",\"utterance_id\":%lld,\"text\":\"%s\"%s,\"t0\":%.3f,\"t1\":%.3f}\n",
                            (long long)utterance_id, crispasr_json_escape(final_text).c_str(), spk_field.c_str(), t0,
                            t1);
                    fflush(stdout);
                    emitted_event_this_step = true;
                    // Round 3 (CKwasd #1): bookmark the finalized
                    // boundary so the slice loop can skip slices the
                    // rolling window still holds from this utterance.
                    // Monotonic: finalized_until_sample only grows.
                    if (last_speech_end_sample > finalized_until_sample)
                        finalized_until_sample = last_speech_end_sample;
                    have_open_utterance = false;
                    utterance_pcm.clear();
                    prefix_committed.clear();
                    last_partial_text.clear();
                    if (backend->capabilities() & CAP_STREAM_UTTERANCE) { backend->clear_utterance_cross_kv(); utterance_pcm_size_at_save = 0; }
                };

                auto open_utterance_at = [&](int window_offset, int64_t stream_start) {
                    utterance_id++;
                    have_open_utterance = true;
                    utterance_start_sample = stream_start;
                    last_speech_end_sample = stream_start;
                    if (window_offset < 0)
                        window_offset = 0;
                    if (window_offset > (int)pcm_window_size())
                        window_offset = (int)pcm_window_size();
                    {
                        const float* base = pcm_window_data();
                        utterance_pcm.assign(base + window_offset, base + pcm_window_size());
                    }
                    prefix_committed.clear();
                    last_partial_text.clear();
                    if (backend->capabilities() & CAP_STREAM_UTTERANCE) { backend->clear_utterance_cross_kv(); utterance_pcm_size_at_save = 0; }
                };

                auto on_partial_text = [&](const std::string& new_text) {
                    if (new_text.empty() || new_text == last_partial_text)
                        return;
                    // Maintain the prefix accumulator so prefix-mode
                    // finalization has accumulated state ready (also
                    // cheap to keep updated when redecode is the active
                    // mode — it's just string compares).
                    if (last_partial_text.empty()) {
                        // first partial of this utterance — nothing to commit yet
                    } else if (new_text.size() + 16 < last_partial_text.size()) {
                        // Window rolled past: the previous partial had
                        // content that's now gone. Commit it as stable.
                        if (last_partial_text.size() > prefix_committed.size())
                            prefix_committed = last_partial_text;
                    } else {
                        size_t lcp = 0;
                        const size_t lim = std::min(last_partial_text.size(), new_text.size());
                        while (lcp < lim && last_partial_text[lcp] == new_text[lcp])
                            ++lcp;
                        if (lcp > prefix_committed.size())
                            prefix_committed = last_partial_text.substr(0, lcp);
                    }
                    last_partial_text = new_text;
                    const double t0 = (double)utterance_start_sample / (double)SR;
                    const double t1 = (double)cumulative_samples / (double)SR;
                    fprintf(stdout,
                            "{\"type\":\"partial\",\"utterance_id\":%lld,\"text\":\"%s\",\"t0\":%.3f,\"t1\":%.3f}\n",
                            (long long)utterance_id, crispasr_json_escape(new_text).c_str(), t0, t1);
                    fflush(stdout);
                };

                const bool was_open_at_step_start = have_open_utterance;
                bool utterance_just_opened = false;

                // Round 3 (CKwasd #3): coalesce slice texts into a single
                // partial per utterance per step. Multiple slices in one
                // step belonging to the same open utterance previously
                // emitted one partial each (competing hypotheses to the
                // wrapper); now they're concatenated into one partial.
                // A mid-step finalize flushes the pending partial first
                // so the prior utterance's partial isn't dropped.
                std::string step_open_partial;
                auto flush_pending_partial = [&]() {
                    if (have_open_utterance && !step_open_partial.empty()) {
                        on_partial_text(step_open_partial);
                        emitted_event_this_step = true;
                    }
                    step_open_partial.clear();
                };

                // Drive the utterance state machine over the per-slice
                // results we collected above. For VAD-on, each slice is
                // a real VAD speech region; for VAD-off, there's at
                // most one synthetic "whole window" slice.
                for (const auto& [sl, sl_text] : step_slice_text) {
                    const int64_t s_start = window_start_sample + (int64_t)sl.start;
                    const int64_t s_end = window_start_sample + (int64_t)sl.end;

                    // Round 3 (CKwasd #1): skip slices fully inside the
                    // already-finalized region. Without this guard the
                    // rolling pcm_window's lingering tail of a finalized
                    // utterance re-opens a fresh utterance_id seeded with
                    // the previous text.
                    if (s_end <= finalized_until_sample)
                        continue;
                    // Straddling slice (s_start < finalized < s_end) with
                    // a post-finalized tail shorter than the encoder min
                    // (kStraddleMinSamples in the slice-building loop)
                    // can't be re-decoded standalone, so its sl_text was
                    // suppressed above. Skip it here too: opening a new
                    // utterance from a straddling slice while its new
                    // content is below the encoder min would create a
                    // spurious utterance with empty final.text. Wait
                    // for a later step where enough new audio has
                    // accumulated to clear the gate.
                    if (s_start < finalized_until_sample && (s_end - finalized_until_sample) < 32000)
                        continue;

                    // Silence-driven finalize: if the gap between the
                    // last speech end and this slice's start is wider
                    // than the threshold, close the open utterance
                    // before opening a new one for this slice. Flush
                    // the pending partial first so its text reaches
                    // the wrapper before the final lands.
                    if (have_open_utterance && params.stream_final_silence_ms > 0 &&
                        (s_start - last_speech_end_sample) * 1000 / SR >= params.stream_final_silence_ms) {
                        flush_pending_partial();
                        finalize_utterance();
                    }

                    if (!have_open_utterance) {
                        // Open utterance starting from this slice's
                        // window-relative start, clamped to the
                        // finalized boundary so a straddling slice
                        // (VAD min-silence wider than ours, so VAD did
                        // not break across the just-finalized utterance)
                        // doesn't pull the prior speech into the new
                        // utterance's redecode buffer. (For the no-VAD
                        // synthetic slice this still degrades cleanly:
                        // open_start = max(window_start, finalized).)
                        const int64_t open_start = std::max(s_start, finalized_until_sample);
                        int window_offset = (int)(open_start - window_start_sample);
                        if (window_offset < (int)sl.start)
                            window_offset = (int)sl.start;
                        open_utterance_at(window_offset, open_start);
                        utterance_just_opened = true;
                    }

                    if (s_end > last_speech_end_sample)
                        last_speech_end_sample = s_end;

                    // Round 3 (CKwasd #3): accumulate into the per-step
                    // partial buffer instead of emitting per slice.
                    if (!sl_text.empty()) {
                        if (!step_open_partial.empty())
                            step_open_partial += ' ';
                        step_open_partial += sl_text;
                    }
                }
                flush_pending_partial();

                // Append this step's NEW samples to utterance_pcm (only
                // when the utterance was already open at step start —
                // otherwise the open path already copied the relevant
                // tail of pcm_window, which includes the new samples).
                if (have_open_utterance && was_open_at_step_start && !utterance_just_opened) {
                    const float* base = pcm_window_data();
                    const size_t wsz = pcm_window_size();
                    utterance_pcm.insert(utterance_pcm.end(), base + wsz - n_new, base + wsz);
                }

                // Cap the per-utterance buffer so monologues don't OOM.
                // When exceeded, force-finalize and let the next step's
                // speech open a fresh utterance with a new id.
                if (have_open_utterance && (int64_t)utterance_pcm.size() > utterance_max_samples) {
                    last_speech_end_sample = now_sample;
                    finalize_utterance();
                }

                // End-of-step trailing-silence check.
                // Round 3 (CKwasd #2): no longer gated on
                // `step_slice_text.empty()`. The rolling pcm_window
                // keeps the last utterance's speech for up to
                // `--stream-length` ms after the speaker stops; VAD
                // keeps re-discovering that lingering slice every
                // step, which made `step_slice_text` non-empty and
                // blocked finalization for the full window length
                // (~18 s for `--stream-length 18000`). With the
                // round-3 issue-1 skip, those lingering slices are
                // already filtered out of the loop above, so by the
                // time we get here `last_speech_end_sample` is the
                // genuine end of the open utterance and the gap to
                // `now_sample` is the actual trailing silence. Fire
                // finalize whenever that gap crosses the threshold,
                // independent of whether anything was decoded this
                // step.
                if (have_open_utterance && params.stream_final_silence_ms > 0 && last_speech_end_sample > 0 &&
                    (now_sample - last_speech_end_sample) * 1000 / SR >= params.stream_final_silence_ms) {
                    finalize_utterance();
                }

                // Heartbeat silence event for consumers that need
                // timing even during pauses. Round 3: gate on
                // "no partial/final this step" rather than
                // "step_slice_text empty" so the wrapper still gets
                // a heartbeat when VAD found a slice but it was
                // filtered (fully-old) or its straddling subrange was
                // too short for the encoder.
                if (!emitted_event_this_step) {
                    fprintf(stdout, "{\"type\":\"silence\",\"t\":%.3f}\n", (double)now_sample / (double)SR);
                    fflush(stdout);
                }

                if (params.stream_monitor) {
                    fprintf(stderr, emitted_event_this_step ? "\xE2\x9C\x93" : "");
                    fflush(stderr);
                }
                continue;
            }

            if (segs.empty())
                continue;

            // Build output text (native diarization labels prefixed inline,
            // matching file-mode text/srt/vtt; no-op for non-diarizers).
            std::string text;
            for (const auto& s : segs)
                crispasr_stream_append_seg(text, s);

            // Output depends on mode:
            // Continuous: print each non-empty result as a new line
            // Normal: overwrite current line (dedup by text content)
            if (params.stream_continuous) {
                if (!text.empty()) {
                    fprintf(stdout, "%s\n", text.c_str());
                    fflush(stdout);
                }
            } else {
                if (!text.empty() && text != prev_text) {
                    // When --monitor is active, use newlines instead of
                    // in-place overwrite so the ✓ on stderr isn't erased
                    // by the \33[2K\r on stdout (they share one cursor).
                    if (params.stream_monitor) {
                        fprintf(stdout, "%s\n", text.c_str());
                    } else {
                        fprintf(stdout, "\33[2K\r%s", text.c_str());
                    }
                    fflush(stdout);
                    prev_text = text;
                }
            }

            // Print ✓ AFTER the stdout text so it isn't erased.
            if (params.stream_monitor) {
                fprintf(stderr, "\xE2\x9C\x93"); // ✓ = got text
                fflush(stderr);
            }
        }
        // Issue #84 round 2: flush any open utterance as a final on
        // EOF so wrappers don't miss the tail of the last spoken
        // region. Same dual-path (redecode / prefix) as the in-loop
        // finalize. `t1` falls back to `cumulative_samples / SR` when
        // we never saw a "speech ended" event before the pipe closed.
        if (params.stream_json && have_open_utterance) {
            // Round 3 (CKwasd #4): same trim as the in-loop finalize
            // so the EOF-flushed final's text covers exactly its
            // advertised [t0..t1] interval. If we never saw a VAD
            // speech-end before EOF (last_speech_end_sample == 0),
            // skip the trim and let the full buffer drive the
            // redecode — t1 also falls back to `cumulative_samples`
            // below in that case.
            if (last_speech_end_sample > utterance_start_sample) {
                const int64_t want = last_speech_end_sample - utterance_start_sample;
                if ((int64_t)utterance_pcm.size() > want)
                    utterance_pcm.resize((size_t)want);
            }
            // EOF path: identical redecode→stitch-fallback contract to the
            // in-loop finalize_utterance. See crispasr_stream_finalize.h.
            std::string final_text;
            std::string final_speaker; // native diarization label (single-speaker utterance); see in-loop finalize
            bool final_text_from_redecode = false;
            if (params.stream_final_mode == "redecode") {
                if ((int)utterance_pcm.size() >= crispasr::kStreamRedecodeMinSamples) {
                    whisper_params decode_params = params;
                    decode_params.vad = false;
                    decode_params.vad_model.clear();
                    // EOF also benefits from utterance-level delta
                    if ((backend->capabilities() & CAP_STREAM_UTTERANCE) && utterance_pcm_size_at_save > 0) {
                        const int n_new2 = (int)utterance_pcm.size() - (int)utterance_pcm_size_at_save;
                        backend->restore_utterance_cross_kv(n_new2, utterance_id);
                    }
                    auto utt_segs =
                        backend->transcribe(utterance_pcm.data(), (int)utterance_pcm.size(), 0, decode_params);
                    if (stream_punc_finals_enabled(params))
                        apply_punc_model(punc_ctx.get(), utt_segs);
                    apply_truecase_model(tc_ctx.get(), utt_segs);
                    apply_truecase_crf_model(tc_crf_ctx.get(), utt_segs);
                    apply_truecase_lstm_model(tc_lstm_ctx.get(), utt_segs);
                    apply_pcs_model(pcs_ctx.get(), utt_segs);
                    if (!params.punctuation) {
                        for (auto& seg : utt_segs)
                            crispasr_strip_punctuation(seg);
                    }
                    for (const auto& s : utt_segs)
                        final_text += s.text;
                    final_speaker = crispasr_stream_common_speaker(utt_segs);
                    final_text_from_redecode = !final_text.empty();
                }
                if (final_text.empty())
                    final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
            } else {
                final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
            }
            if (stream_punc_finals_enabled(params) && !final_text_from_redecode)
                final_text = apply_punc_text(punc_ctx.get(), final_text);
            const double t0 = (double)utterance_start_sample / (double)SR;
            const double t1 = last_speech_end_sample > 0 ? (double)last_speech_end_sample / (double)SR
                                                         : (double)cumulative_samples / (double)SR;
            std::string spk_field;
            if (!final_speaker.empty())
                spk_field = ",\"speaker\":\"" + crispasr_json_escape(final_speaker) + "\"";
            fprintf(stdout, "{\"type\":\"final\",\"utterance_id\":%lld,\"text\":\"%s\"%s,\"t0\":%.3f,\"t1\":%.3f}\n",
                    (long long)utterance_id, crispasr_json_escape(final_text).c_str(), spk_field.c_str(), t0, t1);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
        if (mic_pipe) {
            crispasr::crispasr_pclose(mic_pipe);
        }
        return 0;
    }

    // Process every input file.
    //
    // n_processors == 1 (default): sequential, single backend instance.
    // Bit-identical with the historical CrispASR behaviour.
    //
    // n_processors > 1: spawn N-1 EXTRA backend instances (model-load
    //                   cost paid N times — beware), then dispatch
    //                   files across N worker threads. Best when you
    //                   have many independent input files; useless on
    //                   single-file runs because both workers would
    //                   race on the same audio.
    int rc = 0;
    const int nproc = std::max(1, params.n_processors);
    if (nproc > 1 && params.fname_inp.size() > 1) {
        // Pre-load N-1 EXTRA backend instances (we already have one).
        // Failure to load any worker is fatal — better to bail than to
        // silently fall back to single-thread, which would surprise
        // batch users with much slower runs.
        std::vector<std::unique_ptr<CrispasrBackend>> pool;
        pool.reserve(nproc);
        pool.emplace_back(std::move(backend));
        for (int i = 1; i < nproc; i++) {
            auto extra = crispasr_create_backend(backend_name);
            if (!extra || !extra->init(params)) {
                fprintf(stderr,
                        "crispasr: error: failed to spin up worker %d/%d "
                        "(extra backend init failed). Try fewer --processors.\n",
                        i + 1, nproc);
                return 14;
            }
            pool.emplace_back(std::move(extra));
        }
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: parallel mode: %d worker(s), %zu input file(s)\n", nproc,
                    params.fname_inp.size());
        }

        // Shared work queue: index into params.fname_inp. std::atomic
        // counter is enough — no need for a real queue since each
        // worker just claims the next index.
        std::atomic<int> next_idx{0};
        std::atomic<int> agg_rc{0};
        const int n_files = (int)params.fname_inp.size();
        std::vector<std::thread> workers;
        workers.reserve((size_t)nproc);
        for (int w = 0; w < nproc; w++) {
            workers.emplace_back([&, w]() {
                CrispasrBackend& be = *pool[w];
                while (true) {
                    const int idx = next_idx.fetch_add(1);
                    if (idx >= n_files)
                        break;
                    const std::string fout =
                        (idx < (int)params.fname_out.size()) ? params.fname_out[idx] : std::string{};
                    const int file_rc =
                        process_one_input(be, params.fname_inp[idx], fout, params, punc_ctx.get(), tc_ctx.get(),
                                          pcs_ctx.get(), tc_crf_ctx.get(), tc_lstm_ctx.get());
                    if (file_rc != 0)
                        agg_rc.store(file_rc);
                }
            });
        }
        for (auto& t : workers)
            t.join();

        for (auto& be : pool)
            be->shutdown();
        punc_ctx.reset();
        return agg_rc.load();
    }

    for (size_t i = 0; i < params.fname_inp.size(); i++) {
        const std::string& fname_inp = params.fname_inp[i];
        const std::string fout = (i < params.fname_out.size()) ? params.fname_out[i] : std::string{};
        const int file_rc = process_one_input(*backend, fname_inp, fout, params, punc_ctx.get(), tc_ctx.get(),
                                              pcs_ctx.get(), tc_crf_ctx.get(), tc_lstm_ctx.get());
        if (file_rc != 0)
            rc = file_rc;
    }
    punc_ctx.reset();
    backend->shutdown();
    return rc;
}

#if 0
// Legacy in-place per-file loop body. Moved into process_one_input()
// above. Kept here under #if 0 only for diff/blame archaeology — the
// linker drops it.
{
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
        // Request stereo split when --diarize is set. Diarize is now
        // a generic dispatcher post-step (crispasr_diarize.cpp), so we
        // try it for every backend rather than only those that
        // advertise CAP_DIARIZE — the backend itself doesn't have to
        // know anything about stereo; the dispatcher labels its
        // segments after transcribe() returns.
        const bool want_stereo = params.diarize;
        if (!read_audio_data(fname_inp, samples, stereo, want_stereo)) {
            fprintf(stderr, "crispasr: error: failed to read audio '%s'\n",
                    fname_inp.c_str());
            rc = 20;
            continue;
        }
        bool have_stereo = want_stereo &&
            stereo.size() == 2 &&
            !stereo[0].empty() &&
            stereo[0].size() == stereo[1].size();
        // miniaudio duplicates mono -> both channels when we ask for
        // stereo, so a mono input file gives us pcmf32s[0] == pcmf32s[1].
        // Detect that and downgrade to mono so the diarize post-step
        // takes the mono-friendly path (vad-turns) instead of the
        // tie-only energy path.
        if (have_stereo) {
            const size_t n = stereo[0].size();
            const size_t check = std::min<size_t>(n, 4096);
            bool channels_equal = true;
            for (size_t i = 0; i < check; i++) {
                if (stereo[0][i] != stereo[1][i]) { channels_equal = false; break; }
            }
            if (channels_equal) have_stereo = false;
        }

        constexpr int SR = 16000;
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n",
                    (int)samples.size(),
                    (double)samples.size() / SR, SR, params.n_threads);
        }

        // Optional language-identification pre-step. Fires only when the
        // user asked for auto language (either --detect-language or
        // --language auto) AND the chosen backend can't detect language
        // natively (qwen3/whisper/parakeet already do). The detected ISO
        // code is written into `params.language` and, if empty, into
        // `params.source_lang` so canary can pick it up as well.
        const bool want_auto_lang = params.detect_language ||
                                    params.language == "auto";
        const bool has_native_lid = (backend->capabilities() & CAP_LANGUAGE_DETECT) != 0;
        const bool lid_disabled   = params.lid_backend == "off" ||
                                    params.lid_backend == "none";
        if (want_auto_lang && !has_native_lid && !lid_disabled) {
            crispasr_lid_result lid;
            // Backend self-probe first (see crispasr_lid_cli.h); external LID
            // only when it declines.
            const bool probed = crispasr_backend_probe_language(*backend, samples.data(),
                                                                (int)samples.size(), params, lid);
            if (probed ||
                crispasr_detect_language_cli(samples.data(), (int)samples.size(),
                                          params, lid)) {
                params.language = lid.lang_code;
                if (params.source_lang.empty()) {
                    params.source_lang = lid.lang_code;
                }
                if (!params.no_prints) {
                    fprintf(stderr,
                            "crispasr: LID -> language = '%s' (%s, p=%.3f)\n",
                            lid.lang_code.c_str(), lid.source.c_str(),
                            lid.confidence);
                }
            } else if (!params.no_prints) {
                fprintf(stderr,
                        "crispasr: LID failed, falling back to params.language='%s'\n",
                        params.language.c_str());
            }
        }
        // Free LID cache to reclaim GPU VRAM before ASR model loads
        crispasr_lid_free_cache();

        // Slice into chunks (VAD or fixed-window fallback).
        const auto slices = crispasr_compute_audio_slices(
            samples.data(), (int)samples.size(), SR,
            params.chunk_seconds, params);

        if (slices.empty()) {
            fprintf(stderr, "crispasr: warning: no speech detected in '%s'\n",
                    fname_inp.c_str());
            continue;
        }

        if (!params.no_prints && slices.size() > 1) {
            fprintf(stderr, "crispasr: processing %zu slice(s)\n", slices.size());
        }

        // Transcribe each slice.
        std::vector<std::vector<crispasr_segment>> per_slice;
        std::vector<crispasr_ctc_logits> per_slice_logits;
        per_slice.reserve(slices.size());
        per_slice_logits.reserve(slices.size());
        for (size_t i = 0; i < slices.size(); i++) {
            const auto & sl = slices[i];
            // Always transcribe in mono — every backend takes mono PCM
            // and the diarize step happens later as a generic post-pass.
            std::vector<crispasr_segment> segs = backend->transcribe(
                samples.data() + sl.start,
                sl.end - sl.start,
                sl.t0_cs,
                params);
            if (params.return_logits) {
                if (const auto* logits = backend->last_ctc_logits())
                    per_slice_logits.push_back(*logits);
                else
                    per_slice_logits.push_back({});
            }

            // Issue #267: run CTC alignment BEFORE diarization so word
            // timestamps are available for speaker-turn splitting.
            //
            // Issue #62: --force-aligner overrides both gates so users
            // can prefer aligner timing over native timestamps.
            const bool want_align =
                !params.aligner_model.empty() &&
                ((backend->capabilities() & CAP_TIMESTAMPS_CTC) || params.force_aligner);
            if (params.verbose) {
                fprintf(stderr,
                        "crispasr[verbose]: align: aligner='%s' caps_ctc=%d force=%d -> want=%d\n",
                        params.aligner_model.c_str(),
                        !!(backend->capabilities() & CAP_TIMESTAMPS_CTC),
                        params.force_aligner ? 1 : 0,
                        want_align ? 1 : 0);
            }
            if (want_align) {
                for (auto & seg : segs) {
                    if (!seg.words.empty() && !params.force_aligner) continue; // already aligned
                    auto words = crispasr_ctc_align(
                        params.aligner_model,
                        seg.text,
                        samples.data() + sl.start,
                        sl.end - sl.start,
                        sl.t0_cs,
                        params.n_threads);
                    if (crispasr_words_have_positive_span(words)) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }

            // Issue #267: diarize AFTER alignment so word timestamps
            // (native or externally aligned) are available for
            // speaker-turn splitting. Without words, falls back to
            // segment-level dominant-speaker assignment.
            if (params.diarize && !segs.empty()) {
                if (have_stereo) {
                    std::vector<float> sl_l(stereo[0].begin() + sl.start,
                                            stereo[0].begin() + sl.end);
                    std::vector<float> sl_r(stereo[1].begin() + sl.start,
                                            stereo[1].begin() + sl.end);
                    crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true,
                                           sl.t0_cs, segs, params);
                } else {
                    std::vector<float> mono_slice(samples.begin() + sl.start,
                                                  samples.begin() + sl.end);
                    crispasr_apply_diarize(mono_slice, mono_slice,
                                           /*is_stereo=*/false,
                                           sl.t0_cs, segs, params);
                }
            }

            per_slice.push_back(std::move(segs));

            if (params.print_progress && slices.size() > 1) {
                const int pct = (int)((i + 1) * 100 / slices.size());
                fprintf(stderr, "crispasr: progress = %3d%% (%zu/%zu slices)\n",
                        pct, i + 1, slices.size());
            }
        }
        auto all_segs = merge_segments(std::move(per_slice), slices);

        apply_punc_model(punc_ctx, all_segs);
        apply_truecase_model(tc_ctx, all_segs);
        apply_truecase_crf_model(tc_crf_ctx, all_segs);
        apply_truecase_lstm_model(tc_lstm_ctx, all_segs);
        
        apply_pcs_model(pcs_ctx, all_segs);

        // Optional post-processing: strip punctuation when --no-punctuation
        // is set. Cohere and canary pass p.punctuation through to their C
        // APIs natively and will usually return text that's already clean,
        // but this second pass is idempotent so the double application is
        // harmless. For the LLM backends (voxtral/voxtral4b/qwen3/granite)
        // this is the only way punctuation control happens — the models
        // don't take a "no punctuation" flag, they just generate whatever
        // the prompt pushes them towards.
        if (!params.punctuation) {
            for (auto & seg : all_segs) {
                crispasr_strip_punctuation(seg);
            }
        }

        // Build display segments.
        const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

        // Print to stdout.
        const bool show_timestamps =
            !params.no_timestamps &&
            (params.output_srt || params.output_vtt ||
             params.max_len > 0  || params.print_colors ||
             params.diarize);
        crispasr_print_stdout(disp, show_timestamps);
        if (params.print_confidence)
            crispasr_print_confidence(all_segs);

        // Write output files.
        if (params.output_txt)
            crispasr_write_txt(out_path(".txt"), disp);
        if (params.output_srt)
            crispasr_write_srt(out_path(".srt"), disp);
        if (params.output_vtt)
            crispasr_write_vtt(out_path(".vtt"), disp);
        if (params.output_csv)
            crispasr_write_csv(out_path(".csv"), disp);
        if (params.output_lrc)
            crispasr_write_lrc(out_path(".lrc"), disp);
        if (params.output_jsn)
            crispasr_write_json(
                out_path(".json"),
                all_segs, backend->name(), params.model, params.language,
                params.output_jsn_full, nullptr);
        if (params.return_logits) {
            crispasr_ctc_logits merged;
            for (const auto& lg : per_slice_logits) {
                if (lg.data.empty() || lg.n_vocab <= 0 || lg.n_frames <= 0)
                    continue;
                if (merged.n_vocab == 0) {
                    merged.n_vocab = lg.n_vocab;
                    merged.normalization = lg.normalization;
                    merged.vocab = lg.vocab;
                }
                if (merged.n_vocab != lg.n_vocab)
                    continue;
                merged.data.insert(merged.data.end(), lg.data.begin(), lg.data.end());
                merged.n_frames += lg.n_frames;
            }
            if (!merged.data.empty())
                crispasr_write_ctc_logits_json(out_path(".ctc-logits.json"), merged, backend->name());
        }
    }

    if (punc_ctx) fireredpunc_free(punc_ctx);
    return 0;
}
#endif
