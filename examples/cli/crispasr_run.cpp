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
#include "crispasr_mic_cli.h"
#include "crispasr_popen.h"
#include "crispasr_vad_cli.h"
#include "crispasr_output.h"
#include "crispasr_punctuation_policy.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_aligner_cli.h"
#include "crispasr_lid_cli.h"
#include "crispasr_lid.h" // crispasr_lid_free_cache()
#include "crispasr_diarize_cli.h"
#include "crispasr_mem.h"
#include "crispasr_stream_finalize.h"
#include "whisper_params.h"
#include "fireredpunc.h"
#include "truecaser.h"
#include "truecaser_crf.h"
#include "truecaser_lstm.h"
#include "pcs.h"
#include "titanet.h"
#include "speaker_db.h"

#include "common-crispasr.h" // read_audio_data

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

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
    return out;
}

// Stdout serialization mutex. Used by the parallel-processors path to
// keep stdout transcript lines from interleaving across worker threads.
// The single-threaded path acquires it too — no measurable cost since
// it's an uncontended lock when n_processors == 1.
std::mutex g_stdout_mutex;

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
    if (!read_audio_data(fname_inp, samples, stereo, want_stereo)) {
        fprintf(stderr, "crispasr: error: failed to read audio '%s'\n", fname_inp.c_str());
        return 20;
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
        double dur = (double)samples.size() / 16000.0;
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

    constexpr int SR = 16000;
    if (!params.no_prints) {
        fprintf(stderr, "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n", (int)samples.size(),
                (double)samples.size() / SR, SR, params.n_threads);
    }

    // Speaker enrollment mode: extract TitaNet embedding, save to DB, exit.
    if (!params.enroll_speaker.empty()) {
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
        if (!speaker_db_enroll(db_dir.c_str(), params.enroll_speaker.c_str(), emb, dim)) {
            fprintf(stderr, "crispasr: error: failed to enroll speaker '%s'\n", params.enroll_speaker.c_str());
            return 24;
        }
        return 0;
    }

    // Optional language-identification pre-step.
    const bool want_auto_lang = params.detect_language || params.language == "auto";
    const bool has_native_lid = (backend.capabilities() & CAP_LANGUAGE_DETECT) != 0;
    const bool lid_disabled = params.lid_backend == "off" || params.lid_backend == "none";
    crispasr_lid_info lid_info; // stored for JSON output
    if (want_auto_lang && !has_native_lid && !lid_disabled) {
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

    const auto slices =
        crispasr_compute_audio_slices(samples.data(), (int)samples.size(), SR, params.chunk_seconds, params);

    if (slices.empty()) {
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
                    auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + s, e - s, seg.t0,
                                                    params.n_threads);
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }
        }

        // Fall through to the shared output path below by wrapping
        // the stitched result into per_slice / all_segs.
        std::vector<std::vector<crispasr_segment>> stitched_per_slice(1);
        stitched_per_slice[0] = std::move(segs);
        auto all_segs = merge_segments(std::move(stitched_per_slice), slices);

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
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            crispasr_print_stdout(disp, show_timestamps);
            if (params.show_alternatives)
                crispasr_print_alternatives(all_segs, params.n_alternatives);
        }
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
        return 0;
    }

    // --------------- Per-slice path (non-VAD or single slice) ---------------
    // Process VAD slices — parallel when multiple slices AND n_processors > 1
    std::vector<std::vector<crispasr_segment>> per_slice(slices.size());

    auto process_slice = [&](size_t i, CrispasrBackend& be) {
        const auto& sl = slices[i];
        std::vector<crispasr_segment> segs =
            be.transcribe(samples.data() + sl.start, sl.end - sl.start, sl.t0_cs, params);

        if (params.diarize && !segs.empty()) {
            if (have_stereo) {
                std::vector<float> sl_l(stereo[0].begin() + sl.start, stereo[0].begin() + sl.end);
                std::vector<float> sl_r(stereo[1].begin() + sl.start, stereo[1].begin() + sl.end);
                crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true, sl.t0_cs, segs, params);
            } else {
                std::vector<float> mono_slice(samples.begin() + sl.start, samples.begin() + sl.end);
                crispasr_apply_diarize(mono_slice, mono_slice,
                                       /*is_stereo=*/false, sl.t0_cs, segs, params);
            }
        }

        // Speaker identification: match diarized speakers against profile DB.
        // Also supports standalone speaker ID (without diarize) when --speaker-db is set.
        if (!params.speaker_db.empty() && !segs.empty()) {
            static titanet_context* spk_ctx = nullptr;
            static speaker_db* spk_db = nullptr;
            if (!spk_ctx) {
                std::string tm = params.titanet_model;
                if (tm.empty() || tm == "auto")
                    tm = crispasr_resolve_model("auto", "titanet", params.no_prints, params.cache_dir,
                                                params.auto_download, "");
                if (!tm.empty())
                    spk_ctx = titanet_init(tm.c_str(), params.n_threads);
            }
            if (!spk_db)
                spk_db = speaker_db_load(params.speaker_db.c_str());
            if (spk_ctx && spk_db && speaker_db_count(spk_db) > 0) {
                // Extract embedding for the whole slice and match
                float emb[192];
                int dim = titanet_embed(spk_ctx, samples.data() + sl.start, sl.end - sl.start, emb);
                if (dim > 0) {
                    float score = 0;
                    const char* name = speaker_db_match(spk_db, emb, dim, params.speaker_threshold, &score);
                    if (name) {
                        std::string label = std::string("(") + name + ") ";
                        for (auto& seg : segs)
                            seg.speaker = label;
                    }
                }
            }
        }

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
                auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + sl.start,
                                                sl.end - sl.start, sl.t0_cs, params.n_threads);
                if (!words.empty()) {
                    seg.t0 = words.front().t0;
                    seg.t1 = words.back().t1;
                    seg.words = std::move(words);
                }
            }
        }

        per_slice[i] = std::move(segs);
    };

    const int n_workers = std::min(params.n_processors, (int32_t)slices.size());

    if (n_workers > 1 && slices.size() > 1) {
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
        int srt_index = 1; // running SRT entry counter
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
            apply_punc_model(punc_ctx, all_segs);
            apply_truecase_model(tc_ctx, all_segs);
            apply_truecase_crf_model(tc_crf_ctx, all_segs);
            apply_truecase_lstm_model(tc_lstm_ctx, all_segs);

            apply_pcs_model(pcs_ctx, all_segs);
            if (!params.punctuation)
                for (auto& seg : all_segs)
                    crispasr_strip_punctuation(seg);
            auto disp_all = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

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
        }
        return 0;
    } else {
        // Sequential (single slice or n_processors == 1)
        for (size_t i = 0; i < slices.size(); i++)
            process_slice(i, backend);
    }
    auto all_segs = merge_segments(std::move(per_slice), slices);

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

        // Serialize stdout across parallel workers so multi-file
        // transcripts don't interleave line-by-line.
        std::lock_guard<std::mutex> lock(g_stdout_mutex);
        crispasr_print_stdout(disp, show_timestamps);
        if (params.show_alternatives) {
            crispasr_print_alternatives(all_segs, params.n_alternatives);
        }
    }

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

    return 0;
}

} // namespace

int crispasr_run_backend(const whisper_params& params_in) {
    whisper_params params = params_in;

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

    // ---- TTS mode: synthesize speech from text ----
    if (!params.tts_text.empty()) {
        if (!(backend->capabilities() & CAP_TTS)) {
            fprintf(stderr, "crispasr: error: backend '%s' does not support TTS\n", backend_name.c_str());
            return 14;
        }

        auto audio = backend->synthesize(params.tts_text, params);
        if (audio.empty()) {
            fprintf(stderr, "crispasr: error: TTS synthesis failed\n");
            return 15;
        }

        // Optional leading-silence trim. RMS gate over a 20 ms window;
        // drop frames below -50 dBFS (≈ 0.0032 RMS) until the gate
        // opens, then back off 50 ms so we don't clip the consonant onset.
        if (params.tts_trim_silence) {
            const int sr_in = 24000;
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

        // Write output WAV (24 kHz mono)
        std::string out_path = params.tts_output.empty() ? "tts_output.wav" : params.tts_output;
        FILE* fout = fopen(out_path.c_str(), "wb");
        if (!fout) {
            fprintf(stderr, "crispasr: error: cannot write '%s'\n", out_path.c_str());
            return 16;
        }
        // WAV header: 24 kHz, mono, 16-bit PCM
        int32_t sr = 24000;
        int16_t channels = 1;
        int16_t bits = 16;
        int32_t data_size = (int32_t)audio.size() * 2;
        int32_t file_size = 36 + data_size;
        fwrite("RIFF", 1, 4, fout);
        fwrite(&file_size, 4, 1, fout);
        fwrite("WAVEfmt ", 1, 8, fout);
        int32_t fmt_size = 16;
        fwrite(&fmt_size, 4, 1, fout);
        int16_t fmt_tag = 1; // PCM
        fwrite(&fmt_tag, 2, 1, fout);
        fwrite(&channels, 2, 1, fout);
        fwrite(&sr, 4, 1, fout);
        int32_t byte_rate = sr * channels * (bits / 8);
        fwrite(&byte_rate, 4, 1, fout);
        int16_t block_align = channels * (bits / 8);
        fwrite(&block_align, 2, 1, fout);
        fwrite(&bits, 2, 1, fout);
        fwrite("data", 1, 4, fout);
        fwrite(&data_size, 4, 1, fout);
        // Convert float → int16
        for (size_t i = 0; i < audio.size(); i++) {
            float s = audio[i];
            if (s > 1.0f)
                s = 1.0f;
            if (s < -1.0f)
                s = -1.0f;
            int16_t v = (int16_t)(s * 32767.0f);
            fwrite(&v, 2, 1, fout);
        }
        fclose(fout);

        if (!params.no_prints)
            fprintf(stderr, "crispasr: TTS output written to '%s' (%zu samples, %.2f sec)\n", out_path.c_str(),
                    audio.size(), audio.size() / 24000.0);
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
    std::unique_ptr<fireredpunc_context, decltype(&fireredpunc_free)> punc_ctx(nullptr, fireredpunc_free);
    {
        std::string punc_path = params.punc_model;
        if (punc_path == "none" || punc_path == "off")
            punc_path.clear();
        if (punc_path == "auto" || punc_path == "firered") {
            punc_path = crispasr_cache::ensure_cached_file(
                "fireredpunc-q4_k.gguf",
                "https://huggingface.co/cstr/fireredpunc-GGUF/resolve/main/fireredpunc-q4_k.gguf", params.no_prints,
                "crispasr[punc]", params.cache_dir);
        } else if (punc_path == "fullstop") {
            punc_path = crispasr_cache::ensure_cached_file(
                "fullstop-punc-q4_k.gguf",
                "https://huggingface.co/cstr/fullstop-punc-multilang-GGUF/resolve/main/fullstop-punc-q4_k.gguf",
                params.no_prints, "crispasr[punc]", params.cache_dir);
        } else if (punc_path == "punctuate-all") {
            punc_path = crispasr_cache::ensure_cached_file(
                "punctuate-all-q4_k.gguf",
                "https://huggingface.co/cstr/punctuate-all-GGUF/resolve/main/punctuate-all-q4_k.gguf", params.no_prints,
                "crispasr[punc]", params.cache_dir);
        } else if (punc_path == "pcs" || punc_path.find("pcs") != std::string::npos) {
            // PCS is handled separately below — skip fireredpunc loading
            punc_path.clear();
        }
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
    // PCS: detect from keyword or filename containing "pcs"
    std::unique_ptr<pcs_context, decltype(&pcs_free)> pcs_ctx(nullptr, pcs_free);
    {
        std::string pcs_path;
        if (params.punc_model == "pcs") {
            pcs_path = crispasr_cache::ensure_cached_file(
                "pcs-xlmr-base-q4_k.gguf",
                "https://huggingface.co/cstr/pcs-xlmr-base-GGUF/resolve/main/pcs-xlmr-base-q4_k.gguf", params.no_prints,
                "crispasr[pcs]", params.cache_dir);
        } else if (params.punc_model.find("pcs") != std::string::npos &&
                   params.punc_model.find(".gguf") != std::string::npos) {
            // Direct path to a PCS GGUF file
            pcs_path = params.punc_model;
        }
        if (!pcs_path.empty()) {
            pcs_ctx.reset(pcs_init(pcs_path.c_str()));
            if (pcs_ctx) {
                punc_ctx.reset(); // PCS replaces fireredpunc
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: loaded PCS model '%s'\n", pcs_path.c_str());
            }
        }
    }

    // Optional truecaser post-processor.
    // `--truecase-model auto` / `de` → statistical German truecaser (~9 MB).
    // `--truecase-model crf`  → CRF German truecaser with context (~25 MB).
    // `--truecase-model lstm` → BiLSTM character-level truecaser (~3 MB, best quality).
    std::unique_ptr<truecaser_context, decltype(&truecaser_free)> tc_ctx(nullptr, truecaser_free);
    std::unique_ptr<truecaser_crf_context, decltype(&truecaser_crf_free)> tc_crf_ctx(nullptr, truecaser_crf_free);
    std::unique_ptr<truecaser_lstm_context, decltype(&truecaser_lstm_free)> tc_lstm_ctx(nullptr, truecaser_lstm_free);
    {
        std::string tc_path = params.truecase_model;
        if (tc_path == "none" || tc_path == "off")
            tc_path.clear();
        if (tc_path == "lstm" || tc_path == "lstm-de" || tc_path == "lstm-en" || tc_path == "lstm-es" ||
            tc_path == "lstm-ru") {
            // Map language suffix to filename
            std::string lang = "de";
            if (tc_path == "lstm-en")
                lang = "en";
            else if (tc_path == "lstm-es")
                lang = "es";
            else if (tc_path == "lstm-ru")
                lang = "ru";
            std::string fname = "truecaser-lstm-" + lang + ".bin";
            std::string url = "https://huggingface.co/cstr/truecaser-de/resolve/main/" + fname;
            tc_path =
                crispasr_cache::ensure_cached_file(fname, url, params.no_prints, "crispasr[tc]", params.cache_dir);
            if (!tc_path.empty()) {
                tc_lstm_ctx.reset(truecaser_lstm_init(tc_path.c_str()));
                if (tc_lstm_ctx && !params.no_prints)
                    fprintf(stderr, "crispasr: loaded BiLSTM truecaser '%s'\n", tc_path.c_str());
            }
            tc_path.clear();
        } else if (tc_path == "crf" || tc_path == "crf-de") {
            tc_path = crispasr_cache::ensure_cached_file(
                "truecaser-crf-de.bin", "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-crf-de.bin",
                params.no_prints, "crispasr[tc]", params.cache_dir);
            if (!tc_path.empty()) {
                tc_crf_ctx.reset(truecaser_crf_init(tc_path.c_str()));
                if (tc_crf_ctx && !params.no_prints)
                    fprintf(stderr, "crispasr: loaded CRF truecaser '%s'\n", tc_path.c_str());
            }
            tc_path.clear();
        } else if (!tc_path.empty() && tc_path != "auto" && tc_path != "de") {
            // Direct path — detect format by magic bytes
            FILE* probe = fopen(tc_path.c_str(), "rb");
            if (probe) {
                char magic[4] = {};
                fread(magic, 1, 4, probe);
                fclose(probe);
                if (memcmp(magic, "LSTM", 4) == 0) {
                    tc_lstm_ctx.reset(truecaser_lstm_init(tc_path.c_str()));
                    if (tc_lstm_ctx && !params.no_prints)
                        fprintf(stderr, "crispasr: loaded BiLSTM truecaser '%s'\n", tc_path.c_str());
                    tc_path.clear();
                } else if (memcmp(magic, "CRF1", 4) == 0) {
                    tc_crf_ctx.reset(truecaser_crf_init(tc_path.c_str()));
                    if (tc_crf_ctx && !params.no_prints)
                        fprintf(stderr, "crispasr: loaded CRF truecaser '%s'\n", tc_path.c_str());
                    tc_path.clear();
                }
            }
        }
        if (tc_path == "auto" || tc_path == "de") {
            tc_path = crispasr_cache::ensure_cached_file(
                "truecaser-de.bin", "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-de.bin",
                params.no_prints, "crispasr[tc]", params.cache_dir);
        }
        if (!tc_path.empty()) {
            tc_ctx.reset(truecaser_init(tc_path.c_str()));
            if (!tc_ctx) {
                fprintf(stderr, "crispasr: warning: failed to load truecaser '%s' — continuing without\n",
                        tc_path.c_str());
            } else if (!params.no_prints) {
                fprintf(stderr, "crispasr: loaded truecaser '%s'\n", tc_path.c_str());
            }
        }
    }

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
        int64_t cumulative_samples = 0;
        const int64_t utterance_max_samples = (int64_t)params.stream_utterance_max_sec * SR;
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

            // Convert s16le to float
            const size_t n_new = n_read;
            const size_t prev_size = pcm_window.size();
            pcm_window.resize(prev_size + n_new);
            for (size_t i = 0; i < n_new; i++)
                pcm_window[prev_size + i] = read_buf[i] / 32768.0f;

            // Issue #84: enforce the rolling cap by dropping the
            // oldest samples once we exceed `--stream-length`. This
            // is the only place the buffer can shrink — there is no
            // separate "keep" tail because the whole tail up to
            // `length_samples` is now the context window. The legacy
            // `--stream-keep` flag is accepted for compatibility but
            // is no longer wired in (see help text + docs/streaming.md).
            if ((int)pcm_window.size() > length_samples) {
                pcm_window.erase(pcm_window.begin(), pcm_window.end() - length_samples);
            }
            cumulative_samples += (int64_t)n_new;
            (void)keep_samples; // legacy, intentionally unused

            // Monitor: show progress during processing
            if (params.stream_monitor) {
                fprintf(stderr, "\xE2\x96\xB6"); // ▶ = processing chunk
                fflush(stderr);
            }

            std::vector<crispasr_segment> segs;
            // Per-slice text, kept alongside the aggregate `segs` for the
            // round-2 JSON state machine. Each entry is `(slice, text)`
            // where the text has had `apply_punc_model` + (optionally)
            // strip-punctuation applied, matching what the aggregate
            // `segs` will look like after the post-loop processing —
            // the aggregate is the source of truth for the legacy plain
            // text path; this side-channel exists so the JSON path can
            // attribute each slice's text to the right utterance.
            std::vector<std::pair<crispasr_audio_slice, std::string>> step_slice_text;
            if (!stream_vad_path.empty()) {
                const auto slices = crispasr_compute_vad_slices(pcm_window.data(), (int)pcm_window.size(), SR,
                                                                stream_vad_path.c_str(), stream_vad_opts);
                // Snapshot for the straddling-slice subrange decode below.
                // `cumulative_samples` has already been advanced by
                // `n_new` for this step, so the rolling window currently
                // covers [window_start_sample_now, cumulative_samples).
                // finalized_until_sample is updated inside the JSON state
                // machine *after* this loop runs, so it reflects the
                // upper bound of all utterances finalized before this step.
                const int64_t window_start_sample_now = cumulative_samples - (int64_t)pcm_window.size();
                constexpr int kStraddleMinSamples = 32000; // 2 s @ 16 kHz; backend-safe tail decode floor.
                for (const auto& sl : slices) {
                    const int64_t s_start_abs = window_start_sample_now + (int64_t)sl.start;
                    const int64_t s_end_abs = window_start_sample_now + (int64_t)sl.end;
                    if (params.stream_json && s_end_abs <= finalized_until_sample)
                        continue;

                    std::vector<crispasr_segment> slice_segs;
                    if (params.stream_json) {
                        // Apply punc/strip on a copy so the per-slice text
                        // is in its final form before we hand it to the
                        // utterance state machine. JSON mode can filter or
                        // trim finalized rolling-window slices before decode;
                        // plain-text mode below still decodes full slices.
                        std::vector<crispasr_segment> sl_for_text;

                        // Round 3 (CKwasd #1 corner): if the VAD slice
                        // straddles a previously-finalized boundary
                        // (s_start < finalized_until_sample < s_end), the
                        // full-slice decode covers audio that belongs to
                        // the prior utterance — emitting it as a partial
                        // for the new utterance_id leaks prior text into
                        // the live stream. Decode just the post-finalized
                        // subrange so partial.text describes only the new
                        // utterance's interval.
                        // Some backends (moonshine's stacked conv1d
                        // encoder, for example) abort on inputs shorter
                        // than a few hundred ms — `OW > 0` from
                        // `ggml_im2col`. Gate the subrange decode on a
                        // generous min so we don't crash; fall back to
                        // suppressing the partial in that case so we
                        // don't leak the prior utterance's text either.
                        // 32000 samples = 2 s @ 16 kHz, comfortably
                        // above every supported backend's encoder
                        // minimum.
                        if (s_start_abs < finalized_until_sample) {
                            const int sub_start = (int)(finalized_until_sample - window_start_sample_now);
                            const int sub_end = (int)sl.end;
                            const int sub_len = sub_end - sub_start;
                            if (sub_start >= 0 && sub_len >= kStraddleMinSamples && sub_end <= (int)pcm_window.size()) {
                                whisper_params decode_params = params;
                                decode_params.vad = false;
                                decode_params.vad_model.clear();
                                sl_for_text =
                                    backend->transcribe(pcm_window.data() + sub_start, sub_len, 0, decode_params);
                                slice_segs = sl_for_text;
                            }
                            // else: sl_for_text stays empty → empty
                            // partial text for this slice, which
                            // `flush_pending_partial` will skip. The
                            // straddling slice's prior-utterance audio
                            // never reaches the wrapper; the genuine
                            // new audio will be picked up on a later
                            // step once the subrange exceeds the min.
                        } else {
                            slice_segs =
                                backend->transcribe(pcm_window.data() + sl.start, sl.end - sl.start, sl.t0_cs, params);
                            sl_for_text = slice_segs;
                        }
                        apply_punc_model(punc_ctx.get(), sl_for_text);
                        apply_truecase_model(tc_ctx.get(), sl_for_text);
                        apply_truecase_crf_model(tc_crf_ctx.get(), sl_for_text);
                        apply_truecase_lstm_model(tc_lstm_ctx.get(), sl_for_text);
                        apply_pcs_model(pcs_ctx.get(), sl_for_text);
                        if (!params.punctuation) {
                            for (auto& seg : sl_for_text)
                                crispasr_strip_punctuation(seg);
                        }
                        std::string sl_text;
                        for (const auto& s : sl_for_text)
                            sl_text += s.text;
                        step_slice_text.emplace_back(sl, std::move(sl_text));
                    } else {
                        slice_segs =
                            backend->transcribe(pcm_window.data() + sl.start, sl.end - sl.start, sl.t0_cs, params);
                    }
                    segs.insert(segs.end(), std::make_move_iterator(slice_segs.begin()),
                                std::make_move_iterator(slice_segs.end()));
                }
            } else {
                segs = backend->transcribe(pcm_window.data(), (int)pcm_window.size(), 0, params);
            }

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
            // No-VAD JSON path: synthesize a single "slice" covering
            // the whole window so the same state machine handles both
            // VAD-on and VAD-off cases.
            if (params.stream_json && stream_vad_path.empty() && !segs.empty()) {
                std::string all_text;
                for (const auto& s : segs)
                    all_text += s.text;
                crispasr_audio_slice fake_sl{0, (int)pcm_window.size(), 0, 0};
                step_slice_text.emplace_back(fake_sl, std::move(all_text));
            }

            if (params.stream_monitor && segs.empty()) {
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
            // by accumulating a longest-common-prefix across partials
            // instead of re-decoding.
            if (params.stream_json) {
                const int64_t now_sample = cumulative_samples;
                const int64_t window_start_sample = cumulative_samples - (int64_t)pcm_window.size();
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
                    if (params.stream_final_mode == "redecode") {
                        if ((int)utterance_pcm.size() >= crispasr::kStreamRedecodeMinSamples) {
                            // Disable nested VAD: utterance_pcm is already
                            // the speech region we identified, no need to
                            // re-segment it (and the nested VAD path would
                            // discover the same slices we're collapsing here).
                            whisper_params decode_params = params;
                            decode_params.vad = false;
                            decode_params.vad_model.clear();
                            auto utt_segs =
                                backend->transcribe(utterance_pcm.data(), (int)utterance_pcm.size(), 0, decode_params);
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
                        }
                        if (final_text.empty())
                            final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
                    } else {
                        // prefix mode: stitch committed prefix + last partial tail
                        final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
                    }
                    const double t0 = (double)utterance_start_sample / (double)SR;
                    const double t1 = (double)last_speech_end_sample / (double)SR;
                    fprintf(stdout,
                            "{\"type\":\"final\",\"utterance_id\":%lld,\"text\":\"%s\",\"t0\":%.3f,\"t1\":%.3f}\n",
                            (long long)utterance_id, crispasr_json_escape(final_text).c_str(), t0, t1);
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
                };

                auto open_utterance_at = [&](int window_offset, int64_t stream_start) {
                    utterance_id++;
                    have_open_utterance = true;
                    utterance_start_sample = stream_start;
                    last_speech_end_sample = stream_start;
                    if (window_offset < 0)
                        window_offset = 0;
                    if (window_offset > (int)pcm_window.size())
                        window_offset = (int)pcm_window.size();
                    utterance_pcm.assign(pcm_window.begin() + window_offset, pcm_window.end());
                    prefix_committed.clear();
                    last_partial_text.clear();
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
                    utterance_pcm.insert(utterance_pcm.end(), pcm_window.end() - n_new, pcm_window.end());
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

            // Build output text
            std::string text;
            for (const auto& s : segs)
                text += s.text;

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
            if (params.stream_final_mode == "redecode") {
                if ((int)utterance_pcm.size() >= crispasr::kStreamRedecodeMinSamples) {
                    whisper_params decode_params = params;
                    decode_params.vad = false;
                    decode_params.vad_model.clear();
                    auto utt_segs =
                        backend->transcribe(utterance_pcm.data(), (int)utterance_pcm.size(), 0, decode_params);
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
                }
                if (final_text.empty())
                    final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
            } else {
                final_text = crispasr::stitch_partial_accumulator(prefix_committed, last_partial_text);
            }
            const double t0 = (double)utterance_start_sample / (double)SR;
            const double t1 = last_speech_end_sample > 0 ? (double)last_speech_end_sample / (double)SR
                                                         : (double)cumulative_samples / (double)SR;
            fprintf(stdout, "{\"type\":\"final\",\"utterance_id\":%lld,\"text\":\"%s\",\"t0\":%.3f,\"t1\":%.3f}\n",
                    (long long)utterance_id, crispasr_json_escape(final_text).c_str(), t0, t1);
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
            if (crispasr_detect_language_cli(samples.data(), (int)samples.size(),
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
        per_slice.reserve(slices.size());
        for (size_t i = 0; i < slices.size(); i++) {
            const auto & sl = slices[i];
            // Always transcribe in mono — every backend takes mono PCM
            // and the diarize step happens later as a generic post-pass.
            std::vector<crispasr_segment> segs = backend->transcribe(
                samples.data() + sl.start,
                sl.end - sl.start,
                sl.t0_cs,
                params);

            // Apply the generic diarize post-step. Stereo-only methods
            // (energy, xcorr) need have_stereo == true; mono-friendly
            // methods (vad-turns, future sherpa/pyannote) work either
            // way. Pass both channel buffers and an is_stereo hint;
            // when have_stereo is false we point both at the mono
            // buffer so the helper has data to look at without
            // special-casing.
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

            // Optional CTC forced alignment to attach word-level timestamps.
            // Applies to backends that expose CAP_TIMESTAMPS_CTC and don't
            // already have words populated. Runs per slice so absolute
            // timestamps come out right.
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
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }

            per_slice.push_back(std::move(segs));
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
    }

    if (punc_ctx) fireredpunc_free(punc_ctx);
    return 0;
}
#endif
