// crispasr-repack-probe — does selecting ggml's CPU repack extra buffer type
// make quantised MUL_MAT faster on *this* host, and for which quant types?
//
// Background: docs/ggml-optimisation-playbook.md §4 and §8. ggml's repacked
// int8 GEMM (ggml/src/ggml-cpu/repack.cpp) is reachable only through the CPU
// device's *extra* buffer types (ggml_backend_dev_get_extra_bufts). Only
// src/crispasr.cpp requests them; everything loading through
// core_gguf::load_weights uses the default buffer type and cannot.
//
// This tool answers three questions without needing a model:
//   1. Is a repack buffer type offered at all on this host's ISA?
//   2. For a given quant type and weight shape, does the repack buft actually
//      accept the tensor (ggml_repack_get_optimal_repack_type != nullptr)?
//   3. If it does, how much faster is the MUL_MAT?
//
// Arms are interleaved round-robin, not run sequentially, so that load on a
// contended box perturbs both arms equally (playbook §6.9).
//
// Usage:
//   crispasr-repack-probe [--threads N] [--reps N] [--shape K,N,M]...
//                         [--arm both|default|repack] [--type f32|q8_0|...]
//
// `--arm` runs ONE arm only and prints a single machine-readable line per
// (shape, type). The tree's A/B discipline (BASIC_PITCH_CONV_PERF.md, and
// .github/workflows/basic-pitch-conv-ab.yml) wants every arm in its OWN
// PROCESS, because a shared process warms caches across arms and because a
// process-local static can make the second arm not be the arm you think. The
// default `both` mode interleaves in-process, which is the right tool on a
// contended box where between-process drift is the larger error; on a clean
// runner, prefer one process per arm and discard the cold run.
//
// NOTE: report the ISA with every number. A negative result on a machine
// without AVX-512 VNNI / AMX (x86) or dotprod / i8mm (arm64) does not
// generalise to one that has them.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Shape {
    int64_t K; // weight ne[0] — reduction dim
    int64_t N; // weight ne[1] — output rows
    int64_t M; // batch columns
};

// One arm of the A/B: a backend, a weight in a chosen buffer type, and a
// single-node MUL_MAT graph, all kept alive so repeats can be interleaved.
struct Arm {
    std::string label;
    ggml_backend_t backend = nullptr;
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_context* cctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* y = nullptr;
    bool usable = false;
    bool repacked = false;
    const char* buft_name = "";
    std::vector<double> times;
    double checksum = 0;

    ~Arm() {
        if (alloc)
            ggml_gallocr_free(alloc);
        if (cctx)
            ggml_free(cctx);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (wctx)
            ggml_free(wctx);
        if (backend)
            ggml_backend_free(backend);
    }
};

// Build one arm. `extra_buft` is null for the default-buffer-type arm.
bool build_arm(Arm& a, ggml_type wt, Shape s, int n_threads, ggml_backend_buffer_type_t extra_buft, const char* label) {
    a.label = label;
    a.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(a.backend, n_threads);

    ggml_backend_buffer_type_t wbuft = extra_buft ? extra_buft : ggml_backend_get_default_buffer_type(a.backend);
    a.buft_name = ggml_backend_buft_name(wbuft);

    ggml_init_params wp = {ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true};
    a.wctx = ggml_init(wp);
    ggml_tensor* w = ggml_new_tensor_2d(a.wctx, wt, s.K, s.N);
    ggml_set_name(w, "w");

    a.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(a.wctx, wbuft);
    if (!a.wbuf) {
        fprintf(stderr, "%s: weight alloc failed\n", label);
        return false;
    }

    // The repack buffer type's init_tensor sets tensor->extra to the repack
    // traits for this (type, shape, ISA) triple, or leaves it null when there
    // is no repacked kernel. Its set_tensor then dereferences tensor->extra
    // *unconditionally* (ggml/src/ggml-cpu/repack.cpp: repack_buffer_set_tensor),
    // so writing a tensor the buft declined is a null dereference, not a
    // graceful fallback. Any loader that selects this buft must check first.
    a.repacked = (w->extra != nullptr);
    if (extra_buft && !a.repacked) {
        a.usable = false;
        return true; // not an error: a reportable "declined"
    }

    {
        std::vector<float> src((size_t)s.K * s.N);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& v : src)
            v = d(rng);
        if (wt == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, src.data(), 0, ggml_nbytes(w));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(w));
            ggml_quantize_chunk(wt, src.data(), q.data(), 0, s.N, s.K, nullptr);
            ggml_backend_tensor_set(w, q.data(), 0, ggml_nbytes(w));
        }
    }

    ggml_init_params cp = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    a.cctx = ggml_init(cp);
    ggml_tensor* x = ggml_new_tensor_2d(a.cctx, GGML_TYPE_F32, s.K, s.M);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    a.y = ggml_mul_mat(a.cctx, w, x);
    ggml_set_output(a.y);
    a.gf = ggml_new_graph(a.cctx);
    ggml_build_forward_expand(a.gf, a.y);

    a.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(a.backend));
    if (!ggml_gallocr_alloc_graph(a.alloc, a.gf)) {
        fprintf(stderr, "%s: graph alloc failed\n", label);
        return false;
    }
    {
        std::vector<float> xs((size_t)s.K * s.M);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (auto& v : xs)
            v = d(rng);
        ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x));
    }
    a.usable = true;
    return true;
}

void finish_arm(Arm& a) {
    if (!a.usable)
        return;
    std::vector<float> out(ggml_nelements(a.y));
    ggml_backend_tensor_get(a.y, out.data(), 0, ggml_nbytes(a.y));
    double sum = 0;
    for (float v : out)
        sum += v;
    a.checksum = sum;
}

struct Stats {
    double best, med, mean;
};

Stats summarise(std::vector<double> t) {
    std::sort(t.begin(), t.end());
    double mean = 0;
    for (double v : t)
        mean += v;
    return {t.front(), t[t.size() / 2], mean / (double)t.size()};
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    int n_threads = 1;
    int reps = 25;
    std::string arm_sel = "both";
    std::string type_sel;
    std::vector<Shape> shapes;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--threads") {
            const char* v = next();
            if (v)
                n_threads = atoi(v);
        } else if (a == "--reps") {
            const char* v = next();
            if (v)
                reps = atoi(v);
        } else if (a == "--arm") {
            const char* v = next();
            if (v)
                arm_sel = v;
        } else if (a == "--type") {
            const char* v = next();
            if (v)
                type_sel = v;
        } else if (a == "--shape") {
            const char* v = next();
            long long K = 0, N = 0, M = 0;
            if (v && sscanf(v, "%lld,%lld,%lld", &K, &N, &M) == 3)
                shapes.push_back({K, N, M});
        } else if (a == "-h" || a == "--help") {
            printf("usage: crispasr-repack-probe [--threads N] [--reps N] [--shape K,N,M]...\n"
                   "                              [--arm both|default|repack] [--type NAME]\n");
            return 0;
        }
    }
    if (shapes.empty()) {
        // hFT-Transformer-ish FFN and attention projection shapes, plus one
        // wide one, at a batch of frames a transcription model actually sees.
        shapes.push_back({256, 256, 128});
        shapes.push_back({512, 2048, 256});
        shapes.push_back({2048, 512, 256});
    }

    printf("== crispasr-repack-probe ==\n");
    printf("ISA as ggml sees it: avx2=%d avx512=%d avx512_vnni=%d amx_int8=%d "
           "neon=%d dotprod=%d matmul_int8(i8mm)=%d sve=%d\n",
           ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(), ggml_cpu_has_amx_int8(),
           ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve());

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra =
        (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

    std::vector<ggml_backend_buffer_type_t> extra;
    if (!get_extra) {
        printf("extra-buft proc address: ABSENT (repack not compiled in)\n");
    } else {
        printf("extra-buft proc address: present\n");
        for (ggml_backend_buffer_type_t* b = get_extra(dev); b && *b; ++b)
            extra.push_back(*b);
        printf("extra buffer types offered: %zu\n", extra.size());
        for (auto b : extra)
            printf("  - %s\n", ggml_backend_buft_name(b));
        // The CPU device can offer several, and the order is not neutral:
        // ggml pushes AMX first on a build with __AMX_INT8__ && __AVX512VNNI__.
        // Select by name so a run on Emerald Rapids measures what it says it
        // measures; --buft picks a different one.
        const char* want = getenv("CRISPASR_EXTRA_BUFT");
        if (!want)
            want = "CPU_REPACK";
        std::vector<ggml_backend_buffer_type_t> sel;
        for (auto b : extra) {
            const char* n = ggml_backend_buft_name(b);
            if (n && strcmp(n, want) == 0)
                sel.push_back(b);
        }
        if (sel.empty() && !extra.empty()) {
            printf("NOTE: '%s' not offered here; falling back to '%s'\n", want, ggml_backend_buft_name(extra[0]));
        } else {
            extra = sel;
        }
        printf("measuring extra buffer type: %s\n", extra.empty() ? "(none)" : ggml_backend_buft_name(extra[0]));
    }
    if (extra.empty()) {
        printf("\nRESULT: no extra buffer type on this host. The repack fast path\n"
               "cannot be selected here at all; the question is untestable on this ISA.\n");
        return 0;
    }

    const struct {
        ggml_type t;
        const char* n;
    } types[] = {
        {GGML_TYPE_F32, "f32"},   {GGML_TYPE_Q8_0, "q8_0"}, {GGML_TYPE_Q4_0, "q4_0"},
        {GGML_TYPE_Q4_K, "q4_K"}, {GGML_TYPE_Q6_K, "q6_K"},
    };

    if (arm_sel == "default" || arm_sel == "repack") {
        // One arm, one process. Machine-readable:
        //   ARM <arm> <type> <K> <N> <M> <threads> <best_ms> <median_ms> <checksum> <status>
        // status is `ok`, or `declined` when the repack buffer type has no
        // kernel for that (type, shape, ISA) — which is a result, not a skip.
        const bool want_repack = (arm_sel == "repack");
        for (Shape s : shapes) {
            for (auto ty : types) {
                if (!type_sel.empty() && type_sel != ty.n)
                    continue;
                Arm a;
                if (!build_arm(a, ty.t, s, n_threads, want_repack ? extra[0] : nullptr, arm_sel.c_str()))
                    return 1;
                if (!a.usable) {
                    printf("ARM %s %s %lld %lld %lld %d - - - declined\n", arm_sel.c_str(), ty.n, (long long)s.K,
                           (long long)s.N, (long long)s.M, n_threads);
                    continue;
                }
                for (int i = 0; i < 3; i++)
                    ggml_backend_graph_compute(a.backend, a.gf);
                for (int i = 0; i < reps; i++) {
                    double t0 = now_ms();
                    ggml_backend_graph_compute(a.backend, a.gf);
                    a.times.push_back(now_ms() - t0);
                }
                finish_arm(a);
                Stats st = summarise(a.times);
                printf("ARM %s %s %lld %lld %lld %d %.4f %.4f %.6f ok\n", arm_sel.c_str(), ty.n, (long long)s.K,
                       (long long)s.N, (long long)s.M, n_threads, st.best, st.med, a.checksum);
            }
        }
        return 0;
    }

    for (Shape s : shapes) {
        printf("\n== shape K=%lld N=%lld M=%lld, threads=%d, reps=%d (interleaved) ==\n", (long long)s.K,
               (long long)s.N, (long long)s.M, n_threads, reps);
        printf("%-6s  %-9s  %-9s  %-9s  %-9s  %s\n", "type", "def best", "rep best", "def med", "rep med", "verdict");

        double f32_best = 0, f32_med = 0;
        for (auto ty : types) {
            Arm def, rep;
            if (!build_arm(def, ty.t, s, n_threads, nullptr, "default"))
                continue;
            if (!build_arm(rep, ty.t, s, n_threads, extra[0], "repack"))
                continue;

            // Even when the repack buft declines the tensor, time the default
            // arm: the f32 row is the baseline the quantised rows are judged
            // against, and "is generic-path q8_0 slower than f32 for this
            // shape?" is the question behind the hFT/O&F disagreement.
            if (!rep.usable) {
                for (int i = 0; i < 3; i++)
                    ggml_backend_graph_compute(def.backend, def.gf);
                for (int i = 0; i < reps; i++) {
                    double t0 = now_ms();
                    ggml_backend_graph_compute(def.backend, def.gf);
                    def.times.push_back(now_ms() - t0);
                }
                finish_arm(def);
                Stats ds = summarise(def.times);
                if (ty.t == GGML_TYPE_F32)
                    f32_best = ds.best, f32_med = ds.med;
                char v[160];
                if (f32_best > 0 && ty.t != GGML_TYPE_F32)
                    snprintf(v, sizeof v, "repack DECLINED; generic path is %.2fx f32 on best, %.2fx on median",
                             ds.best / f32_best, ds.med / f32_med);
                else
                    snprintf(v, sizeof v, "%s",
                             ty.t == GGML_TYPE_F32
                                 ? "f32 baseline (repack is quant-only by design)"
                                 : "repack buft DECLINED this tensor (no kernel for this type/shape/ISA)");
                printf("%-6s  %9.3f  %-9s  %9.3f  %-9s  %s\n", ty.n, ds.best, "-", ds.med, "-", v);
                continue;
            }

            for (int i = 0; i < 3; i++) { // warmup, both arms
                ggml_backend_graph_compute(def.backend, def.gf);
                ggml_backend_graph_compute(rep.backend, rep.gf);
            }
            // Interleaved round-robin, alternating which arm goes first.
            for (int i = 0; i < reps; i++) {
                Arm* first = (i % 2) ? &rep : &def;
                Arm* second = (i % 2) ? &def : &rep;
                double t0 = now_ms();
                ggml_backend_graph_compute(first->backend, first->gf);
                first->times.push_back(now_ms() - t0);
                double t1 = now_ms();
                ggml_backend_graph_compute(second->backend, second->gf);
                second->times.push_back(now_ms() - t1);
            }
            finish_arm(def);
            finish_arm(rep);

            Stats ds = summarise(def.times);
            Stats rs = summarise(rep.times);
            double rel_err = def.checksum == 0 ? 0 : std::abs(rep.checksum - def.checksum) / std::abs(def.checksum);
            char verdict[200];
            if (f32_best > 0)
                snprintf(verdict, sizeof verdict,
                         "repack %.2fx vs generic; generic %.2fx f32, repack %.2fx f32 (best; sum rel-diff %.1e)",
                         ds.best / rs.best, ds.best / f32_best, rs.best / f32_best, rel_err);
            else
                snprintf(verdict, sizeof verdict, "repack %.2fx on best, %.2fx on median  (sum rel-diff %.2e)",
                         ds.best / rs.best, ds.med / rs.med, rel_err);
            printf("%-6s  %9.3f  %9.3f  %9.3f  %9.3f  %s\n", ty.n, ds.best, rs.best, ds.med, rs.med, verdict);
        }
    }
    return 0;
}
