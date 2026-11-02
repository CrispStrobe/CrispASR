// mossformergan-se-parity — compare the native MossFormerGAN_SE_16K port with
// PyTorch reference dumps from tools/dump_mossformergan_se_reference.py.
//
//   mossformergan-se-parity model.gguf ref_dir [--threads N] [--min-cos 0.999] [--no-stages]
//
// Prints cosine / max-abs per network stage (one unsegmented forward on the
// reference's own stage_input_spec) and for the end-to-end decode
// (input_wav.npy -> wave.npy, including segmentation for > 10 s inputs).
// Exit status 1 when any compared cosine is below --min-cos.

#include "mossformergan_se.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_npy(const std::string& path, std::vector<float>& data, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return false;
    unsigned char ver[2];
    f.read((char*)ver, 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t h = 0;
        f.read((char*)&h, 2);
        hlen = h;
    } else {
        f.read((char*)&hlen, 4);
    }
    std::string header(hlen, '\0');
    f.read(&header[0], hlen);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) {
        std::fprintf(stderr, "%s: only little-endian float32 C-order arrays are supported\n", path.c_str());
        return false;
    }
    shape.clear();
    const size_t a = header.find('('), b = header.find(')');
    std::string dims = header.substr(a + 1, b - a - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ','))
            pos++;
        if (pos >= dims.size())
            break;
        shape.push_back(std::strtoll(dims.c_str() + pos, nullptr, 10));
        while (pos < dims.size() && dims[pos] != ',')
            pos++;
    }
    int64_t n = 1;
    for (int64_t d : shape)
        n *= d;
    data.resize((size_t)n);
    f.read((char*)data.data(), n * sizeof(float));
    return (bool)f;
}

static void compare(const float* a, const float* b, size_t n, double& cos, double& maxabs, double& ref_rms) {
    double ab = 0, aa = 0, bb = 0;
    maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
        maxabs = std::max(maxabs, (double)std::fabs(a[i] - b[i]));
    }
    cos = ab / (std::sqrt(aa * bb) + 1e-30);
    ref_rms = std::sqrt(bb / (double)std::max<size_t>(n, 1));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf ref_dir [--threads N] [--min-cos X] [--no-stages]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], dir = argv[2];
    int threads = 4;
    double min_cos = 0.999;
    bool stages_on = true;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--min-cos") && i + 1 < argc)
            min_cos = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-stages"))
            stages_on = false;
    }
    mossformergan_se_context* ctx = mossformergan_se_init(model.c_str(), threads, true);
    if (!ctx) {
        std::fprintf(stderr, "failed to load %s\n", model.c_str());
        return 2;
    }
    int fails = 0;
    if (stages_on) {
        std::vector<float> spec;
        std::vector<int64_t> shape;
        if (!read_npy(dir + "/stage_input_spec.npy", spec, shape) || shape.size() != 3) {
            std::fprintf(stderr, "missing %s/stage_input_spec.npy\n", dir.c_str());
            return 2;
        }
        std::vector<mossformergan_se_stage> st;
        auto t0 = std::chrono::steady_clock::now();
        if (!mossformergan_se_forward_stages(ctx, spec.data(), (int)shape[1], st)) {
            std::fprintf(stderr, "forward failed\n");
            return 2;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("forward T=%lld: %.2f s\n", (long long)shape[1], dt);
        for (auto& s : st) {
            std::vector<float> ref;
            std::vector<int64_t> rs;
            if (!read_npy(dir + "/" + s.name + ".npy", ref, rs)) {
                std::printf("%-30s (no reference)\n", s.name.c_str());
                continue;
            }
            if (rs != s.shape) {
                std::printf("%-30s SHAPE MISMATCH ref", s.name.c_str());
                for (auto d : rs)
                    std::printf(" %lld", (long long)d);
                std::printf(" ours");
                for (auto d : s.shape)
                    std::printf(" %lld", (long long)d);
                std::printf("\n");
                fails++;
                continue;
            }
            double cos, mx, rms;
            compare(s.data.data(), ref.data(), ref.size(), cos, mx, rms);
            const bool ok = cos >= min_cos;
            fails += !ok;
            std::printf("%-30s cos %.8f  max|d| %.3e  ref_rms %.3e %s\n", s.name.c_str(), cos, mx, rms,
                        ok ? "" : "  <-- FAIL");
        }
    }
    std::vector<float> wav, ref;
    std::vector<int64_t> s1, s2;
    if (stages_on && read_npy(dir + "/input_wav.npy", wav, s1)) {
        // STFT front end on the first decode window (what stage_input_spec was taken from)
        std::vector<float> spec, rspec;
        std::vector<int64_t> rs;
        int T = 0;
        const int n = std::min<int>((int)wav.size(), 160000);
        if (read_npy(dir + "/stage_input_spec.npy", rspec, rs) &&
            mossformergan_se_input_spec(ctx, wav.data(), n, spec, &T) && spec.size() == rspec.size()) {
            double cos, mx, rms;
            compare(spec.data(), rspec.data(), rspec.size(), cos, mx, rms);
            const bool ok = cos >= min_cos;
            fails += !ok;
            std::printf("%-30s cos %.8f  max|d| %.3e  ref_rms %.3e %s\n", "stage_input_spec(stft)", cos, mx, rms,
                        ok ? "" : "  <-- FAIL");
        } else {
            std::printf("stage_input_spec(stft)         could not compare\n");
            fails++;
        }
    }
    if (read_npy(dir + "/input_wav.npy", wav, s1) && read_npy(dir + "/wave.npy", ref, s2)) {
        std::vector<float> out;
        auto t0 = std::chrono::steady_clock::now();
        if (!mossformergan_se_enhance(ctx, wav.data(), (int)wav.size(), out)) {
            std::fprintf(stderr, "enhance failed\n");
            return 2;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double cos, mx, rms;
        compare(out.data(), ref.data(), ref.size(), cos, mx, rms);
        const bool ok = out.size() == ref.size() && cos >= min_cos;
        fails += !ok;
        std::printf("%-30s cos %.8f  max|d| %.3e  ref_rms %.3e  (%zu samples, %.2f s)%s\n", "wave(end-to-end)", cos, mx,
                    rms, out.size(), dt, ok ? "" : "  <-- FAIL");
        std::printf("TIME_ENHANCE %.3f\n", dt);
    }
    mossformergan_se_free(ctx);
    std::printf("%s\n", fails ? "PARITY FAIL" : "PARITY OK");
    return fails ? 1 : 0;
}
