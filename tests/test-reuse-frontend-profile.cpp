#include "../examples/cli/reuse_frontend_profile.h"
#include <cmath>
#include <cstdio>
#include <limits>
#ifdef CRISPASR_REUSE_HAVE_MKL
#include <mkl.h>
#endif

int main(int argc, char** argv) {
    reuse_frontend_profile::Analyze selected = nullptr;
    const char* label = nullptr;
    if (argc == 2 && std::strcmp(argv[1], "--select-avx2") == 0) {
        if (!reuse_frontend_profile::select("mkl-avx2", selected, label))
            return 2;
        if (!selected || !label || std::strcmp(label, "mkl-avx2"))
            return 8;
        std::puts("REUSE_MKL_AVX2_PROFILE_SELECTED");
        return 0;
    }
    for (const char* name : {static_cast<const char*>(nullptr), "", "native"}) {
        if (!reuse_frontend_profile::select(name, selected, label) || selected != reuse_frontend::analyze ||
            std::strcmp(label, "native"))
            return 1;
    }
    if (reuse_frontend_profile::select("invented", selected, label) || selected || label)
        return 2;
    const bool available = reuse_frontend_profile::select("mkl", selected, label);
#ifdef CRISPASR_REUSE_HAVE_MKL
    if (!available || selected != reuse_frontend_mkl::analyze || std::strcmp(label, "mkl"))
        return 3;
    const int previous_threads = MKL_Set_Num_Threads_Local(1);
    reuse_frontend::Spectrum output;
    output.sample_rate = 123;
    reuse_frontend::Diagnostics diagnostics;
    diagnostics.window = {42};
    std::vector<float> invalid(400, std::numeric_limits<float>::quiet_NaN());
    if (selected(invalid, 16000, output, &diagnostics) || output.sample_rate != 123 ||
        diagnostics.window != std::vector<float>{42} || selected(std::vector<float>(400), 8000, output, &diagnostics) ||
        MKL_Set_Num_Threads_Local(1) != 1)
        return 4;
    const std::vector<float> constant(1600, -0.25f);
    if (!selected(constant, 16000, output, &diagnostics) || output.frames != 21 || output.bins != 321 ||
        diagnostics.window.size() != 640 || diagnostics.real.size() != output.phase.size())
        return 5;
    for (size_t frame = 0; frame < output.frames; ++frame)
        if (output.phase[frame * output.bins] != static_cast<float>(std::acos(-1.0)))
            return 6;
    if (MKL_Set_Num_Threads_Local(1) != 1)
        return 7;
    MKL_Set_Num_Threads_Local(previous_threads);
    std::puts("RE-USE optional oneMKL selection, invalid-input ownership and raw phase controls passed");
#else
    if (available || selected || label)
        return 3;
    if (reuse_frontend_profile::select("mkl-avx2", selected, label) || selected || label)
        return 8;
    std::puts("RE-USE default selection and unavailable oneMKL rejection passed without an SDK");
#endif
    return 0;
}
