// Explicit bounded five-second research entry point; not a product API.
#include "reuse_diff.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: reuse-long-diff model.gguf ref.gguf [native-waveform.f32]\n");
        return 2;
    }
    return reuse_diff::run(argv[1], argv[2], argc == 4 ? argv[3] : nullptr, 80000);
}
