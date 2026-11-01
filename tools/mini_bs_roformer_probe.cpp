// Minimal parity runner: same model-specific argv as crispasr-diff, without
// linking unrelated speech backends. The production runtime owns all math.
#include "mini_bs_roformer.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "mini-bs-roformer") == 0)
        return mini_bs_roformer_diff(argv[2], argv[3], argv[4], 2);
    if (argc == 4 && std::strcmp(argv[1], "mini-bs-roformer") == 0)
        return mini_bs_roformer_diff(argv[2], argv[3], nullptr, 2);
    std::fprintf(stderr, "usage: %s mini-bs-roformer <model.gguf> <reference.gguf> [audio.wav]\n", argv[0]);
    return 2;
}
