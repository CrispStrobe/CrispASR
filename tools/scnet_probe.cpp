// Minimal parity runner: same model-specific argv as crispasr-diff, without
// linking unrelated speech backends. The production runtime owns all math.
#include "scnet.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "scnet") == 0)
        return scnet_diff(argv[2], argv[3], argv[4], 2);
    if (argc == 4 && std::strcmp(argv[1], "scnet") == 0)
        return scnet_diff(argv[2], argv[3], nullptr, 2);
    std::fprintf(stderr, "usage: %s scnet <model.gguf> <reference.gguf> [audio.wav]\n", argv[0]);
    return 2;
}
