// Isolated entry point for the same research implementation used by crispasr-diff.
#include "reuse_diff.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: reuse-research-diff model.gguf ref.gguf\n");
        return 2;
    }
    return reuse_diff::run(argv[1], argv[2]);
}
