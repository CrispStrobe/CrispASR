// crispasr-lid — standalone text language identification.
//
// Companion to crispasr (audio ASR) for tagging the language of a
// transcript or arbitrary text. Reads UTF-8 text from --text or stdin
// and emits the top-k predictions one per line (label\tscore).
//
// Auto-routes between fastText (GlotLID-V3 / LID-176) and CLD3
// based on the GGUF's `general.architecture` key — one binary, any
// supported text-LID GGUF.
//
// Usage:
//
//   crispasr-lid -m /path/to/lid-glotlid-f16.gguf --text "Hello world"
//   crispasr-lid -m /path/to/cld3-f16.gguf --text "Hallo Welt"
//   echo "Bonjour le monde" | crispasr-lid -m lid-glotlid-f16.gguf
//   crispasr-lid -m cld3-f16.gguf -k 5 < transcript.txt
//
// Output (default top-1, single line):
//
//   eng_Latn\t0.9997    (fastText / GlotLID — ISO 639-3 + script)
//   en\t0.9997          (CLD3 — ISO 639-1)
//
// Output with -k > 1 (one entry per line):
//
//   eng_Latn\t0.9997
//   sco_Latn\t0.0001
//   ...

#include "text_lid_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(FILE* out, const char* argv0) {
    std::fprintf(out,
                 "usage: %s -m <model.gguf> [--text \"...\"] [-k N] [--quiet]\n"
                 "\n"
                 "Identify the language of UTF-8 text using any text-LID GGUF.\n"
                 "Auto-routes between fastText (GlotLID-V3, LID-176) and Google\n"
                 "CLD3 by reading the GGUF's general.architecture key. Defaults\n"
                 "to reading text from stdin if --text is omitted.\n"
                 "\n"
                 "Options:\n"
                 "  -m, --model PATH    LID GGUF (required)\n"
                 "      --text  STR     input text (otherwise read from stdin)\n"
                 "  -k, --topk  N       number of top predictions to print (default 1)\n"
                 "      --quiet         omit the trailing variant/dim summary\n"
                 "  -h, --help          show this message\n",
                 argv0);
}

std::string read_stdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    // Strip trailing newline so the LID forward sees a clean line.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
        s.pop_back();
    return s;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string text;
    bool have_text = false;
    int topk = 1;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "crispasr-lid: missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(stdout, argv[0]);
            return 0;
        } else if (a == "-m" || a == "--model") {
            model_path = need_arg(a.c_str());
        } else if (a == "--text") {
            text = need_arg(a.c_str());
            have_text = true;
        } else if (a == "-k" || a == "--topk") {
            topk = std::atoi(need_arg(a.c_str()));
            if (topk <= 0)
                topk = 1;
        } else if (a == "--quiet") {
            quiet = true;
        } else {
            std::fprintf(stderr, "crispasr-lid: unknown arg '%s'\n", a.c_str());
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "crispasr-lid: -m/--model is required\n");
        usage(stderr, argv[0]);
        return 2;
    }
    if (!have_text) {
        text = read_stdin();
    }
    if (text.empty()) {
        std::fprintf(stderr, "crispasr-lid: empty input text\n");
        return 2;
    }

    // Resolve `-m auto[:variant]` and bare-filename inputs against the
    // registry, downloading into ~/.cache/crispasr/ on first use.
    const std::string resolved = text_lid_resolve_path(model_path, /*cache_dir_override=*/"", /*quiet=*/quiet);
    if (resolved.empty()) {
        return 1;
    }
    text_lid_context* ctx = text_lid_init_from_file(resolved.c_str(), 1);
    if (!ctx) {
        // Errors already printed to stderr by the dispatcher / loader.
        return 1;
    }

    if (topk == 1) {
        float conf = 0.f;
        const char* lab = text_lid_predict(ctx, text.c_str(), &conf);
        if (!lab) {
            std::fprintf(stderr, "crispasr-lid: prediction failed\n");
            text_lid_free(ctx);
            return 1;
        }
        std::printf("%s\t%.6f\n", lab, conf);
    } else {
        std::vector<const char*> labs(static_cast<size_t>(topk), nullptr);
        std::vector<float> scores(static_cast<size_t>(topk), 0.f);
        const int n = text_lid_predict_topk(ctx, text.c_str(), topk, labs.data(), scores.data());
        if (n <= 0) {
            std::fprintf(stderr, "crispasr-lid: prediction failed\n");
            text_lid_free(ctx);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            std::printf("%s\t%.6f\n", labs[i], scores[i]);
        }
    }

    if (!quiet) {
        std::fprintf(stderr, "crispasr-lid: backend=%s variant=%s dim=%d %d labels\n", text_lid_backend(ctx),
                     text_lid_variant(ctx), text_lid_dim(ctx), text_lid_n_labels(ctx));
    }

    text_lid_free(ctx);
    return 0;
}
