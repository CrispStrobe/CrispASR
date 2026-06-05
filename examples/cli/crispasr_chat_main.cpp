// crispasr_chat_main.cpp — reference CLI for the crispasr_chat_* C ABI.
//
// Two operating modes share one binary:
//
//   interactive  (default when stdin is a TTY)
//     Prompts ">> " for each user turn. Generates streamingly,
//     printing assistant deltas as they arrive. Empty input quits.
//
//   one-shot     (default when stdin is a pipe / file)
//     Reads all of stdin as a single user message, prints the
//     assistant reply on stdout, exits.
//
// Intentionally minimal: no chat history persistence, no fancy
// terminal handling. Aimed at:
//   • smoke-testing the ABI from a release build
//   • being the smallest reference for downstream wrappers
//
// Usage:
//   crispasr-chat -m model.gguf [--system "..."] [-c 4096]
//                 [--temp 0.8] [--top-p 0.95] [--top-k 40]
//                 [--seed N] [--max-tokens 256] [--threads N]
//                 [--no-color] [--one-shot]

#include "crispasr_chat.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace {

struct cli_args {
    std::string model;
    std::string system_prompt;
    int32_t n_ctx = 4096;
    int32_t n_threads = 0; // 0 → ABI default
    int32_t max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    uint32_t seed = 0;
    bool no_color = false;
    bool force_one_shot = false;
    bool help = false;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s -m <model.gguf> [options]\n"
                 "  -m, --model PATH       GGUF chat model (required)\n"
                 "      --system TEXT      system prompt\n"
                 "  -c, --ctx N            context window in tokens (default 4096)\n"
                 "  -t, --threads N        generation threads\n"
                 "      --max-tokens N     cap tokens per reply (default 256)\n"
                 "      --temp F           sampling temperature (default 0.8)\n"
                 "      --top-p F          top-p / nucleus (default 0.95)\n"
                 "      --top-k N          top-k (default 40, 0 = off)\n"
                 "      --seed N           RNG seed (default 0 = random)\n"
                 "      --one-shot         read all stdin as one user turn\n"
                 "      --no-color         disable ANSI colour\n"
                 "  -h, --help             show this help\n",
                 argv0);
}

bool parse_args(int argc, char** argv, cli_args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", s.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        if (s == "-m" || s == "--model") {
            if (auto* v = next()) {
                a.model = v;
            } else {
                return false;
            }
        } else if (s == "--system") {
            if (auto* v = next()) {
                a.system_prompt = v;
            } else {
                return false;
            }
        } else if (s == "-c" || s == "--ctx") {
            if (auto* v = next()) {
                a.n_ctx = std::atoi(v);
            } else {
                return false;
            }
        } else if (s == "-t" || s == "--threads") {
            if (auto* v = next()) {
                a.n_threads = std::atoi(v);
            } else {
                return false;
            }
        } else if (s == "--max-tokens") {
            if (auto* v = next()) {
                a.max_tokens = std::atoi(v);
            } else {
                return false;
            }
        } else if (s == "--temp") {
            if (auto* v = next()) {
                a.temperature = (float)std::atof(v);
            } else {
                return false;
            }
        } else if (s == "--top-p") {
            if (auto* v = next()) {
                a.top_p = (float)std::atof(v);
            } else {
                return false;
            }
        } else if (s == "--top-k") {
            if (auto* v = next()) {
                a.top_k = std::atoi(v);
            } else {
                return false;
            }
        } else if (s == "--seed") {
            if (auto* v = next()) {
                a.seed = (uint32_t)std::atoll(v);
            } else {
                return false;
            }
        } else if (s == "--one-shot") {
            a.force_one_shot = true;
        } else if (s == "--no-color") {
            a.no_color = true;
        } else if (s == "-h" || s == "--help") {
            a.help = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", s.c_str());
            return false;
        }
    }
    return true;
}

void stream_cb(const char* utf8, void* user) {
    auto* out = static_cast<std::string*>(user);
    std::fwrite(utf8, 1, std::strlen(utf8), stdout);
    std::fflush(stdout);
    out->append(utf8);
}

std::string read_all_stdin() {
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    return oss.str();
}

bool stdin_is_tty() {
    return isatty(fileno(stdin)) != 0;
}

} // namespace

int main(int argc, char** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }
    if (args.help || args.model.empty()) {
        print_usage(argv[0]);
        return args.help ? 0 : 2;
    }

    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_ctx = args.n_ctx;
    if (args.n_threads > 0) {
        op.n_threads = args.n_threads;
        op.n_threads_batch = args.n_threads;
    }

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(args.model.c_str(), &op, &err);
    if (!s) {
        std::fprintf(stderr, "crispasr_chat_open failed: %s\n", err.message);
        return 1;
    }

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = args.max_tokens;
    gp.temperature = args.temperature;
    gp.top_p = args.top_p;
    gp.top_k = args.top_k;
    gp.seed = args.seed;

    const bool one_shot = args.force_one_shot || !stdin_is_tty();
    const char* color_user = args.no_color ? "" : "\x1b[36m";
    const char* color_bot = args.no_color ? "" : "\x1b[32m";
    const char* color_off = args.no_color ? "" : "\x1b[0m";

    if (one_shot) {
        const std::string user = read_all_stdin();
        std::vector<crispasr_chat_message> msgs;
        if (!args.system_prompt.empty()) {
            msgs.push_back({"system", args.system_prompt.c_str()});
        }
        msgs.push_back({"user", user.c_str()});

        std::string acc;
        if (crispasr_chat_generate_stream(s, msgs.data(), msgs.size(), &gp, stream_cb, &acc, &err) != 0) {
            std::fprintf(stderr, "\ngenerate failed: %s\n", err.message);
            crispasr_chat_close(s);
            return 1;
        }
        std::fputc('\n', stdout);
        crispasr_chat_close(s);
        return 0;
    }

    // Interactive REPL — re-issue the full history each turn so the
    // ABI's history-cache fast path kicks in (no re-prefill).
    std::vector<crispasr_chat_message> msgs;
    std::vector<std::string> owned; // backing storage for c_str()
    if (!args.system_prompt.empty()) {
        owned.push_back(args.system_prompt);
        msgs.push_back({"system", owned.back().c_str()});
    }

    std::fprintf(stderr, "Loaded %s — template '%s', ctx %d. Empty line to quit.\n", args.model.c_str(),
                 crispasr_chat_template_name(s), crispasr_chat_n_ctx(s));

    while (true) {
        std::fprintf(stdout, "%s>> %s", color_user, color_off);
        std::fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) {
            break;
        }
        owned.push_back(line);
        msgs.push_back({"user", owned.back().c_str()});

        std::fprintf(stdout, "%s", color_bot);
        std::fflush(stdout);
        std::string reply;
        if (crispasr_chat_generate_stream(s, msgs.data(), msgs.size(), &gp, stream_cb, &reply, &err) != 0) {
            std::fprintf(stderr, "\n%sgenerate failed: %s%s\n", color_off, err.message, color_off);
            owned.pop_back();
            msgs.pop_back();
            continue;
        }
        std::fprintf(stdout, "%s\n", color_off);

        owned.push_back(reply);
        msgs.push_back({"assistant", owned.back().c_str()});
    }

    crispasr_chat_close(s);
    return 0;
}
