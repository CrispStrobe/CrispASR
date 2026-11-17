// Exercise the production mmap and fread paths without models or ggml linkage.
#include "../src/core/gguf_file.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
int checks = 0;
void check(bool condition) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "GGUF file check %d failed\n", checks);
        std::exit(1);
    }
}

struct FixtureDir {
    std::filesystem::path path;
    FixtureDir() {
        path = std::filesystem::temp_directory_path() /
               ("crispasr-gguf-file-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(path));
    }
    ~FixtureDir() { std::filesystem::remove_all(path); }
};
} // namespace

int main() {
    using core_gguf::detail::MappedFile;
    using core_gguf::detail::open_read;
    FixtureDir fixtures;
    const char payload[] = "GGUF\0binary fixture\xff";
    // Universal escapes keep the test independent of the compiler source code page.
    for (const char* name : {"ascii", u8"\u6a21\u578b", u8"caf\u00e9", u8"model-\U0001f680"}) {
        const auto dir = fixtures.path / std::filesystem::u8path(name);
        check(std::filesystem::create_directory(dir));
        const auto path =
            dir / std::filesystem::u8path(std::strcmp(name, "ascii") == 0 ? "weights.gguf" : u8"weights-\u00e9.gguf");
        {
            // std::filesystem uses native wide paths on Windows to create the oracle fixture.
            std::ofstream out(path, std::ios::binary);
            out.write(payload, sizeof(payload));
            check(out.good());
        }
        const std::string utf8 = path.u8string();
        {
            MappedFile mapping(utf8.c_str());
            check(mapping.ok);
            check(mapping.size == sizeof(payload));
            check(std::memcmp(mapping.base, payload, sizeof(payload)) == 0);
        }
        {
            MappedFile mapping(utf8.c_str(), true);
            check(mapping.ok);
            check(mapping.size == sizeof(payload));
            check(std::memcmp(mapping.base, payload, sizeof(payload)) == 0);
            static_cast<char*>(mapping.base)[0] = 'X';
            check(static_cast<char*>(mapping.base)[0] == 'X');
        }
        // Explicitly exercise the fallback even when mmap succeeds, and prove COW
        // did not alter the bytes on disk.
        FILE* fp = open_read(utf8.c_str());
        check(fp != nullptr);
        char actual[sizeof(payload)] = {};
        check(std::fread(actual, 1, sizeof(actual), fp) == sizeof(payload));
        check(std::memcmp(actual, payload, sizeof(payload)) == 0);
        check(std::fclose(fp) == 0);
    }
    const auto missing = (fixtures.path / "missing.gguf").u8string();
    check(!MappedFile(missing.c_str()).ok);
    check(open_read(missing.c_str()) == nullptr);
    check(!MappedFile("").ok);
    check(open_read("") == nullptr);

    const auto empty = fixtures.path / "empty.gguf";
    { std::ofstream out(empty, std::ios::binary); }
    check(!MappedFile(empty.u8string().c_str()).ok);
    FILE* fp = open_read(empty.u8string().c_str());
    check(fp != nullptr);
    check(std::fgetc(fp) == EOF);
    check(std::fclose(fp) == 0);
#if defined(_WIN32)
    using core_gguf::detail::utf8_to_wide;
    check(utf8_to_wide(u8"C:\\caf\u00e9\\\u6a21\u578b\\\U0001f680.gguf") ==
          L"C:\\caf\u00e9\\\u6a21\u578b\\\U0001f680.gguf");
    for (const char* invalid : {"\xc0\xaf", "\xe6\xa8", "\xed\xa0\x80", "\xf4\x90\x80\x80"}) {
        check(utf8_to_wide(invalid).empty());
        check(open_read(invalid) == nullptr);
        check(!MappedFile(invalid).ok);
    }
#endif
    std::printf("GGUF file: %d checks passed\n", checks);
}
