// crispasr_cache.cpp — implementation of crispasr_cache.h.
// See header for the contract.

#include "crispasr_cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#define F_OK 0
#define mkdir(d, m) _mkdir(d)
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace crispasr_cache {

// Anonymous helpers
namespace {

std::string sh_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

} // namespace

std::string dir() {
    const char * home = std::getenv("HOME");
    std::string d = (home && *home) ? home : "/tmp";
    d += "/.cache";
    mkdir(d.c_str(), 0755); // ignore EEXIST
    d += "/crispasr";
    mkdir(d.c_str(), 0755);
    return d;
}

bool file_present(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return st.st_size > 0;
}

bool fetch(const std::string & url, const std::string & dest, bool quiet) {
    // curl: -fL fails on HTTP error and follows redirects.
    std::string curl_cmd = "curl -fL ";
    curl_cmd += quiet ? "-s " : "--progress-bar ";
    curl_cmd += "-o " + sh_quote(dest) + " " + sh_quote(url);

    int rc = std::system(curl_cmd.c_str());
    if (rc == 0 && file_present(dest)) return true;

    // Fall back to wget. Some distros / containers ship one but not the
    // other.
    std::string wget_cmd = "wget ";
    wget_cmd += quiet ? "-q " : "--show-progress ";
    wget_cmd += "-O " + sh_quote(dest) + " " + sh_quote(url);

    rc = std::system(wget_cmd.c_str());
    if (rc == 0 && file_present(dest)) return true;

    fprintf(stderr,
            "crispasr: download failed (curl + wget both rejected). "
            "Install one of them, or fetch manually:\n  %s\n  -> %s\n",
            url.c_str(), dest.c_str());
    return false;
}

std::string ensure_cached_file(const std::string & filename,
                               const std::string & url,
                               bool quiet,
                               const char * pretty_label) {
    const std::string dst = dir() + "/" + filename;
    if (file_present(dst)) {
        if (!quiet) {
            fprintf(stderr, "%s: using cached %s\n", pretty_label, dst.c_str());
        }
        return dst;
    }
    if (!quiet) {
        fprintf(stderr, "%s: downloading %s\n", pretty_label, url.c_str());
    }
    if (!fetch(url, dst, quiet)) return "";
    return dst;
}

} // namespace crispasr_cache
