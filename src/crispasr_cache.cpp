// crispasr_cache.cpp — implementation of crispasr_cache.h.
// See header for the contract.

#include "crispasr_cache.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#define access _access
#define F_OK 0
#define mkdir(d, m) _mkdir(d)
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef CRISPASR_USE_LIBCURL
#include <curl/curl.h>
#endif

namespace crispasr_cache {

// ─── anonymous helpers ───────────────────────────────────────────────────────
namespace {

// shell_quote: wrap a string so that a shell (or cmd.exe) treats it as a
// single argument even if it contains spaces or special characters.
//
//   POSIX sh  — single-quote the string; embed literal ' as '\''
//   Windows   — double-quote the string; embed literal " as ""
//
// NOTE: on Windows, double-quoting does NOT protect bare & | < > ^ from
// cmd.exe expansion *outside* the quoted region, but since we quote the
// entire argument they will be treated literally inside.  That is sufficient
// for paths and HTTPS URLs (which only contain & inside the query string,
// already wrapped in the outer "...").

#ifdef _WIN32
static std::string shell_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += "\"\""; // "" inside "…"
        else
            out += c;
    }
    out += "\"";
    return out;
}

// ─── WinHTTP helpers (Windows-only) ─────────────────────────────────────────

static std::wstring to_wide(const std::string& s) {
    if (s.empty())
        return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0)
        return {};
    // MultiByteToWideChar writes the trailing NUL because cbMultiByte is -1.
    // Keep room for it, then remove it from the returned C++ string.
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.pop_back();
    return w;
}

static std::string to_utf8(const std::wstring& w) {
    if (w.empty())
        return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0)
        return {};
    // WideCharToMultiByte likewise includes the trailing NUL for cchWideChar=-1.
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, nullptr, nullptr);
    s.pop_back();
    return s;
}

// Fetch url → dest using the WinHTTP native Windows HTTPS stack.
// Manually follows up to 10 redirects so cross-host hops (e.g.
// huggingface.co → CDN) are handled without relying on shell utilities.
// Returns true iff the file was written and is non-empty.
static bool fetch_winhttp(const std::string& url, const std::string& dest, bool quiet) {
    std::string current_url = url;

    for (int hop = 0; hop < 10; ++hop) {
        std::wstring wurl = to_wide(current_url);

        // Parse the URL into components.
        URL_COMPONENTS uc = {};
        uc.dwStructSize = sizeof(uc);
        wchar_t scheme[16] = {};
        wchar_t host[512] = {};
        wchar_t path[4096] = {};
        wchar_t extra[4096] = {};
        uc.lpszScheme = scheme;
        uc.dwSchemeLength = _countof(scheme);
        uc.lpszHostName = host;
        uc.dwHostNameLength = _countof(host);
        uc.lpszUrlPath = path;
        uc.dwUrlPathLength = _countof(path);
        uc.lpszExtraInfo = extra;
        uc.dwExtraInfoLength = _countof(extra);

        if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) {
            if (!quiet)
                fprintf(stderr, "crispasr: WinHTTP: WinHttpCrackUrl failed for %s\n", current_url.c_str());
            return false;
        }

        HINTERNET hSess = WinHttpOpen(L"CrispASR/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSess)
            return false;

        const bool is_https = (_wcsicmp(scheme, L"https") == 0);
        INTERNET_PORT port =
            uc.nPort ? uc.nPort : (is_https ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT);

        HINTERNET hConn = WinHttpConnect(hSess, host, port, 0);
        if (!hConn) {
            WinHttpCloseHandle(hSess);
            return false;
        }

        // Full request path = path + optional query string.
        std::wstring req_path = path;
        if (extra[0])
            req_path += extra;

        DWORD req_flags = is_https ? WINHTTP_FLAG_SECURE : 0;
        HINTERNET hReq = WinHttpOpenRequest(hConn, L"GET", req_path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, req_flags);
        if (!hReq) {
            WinHttpCloseHandle(hConn);
            WinHttpCloseHandle(hSess);
            return false;
        }

        // Disable auto-redirect so we handle cross-host hops ourselves.
        DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        WinHttpSetOption(hReq, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof(policy));

        const bool sent =
            WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) != FALSE &&
            WinHttpReceiveResponse(hReq, nullptr) != FALSE;

        DWORD status = 0;
        if (sent) {
            DWORD sz = sizeof(status);
            WinHttpQueryHeaders(hReq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz, WINHTTP_NO_HEADER_INDEX);
        }

        if (!sent || status == 0) {
            WinHttpCloseHandle(hReq);
            WinHttpCloseHandle(hConn);
            WinHttpCloseHandle(hSess);
            return false;
        }

        if (status == 200) {
            // Stream the response body to dest.
            FILE* f = fopen(dest.c_str(), "wb");
            bool dl_ok = (f != nullptr);
            if (!f && !quiet) {
                fprintf(stderr, "crispasr: WinHTTP: cannot create '%s': %s\n", dest.c_str(), strerror(errno));
            }
            if (f) {
                BYTE buf[65536];
                DWORD got = 0;
                while (WinHttpReadData(hReq, buf, sizeof(buf), &got) && got > 0) {
                    fwrite(buf, 1, got, f);
                }
                fclose(f);
            }
            WinHttpCloseHandle(hReq);
            WinHttpCloseHandle(hConn);
            WinHttpCloseHandle(hSess);
            return dl_ok;
        }

        // 3xx → follow the Location header.
        if (status >= 300 && status < 400) {
            DWORD loc_bytes = 0;
            // First call: gets required byte count in loc_bytes.
            WinHttpQueryHeaders(hReq, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &loc_bytes,
                                WINHTTP_NO_HEADER_INDEX);
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && loc_bytes > 0) {
                std::wstring loc(loc_bytes / sizeof(wchar_t), L'\0');
                WinHttpQueryHeaders(hReq, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, &loc[0], &loc_bytes,
                                    WINHTTP_NO_HEADER_INDEX);
                while (!loc.empty() && loc.back() == L'\0')
                    loc.pop_back();
                // Hugging Face's first /resolve/ redirect is commonly a
                // root-relative /api/resolve-cache/... URL. WinHttpCrackUrl
                // requires an absolute URL, so retain the current origin.
                if (loc.rfind(L"//", 0) == 0) {
                    current_url = to_utf8(std::wstring(scheme) + L":" + loc);
                } else if (!loc.empty() && loc.front() == L'/') {
                    std::wstring absolute = std::wstring(scheme) + L"://" + host;
                    const bool default_port = (is_https && port == INTERNET_DEFAULT_HTTPS_PORT) ||
                                              (!is_https && port == INTERNET_DEFAULT_HTTP_PORT);
                    if (!default_port)
                        absolute += L":" + std::to_wstring(port);
                    current_url = to_utf8(absolute + loc);
                } else if (loc.find(L"://") != std::wstring::npos) {
                    current_url = to_utf8(loc);
                } else {
                    const size_t slash = current_url.find_last_of('/');
                    current_url = current_url.substr(0, slash == std::string::npos ? 0 : slash + 1) + to_utf8(loc);
                }
            }
            WinHttpCloseHandle(hReq);
            WinHttpCloseHandle(hConn);
            WinHttpCloseHandle(hSess);
            if (current_url.empty())
                return false;
            continue; // retry with new URL
        }

        // Any other status → failure.
        if (!quiet)
            fprintf(stderr, "crispasr: WinHTTP: server returned %lu for %s\n", static_cast<unsigned long>(status),
                    current_url.c_str());
        WinHttpCloseHandle(hReq);
        WinHttpCloseHandle(hConn);
        WinHttpCloseHandle(hSess);
        return false;
    }

    if (!quiet)
        fprintf(stderr, "crispasr: WinHTTP: too many redirects\n");
    return false;
}

#else // !_WIN32

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += "'";
    return out;
}

#endif // _WIN32

#ifdef CRISPASR_USE_LIBCURL
// Write callback: appends received bytes to the open FILE*.
static size_t curl_write_cb(void* ptr, size_t /*size*/, size_t nmemb, void* userdata) {
    return fwrite(ptr, 1, nmemb, static_cast<FILE*>(userdata));
}

// Optional progress trampoline (only compiled in when !quiet).
struct CurlProgress {
    bool quiet;
    double last_pct = -1.0;
};
static int curl_progress_cb(void* userdata, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/,
                            curl_off_t /*ulnow*/) {
    auto* p = static_cast<CurlProgress*>(userdata);
    if (p->quiet || dltotal <= 0)
        return 0;
    const double pct = 100.0 * static_cast<double>(dlnow) / static_cast<double>(dltotal);
    if (pct - p->last_pct >= 5.0) {
        fprintf(stderr, "\r  %.0f%%", pct);
        fflush(stderr);
        p->last_pct = pct;
    }
    return 0;
}

// Fetch url → dest using the libcurl C API.
// Follows redirects automatically (CURLOPT_FOLLOWLOCATION).
// Returns true iff the file was written and is non-empty.
static bool fetch_libcurl(const std::string& url, const std::string& dest, bool quiet) {
    CURL* curl = curl_easy_init();
    if (!curl)
        return false;

    FILE* f = fopen(dest.c_str(), "wb");
    if (!f) {
        if (!quiet)
            fprintf(stderr, "crispasr: libcurl: cannot create '%s': %s\n", dest.c_str(), strerror(errno));
        curl_easy_cleanup(curl);
        return false;
    }

    CurlProgress prog{quiet};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // follow redirects
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L); // fail on 4xx/5xx
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "CrispASR/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L); // 5 min max for large models
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);

    // HF xet storage: add Accept header for binary downloads + auth token if available
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Accept: application/octet-stream");
    const char* hf_token = getenv("HF_TOKEN");
    if (!hf_token)
        hf_token = getenv("HUGGING_FACE_HUB_TOKEN");
    if (hf_token && hf_token[0]) {
        std::string auth = "Authorization: Bearer " + std::string(hf_token);
        headers = curl_slist_append(headers, auth.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (!quiet) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    const CURLcode res = curl_easy_perform(curl);
    fclose(f);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (!quiet)
        fprintf(stderr, "\n"); // end progress line
    if (res != CURLE_OK) {
        if (!quiet)
            fprintf(stderr, "crispasr: libcurl: %s\n", curl_easy_strerror(res));
        return false;
    }
    return true;
}
#endif // CRISPASR_USE_LIBCURL

} // anonymous namespace

// ─── public API ──────────────────────────────────────────────────────────────

namespace {

std::string platform_default_dir() {
#ifdef __EMSCRIPTEN__
    return "/models";
#endif

    const char* home = std::getenv("HOME");
#ifdef _WIN32
    // In a plain cmd.exe shell HOME is typically unset; USERPROFILE is
    // always set by Windows itself. Fall through to LOCALAPPDATA.
    if (!home || !*home)
        home = std::getenv("USERPROFILE");
    if (!home || !*home)
        home = std::getenv("LOCALAPPDATA");
#endif

    std::string d = (home && *home) ? home : ".";
    d += "/.cache";
    mkdir(d.c_str(), 0755); // ignore EEXIST
    d += "/crispasr";
    mkdir(d.c_str(), 0755);
    return d;
}

void append_unique(std::vector<std::string>& dirs, const std::string& candidate) {
    if (candidate.empty())
        return;
    if (std::find(dirs.begin(), dirs.end(), candidate) == dirs.end())
        dirs.push_back(candidate);
}

// ─── source-identity helpers (issue #250) ────────────────────────────────────

static unsigned long current_pid() {
#ifdef _WIN32
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

// Unique temp path beside `dest` (same directory → same filesystem, so the
// subsequent rename is atomic). Per-process + per-call so concurrent downloads
// of the same target never write the same partial file.
static std::string unique_temp_path(const std::string& dest) {
    static int counter = 0;
    return dest + ".part." + std::to_string(current_pid()) + "." + std::to_string(counter++);
}

// Atomically replace `dest` with `tmp`. POSIX rename() already replaces; on
// Windows plain rename() fails if the target exists, so use MoveFileEx.
static bool atomic_rename(const std::string& tmp, const std::string& dest) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dest.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return ::rename(tmp.c_str(), dest.c_str()) == 0;
#endif
}

// The source-record sidecar path for a cached file (records the URL it came
// from, so a later request for a DIFFERENT url with the same basename is not
// silently served the wrong bytes).
static std::string sidecar_path(const std::string& file_path) {
    return file_path + ".src";
}

static std::string read_text_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return "";
    std::string out;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    // trim trailing whitespace/newline
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

static void write_text_file(const std::string& path, const std::string& text) {
    const std::string tmp = unique_temp_path(path);
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f)
        return;
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);
    if (!atomic_rename(tmp, path))
        ::remove(tmp.c_str());
}

} // namespace

std::string dir(const std::string& cache_dir_override) {
#ifdef __EMSCRIPTEN__
    // WASM: use Emscripten MEMFS. Models are written by JS via FS.writeFile.
    return cache_dir_override.empty() ? platform_default_dir() : cache_dir_override;
#endif
    std::string selected = cache_dir_override;
    if (selected.empty()) {
        if (const char* env = std::getenv("CRISPASR_CACHE_DIR"); env && *env)
            selected = env;
    }
    if (selected.empty()) {
        if (const char* env = std::getenv("CRISPASR_MODELS_DIR"); env && *env)
            selected = env;
    }
    if (selected.empty())
        selected = platform_default_dir();
    mkdir(selected.c_str(), 0755); // create leaf dir if absent
    return selected;
}

bool file_present(const std::string& path) {
    // std::filesystem, not stat(): on MSVC `stat` resolves to `_stat64i32`,
    // whose st_size is a 32-bit field, and the call FAILS outright for a file
    // larger than 2 GiB. Every GGUF worth caching is bigger than that, so the
    // probe reported "missing" for a model that was sitting right there and
    // -m auto re-downloaded it (#393); the same helper validates a finished
    // download, so a >2 GiB fetch could also be judged failed after it
    // succeeded. Same reasoning as the file_size() note in chat.cpp.
    // Plain path(std::string), not u8path(): these paths come from getenv /
    // argv, i.e. the platform's narrow encoding, which is what path() assumes
    // (u8path would misread a non-ASCII Windows profile dir, and is deprecated
    // in C++20). Matches chat.cpp's construction.
    std::error_code ec;
    const std::filesystem::path fp(path);
    if (!std::filesystem::is_regular_file(fp, ec) || ec)
        return false;
    const std::uintmax_t sz = std::filesystem::file_size(fp, ec);
    return !ec && sz > 0;
}

// Worker: download `url` straight into `dest` (which the public fetch() sets to
// a unique temp path). Split out so the public wrapper can add atomic rename.
static bool fetch_download(const std::string& url, const std::string& dest, bool quiet) {
#ifdef __EMSCRIPTEN__
    // WASM: no network from C++ side. Models are pre-loaded by JS into MEMFS.
    (void)url;
    (void)dest;
    (void)quiet;
    fprintf(stderr, "crispasr: download not available in WASM — load model via JS FS.writeFile\n");
    return false;
#endif
#ifdef _WIN32
    // ── WinHTTP: built-in Windows HTTPS stack, no shell-quoting issues ──────
    if (!quiet)
        fprintf(stderr, "crispasr: downloading via WinHTTP...\n");
    if (fetch_winhttp(url, dest, quiet) && file_present(dest))
        return true;
    if (!quiet)
        fprintf(stderr, "crispasr: WinHTTP failed, falling back...\n");
#endif

#ifdef CRISPASR_USE_LIBCURL
    // ── libcurl API (preferred on POSIX; fallback after WinHTTP on Windows) ──
    if (!quiet)
        fprintf(stderr, "crispasr: downloading via libcurl...\n");
    if (fetch_libcurl(url, dest, quiet) && file_present(dest))
        return true;
    if (!quiet)
        fprintf(stderr, "crispasr: libcurl failed, falling back to curl CLI...\n");
#endif

        // ── curl/wget CLI (not available on iOS/Android) ────────────────────────
#if !defined(__APPLE__) || !defined(TARGET_OS_IPHONE) || !TARGET_OS_IPHONE
    {
        std::string curl_cmd = "curl -fL ";
        curl_cmd += quiet ? "-s " : "--progress-bar ";
        curl_cmd += "-H " + shell_quote("Accept: application/octet-stream") + " ";
        {
            const char* tok = getenv("HF_TOKEN");
            if (!tok)
                tok = getenv("HUGGING_FACE_HUB_TOKEN");
            if (tok && tok[0])
                curl_cmd += "-H " + shell_quote("Authorization: Bearer " + std::string(tok)) + " ";
        }
        curl_cmd += "-o " + shell_quote(dest) + " " + shell_quote(url);

        int rc = std::system(curl_cmd.c_str());
        if (rc == 0 && file_present(dest))
            return true;

        std::string wget_cmd = "wget ";
        wget_cmd += quiet ? "-q " : "--show-progress ";
        wget_cmd += "--header=" + shell_quote("Accept: application/octet-stream") + " ";
        {
            const char* tok = getenv("HF_TOKEN");
            if (!tok)
                tok = getenv("HUGGING_FACE_HUB_TOKEN");
            if (tok && tok[0])
                wget_cmd += "--header=" + shell_quote("Authorization: Bearer " + std::string(tok)) + " ";
        }
        wget_cmd += "-O " + shell_quote(dest) + " " + shell_quote(url);

        rc = std::system(wget_cmd.c_str());
        if (rc == 0 && file_present(dest))
            return true;
    }
#endif

    fprintf(stderr,
#if defined(_WIN32) && defined(CRISPASR_USE_LIBCURL)
            "crispasr: download failed (WinHTTP + libcurl + curl + wget all rejected). "
#elif defined(_WIN32)
            "crispasr: download failed (WinHTTP + curl + wget all rejected). "
#elif defined(CRISPASR_USE_LIBCURL)
            "crispasr: download failed (libcurl + curl + wget all rejected). "
#else
            "crispasr: download failed (curl + wget both rejected). "
#endif
            "Install curl or wget, or fetch manually:\n  %s\n  -> %s\n",
            url.c_str(), dest.c_str());
    return false;
}

// Public fetch: download atomically (issue #250 Q10) — the backends write to a
// unique temp path beside `dest`, and only a complete download is renamed into
// place. An interrupted transfer therefore never leaves a partial non-zero file
// at `dest` (which file_present() would later mistake for a valid cache hit),
// and concurrent downloads to the same `dest` use distinct temps.
bool fetch(const std::string& url, const std::string& dest, bool quiet) {
    std::error_code ec;
    const auto parent = std::filesystem::path(dest).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent, ec);
    if (ec)
        return false;
    const std::string tmp = unique_temp_path(dest);
    if (fetch_download(url, tmp, quiet) && file_present(tmp) && atomic_rename(tmp, dest) && file_present(dest))
        return true;
    ::remove(tmp.c_str()); // clean up any partial/temp bytes
    return false;
}

// Build the well-known search list for already-on-disk model files.
// Caller's cache_dir_override (or the canonical ~/.cache/crispasr) is
// always tried first. Then we probe a small set of common locations
// users typically already have populated, so an offline / flaky-network
// invocation can succeed without re-downloading multi-GB GGUFs:
//
//   1. The dispatcher's chosen cache dir.
//   2. $CRISPASR_MODELS_DIR (set by users with a dedicated model SSD).
//   3. ~/.cache/crispasr-models           (legacy alt cache).
//   4. ~/.cache/huggingface/hub           (raw HF download cache —
//      filename match is rough, but worth a glance).
//
// Deliberately NO absolute machine-specific defaults: this list used to
// carry two maintainer paths, which shipped one developer's directory
// layout to every user, made a unit test depend on whether that volume
// happened to be mounted, and probed directories no user has. Anyone with
// a dedicated model volume points $CRISPASR_MODELS_DIR at it.
//
// The list is platform-agnostic; non-existent dirs are skipped silently.
static std::vector<std::string> well_known_search_dirs(const std::string& cache_dir_override) {
    std::vector<std::string> dirs;
    append_unique(dirs, dir(cache_dir_override));

    if (const char* env = std::getenv("CRISPASR_MODELS_DIR"); env && *env) {
        append_unique(dirs, env);
    }
    append_unique(dirs, platform_default_dir());

    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || !*home)
        home = std::getenv("USERPROFILE");
#endif
    if (home && *home) {
        append_unique(dirs, std::string(home) + "/.cache/crispasr-models");
        append_unique(dirs, std::string(home) + "/.cache/huggingface/hub");
    }
    return dirs;
}

std::string probe_cached_file(const std::string& filename, const std::string& cache_dir_override) {
    for (const auto& d : well_known_search_dirs(cache_dir_override)) {
        const std::string p = d + "/" + filename;
        if (file_present(p))
            return p;
    }
    return "";
}

std::string ensure_cached_file(const std::string& filename, const std::string& url, bool quiet,
                               const char* pretty_label, const std::string& cache_dir_override) {
    // Probe all well-known locations. A hit is reused unless its `.src` sidecar
    // records a DIFFERENT origin url — that is a same-basename file from another
    // repository (issue #250) and must not be handed to the loader. No sidecar =
    // a pre-#250 cache or a hand-managed model dir → trusted as-is (backward
    // compat; keeps dedicated model SSDs working without re-download).
    const std::string canonical = dir(cache_dir_override);
    for (const auto& d : well_known_search_dirs(cache_dir_override)) {
        const std::string p = d + "/" + filename;
        if (!file_present(p))
            continue;
        const std::string sc = sidecar_path(p);
        if (file_present(sc) && read_text_file(sc) != url) {
            if (!quiet)
                fprintf(stderr, "%s: cached %s is from a different source; not reusing\n", pretty_label, p.c_str());
            continue; // wrong source — keep probing, then re-download below
        }
        if (!quiet)
            fprintf(stderr, "%s: using cached %s\n", pretty_label, p.c_str());
        return p;
    }

    // No source-consistent hit. Download into the canonical cache dir (atomically,
    // overwriting any wrong-source file that lives there) and record the origin.
    const std::string dst = canonical + "/" + filename;
    if (!quiet)
        fprintf(stderr, "%s: downloading %s\n", pretty_label, url.c_str());
    if (!fetch(url, dst, quiet))
        return "";
    write_text_file(sidecar_path(dst), url); // record source so a later same-basename request validates
    return dst;
}

} // namespace crispasr_cache
