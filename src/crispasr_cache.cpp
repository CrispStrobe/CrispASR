// crispasr_cache.cpp — implementation of crispasr_cache.h.
// See header for the contract.

#include "crispasr_cache.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
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
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string to_utf8(const std::wstring& w) {
    if (w.empty())
        return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0)
        return {};
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, nullptr, nullptr);
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
                current_url = to_utf8(loc);
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
    if (!quiet) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    const CURLcode res = curl_easy_perform(curl);
    fclose(f);
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

std::string dir(const std::string& cache_dir_override) {
    if (!cache_dir_override.empty()) {
        mkdir(cache_dir_override.c_str(), 0755); // create leaf dir if absent
        return cache_dir_override;
    }

    const char* home = std::getenv("HOME");
#ifdef _WIN32
    // In a plain cmd.exe shell HOME is typically unset; USERPROFILE is
    // always set by Windows itself.  Fall through to LOCALAPPDATA as a
    // last resort before the /tmp fallback (which is meaningless on Windows).
    if (!home || !*home)
        home = std::getenv("USERPROFILE");
    if (!home || !*home)
        home = std::getenv("LOCALAPPDATA");
#endif

    std::string d = (home && *home) ? home : "/tmp";
    d += "/.cache";
    mkdir(d.c_str(), 0755); // ignore EEXIST
    d += "/crispasr";
    mkdir(d.c_str(), 0755);
    return d;
}

bool file_present(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    return st.st_size > 0;
}

bool fetch(const std::string& url, const std::string& dest, bool quiet) {
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
        curl_cmd += "-o " + shell_quote(dest) + " " + shell_quote(url);

        int rc = std::system(curl_cmd.c_str());
        if (rc == 0 && file_present(dest))
            return true;

        std::string wget_cmd = "wget ";
        wget_cmd += quiet ? "-q " : "--show-progress ";
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

std::string ensure_cached_file(const std::string& filename, const std::string& url, bool quiet,
                               const char* pretty_label, const std::string& cache_dir_override) {
    const std::string dst = dir(cache_dir_override) + "/" + filename;
    if (file_present(dst)) {
        if (!quiet) {
            fprintf(stderr, "%s: using cached %s\n", pretty_label, dst.c_str());
        }
        return dst;
    }
    if (!quiet) {
        fprintf(stderr, "%s: downloading %s\n", pretty_label, url.c_str());
    }
    if (!fetch(url, dst, quiet))
        return "";
    return dst;
}

} // namespace crispasr_cache
