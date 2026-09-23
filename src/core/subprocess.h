// core/subprocess.h — run a program with an argv vector and read its stdout,
// WITHOUT a shell.
//
// popen()/system() hand one string to /bin/sh (or cmd.exe). Any argument that
// came from a request — an uploaded file's name or bytes, a TTS language code,
// the text to phonemise — can then close the quoting and run a command. Quoting
// helpers only move the problem: POSIX single-quote escaping is sound, but
// cmd.exe has no quoting that is safe for arbitrary text. This helper never
// builds a command line for a shell:
//
//   POSIX:   pipe + posix_spawnp(argv) — the kernel receives argv as-is.
//   Windows: CreateProcessW with each argument quoted by the MSVC CRT rules
//            (the rules CommandLineToArgvW / the child's CRT parse with), so
//            every argument round-trips to the child verbatim. cmd.exe is
//            never involved.
//
// Usage:
//   core_subprocess::ReadPipe p;
//   if (p.open({"ffmpeg", "-i", path, "-f", "s16le", "-"})) {
//       ... fread(buf, 1, n, p.out) ...
//       int rc = p.close();   // child exit code, or -1
//   }
//
// stderr of the child is discarded unless keep_stderr is set.

#pragma once

#include <cstdio>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
// iOS forbids spawning processes; Android has posix_spawn only from API 28;
// Emscripten has no posix_spawnp at all (wasm-ld: undefined symbol). None of
// them ships ffmpeg / espeak-ng binaries, so there open() just fails.
#if (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) || (defined(__ANDROID_API__) && __ANDROID_API__ < 28) ||           \
    defined(__EMSCRIPTEN__)
#define CORE_SUBPROCESS_UNAVAILABLE 1
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#elif !defined(CORE_SUBPROCESS_UNAVAILABLE)
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace core_subprocess {

#if defined(_WIN32)
// Quote one argument so the MSVC CRT parser in the child reconstructs it
// exactly: backslashes are literal except before a double quote, where each
// backslash and the quote itself must be escaped.
inline std::wstring quote_arg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return a;
    }
    std::wstring out = L"\"";
    for (size_t i = 0;; ++i) {
        size_t n_bs = 0;
        while (i < a.size() && a[i] == L'\\') {
            ++i;
            ++n_bs;
        }
        if (i == a.size()) {
            out.append(n_bs * 2, L'\\');
            break;
        }
        if (a[i] == L'"') {
            out.append(n_bs * 2 + 1, L'\\');
            out.push_back(L'"');
        } else {
            out.append(n_bs, L'\\');
            out.push_back(a[i]);
        }
    }
    out.push_back(L'"');
    return out;
}

inline bool widen_utf8(const std::string& in, std::wstring& out) {
    out.clear();
    if (in.empty()) {
        return true;
    }
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, in.data(), (int)in.size(), nullptr, 0);
    if (n <= 0) {
        return false;
    }
    out.resize((size_t)n);
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, in.data(), (int)in.size(), &out[0], n) == n;
}
#endif

struct ReadPipe {
    FILE* out = nullptr; // child's stdout, binary

#if defined(_WIN32)
    HANDLE proc = nullptr;
#elif !defined(CORE_SUBPROCESS_UNAVAILABLE)
    pid_t pid = -1;
#endif

    ReadPipe() = default;
    ReadPipe(const ReadPipe&) = delete;
    ReadPipe& operator=(const ReadPipe&) = delete;
    ~ReadPipe() { close(); }

    // argv[0] is looked up on PATH. Returns false if the program could not be
    // started (the caller treats that like "tool not installed").
    bool open(const std::vector<std::string>& argv, bool keep_stderr = false) {
        if (argv.empty() || out) {
            return false;
        }
        for (const auto& a : argv) {
            if (a.find('\0') != std::string::npos) {
                return false;
            }
        }
#if defined(_WIN32)
        std::wstring cmdline;
        for (size_t i = 0; i < argv.size(); ++i) {
            std::wstring w;
            if (!widen_utf8(argv[i], w)) {
                return false;
            }
            if (i) {
                cmdline.push_back(L' ');
            }
            cmdline += quote_arg(w);
        }
        SECURITY_ATTRIBUTES sa = {sizeof(sa), nullptr, TRUE};
        HANDLE rd = nullptr, wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0)) {
            return false;
        }
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
        HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = wr;
        si.hStdError = keep_stderr ? GetStdHandle(STD_ERROR_HANDLE) : nul;
        PROCESS_INFORMATION pi = {};
        const BOOL ok =
            CreateProcessW(nullptr, &cmdline[0], nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(wr);
        if (nul != INVALID_HANDLE_VALUE) {
            CloseHandle(nul);
        }
        if (!ok) {
            CloseHandle(rd);
            return false;
        }
        CloseHandle(pi.hThread);
        proc = pi.hProcess;
        const int fd = _open_osfhandle((intptr_t)rd, _O_RDONLY | _O_BINARY);
        if (fd < 0) {
            CloseHandle(rd);
            close();
            return false;
        }
        out = _fdopen(fd, "rb");
        if (!out) {
            _close(fd);
            close();
            return false;
        }
        return true;
#elif defined(CORE_SUBPROCESS_UNAVAILABLE)
        (void)keep_stderr;
        return false;
#else
        // Close-on-exec on both ends: a child spawned concurrently by another
        // server thread must not inherit our write end, or EOF never arrives.
        // dup2 onto STDOUT_FILENO in the file actions clears the flag for the
        // one child that should have it.
        int fds[2];
#if defined(__linux__)
        if (pipe2(fds, O_CLOEXEC) != 0) {
            return false;
        }
#else
        if (pipe(fds) != 0) {
            return false;
        }
        fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        fcntl(fds[1], F_SETFD, FD_CLOEXEC);
#endif
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&fa, fds[0]);
        posix_spawn_file_actions_addclose(&fa, fds[1]);
        if (!keep_stderr) {
            posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
        }
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) {
            cargv.push_back(const_cast<char*>(a.c_str()));
        }
        cargv.push_back(nullptr);
        const int rc = posix_spawnp(&pid, cargv[0], &fa, nullptr, cargv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        ::close(fds[1]);
        if (rc != 0) {
            ::close(fds[0]);
            pid = -1;
            return false;
        }
        out = fdopen(fds[0], "r");
        if (!out) {
            ::close(fds[0]);
            close();
            return false;
        }
        return true;
#endif
    }

    // Closes the pipe and reaps the child. Returns its exit code, or -1 if it
    // was not started, was killed by a signal, or could not be waited for.
    int close() {
        if (out) {
            fclose(out);
            out = nullptr;
        }
#if defined(_WIN32)
        if (!proc) {
            return -1;
        }
        WaitForSingleObject(proc, INFINITE);
        DWORD code = (DWORD)-1;
        const bool got = GetExitCodeProcess(proc, &code) != 0;
        CloseHandle(proc);
        proc = nullptr;
        return got ? (int)code : -1;
#elif defined(CORE_SUBPROCESS_UNAVAILABLE)
        return -1;
#else
        if (pid <= 0) {
            return -1;
        }
        int status = 0;
        pid_t r;
        do {
            r = waitpid(pid, &status, 0);
        } while (r < 0 && errno == EINTR);
        pid = -1;
        if (r < 0 || !WIFEXITED(status)) {
            return -1;
        }
        return WEXITSTATUS(status);
#endif
    }
};

} // namespace core_subprocess
