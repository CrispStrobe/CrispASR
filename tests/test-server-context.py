#!/usr/bin/env python3
"""Weight-free #588 controls for the production HTTP context routing.

Compile both complete request-parameter blocks, form readers, streaming callback
and ignored-context warning from crispasr_server.cpp unchanged. Only audio/model
inference and unrelated response validation are replaced by a recording backend.
This checks server parameter ownership, not model language-selection quality.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(text, signature):
    start = text.index(signature)
    return text[start:text.index("\n}", start) + 2]


def source():
    server = (ROOT / "examples/cli/crispasr_server.cpp").read_text()
    native = server[server.index('svr.Post("/inference",'):server.index('svr.Post("/v1/audio/transcriptions",')]
    openai = server[server.index('svr.Post("/v1/audio/transcriptions",'):server.index('svr.Post("/load",')]
    native_params = native[native.index("whisper_params rp = params;"):native.index("// #261: soft-degrade")]
    openai_params = openai[openai.index("std::string response_format ="):openai.index("// #261: soft-degrade")]
    readers = "\n".join(function(server, signature) for signature in (
        "static bool req_has_part(", "static form_part req_part(", "static std::string form_string(",
        "static float form_float(", "static int form_int(", "static uint64_t form_u64(", "static bool form_bool("))
    warning_start = server.index("// #588: `context` only biases")
    warning_end = server.index("const bool want_auto_lang", warning_start)
    buffered_warning = server[warning_start:warning_end]
    # A shared warning function is optional here so the original implementation
    # compiles and the streaming regression is observed as a behavioral failure.
    helper = (function(server, "static void warn_ignored_asr_context(")
              if "static void warn_ignored_asr_context(" in server else "")
    stream_start = openai.index("[pcmf32,")
    stream_end = openai.index("\n                });", stream_start)
    streaming_provider = openai[stream_start:stream_end] + "\n                }"
    # Compile the actual capability declarations without instantiating models.
    def capability(filename):
        text = (ROOT / "examples/cli" / filename).read_text()
        return re.search(r"bool honours_context\(\) const override \{[^}]+\}", text).group()

    return r'''
#include "whisper_params.h"
#include "crispasr_backend.h"
#include "core/asr_sensitivity.h"
#include "httplib.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

// whisper_params asks this unrelated default constructor for best_of only.
extern "C" whisper_full_params whisper_full_default_params(whisper_sampling_strategy) {
    whisper_full_params p{}; p.greedy.best_of = 5; return p;
}
using form_part = httplib::FormData;
''' + readers + r'''
static std::string validate_request_language(const std::string&, const std::string&) { return {}; }
template<class... Args> void json_error(Args&&...) { std::abort(); }
static std::string crispasr_json_escape(const std::string& value) { return value; }
static void require(bool condition) { if (!condition) std::abort(); }
struct RecordingBackend : CrispasrBackend {
    std::string seen;
    const char* name() const override { return "recording-unsupported"; }
    uint32_t capabilities() const override { return CAP_STREAMING; }
    bool init(const whisper_params&) override { return true; }
    void shutdown() override {}
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params& p) override {
        seen = p.context; return {};
    }
    void transcribe_streaming(const float*, int, int64_t, const whisper_params& p,
                              crispasr_stream_callback callback) override {
        seen = p.context; callback("recorded", true);
    }
};
struct QwenBackend : RecordingBackend {
    bool context_enabled_ = true;
''' + capability("crispasr_backend_qwen3.cpp") + r'''
};
struct VibeBackend : RecordingBackend {
''' + capability("crispasr_backend_vibevoice.cpp") + r'''
};
''' + helper + r'''
static whisper_params parse(int endpoint, const httplib::Request& req, const whisper_params& params) {
    whisper_params output;
    httplib::Response res;
    // Keep the route's early-return validation flow unchanged.
    auto handler = [&] {
        if (endpoint == 0) {
''' + native_params + r'''
            output = rp;
        } else {
''' + openai_params + r'''
            output = rp;
        }
    };
    handler();
    return output;
}
static void buffered(RecordingBackend* backend, const whisper_params& rp) {
''' + buffered_warning + r'''
    backend->transcribe(nullptr, 0, 0, rp);
}
static void streaming(RecordingBackend* borrowed, const whisper_params& input) {
    std::unique_ptr<RecordingBackend, void (*)(RecordingBackend*)> backend(borrowed, [](RecordingBackend*) {});
    auto rp = input;
    std::vector<float> pcmf32(160);
    std::mutex model_mutex;
    httplib::DataSink sink;
    sink.write = [](const char*, size_t) { return true; };
    sink.done = [] {};
    auto provider = ''' + streaming_provider + r''';
    // HTTP providers run later, after request parsing: keep their own params.
    rp.context = "context changed after callback setup";
    provider(0, sink);
    require(backend->seen == input.context);
}
static httplib::Request request(int channel, const char* value) {
    httplib::Request req;
    if (!value) return req;
    if (channel == 0) req.form.fields.emplace("context", httplib::FormField{"context", value, {}});
    if (channel == 1) {
        httplib::FormData part; part.name = "context"; part.content = value;
        part.filename = "context.txt"; req.form.files.emplace("context", part);
    }
    if (channel == 2) req.params.emplace("context", value);
    return req;
}
int main(int argc, char** argv) {
    require(argc == 2);
    const std::string test = argv[1];
    whisper_params params;
    params.context = "startup context";
    params.prompt = "startup prompt";
    params.hotwords = "startup hotwords";
    if (test == "defaults") {
        for (int endpoint : {0, 1}) {
            auto out = parse(endpoint, request(0, nullptr), params);
            require(out.context == params.context);
            for (int channel : {0, 1, 2}) {
                for (const char* empty : {"", " \t "}) {
                    out = parse(endpoint, request(channel, empty), params);
                    require(out.context == params.context);
                }
            }
            whisper_params empty;
            require(parse(endpoint, request(0, nullptr), empty).context.empty());
        }
    } else if (test == "override") {
        for (int endpoint : {0, 1}) for (int channel : {0, 1, 2}) {
            auto out = parse(endpoint, request(channel, " \tEnglish\t "), params);
            require(out.context == "English");
            require(out.prompt == "startup prompt" && out.hotwords == "startup hotwords");
            require(out.language == "auto" && !out.translate);
            auto exact = parse(endpoint, request(channel, "English\nРусский: verbatim."), params);
            require(exact.context == "English\nРусский: verbatim.");
        }
        // Multipart has the same precedence as every other server form field.
        auto req = request(0, "multipart"); req.params.emplace("context", "query");
        for (int endpoint : {0, 1}) require(parse(endpoint, req, params).context == "multipart");
    } else if (test == "prompt") {
        auto req = request(0, "English");
        req.form.fields.emplace("prompt", httplib::FormField{"prompt", "decoder prefix", {}});
        req.form.fields.emplace("hotwords", httplib::FormField{"hotwords", "independent hotwords", {}});
        auto out = parse(1, req, params);
        require(out.context == "English" && out.prompt == "decoder prefix");
        require(out.hotwords == "independent hotwords");
        req.form.fields.erase("context");
        require(parse(1, req, params).context == "startup context");
    } else if (test == "consecutive") {
        for (const std::string startup : {"startup context", ""}) for (int endpoint : {0, 1}) {
            auto defaults = params; defaults.context = startup;
            RecordingBackend backend;
            for (const char* value : {"English", "Русский", "", "English"}) {
                auto out = parse(endpoint, request(0, value), defaults);
                backend.transcribe(nullptr, 0, 0, out);
                require(backend.seen == (*value ? std::string(value) : startup));
                require(defaults.context == startup);
            }
            require(parse(endpoint, request(0, nullptr), defaults).context == startup);
        }
    } else if (test == "concurrent") {
        constexpr int count = 12;
        std::mutex mutex; std::condition_variable cv; int waiting = 0;
        std::vector<std::thread> threads;
        for (int i = 0; i < count; ++i) threads.emplace_back([&, i] {
            const std::string expected = i % 3 ? "request-" + std::to_string(i) : "startup context";
            auto req = request(i % 3, i % 3 ? expected.c_str() : nullptr);
            auto rp = parse(i % 2, req, params);
            { std::unique_lock<std::mutex> lock(mutex); ++waiting; cv.notify_all();
              cv.wait(lock, [&] { return waiting == count; }); }
            RecordingBackend backend;
            for (int repeat = 0; repeat < 100; ++repeat) {
                backend.transcribe(nullptr, 0, 0, rp);
                require(backend.seen == expected);
                require(parse(i % 2, req, params).context == expected);
            }
        });
        for (auto& thread : threads) thread.join();
        require(params.context == "startup context");
    } else if (test == "capabilities") {
        RecordingBackend unsupported; QwenBackend qwen; VibeBackend vibe;
        require(!unsupported.honours_context() && qwen.honours_context() && vibe.honours_context());
        qwen.context_enabled_ = false; require(!qwen.honours_context());
    } else if (test == "stream-isolation") {
        QwenBackend backend;
        for (const char* context : {"English", "Русский", ""}) {
            auto rp = parse(1, request(0, context), params);
            streaming(&backend, rp);
            require(backend.seen == (*context ? context : "startup context"));
            require(params.context == "startup context");
        }
    } else if (test == "warning-mixed") {
        RecordingBackend backend;
        buffered(&backend, params); streaming(&backend, params);
    } else if (test.rfind("warning-", 0) == 0) {
        const bool stream = test.find("stream") != std::string::npos;
        auto dispatch = stream ? streaming : buffered;
        RecordingBackend unsupported; QwenBackend qwen; VibeBackend vibe;
        whisper_params empty;
        dispatch(&unsupported, empty); // must not consume the one-time warning
        dispatch(&qwen, params); dispatch(&vibe, params); // supported must not warn
        require(qwen.seen == params.context && vibe.seen == params.context);
        if (test.find("supported") == std::string::npos) {
            if (test.find("concurrent") != std::string::npos) {
                std::vector<std::thread> threads;
                for (int i = 0; i < 12; ++i) threads.emplace_back([&] {
                    RecordingBackend own; dispatch(&own, params); require(own.seen == params.context);
                });
                for (auto& thread : threads) thread.join();
            } else {
                dispatch(&unsupported, params); dispatch(&unsupported, params);
                require(unsupported.seen == params.context);
            }
        }
    } else std::abort();
    std::cout << test << " passed\n";
}
'''


class ServerContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "")) or [shutil.which("c++") or "c++"]
        cls.temp = tempfile.TemporaryDirectory(prefix="server-context-")
        work = Path(cls.temp.name)
        cpp = work / "probe.cpp"
        cpp.write_text(source())
        cls.binary = work / "probe"
        command = compiler + ["-std=c++17", "-O0", "-pthread", str(cpp), "-o", str(cls.binary)]
        for folder in ["examples/cli", "examples/server", "src", "include", "ggml/include"]:
            command += ["-I", str(ROOT / folder)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_case(self, case, warnings=0):
        result = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), case + " passed")
        self.assertEqual(result.stderr.count("ignores 'context'"), warnings, result.stderr)
        if warnings:
            self.assertIn("recording-unsupported", result.stderr)
            self.assertNotIn("startup context", result.stderr)

    def test_defaults_and_empty_context(self):
        self.run_case("defaults")

    def test_both_endpoints_and_all_form_channels(self):
        self.run_case("override")

    def test_prompt_hotwords_and_language_stay_independent(self):
        self.run_case("prompt")

    def test_consecutive_request_isolation(self):
        self.run_case("consecutive")

    def test_concurrent_request_isolation(self):
        self.run_case("concurrent")

    def test_streaming_callback_captures_context_by_value(self):
        self.run_case("stream-isolation")

    def test_real_backend_capability_declarations(self):
        self.run_case("capabilities")

    def test_supported_and_empty_context_do_not_warn(self):
        for mode in ["buffered", "stream"]:
            with self.subTest(mode=mode):
                self.run_case("warning-" + mode + "-supported")

    def test_unsupported_context_warns_once(self):
        self.run_case("warning-buffered", warnings=1)

    def test_unsupported_streaming_context_warns_once(self):
        self.run_case("warning-stream", warnings=1)

    def test_buffered_and_streaming_share_the_one_time_warning(self):
        self.run_case("warning-mixed", warnings=1)

    def test_concurrent_unsupported_context_warns_once(self):
        for mode in ["buffered", "stream"]:
            with self.subTest(mode=mode):
                self.run_case("warning-" + mode + "-concurrent", warnings=1)


if __name__ == "__main__":
    unittest.main()
