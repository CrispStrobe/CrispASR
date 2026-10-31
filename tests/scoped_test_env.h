#pragma once

#include <cstdlib>
#include <string>

#include "portable_env.h"

// Keep test controls local even when a REQUIRE aborts a test case. Environment
// tests run in separate CTest processes; do not mutate these from worker threads.
class ScopedTestEnv {
    std::string name_;
    std::string saved_;
    bool existed_;

public:
    explicit ScopedTestEnv(const char* name, const char* value = nullptr) : name_(name) {
        const char* old = std::getenv(name);
        existed_ = old != nullptr;
        if (old)
            saved_ = old;
        set(value);
    }
    ~ScopedTestEnv() {
        if (existed_)
            setenv(name_.c_str(), saved_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }
    void set(const char* value) {
        if (value)
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
    }
    ScopedTestEnv(const ScopedTestEnv&) = delete;
    ScopedTestEnv& operator=(const ScopedTestEnv&) = delete;
};
