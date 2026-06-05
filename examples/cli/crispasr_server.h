#pragma once

#include "whisper_params.h"
#include <string>

// Run CrispASR as an HTTP server with persistent model.
// Returns 0 on clean shutdown, non-zero on error.
int crispasr_run_server(whisper_params& params, const std::string& host = "127.0.0.1", int port = 8080);
