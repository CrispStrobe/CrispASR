// crispasr_separate_cli.h — CLI dispatcher for the `--separate` task (§248).
//
// Source separation is its own task (audio out, N named stems), not a
// transcribe backend. This dispatcher resolves the model, detects its
// architecture from the GGUF, runs the matching separation backend
// (scnet / mini-bs-roformer / mel-band-roformer / htdemucs), and writes one WAV per selected stem through
// the shared surface in src/core/separation_io.h. See
// docs/source-separation-surface.md.

#pragma once

struct whisper_params;

// Run the --separate task for every input file in `params.fname_inp`. Returns
// a process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_separate(const whisper_params& params);
