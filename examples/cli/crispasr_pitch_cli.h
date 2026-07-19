// crispasr_pitch_cli.h — CLI dispatcher for the `--pitch` task.
//
// Pitch (F0) estimation is its own task: audio in, a pitch track out — not
// crispasr_segments — so per docs/source-separation-surface.md it gets its own
// early dispatcher rather than being layered onto transcribe(). This resolves
// the model, detects its architecture from the GGUF ("crepe"), decodes the
// audio to 16 kHz mono and prints one frame per line (or JSON).

#pragma once

struct whisper_params;

// Run the --pitch task for every input file in `params.fname_inp`. Returns a
// process exit code (0 = success). Called from crispasr_run_backend() before
// any transcribe backend is constructed.
int crispasr_run_pitch(const whisper_params& params);
