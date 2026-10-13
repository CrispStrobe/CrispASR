# Dia late decoder stage diagnostic

The unchanged fixed F32-weights/F32-cross-cache speech control fails, despite
all 40 canonical replay stages passing. Its first remaining differing draw is
step 978/head 2: tiny neural differences swap tokens 46/904 at the nucleus
boundary. Original/native filtering agrees on identical retained logits.

`tools/ci-heavy/dia_late_stage_control.py` captures actual original decoder
layers at step 978 and compares the native model under the same complete
original input history. `DIA_CAPTURE_STEP` selects original hooks;
`CRISPASR_DIA_DUMP_STEP` selects native dumps. Both default to step zero.
The diagnostic reference records `dia_decoder_capture_step`; the diff validates
that metadata, selects it explicitly and checks the actual capture marker.
The immutable canonical archive remains unchanged and has no late-step marker.

`CRISPASR_DIA_DIFF_DUMP_DIR` preserves the complete native arrays in a **new**
directory; existing directories are rejected. Without it the diff cleans its
scratch directory normally. The replay environment restores all inherited
capture settings. Missing or invalid indices cannot produce a passing capture.

Local tiny original-model controls compare the requested late layer with the
actual corresponding forward call, require unchanged complete inputs/logits/
codes/PCM, and check default/late/default repetition and restored hooks. Small
compiled contracts check the actual parser/environment and owned dump paths.
These controls use tiny original neural weights and a labelled dummy codec;
they establish capture transparency, not real-model speech quality.

The GH worker uses the pinned original source/model, Torch 2.6 CPU, seed 42,
temperature 1.2, top-p .95, top-k 45 and CFG 3. Original generated input/code
grids must match the retained canonical grids, and actual original DAC/ASR
must pass. It then audits the identical full-F32 native export and critical
DAC, runs canonical/late/canonical native replay, requires unchanged complete
inputs/logits/decoded PCM, and retains scaled-hidden and invalid-index negatives.
All native arrays, source files and failed logs survive in the artifact.

This is layer/teacher-replay diagnosis. It changes no neural or sampling math,
accepts no native free speech or CUDA/Q4 behavior, and promotes no model pin.
