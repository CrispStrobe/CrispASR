# Current-main Dia serial head preparation

The runtime's private `src/dia_serial_head.h` carries the same generic compute
and graph-apply functions as the immutable serial-head experiment3c2fbb5a.
`CRISPASR_DIA_SERIAL_HEAD=off` (also absent/empty) returns the existing matmul
node unchanged. `identity` copies its result; `serial` uses separately rounded
F32 products and sequential F32 addition from actual weights/current hidden.
All heads and steps use the same arithmetic. The ordinary matmul remains a
dependency, so this deliberately computes both paths and claims no speed win.
GPU or non-F32/mismatched heads reject before the encoder for either opt-in
mode. OFF retains the existing model/backend path.

This preparation does not port the old runtime wholesale. Current top-k and
output-protocol fixes stay in place; current cross caches remain F16, its sampler
remains `std::discrete_distribution`, and no portable RNG/source-RoPE diagnostic
is enabled. The pinned full-generation causal experiment remains separate.
No speech/default/GPU/Q4 acceptance follows from these graph checks.

`tests/test-dia-serial-head.cpp` runs the actual graph on tiny analytic tensors,
including a F32 fusion witness, two batches, one/four threads, exact identity
and mode/backend/precision/shape admission. CMake registers it with `unit;dia`.
The synthesis implementation is shared by CLI and C ABI; no binding signatures,
model detection, registry, conversion or quantization policy changes here.

`preflight.py` compiles that test and a captured-input graph probe against the
actual current static ggml libraries. It removes the GPU guard in an isolated
copy and requires the unit test to fail. It SHA-verifies the immutable published
projection fixture and checks all18,504 serial outputs on both hidden states,
OFF/identity/thread agreement, and the whole current runtime's C++ syntax.
All source/ELF/library/array hashes and commands are retained in a fresh output
directory. No full neural model is loaded.

```sh
python tools/ci-heavy/dia-main-head/preflight.py \
  --output FRESH_OUTPUT --build CURRENT_STATIC_BUILD \
  --fixture complete-reduction-cause-v1.zip

gh workflow run heavy-cpu.yml --ref IMMUTABLE_CANDIDATE \
  -f script=tools/ci-heavy/dia_main_head_validate.py \
  -f pip=numpy==1.26.4 -f timeout_minutes=90
```

The hosted validator compiles the real current `dia-tts` library and registered
CMake test target, runs it through ctest and executes this preflight. It preserves
actual static libraries and sources for an independent artifact replay. Further
current-main generation, bounded prompts/seeds, CLI/Session roundtrips, precision
and GPU checks precede any promotion. Linux static builds avoid CIFS symlink
requirements when running the small preflight locally.
