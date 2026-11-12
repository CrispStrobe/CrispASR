# RE-USE intermediate comparison

The RE-USE CPU implementation is experimental research and is not a product
CLI, server or session backend. Its standalone build shares CrispASR's existing
comparison machinery without linking the reference loader into `libcrispasr`.
See the [reconciliation record](reuse-reconciliation-20261112.md). To diagnose it against the pinned original checkpoint, create a
16 kHz, mono, 16-bit PCM WAV with at least 400 samples and a length divisible
by 80. The reference dumper rejects inputs over 16,000 samples; select each
short, non-silent speech excerpt deliberately. One-second comparisons are
substantially slower and create larger reference archives.

```sh
git submodule update --init --recursive
# BUILD_DIR is a caller-selected build directory outside the source tree.
cmake -S tools/ci-heavy/issue579_reuse -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build "$BUILD_DIR" -j2
ctest --test-dir "$BUILD_DIR" --output-on-failure
python tools/dump_reference.py --backend reuse --model-dir /path/to/pinned-model \
  --audio excerpt.wav --output reuse-ref.gguf
"$BUILD_DIR/reuse-research-diff" reuse-f32.gguf reuse-ref.gguf
# Optional: write unconditioned native-endian float32 waveform after both gates pass.
"$BUILD_DIR/reuse-research-diff" reuse-f32.gguf reuse-ref.gguf native-waveform.f32
```

The dumper checks the pinned checkpoint and configuration hashes and uses the
separately audited, non-fused CPU selective scan. The reference archive embeds
the exact input samples, not model weights. The native run consumes that input
from the archive. The standalone command takes only the model and reference
archive; the input WAV is used by the dumper.
Keep reference archives outside the repository.

The report compares raw frontend magnitude and phase, all six dense encoder
boundaries, the temporal and final boundary of each of the 30 Mamba blocks,
the two decoder outputs, and the reconstructed waveform. The `native/` rows
use the native frontend. The `conditioned/` rows repeat the forward pass with
reference phase values, isolating later model stages. Both groups must pass;
the conditioned run is diagnostic and cannot substitute for frontend parity.
The gate checks raw relative RMS, per-row cosine and amplitude ratio, and reports
maximum error. This matters because +π and −π describe the same
complex value but are different inputs to this model.

New reference archives also capture the exact upstream Hann window and the
STFT's real and imaginary values. Optional `diagnostic/` rows compare these
before phase calculation; they do not replace or relax the existing gates.
Older reference archives remain usable and simply omit those rows. For a
cross-environment first-frame investigation, run
`tools/ci-heavy/issue579_reuse/frontend_stability.py --help`: it captures
byte-checked input, window and spectrum, and can replay a second environment's
window. Its output is diagnostic and never declares model parity.

`tools/ci-heavy/issue579_reuse/chunked.py` is a research-only bounded
overlap-add diagnostic. It pads a short or partial final window and returns the
original length. Its one-second windows are chosen to fit the comparator; they
do not establish an NVIDIA-equivalent long-audio or streaming policy.
