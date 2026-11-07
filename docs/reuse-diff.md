# RE-USE intermediate comparison

The RE-USE CPU implementation is experimental and is not a CLI or session
backend. To diagnose it against the pinned original checkpoint, create a
16 kHz, mono, 16-bit PCM WAV with at least 400 samples and a length divisible
by 80. The reference dumper uses at most the first 16,000 samples, so select a
short, non-silent speech excerpt deliberately. One-second comparisons are
substantially slower and create larger reference archives.

```sh
python tools/dump_reference.py --backend reuse --model-dir /path/to/pinned-model \
  --audio excerpt.wav --output reuse-ref.gguf
build/bin/crispasr-diff reuse reuse-f32.gguf reuse-ref.gguf excerpt.wav
```

The dumper checks the pinned checkpoint and configuration hashes and uses the
separately audited, non-fused CPU selective scan. The reference archive embeds
the exact input samples, not model weights. The native run consumes that input
from the archive; the final CLI argument is required by the common interface.
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
