# RE-USE intermediate comparison

The RE-USE CPU implementation is experimental and is not a CLI or session
backend. To diagnose it against the pinned original checkpoint, create a
16 kHz, mono, 16-bit PCM WAV with at least 400 samples and a length divisible
by 80. The reference dumper uses at most the first 1,600 samples, so select a
short, non-silent speech excerpt deliberately.

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
The gate checks raw relative RMS and amplitude ratio as well as reporting
cosine and maximum error. This matters because +π and −π describe the same
complex value but are different inputs to this model.
