# OpenVoice conversion stage diagnostic

This prepared private Kaggle adapter uses the exact accepted F16-Melo-derived
inputs and current-source OpenVoice application checkpoint from the paired
conversion pilot. It enables existing native stage timers and checks every
converted output against the same accepted waveform and cosine threshold.
The run also adds a CPU-only native arm using the same weights and inputs,
which helps distinguish host-transfer effects from CPU compute and vocoder
cost. All arms retain the same output parity gate. Only aggregate stage
timings are added to the receipt; native logs and audio
remain in private worker scratch. Original and native inference boundaries and
device differences are the same as in the paired pilot.

The timing switch adds logging overhead. Compare it with the previously
accepted instrumentation-off run only as a diagnostic, not a controlled
same-session speedup. Model loading and full original Melo synthesis remain
outside scope. Do not change neural math or default device placement on the
basis of these timers alone.
