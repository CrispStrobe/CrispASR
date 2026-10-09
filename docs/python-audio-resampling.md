# Python audio resampling

`CrispASR.transcribe_pcm` and the Session transcription methods (`transcribe`,
`transcribe_chunked`, `transcribe_vad`, `transcribe_with_logits`) accept mono
arrays with an integer `sample_rate`. They prepare contiguous float32 samples
at 16 kHz before calling the native ASR interface. Already-16-kHz float32
samples retain their values, including when the input is a strided view.

Other rates use `crispasr_audio_resample`, an additive C API exposing the
existing shared CLI Kaiser polyphase filter (14 zero crossings, beta 8.6).
Older native libraries use a NumPy implementation of the same filter; SciPy
is not a runtime dependency. The static Whisper WAV helper uses this NumPy
path. Linear interpolation was removed from these paths because it aliases
frequencies above the destination Nyquist frequency into the speech band.

The C API returns a malloc-owned buffer, released by `crispasr_audio_free`.
The Python wrapper copies the result before freeing it. Empty input has zero
samples; other output lengths round up to preserve the trailing sample.
Rates must be positive integers no greater than 384000 Hz. Expansion is
limited to 64 times the input length, and output lengths must fit the native
signed sample count. These bounds also limit filter allocation. A native
failure raises an error rather than silently switching algorithms.

The C ABI **file decoder** `crispasr_audio_load` currently has its own
miniaudio resampling path. It is not the same Kaiser filter as this new
in-memory API or the shared CLI helper. The fixed-audio TTS diagnostics
measure these paths separately; this repair makes no claim that resampling
caused the pending OmniVoice/VoxCPM2 speech failures.

Model-free regression checks compile the actual C API and exercise both
Python implementations, the five transcription entry points and WAV helper:

```sh
python -m unittest discover -s tests -p test_python_audio_resample.py -v
```

The regression workflow runs these checks with ggml headers initialized.
They verify rejection of a 10 kHz tone when converting to 16 kHz, preserved
1 kHz/DC signals, native/fallback agreement within 1e-7 absolute error,
unchanged 16 kHz values, rounded lengths, error resets and independent output
ownership. Full speech acceptance remains subject to the unchanged model
round-trip gates.
