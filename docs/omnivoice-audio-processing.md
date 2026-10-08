# OmniVoice audio processing (#518)

The native reference and decode funnels now mirror the audio/text utilities at
[k2-fsa/OmniVoice 08be0b4](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2).
This branch is awaiting real CLI/session speech acceptance before integration.

Reference order: mono/resample to 24 kHz, measure original RMS, normalize quiet
references to 0.1 RMS, remove silence (200/100/200 ms), reject empty results,
clip to the codec's 960-sample hop, encode, and add missing terminal punctuation
to the reference text. The RVQ disk-cache key hashes the resulting PCM and
encoder fingerprint; punctuation does not invalidate identical audio codes.
Failures still clear the previous speaker before returning.

Output order: remove silence (500/100/100 ms), restore a quiet reference's
original RMS or normalize an unconditioned voice to 0.5 peak, apply linear
100-ms fades, then pad 100 ms per side. Silence removal follows pydub's
PCM16 conversion, integer RMS, sliding windows, overlapping retained ranges
and edge trimming. It is not a 10-ms frame-classification approximation.
Fades are bounded to half the waveform length. Empty output returns no PCM.

Controls are read when a context opens, and work through the CLI, session ABI
and other callers of the same native funnels:

| Environment variable | Default | Effect |
| --- | --- | --- |
| `CRISPASR_OMNIVOICE_PREPROCESS_PROMPT` | 1 | Reference silence cleanup and punctuation; 0 disables |
| `CRISPASR_OMNIVOICE_POSTPROCESS_OUTPUT` | 1 | Output silence cleanup only; 0 disables |
| `CRISPASR_OMNIVOICE_PAD_DURATION` | 0.1 | Seconds of silence per edge; 0 disables |
| `CRISPASR_OMNIVOICE_FADE_DURATION` | 0.1 | Seconds of fade per edge; 0 disables |
| `CRISPASR_OMNIVOICE_AUDIO_LEGACY` | 0 | 1 restores the prior pipeline, including no unconditioned peak normalization |

Explicit individual controls override legacy defaults. Durations must be
finite and between 0 and 10 seconds; invalid environment values keep the
default. Native callers can use `omnivoice_set_audio_processing` before
loading a reference. It validates all inputs before changing state, enables
upstream amplitude handling, and preserves the existing by-value params ABI.
Changing reference preprocessing does not re-encode an already loaded prompt.
As upstream does, disabling output silence cleanup still leaves fades/padding
active. Silence cleanup also quantizes retained PCM to 16 bits.

The independent oracle executes pinned upstream functions with numpy/pydub;
`tools/ci-heavy/omnivoice_audio_parity.py` checks silence, clipping, threshold
boundaries, sub-millisecond/short clips, long pauses, fades and multilingual
punctuation, including a rejected raw-output negative control.
`tools/ci-heavy/omnivoice_audio_acceptance.py` adds same-code raw/processed
waveform parity, actual CLI/session cloning and four unchanged WER <= 0.2
roundtrip gates. Utility parity alone is not speech acceptance. Fades can
attenuate the upstream transient; they do not guarantee removal of every click.
