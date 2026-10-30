# Python shared API coverage

The Python package exposes the public operations in
`include/crispasr_session.h`, with equivalent bulk/superset entry points where
appropriate. `tools/check-python-api-coverage.py` inventories every unique
`crispasr_*` function declaration, rather than the older curated 155-symbol smoke list.
The inventory now reads both `include/crispasr_session.h` and
`include/crispasr.h`: 297 unique `crispasr_*` functions. Upstream `whisper_*`
declarations are outside this census. The eight additional audio/container/
channel omissions from the [initial expanded audit](historical/binding-gaps-2026-10-18.md#python-census-boundary)
are implemented and tested through model-free native probes.
The inventory is a source drift guard; it does not establish model accuracy,
platform support, or successful inference from a particular checkpoint.

## Operations

| Native surface | Python entry point | Output / contract |
| --- | --- | --- |
| Session open / explicit / open-with-params | `Session(...)` | Optional `use_gpu`, `verbosity`, `flash_attn`, `n_gpu_layers`; missing load-option API raises on older libraries |
| ASR, chunked, VAD, CTC logits | `Session.transcribe*`, `ctc_vocab()` | Segments, words, native tokens/raw bytes, available probabilities/timestamps/speakers; word alternatives in `SessionWord.alternatives` |
| Progress / segment / token callbacks | `set_progress_callback`, `set_segment_callback`, `set_token_callback` | Synchronous callbacks; sample counts / seconds / raw token bytes respectively |
| Progress and streamed polling | `get_progress`, `reset_progress`, `get_streamed_*_count`, `drain_streamed_*`, `reset_streamed_*` | Native global polling selects the most recently streaming session, which may differ from the receiver |
| TTS / raw TTS | `synthesize`, `synthesize_raw` | Copied mono PCM at `output_sample_rate()`; native marking rules remain enforced |
| Streaming TTS | `synthesize_streaming(text, callback)` | `callback(owned_pcm, is_final)` per sentence, synchronous; failed/empty native audio chunks raise |
| Voice reference PCM | `set_voice_samples(pcm, sample_rate, ref_text=None)` | Contiguous mono reference, native consent/marking rules |
| TADA companions | `tada_set_makeref_models(encoder=None, aligner=None)` | Overrides or clears native companion paths; backend/gate errors propagate |
| TTS CFG scale / error details | `set_tts_cfg_scale`, `last_synth_error` | CFG scale is separate from `set_cfg_weight` |
| Disclosure helpers | `disclaimer_text`, `get_disclaimer_pcm` | Fetch neutral-voice PCM before installing a cloning voice; returned allocation copied and freed |
| Speech-to-speech | `speech_to_speech` | Copied audio and transcript, both native allocations released even on errors |
| Separation | `separate`, `separate_sample_rate` | Stem dictionary; accepts interleaved stereo or `[frames,2]` input |
| Pitch | `pitch`, `pitch_n_frames`, `pitch_sample_rate` | Copied `[time_ms,f0_hz,voiced_probability]` frames |
| Voice conversion | `convert`, `convert_content_dim`, `convert_n_speakers`, `convert_sample_rate` | ContentVec features `[frames,content_dim]`, parallel Hz F0; copied mono PCM |
| Replay-noise dimensions | `convert_inter_channels`, `convert_upsample_product` | Model-derived sizes for optional `noise_zp[inter_channels,frames]`, `noise_sine[frames,upsample_product]`; defaults use native random draws |
| Piano / multi-instrument notes | `piano`, `piano_sample_rate`, `piano_n_notes` | `PianoResult.notes[onset_ms,offset_ms,midi_note,velocity]`, parallel integer GM `programs`, native sample rate |
| Chords | `chords`, `chords_n_spans`, `chords_vocab_size` | `ChordResult.spans[start_ms,end_ms,label,confidence]`, matching names; native resampling |
| Beats | `beats`, `beats_n_events`, `beats_sample_rate`, `beats_tempo_bpm` | `BeatResult.events[time_s,is_downbeat]`, tempo and sample rate; native rejects rate mismatch |
| Tablature | `tab`, `tab_n_frames`, `tab_silent_class`, `tab_frame_period` | `TabResult.emissions[frame,string,class]` log probabilities, silent class, frame period, open-string MIDI pitches; fingering decoding is caller-owned |
| Punctuation / truecase / PCS | `PuncModel`, `TruecaseModel`, `PCSModel` | Context managers; native allocated UTF-8 text freed after copying, including empty text |
| GPU preference / capabilities / versions | `set_gpu_backend`, `backend_capabilities`, `c_api_version`, `dart_helpers_version` | Process-wide load preference; declared verbs are separate from compiled `available_backends` |
| Legacy Whisper DTW / parallel decode | `CrispASR(..., dtw=True, ...)`, `transcribe_pcm(..., n_processors=...)`, `token_dtw_time` | Compiled pointer adapter required for struct-by-value calls; missing helpers raise instead of an unsafe ctypes call |
| Legacy tokens / decode controls | `CrispASR.tokens`, `Segment.tokens`, `transcribe_pcm(decode_options=...)` | Original bytes, timestamps/probabilities, vocabulary-ID alternatives, DTW times; native stored string pointers retained through inference |
| Audio decode / channel metadata | `load_audio`, `load_audio_stereo`, `Session.input_channels`, `output_channels` | `AudioData.samples` owns mono `[frame]` or stereo `[frame,2]` PCM, rate and original channel flag; optional target rate avoids a 16 kHz intermediate |
| Container / signing | `pcm_to_wav`, `c2pa_sign` | Owned binary bytes; native WAV provenance; signing failure/unavailability raises and does not return unsigned success |
| Existing auxiliary APIs | Package-root exports | Alignment, diarization, speaker embedding/roster DB, VAD/slices, language detection, RNNoise, registry/cache, mic, watermark, Kokoro helpers, GGUF detection, LCS dedup, legacy Parakeet |

Music results and conversion buffers are copied before another native call can
replace their session-owned storage. A successful empty music result remains
empty rather than raising. Model-specific rates, pitch units and class mappings
are preserved; a wrapper does not infer capability from a filename.

Legacy `decode_options` accepts `translate`, `detect_language`,
`token_timestamps`, `n_threads`, `max_len`, `best_of`, `split_on_word`,
`no_context`, `single_segment`, `print_realtime`, `print_progress`,
`print_timestamps`, `print_special`, `suppress_blank`, `temperature`,
`max_tokens`, `initial_prompt`, and `alt_n`. Unknown or unavailable controls
raise. Enable DTW while constructing the context; the default enabled recipe
uses N_TOP_MOST (preset 1), one layer, with model-specific presets 3–14 available.
Custom head arrays are not part of this scalar C ABI.

Callbacks run inside native inference: do not re-enter or close their session.
The wrapper retains registered callback objects and transports exceptions back
to the Python inference caller after native control returns. Streaming-TTS
callback arrays are independent copies. Native inference continues after a
callback exception because this C callback ABI has no cancellation return.
Streams retain their parent session; closing a session first closes its streams,
and feed/flush/text calls after stream close raise. Concurrent inference on one
session remains unsupported; use separate sessions.

## Equivalent or retired entry points

- `crispasr_session_pitch_frame`: covered by the bulk `pitch_frames` read.
- `crispasr_session_transcribe_chunked`: the language-aware superset is used.
- `crispasr_transcribe_parallel`: called through compiled
  `crispasr_transcribe_parallel_ptr`; Python never mirrors Whisper's by-value
  full-parameter struct.
- `crispasr_speaker_db_load` and `crispasr_speaker_db_enroll`: native retired
  stubs, deliberately not revived. `SpeakerDB` uses the current roster/consent
  APIs instead.

## Verification and remaining acceptance

```sh
python tools/check-python-api-coverage.py
python tests/test_python_wrapper_operations.py -v
python tests/test_session_metadata.py -v
python tests/test_parakeet_window_metadata.py -v
```

The wrapper operation suite compiles a C probe against the public header and
runs the actual Python methods. It checks buffer copying/freeing, empty results,
units/layouts, strided inputs, errors/older libraries, callback lifetime,
exception transport and re-entry, load-option layout/pointer width, and the
compiled legacy struct adapter. Metadata fixtures separately compile literal
native result packaging/getters, including word alternatives.

CI supplies `CRISPASR_LIB` from the complete newly built library, verifies the
complete header export set/null dimension contracts, and compiles the helper against the real
headers. Stand-in probes alone do not establish native model inference,
TTS→ASR quality, GPU behavior, long-Japanese ASR ownership, or original-Python
intermediate parity. Keep those gates and exact source/run pins in `PLAN.md`.
This inventory covers Python; other language wrappers need their own audit.
