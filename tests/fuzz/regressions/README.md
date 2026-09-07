# Fuzz regression seeds

Inputs that once crashed a parser, kept as permanent corpus entries. The
smoke-fuzz job copies this directory into its corpus, so every run replays them
— a fuzzing campaign rediscovers a crash only by luck, and a 45 s seeded run is
stochastic (the stb_vorbis crash below failed CI on a *documentation-only*
commit and passed on the four before it).

Each file is the minimal input that reproduces, crafted rather than harvested
where that was practical, so it is deterministic and small enough to read.

**These files are source, and `tests/.gitignore` excludes `*.wav` / `*.ogg`.**
`ogg-huge-comment-count.ogg` below was written, documented here, and named in
`ci.yml` — and never committed, because `git add` obeyed that pattern without
saying so. The glob matched nothing on every run from then on, and an empty
replay set is indistinguishable from a passing one. `.gitignore` in this
directory now re-includes the audio extensions, and the CI step counts what it
copied and fails at zero.

**The lost ogg seed was re-crafted, and testing it produced a correction to the
row below.** Its documented value, `comment_list_length = 0x3FFFFFFF`, does NOT
reproduce anything against current code: `8 * 0x3FFFFFFF` truncates to −8, so
`setup_malloc` fails and the header is rejected cleanly. Restoring that exact
value would have put a seed in the corpus that gates nothing — the same shape as
the empty replay set it was meant to fix. The value that *does* reproduce is
`1646854400`, which is the one committed. `tools/gen-ogg-comment-fuzz-seed.py`
generates either; pass a count to explore the space.

| File | Bug | Fixed in |
|---|---|---|
| `wav-1hz-resample-oom.wav` | 344 KB, harvested from the run that found it (CI run 33840954769). A structurally valid RIFF/WAVE declaring `sampleRate = 1`. miniaudio resamples it to the 16 kHz target — 16000x — so 176 000 stored samples become 2.816e9 output frames, 11.3 GB, and the chunked decode loops doubled their buffer with no ceiling. Reachable from any surface that accepts a user file, including server upload. | `src/crispasr_audio.cpp`, `crispasr_max_decoded_frames()` bounds decoded frames against input size at all three loops |
| `ogg-comment-count-int-overflow.ogg` | 101 bytes, crafted (`tools/gen-ogg-comment-fuzz-seed.py`). Ogg/Vorbis comment header declaring `comment_list_length = 1646854400`. `setup_malloc` takes an **int**, so `sizeof(char*) * length` = 13,174,835,200 truncates to 289,933,312 and allocates that; the `memset` on the next line computes the same product in `size_t` and writes the full 13 GB. The allocation truncates, the consumer does not. Verified: against the pre-fix vendored decoder it reproduces the original report exactly — same write size, same region size, same `start_decoder:3683` / `setup_malloc:960` / `open_memory:5141` / `decode_memory:5419` frames — and is rejected cleanly after. | `examples/stb_vorbis.c`, bound the count before the multiply + reject a negative `sz` in `setup_malloc` |
| ~~`ogg-huge-comment-count.ogg`~~ **(retired — its value gates nothing, see above)** | 102 bytes. Ogg/Vorbis comment header declaring `comment_list_length = 0x3FFFFFFF`. The allocation of `sizeof(char*) * length` fails, and stb_vorbis returned from the error path with the length still set and `comment_list` NULL — `vorbis_deinit` then indexed the null array. ASAN: `SEGV in vorbis_deinit`, reached from `crispasr_audio_load`. | `examples/stb_vorbis.c`, guard in `vorbis_deinit` + reset the length on the error path |

## Adding one

Craft or minimise the input, drop it here, and add a row. Keep them small: they
run on every CI push. To check a seed still reproduces against an unpatched
build, revert the fix and run the single input directly — `libcrispasr` is a
shared library, so rebuild it, not just the harness, or you will test the new
code with an old-looking binary:

```
./build-fuzz/bin/crispasr-fuzz-audio tests/fuzz/regressions/<file>
```
