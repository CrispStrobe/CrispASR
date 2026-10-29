# Whisper loader device classification A/B (prepared, not submitted)

`SOURCES` pins baseline `9df3071a933b2c85ceb3e278596c219c5d00c32e` and candidate
`0a3a97c1e1c43ba3a1a56f1923ef6a74fb362f89`; both must be published before
submission. The baseline contains the shared opt-in placement tracer; the
candidate adds the loader-local device classification cache. The adapter
rejects changed paths beyond `src/crispasr.cpp`, mismatched submodule revisions,
or differing normalized compiler commands. Normalization replaces only absolute
source/build workspace prefixes and the CLI banner definitions
`CRISPASR_GIT_SHA1`, `CRISPASR_GIT_DATE`, `CRISPASR_GIT_SUBJECT` and
`CRISPASR_BUILD_DATE`; all other compiler arguments remain compared. Those four
definitions come from `examples/cli/CMakeLists.txt` and describe source identity
or configure time. Review the retained source patch to
confirm the intended optimization is the only behavioral change.

Both source trees build sequentially on the same Tesla T4 with the same resolved
C/C++ compilers, CUDA toolkit, SM75 target and VMM disabled. It uses the common
Kaggle build helpers, caps CUDA builds at two jobs, checks memory/load/disk and
rejects unrelated GPU compute processes before every build and inference. All
external commands have a 5,400-second timeout. The script exits on another GPU;
do not automatically redraw. Only `run.py` is uploaded for a script kernel; the
common harness is imported from the pinned baseline clone. Network failure is
terminal, with no fictional bundled fallback.

Separate CPU and CUDA placement controls enable
`CRISPASR_WHISPER_PLACEMENT_TRACE=1`. Each successful placement must emit one
ordered line with tensor name, operation, type, four-dimensional shape, device
name and buffer type. The adapter rejects empty/malformed/duplicate-name traces
and compares the complete ordered traces and transcript bytes between commits.
Successful loading also invokes the loader's existing complete-weight count
check. This proves equivalence for this pinned model under these two device
configurations; it does not cover every dtype, model, CPU extra buffer, device
preference or accelerator. Aggregate `CUDA0 total size` and GPU PID evidence alone
would not prove tensor placement equivalence. The trace describes selected weight
placement, not subsequent scheduler/activation placement. CPU and CUDA transcripts
are compared across commits within their device mode, not forced equal between
device modes.

Timing disables placement tracing. One warmup and four measured runs per arm
and profiler mode alternate baseline/candidate and profiler off/on order. Every
timed run must have zero JFK WER, identical transcript bytes, exactly 27 reported
decoder calls, CUDA weight placement/execution, sampled GPU PID and VMM-off
runtime evidence. Model/audio bytes, CLI flags, thread counts and device index
are fixed. `CRISPASR_WHISPER_TIMING_PROFILE=1` uses the existing instrumentation;
the adapter accepts additional unique counters, including threadpool time, while
requiring the existing load/allocation/read/upload/preallocation/vocabulary/list/
selection and decoder counters. No native math is edited by this adapter.

`receipt.json` records separate medians for each profiler mode. Primary speed
comparisons use profiler-off load and process medians; profiler-on selection and
allocation medians diagnose the mechanism. A profiler-on improvement alone is
not a profiler-off speed result. The fresh-process wall boundary includes
startup, loading, inference, exit and concurrent GPU PID monitoring. These are
warm-page-cache model loads, not cold-storage tests. The existing loader timer
excludes backend setup before loader entry. Preallocation contains nested
selection/vocabulary/list work; do not sum overlapping timers. Decode milliseconds
per call are the reported decode timer divided by calls, not CUDA kernel time.
One public base.en clip cannot establish general Whisper quality or throughput.

All command/build/placement/inference logs, transcript files, source loader
copies, uploaded script, commit/tree/submodule/toolchain hashes, compile commands,
CMake caches and source diff are retained under `/kaggle/working`. Receipt writes
are atomic and any exception retains `acceptance:false`, partial runs and a
traceback. Model/build scratch remains under `/kaggle/temp`. Before a new version,
retain the previous terminal logs, output and pulled uploaded source privately,
verify its source digest, increment `VERSION`, and obtain fresh GPU-slot/privacy
guard admission. This preparation performs no build, inference, push or submission.
