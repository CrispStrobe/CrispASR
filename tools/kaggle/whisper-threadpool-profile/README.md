# Whisper CPU-threadpool overhead diagnostic

Prepared, unsubmitted T4 job. It builds the pinned validation source and compares
profiling off and on for the same CrispASR binary, model and JFK input, alongside
the previously measured packaged MIT fork. It checks the CUDA backend and model
placement, GPU PID, WER, matching transcript hashes and 27 decoder calls. The
profiled arm additionally requires the explicit threadpool timing field.

`threadpool_us` is host wall time spent in the existing persistent CPU-threadpool
setup during single-token decoding. It is included in, not additive to, the
existing decoder compute time. The off/on controls expose instrumentation
perturbation. This is one clip, and the packaged fork has different build and
CUDA libraries; it cannot establish causality or a general performance match.
First collect the currently running same-toolchain and VMM comparisons, then
schedule this diagnostic only if the threadpool hypothesis remains relevant.
No changes to threadpool ownership or scheduling are made by this diagnostic.
