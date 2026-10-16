# Echo full-model observer controls

This is a CPU-only hosted preparation worker, based on capture sourced8183cf9
and GGML0a862076. No runtime graph, model, quantization or default changes.

It executes the actual public reference dumper at original HF5d98a34d using
released transformers5.6.0 and externally pinned modeling/class hashes. The
native primary/decoder are the published 2B F16 files atb3453826. Actual weight
SHA256 values are preregistered in the worker and checked against downloaded
bytes and immutable HF metadata. All files receive Git-blob or LFS verification;
every physical GGUF tensor type/shape/size is retained. No models run locally.

Predeclared cases are complete repository JFK and Chinese audio, both targeting
English. Settings are four threads, greedy decoding, capacity2000 and16 cached
logit rows. Original tower/connector/decoder are explicitly F32. Native uses
published F16 weights, CPU and its existing default flash-attention setting.
Original/native numerical differences are distinct from observer transparency.

For each case, original Python runs OFF/ON/OFF through tools/dump_reference.py.
The actual native backend runs OFF/ON/OFF in one loaded context, with complete
owned stage arrays and actual direct-window generation. It verifies generation
prefill stage bytes against its preceding diagnostic prefill. The enabled run
also opens an arbitrarily named primary through the actual automatic Session
C ABI and checks complete decoded cues. CLI output runs OFF/ON/OFF separately.

Within each implementation every existing array must retain exact shape, dtype
and bytes. Original requires69 existing arrays; native requires68. Both require
all144 additional FFN vectors (six boundaries at each of24 decoder layers),
with pinned hidden/intermediate widths2048/6144. Missing/extra stages, scaling,
signed-zero changes, hook leaks, empty/nonfinite data and decoded differences
are failures. Raw source/native text must also match. Public crispasr-diff runs
against both ordinary and extended original references with its unchanged
cosine/magnitude/prompt/cache-ID gates. Preserve all outcomes, including failed
source controls. Hosted success requires a separate official-artifact audit.

Small selftests calibrate the gates and compile the actual native header to
verify all ctypes struct sizes/offsets. They do not establish model acceptance.
The prior actual-class/ggml tiny proof remains separate. Returned artifacts
retain full references, all native arrays, audio, decoded/CLI outputs, sources,
physical/provenance ledgers, native executables/libraries and licenses. Large
weights/builds stay in hosted scratch; local evidence belongs under
/mnt/storage/crispasr/echo-ffn-fullmodel-20261016.

This control is scoped to2B observer transparency. It does not perform
same-original-input projection attribution, validate9B/Q4/GPU, change a shipped
recipe, or claim a performance improvement. Those remain later gates.
