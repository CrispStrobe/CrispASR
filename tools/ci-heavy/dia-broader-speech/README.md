# Broader Dia CPU speech controls

The runtime remains byte-identical to candidate5e04955e. This worker changes
no compute graph, production default, model pin or quantization recipe.

The predeclared cases are the two texts below, each at seeds42 and123:

- `[S1] Hello there, how are you doing today?`
- `[S1] The quick brown fox jumps over the lazy dog.`

The recipe stays temperature1.2, top_p0.95, top_k45, CFG3 and capacity3072.
Native generation uses the generic serial head, F32 cross-cache and portable
sampling. Original Python uses pinned4a9e29b1, original checkpoint257bc72f,
PyTorch2.6 CPU, MKL_COMPATIBLE and the existing public dumper. Original code
executes its own generate/codec path. No original input or decision is injected
into native generation.

Each case retains original reference, complete native inputs/logits/PCM,
original-input prefix/exactness, actual CLI/Session equality and code-to-PCM
replay. Actual original DAC is also run on the native arm's OWN code grid;
its eight stages and PCM are compared with the native decode of that same
input. Original generated-code equality is recorded separately, not assumed.
Initial neural inputs are aligned before comparing encoder, all decoder layers,
selected0/17 attention/block boundaries and logits. These are88 complete
aligned comparisons per case with cosine>=0.9995 and row norm error<0.02.

Actual ASR runs on the original positive and native/original-DAC/CLI outputs.
All four cases must meet WER0 and stage gates, with every outcome saved before
the final quality assertion. A failed original positive remains a failed
control. No seed/phrase selection, threshold relaxation, GPU/Q4 or performance
claim is permitted. Independent full-artifact audit must follow hosted success.

The small selftest executes the actual shared child body with a fake Session
to verify all four text/seed forwards, and calibrates scoring, row/scale/invalid
array guards and nested-receipt inventory. It proves harness behavior only.
The shared support helper now accepts explicit text/seed with legacy defaults,
and indexes nested receipt.json files correctly; only the root receipt itself
is excluded from its inventory. This fixes the earlier producer's metadata
omission without changing the frozen full-model evidence.

Builds/models run on GH CPU. Local scratch and returned evidence belong under
/mnt/storage/crispasr/dia-broader-speech-20261016. Launch only after independent
acceptance of the preceding factorial artifact12049198423. Further prompts,
GPU/Q4 and profiling remain separate acceptance steps.
