# Direct original Dia control

`dia_direct_original.py` runs only on GH CPU. It uses the actual pinned original
`Dia(config, compute_dtype='float32', device=cpu)` constructor, copies the pinned
safetensors state through normal strict `load_state_dict`, attaches the real
pinned DAC and calls untouched `generate()` without observation hooks. Required original DAC WeightNorm pre-hooks are
retained unchanged; an actual tiny weight-normalized convolution checks both
preservation and rejection of an added observer. There is
no meta allocation, `__new__`, parameter assignment or rotary-buffer replacement.

The preregistered legacy seed42 and hello/fox seeds42/123 retain temperature1.2,
top_p0.95, top_k45, CFG3 and cap3072. Raw PCM is compared bytewise against the
original captures sealed in crossed-ASR run38930760374/artifact12060029304. That
artifact has its fixed official SHA checked, retains full parent PCM and ELF
sets, and records passing legacy/JFK but failed expanded recognitions. The same
pinned Nemotron Q4 ASR with old sealed ELF set recognizes each new direct PCM.
Every result remains in the artifact even if the final identity/legacy gate
fails. A green diagnostic can still report four failed speech cases. This does
not promote runtime defaults, speech quality, Q4 or GPU support.

Local `--selftest` exercises comparison guards. `selftest.py`, with pinned
`DIA_SOURCE` and Torch2.6 CPU available, executes real tiny original neural
constructor/generation/capture on all five cases and rejects an injected hook;
its deterministic DAC stub explicitly establishes no codec or speech acceptance.

GH dependencies: torch==2.6.0 torchaudio==2.6.0 numpy==1.26.4
huggingface_hub==1.5.0 safetensors pydantic descript-audio-codec==1.0.0
scipy==1.15.3. No local model download/build is needed. Output is diagnostic,
not a replacement for pinned original→GGUF stage, magnitude and decoded gates.

The hosted checkout is depth one. The worker explicitly fetches the immutable
crossed-ASR source before its exact helper-byte check; no local parent-object
availability is assumed. The source check remains mandatory. Run38936657203
failed at this prerequisite before downloads or synthesis and is retained.
