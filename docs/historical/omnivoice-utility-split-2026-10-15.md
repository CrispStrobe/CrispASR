# OmniVoice standalone utility split from draft PR521

This split preserves only the utility header, standalone C++ probe and pinned
upstream oracle from original commita1ae00dff82185c3ec8a44cef3054a09d3ab70b1,
as present at draft head381f3a3aeb5fc9f5e87d8b2408fcdf69875c0f47. Authorship
and original source bytes are retained in the first split commit. The second
commit adds full-array evidence retention and source/compiler/actual ELF hashes.

No production file includes this helper, and the split changes no runtime
reference preparation, output processing, params/setter ABI, generation or
default. The full PR521 stays a draft. Its unchanged clone WER gate fails2/9
words in both original and native controls. Upstream issue256 attributes the
reported onset to an abruptly cut reference; fades attenuate a transient but
do not prove a reference or speech repair.

Corrected2026-10-22: the2/9 was the reference clip, not the gate or the port.
Pinned `fox.wav` says "fex/dig" under both the Nemotron RNNT and the Parakeet
CTC judge and every clone arm inherited those two words, while the no-reference
control was correct. The speech gate now vets reference audio against its
transcript before synthesizing and clones cross-sentence on a vetted reference;
seven arms pass the unchanged0.2 gate, maximum0.111. See [reference
integrity](omnivoice-clone-reference-integrity-2026-10-22.json) and the [audio
processing method](../omnivoice-audio-processing.md). Do not reuse `fox.wav` as a
cloning reference.

The oracle executes original functions at upstream
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`, with complete source SHA256 checks.
It compares738 waveform calls and18 multilingual text cases. PCM16 silence
cleanup must be byte-exact. Fades/padding require equal lengths and maximum
absolute error≤2e-7. It retains every original input and expected/native output,
including failures, plus source notices, actual compiled probe and compiler/link
closure. Scale and reversed-sample controls accompany the raw/unprocessed
negative witness. These are utility gates only, never model speech acceptance.

The unchanged original oracle already passes738/18 on Python3.12, NumPy1.26.4
and pydub0.25.1. Full-array proof and an independent replay of all hosted arrays pass, as do the
heavy run and all 10 lint checks. See the
[integration receipt](omnivoice-utility-split-proof-2026-10-15.md). All large outputs and isolated reference dependencies
are under ${CRISPASR_STORAGE}/crispasr/omnivoice-utility-split-20261015.

GH reproduction recipe (the validated source is recorded in the receipt):

```sh
gh workflow run heavy-cpu.yml --ref main \
  -f script=tools/ci-heavy/omnivoice_audio_parity.py \
  -f 'pip=numpy==1.26.4 pydub==0.25.1' -f timeout_minutes=30
```

Local execution uses the same script with `HEAVY_OUT`, `HEAVY_SCRATCH` and
`TMPDIR` set under ${CRISPASR_STORAGE}, plus isolated pinned reference dependencies.
The main clone diagnosis is in
[the original control](omnivoice-original-clone-2026-10-09.json) and
[native failure receipt](omnivoice-clone-failure-2026-10-09.json).
