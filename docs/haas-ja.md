# Haas-ja-v1.0 Japanese transcription

Haas-ja-v1.0 is a Japanese speech recognition model from
[third-intelligence](https://huggingface.co/third-intelligence/Haas-ja-v1.0).
CrispASR runs its FastConformer encoder, Kimi Delta Attention predictor and
TDT decoder through the Parakeet implementation. Haas support shipped in
v0.8.50; use a current release for the latest fixes.

## Transcribe audio

```sh
crispasr --backend haas -m auto -f japanese.wav
```

On first use, `-m auto` resolves the registered default and downloads it to the
CrispASR model cache. `haas-ja` and `haas-asr` are also accepted as explicit
backend aliases. To use a model you have already downloaded, pass its path:

```sh
crispasr --backend haas -m /path/to/haas-ja-v1.0-q4_k.gguf -f japanese.wav
```

The [CrispASR GGUF repository](https://huggingface.co/cstr/Haas-ja-v1.0-GGUF)
provides F16 (~1.18 GB), Q8_0 (~626 MB) and Q4_K (~368 MB) files. The model
registry lists Q8_0 first for automatic selection. Sizes are approximate;
choose a specific downloaded file when a particular precision is required.
The model is Apache-2.0; see its upstream model card for its intended use.

For applications using the session C ABI, `haas`, `haas-ja` and `haas-asr`
are supported explicit aliases. Automatic backend detection uses GGUF
architecture metadata; the alias entrypoint and registry are covered by
`tests/test_haas_dispatch.py` and `tests/test-registry.cpp`. These are dispatch
tests, not a general accuracy benchmark for every audio domain or quantization.

## Accuracy

The original model's published scores reproduce, and CrispASR matches them.
`tools/kaggle/haas-ja-bench-verify` rebuilt two subsets of
[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench)
with its own `rebuild.py` and scored them with upstream's evaluation script
(jacer v1.2). CER strict / flex / kana, lower is better:

| | FLEURS (650 utterances) | JSUT basic5000 (5,000) |
|---|---|---|
| Model card | 3.67 / 2.98 / 2.34 % | 5.32 / 2.43 / 1.36 % |
| Original model, NeMo bf16 (T4, batch 16) | 3.67 / 2.98 / 2.35 % | 5.33 / 2.44 / 1.37 % |
| CrispASR, Q8_0 (M1 Metal) | 3.69 / 3.02 / 2.36 % | not measured |

On FLEURS, 605 of 650 CrispASR transcripts are byte-identical to the original
model's; the rest differ by Q8_0 rounding. Before the byte-fallback fix of
2026-11-21 (#605), CrispASR dropped kanji outside the model's 3,072-piece
vocabulary (骨董品 came out as 骨品) and scored 3.77 %.

These are clean, human-transcribed recordings. Noisy audio is a different
matter: on a 3-minute game stream with music and sound effects (#605), even two
whisper models agree with each other on only 72 % of the content, so no score
against them can show 95 % there.

## Long audio

Haas-ja follows its own upstream long-form recipe
(`haas_asr.transcribe_longform` in
[Haas-ASR](https://github.com/third-intelligence/Haas-ASR)), not the one
CrispASR uses for parakeet-ja. Audio longer than 30 s is cut into 30 s chunks,
each cut moved to the quietest 0.1 s within 2 s of its target, and every chunk
is decoded in one pass. No VAD is applied unless you pass `--vad`; then VAD
slices are capped at 30 s, the longest utterance the model was trained on.

Each chunk is decoded as a whole, but its transcript is split into output
segments (subtitle cues) at pauses of 0.6 s or more between words, and before
a segment would run past 12 s. On the #605 recording that gives 27 cues with
a median of 1.4 s and none longer than 9.7 s.

parakeet-ja's workarounds (VAD forced on, 12 s slices, the gap-fill second
pass) were built for an encoder defect Haas-ja does not have (#89). On a noisy
3-minute game stream (#605) they cost about 5 points of content recall, because
VAD discarded speech the model decodes, and gap-fill turned pauses and sound
effects into short, wrong subtitles. With the upstream recipe, CrispASR's
output matches the original NeMo model run on the same chunks, apart from
single characters.

| Setting | Effect |
|---|---|
| `CRISPASR_HAAS_LONGFORM=0` | Use the parakeet-ja path again (VAD, 12 s slices), for comparison. |
| `--chunk-seconds N` | Chunk length instead of 30 s. |
| `CRISPASR_GAP_FILL=1` | Run the gap-fill pass anyway; it keeps only recoveries above `CRISPASR_GAP_FILL_MIN_CONF` (default 0.85). |
