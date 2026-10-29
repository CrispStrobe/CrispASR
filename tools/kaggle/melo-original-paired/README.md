# Melo full original versus native CUDA pilot

This **prepared, unsubmitted** private Kaggle pilot runs the pinned original
`melo.api.TTS` from `myshell-ai/MeloTTS@209145371cff8fc3bd60d7be902ea69cbdb7965a`
against the accepted current-source native F16 Melo model and F16 BERT
companion. Both use the same two English texts, `EN-US`/speaker 0, two threads,
seed 42, `sdp_ratio=0`, `noise_scale=0.667`, `noise_scale_w=0.8`, and
`speed=1.0`. The original API defaults differ and are overridden explicitly.

The original checkpoint is `myshell-ai/MeloTTS-English-v2` at
`a53e3509c4ee4ff16d79272feb2474ff864e18f3` (checkpoint SHA-256
`794226eb7c1745f3ca281b290613d5f39aa5b0d3b16a117009966f4aaf184757`).
The original BERT/tokenizer snapshot is
`google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`;
its safetensors SHA-256 is
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.
The original's floating `bert-base-uncased` calls are redirected to that
local immutable snapshot; its `vocab.txt` is checked against SHA-256
`07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`.
The native model and BERT are pinned to
`cstr/melotts-en-v2-GGUF@35522347b1bed9a913d6894803c5cab929ae42a5`
and checked against the accepted SHA-256 values. Before inference, the worker
compares every inference GGUF tensor against the original checkpoint after
the repository's weight-norm fusion and F16/F32 cast. A mismatch stops the
run; the model card alone is insufficient lineage proof.

The pinned original API eagerly imports cleaners for Japanese, Korean,
French, Spanish and mixed Chinese even when `language="EN_V2"`. Japanese also
imports MeCab and initializes its tagger during import, before its tokenizer
is constructed. The adapter installs a fail-on-use Japanese cleaner module
before importing the original API; the other four cleaners construct foreign
tokenizers, which the adapter replaces with fail-on-use sentinels. The five
foreign tokenizer IDs remain blocked if requested, so no foreign model assets
are fetched. Any attempt to use one of those languages raises. The
English tokenizer and masked-language model are redirected to the pinned
English BERT snapshot; unexpected model IDs raise. This is a wrapper-only
import substitution. The original English TTS implementation and sampling
settings are otherwise used directly. A hosted CPU preflight imports the same
pinned source and exercises the English tokenizer, Japanese cleaner guard,
and tokenizer sentinel failures,
without loading model weights or synthesizing speech.

License notices: pinned [MeloTTS source MIT license](https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/LICENSE),
[MeloTTS-English-v2 model card (MIT)](https://huggingface.co/myshell-ai/MeloTTS-English-v2/blob/a53e3509c4ee4ff16d79272feb2474ff864e18f3/README.md),
and [Google BERT model card (Apache-2.0)](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/README.md).

One warmup and three measured whole synthesis calls per arm and phrase are
planned. Original wrapper time includes text cleaning, BERT, model inference,
50 ms trailing silence, and `torch.cuda.empty_cache()`; a synchronized
original `model.infer` duration is recorded separately. Native C ABI time
includes G2P, BERT, encoder, duration predictor, flow and vocoder and output
copy; native built-in stage counters are retained separately. The original
wrapper and native C ABI have different postprocessing, so no model-only
speedup is inferred from their whole-call medians. Torch and ggml RNGs differ
despite the matched seed. Original weights/BERT run in F32; native GGUF/BERT
are F16 with sensitive tensors retained F32.

The unchanged accepted independent content gate is tiny.en WER ≤0.2 for
**every** warmup/measured output, plus 44.1 kHz mono finite PCM and observed
CUDA execution for both arms. If an arm fails, the receipt remains incomplete;
no threshold, oracle or source text changes. The worker keeps all generated
PCM, model weights, G2P/BERT features, transcripts and verbose stage logs in
private scratch only. Its Kaggle output contains aggregate timing and quality
numbers, hashes, pinned provenance and failure status. Do not publish raw
Kaggle logs without privacy review.

The adapter has only model-free static checks so far. It has not been submitted
or proven on a hosted worker. Run
`tools/ci-heavy/melo_original_imports.py` on the Heavy CPU workflow with
`python_version=3.13` before GPU admission. Review dependencies against the
current Kaggle image, refresh metadata-aware GPU slots and privacy guards,
and preserve all terminal source/log/output before any retry.
