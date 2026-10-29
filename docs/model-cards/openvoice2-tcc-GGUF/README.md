---
license: mit
pipeline_tag: audio-to-audio
base_model: myshell-ai/OpenVoiceV2
inference: false
language:
- en
tags:
- gguf
- voice-conversion
- openvoice-v2
---

# OpenVoice V2 tone-color converter — F32-sensitive GGUF

A converted checkpoint for CrispASR **v0.8.48 or later**. This is a tone-color
converter, not a standalone text-to-speech engine. No new target voice, reference
recording or validation audio is included.

`openvoice2-tcc-f32sens.gguf` is 102,123,872 bytes; SHA-256:
`483fd233035bb2c7cde69519d4d460742b6a70c96c5293e878fe25e72d944255`.
The reference encoder, posterior encoder, flows and one-dimensional tensors
retain F32; other decoder tensors use F16. Weight normalization is fused.
The bundled base-speaker embeddings are the official upstream source tensors;
no new target embedding is added. Full input hashes,
converter source and validation links are in [PROVENANCE.json](PROVENANCE.json).

## Source and license

The original [MyShell OpenVoice V2 model](https://huggingface.co/myshell-ai/OpenVoiceV2/tree/f36e7edfe1684461a8343844af60babc2efbb727)
declares MIT. The unchanged [MyShell MIT notice](LICENSE) is included, from
[the pinned original source](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/LICENSE).
This conversion changes storage and precision; it does not train a new voice.
Melo and BERT are separate models: their licenses remain independent, including
BERT's Apache-2.0 notice. This repository does not bundle them or the original
Python installation's speech/transcription helper dependencies.

The model license does not establish rights in a particular person's identity
or in an operator's reference recording. Supply an authorized reference and
retain CrispASR's required generated-audio markings. Test preset identity stays
unknown; the validation audio and reference captures remain private.

## Validated use and limits

Use the converter with the qualified Melo V2/F16 BERT bundle. The native CLI
finds this exact converter filename beside the Melo model; the Session/C ABI
accepts its path through the codec setter. Both routes require an authorized
reference voice and use the actual source-speaker key. The converter outputs
22.05 kHz PCM; the unconverted Melo baseline is 44.1 kHz.

Original-reference stage/magnitude checks pass on CPU and T4 CUDA. The complete
current application passes both reference cases, exact reseeded repeats,
observed adjacent F16 BERT discovery and all four unchanged Session/CLI speech
and speaker cases. See the pinned CPU/CUDA receipts linked in PROVENANCE.json.
Reference and flow work run on CPU; converter vocoder CUDA work is observed.

Qualification is bounded to the documented synthetic English cases. Broader
languages, natural recordings, other GPU families and automatic VAD selection
need separate validation. These results do not grant voice identity rights.
