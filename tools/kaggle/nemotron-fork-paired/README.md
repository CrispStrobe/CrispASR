# Private Nemotron F16 fork comparison

Prepared only. Do not submit until the maintainer coordinates the T4 slot and
reviews the pins. This v3 kernel uses the published CUDA native library and its
matching pure Python API wheel, with no build. The v0.4.1 native tarball contains
shared libraries, contract metadata, and licenses; it does not contain
`transcribe-cli`. The Python API invokes the same offline `transcribe_run` C path
as the CLI with batch size 1 and exposes the same per-utterance mel, encode, and
decode timers. It writes a private `receipt.json` plus subprocess logs under
`/kaggle/working/nemotron-fork-paired-v3`; the failed v1 and v2 runs remain
diagnostic evidence. The metadata id matches the title-derived Kaggle slug used
by v1, so the successor is a new version of that existing private kernel.

The v0.8.47 CLI strips the audio extension when `-otxt` writes a transcript:
`audio.wav` produces `audio.txt`. The v2 adapter looked for `audio.wav.txt` and
failed after all 27 Crisp R3 cases had completed. V3 removes each expected
`audio.txt` before a run, then requires one nonempty output file and one stdout
line per input. It matches each file's transcript against its corresponding
stdout line, preserving the CLI's sequential input order as a quality gate.

Three offline arms use exactly the same 16 kHz PCM16 WAV files: CrispASR v0.8.47
production default R3, CrispASR v0.8.47 with `CRISPASR_NEMOTRON_CONTEXT_PRESET=3`
(R13), and transcribe.cpp v0.4.1 default R13. The matched comparison is the R13
pair. R3 is a separate Crisp production control. transcribe.cpp's
`--stream-att-right` takes the cache-aware streaming path, so an R3 run through
that option would change processing semantics and is omitted.

The two GGUFs are distinct schemas and byte sequences from the NVIDIA Nemotron
3.5 ASR Streaming 0.6B model family. Both are F16. Crisp's original F32 oracle
uses upstream revision `ea30d66d` and a safetensors artifact; transcribe.cpp's
model card cites upstream revision `24b151a8` and a `.nemo` artifact. Their
upstream artifact bytes have not been proved identical. The upstream
weights are governed by NVIDIA OpenMDW-1.1, including its attribution and use
terms; the transcribe.cpp runtime is MIT. The private run neither publishes nor
redistributes model weights. Preserve the [OpenMDW-1.1 license](../../../docs/nemotron-model-notices/LICENSE-OpenMDW-1.1.txt)
and [model notice](../../../docs/nemotron-model-notices/NOTICE.txt) with any
downstream artifact or report that needs them. Source revisions, model SHA256s,
archive SHA256s, and original-model oracle provenance are in the worker.

Quality is measured as normalized word error rate against both the 26 saved
NVIDIA F32 ASR transcripts and the TTS prompts, plus JFK's reference text.
The 26 original float32 PCM arrays are deterministically converted to a single
set of PCM16 WAVs used by all three arms. This conversion can affect individual
words; the original-model comparison is therefore a contextual oracle, not an
exact parity gate. All transcripts, per-case WERs, input hashes, GPU process
evidence, and stage timings are retained. Acceptance requires complete outputs,
matching CUDA and context logs, positive process-specific GPU SM activity, and
reported WER for every arm. Timing stage boundaries differ between runtimes;
the receipt reports each native stage separately and does not equate their
encoder/decode sub-stages by name alone.

The worker requires Crisp's per-file offline attention-mask log to show L56/R3
or L56/R13 as appropriate. It reads transcribe.cpp's pinned GGUF header and
requires `chunked_limited` with L56/R13; the pinned v0.4.1 offline source uses
those fields as its mask and the Python API omits streaming options. A missing or
different field fails before inference.

The script fails closed on a non-T4 allocation or unavailable pinned asset.
Do not retry an allocation for a preferred GPU. Keep a failed receipt and logs
as diagnostic evidence before preparing any corrected successor.
