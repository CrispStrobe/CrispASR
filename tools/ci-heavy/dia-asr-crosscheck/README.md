# Crossed Dia recognizer replay

This CPU-only diagnostic reuses both actual x86 shared-library/dependency sets
from frozen runs38892690744 and38913941927. It verifies each official artifact
digest and source, then verifies selected library/audio payloads against the
sealed inventory. The original ZIPs stay in hosted scratch; results retain the
libraries, exact PCM, parent receipts, provenance, transcripts, maps and logs.

Each recognizer sees the same six inputs: the old accepted original PCM, the
four predeclared broader original PCMs and repository JFK. Each cell runs in a
fresh process, with its own dependency directory. Actual loaded ggml/crispasr
libraries are checked through /proc/self/maps and hashed. Native SONAME aliases
are reconstructed without modifying the library bytes. The ASR model is the
same immutable Nemotron Q4 file/SHA as both parent runs.

Both recognizers must pass the old WER-zero control and return nonempty JFK
text. The four new WER-zero gates are retained and reported independently;
failed speech remains failed speech even if recognition replay is consistent.
Exact old/new transcript agreement and equality to the sealed ASR observations
are recorded without assuming either in advance. The diagnostic never claims
new speech/model/Q4/GPU acceptance or changes production defaults.

No neural resynthesis, large local models, builds or sampling changes. Heavy
downloads/inference run on GH; local scratch/evidence belong under
/mnt/storage/crispasr/dia-asr-crosscheck-20261016. If the original audio still
fails both working recognizers, the next distinct control is pristine original
generation versus the public dumper under the identical original settings.
