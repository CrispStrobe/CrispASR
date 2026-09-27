"""wav2vec2 / hubert / data2vec reference backend - the transformers *ForCTC
dumper in hf_ctc.py (ctc_logits vs wav2vec2_compute_logits)."""
from .hf_ctc import DEFAULT_STAGES, dump  # noqa: F401
