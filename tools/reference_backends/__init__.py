# reference_backends/ — per-model PyTorch hook modules used by
# tools/dump_reference.py. Each backend exposes:
#
#   DEFAULT_STAGES : list[str]
#   def dump(model_dir: Path, audio: np.ndarray, stages: set[str],
#            max_new_tokens: int) -> dict[str, np.ndarray]
#
# See tools/dump_reference.py for the stage-name contract.

# Every capture a dumper takes with tensor.numpy() is an owned copy, and the
# captures whose source tensor changes afterwards are reported (the aliasing
# that corrupted funasr's mel_features reference). See _safe_capture.py.
from . import _safe_capture as _safe_capture  # noqa: E402

_safe_capture.install()
