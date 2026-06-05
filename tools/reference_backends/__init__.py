# reference_backends/ — per-model PyTorch hook modules used by
# tools/dump_reference.py. Each backend exposes:
#
#   DEFAULT_STAGES : list[str]
#   def dump(model_dir: Path, audio: np.ndarray, stages: set[str],
#            max_new_tokens: int) -> dict[str, np.ndarray]
#
# See tools/dump_reference.py for the stage-name contract.
