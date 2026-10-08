"""Independent Python execution of deployed five-graph Moonshine exports.

Graph contract: moonshine-ai/moonshine core/moonshine-streaming-model.cpp
at 234f60faa0eb388b01cdf7e60aca232af37aefda, plus the pinned export's
streaming_config.json. Whole-utterance BOS greedy decoding captures frontend
state, encoder, absolute-position adapter, cross-KV and first decoder outputs.
This checks native wrapper execution against Python ORT using identical graphs;
it does not certify the exporter against original PyTorch checkpoints.
"""
from pathlib import Path
import json
import numpy as np

DEFAULT_STAGES = ["raw_audio", "graph_outputs"]


def dump(*, model_dir: Path, audio: np.ndarray, stages: set,
         max_new_tokens: int) -> dict:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    primary = Path(model_dir)
    root = primary.parent if primary.suffix == ".onnx" else primary
    config = json.loads((root / "streaming_config.json").read_text())
    quant = primary.name != "encoder.onnx" and (root / "encoder_int8.onnx").exists()
    captures = {"raw_audio": np.array(audio, dtype=np.float32, copy=True)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    graphs = {}
    seen = set()
    for name in ["frontend", "encoder", "adapter", "cross_kv", "decoder_kv"]:
        suffix = "_int8" if quant and name != "frontend" else ""
        graphs[name] = ort.InferenceSession(str(root / (name + suffix + ".onnx")),
                                           options, providers=["CPUExecutionProvider"])

    def run(name, values):
        graph = graphs[name]
        inputs = graph.get_inputs()
        if len(inputs) != len(values):
            raise ValueError(f"{name}: unexpected input count")
        result = graph.run(None, dict(zip((x.name for x in inputs), values)))
        if name not in seen:
            for output, value in zip(graph.get_outputs(), result):
                captures[name + "." + output.name] = np.array(value, dtype=np.float32, copy=True)
            seen.add(name)
        return result

    if not len(audio):
        raise ValueError("Stage parity requires nonempty audio")
    padded = np.pad(np.asarray(audio, dtype=np.float32), (0, (-len(audio)) % 640))[None, :]
    state = [np.zeros(config["frontend_state_shapes"][name],
                      dtype=np.int64 if name in ("sample_len", "frame_count") else np.float32)
             for name in ["sample_buffer", "sample_len", "conv1_buffer", "conv2_buffer", "frame_count"]]
    features = run("frontend", [padded, *state])[0]
    if features.shape[1] > config.get("max_position_embeddings", 4096):
        raise ValueError("Audio exceeds model position capacity")
    encoded = run("encoder", [features])[0]
    memory = run("adapter", [encoded, np.array([0], dtype=np.int64)])[0]
    cross = run("cross_kv", [memory])
    shape = (config["depth"], 1, config["nheads"], 0, config["head_dim"])
    key, value = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    token = np.array([[config["bos_id"]]], dtype=np.int64)
    ids = []
    # Match the deployed utterance budget; max_new_tokens is a generic dumper
    # argument, not the model's decoding policy.
    budget = min(1024, max(4, int(len(audio) / 16000 * 6.5) + 2))
    for _ in range(budget):
        decoded = run("decoder_kv", [token, key, value, *cross])
        logits, key, value = decoded[:3]
        cross = decoded[3:5]
        best = int(np.argmax(logits[0, -1]))
        if best == config["eos_id"]:
            break
        ids.append(best)
        token = np.array([[best]], dtype=np.int64)
        if len(ids) >= 16 and ids[-8:] == ids[-16:-8]:
            break
    captures["generated_text"] = Tokenizer.from_file(str(root / "tokenizer.json")).decode(ids).strip()
    return captures
