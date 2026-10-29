#!/usr/bin/env python3
"""Hosted CPU import preflight for the pinned original Melo English API.

This downloads public source and tokenizer files only. It does not construct a
TTS model, fetch model weights, synthesize audio, or run private inputs.
"""

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "tools/kaggle/melo-original-paired/run.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Hosted-only dependency/import preflight")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "melo-original-imports"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema=1, complete=False, acceptance=False,
                   source=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=ROOT, text=True).strip(),
                   adapter_sha256=sha(ADAPTER), python=sys.version.split()[0], phases={})

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def guard(label):
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        row = dict(available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
                   scratch_free_bytes=shutil.disk_usage(scratch).free,
                   load=list(os.getloadavg()), cpus=os.cpu_count() or 1)
        receipt["phases"][label] = row
        save()
        if (row["available_bytes"] < 3 * 1024**3 or row["scratch_free_bytes"] < 4 * 1024**3
                or row["load"][0] > 2 * row["cpus"]):
            raise RuntimeError("Insufficient hosted headroom before " + label)

    def run(command, label):
        guard(label)
        with (out / (label + ".log")).open("w") as log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, timeout=1800)

    save()
    try:
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError("Select python_version=3.13 for the Kaggle compatibility preflight")
        spec = importlib.util.spec_from_file_location("melo_paired_adapter", ADAPTER)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        receipt["pins"] = dict(melo_source=adapter.MELO_SOURCE,
                               bert_revision=adapter.BERT_REV,
                               bert_vocab_sha256=adapter.BERT_VOCAB_SHA,
                               dependency_count=len(adapter.ORIGINAL_IMPORT_DEPS))
        save()

        # The paired GPU adapter uses the worker's Torch. This CPU-only run
        # installs a compatible matched pair to exercise Python 3.13 imports.
        run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:",
             "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch==2.8.0", "torchaudio==2.8.0"], "install-cpu-torch")
        run([sys.executable, "-m", "pip", "install", "-q",
             *adapter.ORIGINAL_IMPORT_DEPS], "install-melo-import-deps")
        guard("after-dependencies")

        import torch
        import torchaudio
        if torch.cuda.is_available():
            raise RuntimeError("CPU import preflight unexpectedly has CUDA")
        receipt["runtime"] = dict(torch=torch.__version__, torchaudio=torchaudio.__version__)
        save()

        from huggingface_hub import snapshot_download
        tokenizer_dir = Path(snapshot_download(
            "google-bert/bert-base-uncased", revision=adapter.BERT_REV,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                            "vocab.txt", "special_tokens_map.json"],
            local_dir=scratch / "english-tokenizer"))
        if sha(tokenizer_dir / "vocab.txt") != adapter.BERT_VOCAB_SHA:
            raise RuntimeError("Pinned English tokenizer changed")
        receipt["tokenizer_files"] = sorted(p.name for p in tokenizer_dir.iterdir()
                                             if p.is_file())
        save()

        source = scratch / "melo-source"
        adapter.clone_pin("https://github.com/myshell-ai/MeloTTS.git",
                          adapter.MELO_SOURCE, source)
        unused_requests = adapter.pin_original_english_bert(tokenizer_dir)
        sys.path.insert(0, str(source))
        blocked_japanese = adapter.block_unused_japanese_cleaner(source)
        guard("before-original-api-import")
        from melo.api import TTS
        from melo.text import english, japanese
        from melo.text.cleaner import clean_text
        if set(unused_requests) != adapter.EXPECTED_EAGER_TOKENIZER_IDS:
            raise RuntimeError("Pinned source eager tokenizer inventory changed")
        if japanese is not blocked_japanese:
            raise RuntimeError("Japanese cleaner guard was bypassed")
        for blocked_call in (japanese.text_normalize, japanese.g2p,
                             japanese.get_bert_feature):
            try:
                blocked_call("must fail")
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Japanese cleaner guard failed open")
        try:
            clean_text("must fail", "JP")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Japanese clean_text route failed open")
        if isinstance(english.tokenizer, adapter.UnusedLanguageTokenizer):
            raise RuntimeError("English tokenizer was replaced by a sentinel")
        if not english.tokenizer.tokenize("A short English import control."):
            raise RuntimeError("Pinned English tokenizer did not tokenize")
        for model_id in adapter.UNUSED_TOKENIZER_IDS:
            try:
                adapter.UnusedLanguageTokenizer(model_id).tokenize("must fail")
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Unused language sentinel failed open")
        parameters = inspect.signature(TTS.__init__).parameters
        if not {"language", "device", "use_hf", "config_path", "ckpt_path"} <= set(parameters):
            raise RuntimeError("Original TTS constructor signature changed")
        parameters = inspect.signature(TTS.tts_to_file).parameters
        if not {"sdp_ratio", "noise_scale", "noise_scale_w", "speed", "quiet"} <= set(parameters):
            raise RuntimeError("Original TTS synthesis signature changed")
        receipt["unused_tokenizer_ids"] = sorted(set(unused_requests))
        receipt["blocked_japanese_cleaner"] = True
        receipt["api_import"] = "melo.api.TTS"
        receipt["complete"] = receipt["acceptance"] = True
        save()
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)[:500]
        save()
        raise


if __name__ == "__main__":
    main()
