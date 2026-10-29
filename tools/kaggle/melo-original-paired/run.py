#!/usr/bin/env python3
"""Private hosted full-original Melo versus native F16 CUDA timing pilot."""

import ast
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from types import ModuleType

VERSION = "melo-original-paired-v1"
OUT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/melo-original-paired")
SOURCE = "97053abff8201870827027692dc25dc0c412e2a3"
APPLICATION_SHA = "ae93882f0a7854cfb59f519ebbf406d3c0854828d9f451c0ef308e7bc5269e71"
LIBRARY_SHA = "fffd43cd2536fd8d3876fdc7d9d3e4d8e0c730bd0b6f532e37a2123c2a560d57"
MELO_SOURCE = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
ORIGINAL_REV = "a53e3509c4ee4ff16d79272feb2474ff864e18f3"
ORIGINAL_CKPT_SHA = "794226eb7c1745f3ca281b290613d5f39aa5b0d3b16a117009966f4aaf184757"
BERT_REV = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_SHA = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
BERT_VOCAB_SHA = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
NATIVE_REPO = "cstr/melotts-en-v2-GGUF"
NATIVE_REV = "35522347b1bed9a913d6894803c5cab929ae42a5"
NATIVE_MODEL_SHA = "68c82ea2a18fd8e9d01c37a286a17fcfa630dfe7dbbd0bd08e7454e313e78fd8"
NATIVE_BERT_SHA = "f7437929246c43d7300cbf6ff39b4715669a6053661117df99a456bdc5e8ecba"
ASR_REPO = "Systran/faster-whisper-tiny.en"
ASR_REV = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
TEXTS = ("The quick brown fox jumps over the lazy dog.",
         "We enjoy reading books and listening to music in the evening.")
# The original API imports foreign cleaners even for EN_V2. Keep the import
# dependencies in one list shared with the hosted CPU preflight.
ORIGINAL_IMPORT_DEPS = (
    "transformers==4.49.0", "huggingface_hub==0.28.1", "cached-path==1.7.3",
    "g2p-en==2.1.0", "eng-to-ipa==0.0.2", "unidecode==1.4.0",
    "inflect==7.5.0", "nltk==3.9.1", "librosa==0.11.0",
    "soundfile==0.13.1", "num2words==0.5.12", "pykakasi==2.2.1",
    "anyascii==0.3.2", "jamo==0.4.1", "gruut==2.2.3",
    "pypinyin==0.50.0", "cn2an==0.5.22", "jieba==0.42.1",
    "tqdm==4.67.1",
)
UNUSED_TOKENIZER_IDS = frozenset({
    "tohoku-nlp/bert-base-japanese-v3", "kykim/bert-kor-base",
    "dbmdz/bert-base-french-europeana-cased",
    "dccuchile/bert-base-spanish-wwm-uncased",
    "bert-base-multilingual-uncased",
})
JAPANESE_TOKENIZER_ID = "tohoku-nlp/bert-base-japanese-v3"
EXPECTED_EAGER_TOKENIZER_IDS = UNUSED_TOKENIZER_IDS - {JAPANESE_TOKENIZER_ID}


def ensure_english_nltk_resources(nltk_dir):
    import nltk

    nltk_dir = Path(nltk_dir)
    nltk_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nltk_dir)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    for resource_name in ("averaged_perceptron_tagger_eng", "cmudict"):
        if not nltk.download(resource_name, download_dir=str(nltk_dir), quiet=True):
            raise RuntimeError("Failed to fetch NLTK resource: " + resource_name)


def block_unused_japanese_cleaner(source):
    """Block the unused Japanese cleaner before it imports MeCab or models.

    Melo's cleaner imports every language module. Japanese initializes MeCab
    during import, even for English synthesis. This wrapper only replaces the
    Japanese module and fails if a caller tries to use its public entry points.
    """
    module_name = "melo.text.japanese"
    if module_name in sys.modules:
        raise RuntimeError("Melo Japanese cleaner was imported before its guard")
    japanese_path = Path(source) / "melo/text/japanese.py"
    if not japanese_path.is_file():
        raise RuntimeError("Pinned Melo Japanese cleaner is missing")
    selected = [node for node in ast.parse(japanese_path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "distribute_phone"]
    if len(selected) != 1:
        raise RuntimeError("Pinned Melo distribute_phone helper changed")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(japanese_path), "exec"), namespace)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Japanese Melo cleaner is unavailable in this EN_V2 run")

    blocked = ModuleType(module_name)
    blocked.__package__ = "melo.text"
    blocked.distribute_phone = namespace["distribute_phone"]
    blocked.text_normalize = unavailable
    blocked.g2p = unavailable
    blocked.get_bert_feature = unavailable
    sys.modules[module_name] = blocked
    return blocked


class UnusedLanguageTokenizer:
    """Fail closed if Melo's eagerly imported non-English cleaner is used."""

    def __init__(self, model_id):
        self.model_id = model_id

    def __getattr__(self, name):
        raise RuntimeError("Unused Melo language tokenizer was accessed: " + self.model_id)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Unused Melo language tokenizer was called: " + self.model_id)


def pin_original_english_bert(bert_snapshot):
    """Pin English assets while avoiding unused eager foreign tokenizers."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    original_tokenizer = AutoTokenizer.from_pretrained
    original_model = AutoModelForMaskedLM.from_pretrained
    unused_requests = []

    def pinned_tokenizer(cls, model_id, *args, **kwargs):
        if model_id in UNUSED_TOKENIZER_IDS:
            unused_requests.append(model_id)
            return UnusedLanguageTokenizer(model_id)
        if model_id != "bert-base-uncased":
            raise RuntimeError("Original requested unexpected tokenizer: " + str(model_id))
        return original_tokenizer(str(bert_snapshot), *args, local_files_only=True, **kwargs)

    def pinned_model(cls, model_id, *args, **kwargs):
        if model_id != "bert-base-uncased":
            raise RuntimeError("Original requested unexpected BERT weights")
        return original_model(str(bert_snapshot), *args, local_files_only=True, **kwargs)

    AutoTokenizer.from_pretrained = classmethod(pinned_tokenizer)
    AutoModelForMaskedLM.from_pretrained = classmethod(pinned_model)
    return unused_requests


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_exact(name, expected):
    matches = [path for path in Path("/kaggle/input").rglob(name)
               if path.is_file() and sha(path) == expected]
    if len(matches) != 1:
        raise RuntimeError("Expected one pinned mounted " + name)
    return matches[0]


def headroom(label, *, first=False):
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,compute_cap",
         "--format=csv,noheader,nounits"], text=True, timeout=30).strip().split(",")
    row = {"phase": label, "available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
           "scratch_free_bytes": shutil.disk_usage(WORK).free,
           "gpu_free_mib": int(gpu[0].strip()), "compute_cap": gpu[1].strip(),
           "load": list(os.getloadavg()), "cpus": os.cpu_count() or 1}
    if (row["available_bytes"] < 3 * 1024**3 or row["scratch_free_bytes"] < 4 * 1024**3
            or row["gpu_free_mib"] < (8 * 1024 if first else 1024)
            or row["load"][0] > 2 * row["cpus"]):
        raise RuntimeError("Insufficient hosted headroom before " + label)
    return row


def clone_pin(url, revision, target):
    subprocess.run(["git", "init", str(target)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", "origin", revision],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(target), "checkout", "FETCH_HEAD"],
                   check=True, stdout=subprocess.DEVNULL)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError("Original source pin changed")


def prepare_application(archive):
    root = WORK / "application"
    root.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(Path(m.name).is_absolute() or ".." in Path(m.name).parts or
               not (m.isfile() or m.isdir() or m.issym()) or
               (m.issym() and (Path(m.linkname).is_absolute() or ".." in Path(m.linkname).parts))
               for m in members):
            raise RuntimeError("Unsafe application checkpoint")
        bundle.extractall(root, filter="data")
    candidate = root / "candidate"
    if sha(candidate / "libcrispasr.so") != LIBRARY_SHA:
        raise RuntimeError("Native library changed from accepted F16 source")
    provenance = json.loads((candidate / "provenance.json").read_text())
    if provenance.get("source") != SOURCE:
        raise RuntimeError("Native source pin changed")
    return candidate


def verify_lineage(checkpoint, gguf_path):
    """Compare all inference tensors to the pinned original conversion recipe."""
    import numpy as np
    import torch
    from gguf import GGUFReader, GGMLQuantizationType

    original = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    fused = dict(original)
    fused_names = set()
    for name, value in original.items():
        if not name.endswith(".weight_v"):
            continue
        base = name[:-len(".weight_v")]
        scale = original.get(base + ".weight_g")
        if scale is None:
            continue
        dims = tuple(range(1, value.ndim))
        fused[base + ".weight"] = value * (scale / (torch.norm(value, dim=dims, keepdim=True) + 1e-12))
        del fused[name], fused[base + ".weight_g"]
        fused_names.add(base + ".weight")
    inference_names = {name for name in fused if not name.startswith(("enc_q.", "dur_disc."))}
    reader = GGUFReader(gguf_path)
    actual_names = {tensor.name for tensor in reader.tensors}
    if actual_names != inference_names:
        raise RuntimeError("Original and GGUF inference tensor inventories differ")
    direct_exact = 0
    fused_max_relative = 0.0
    for tensor in reader.tensors:
        if tensor.tensor_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            raise RuntimeError("Expected the qualified F16/F32 Melo checkpoint")
        target = np.asarray(tensor.data)
        expected = fused[tensor.name].detach().cpu().numpy().astype(target.dtype, copy=False)
        if expected.size != target.size:
            raise RuntimeError("Melo tensor size changed: " + tensor.name)
        expected = expected.ravel().astype(np.float64)
        target = target.ravel().astype(np.float64)
        if not np.isfinite(expected).all() or not np.isfinite(target).all():
            raise RuntimeError("Non-finite Melo weights")
        if tensor.name in fused_names:
            norm = float(np.linalg.norm(expected))
            relative = float(np.linalg.norm(expected - target) / max(norm, 1e-12))
            fused_max_relative = max(fused_max_relative, relative)
            if relative > 1e-4:
                raise RuntimeError("Fused Melo weight differs from original: " + tensor.name)
        else:
            if not np.array_equal(expected, target):
                raise RuntimeError("Melo weight differs from original: " + tensor.name)
            direct_exact += 1
    return {"tensor_count": len(actual_names), "direct_exact": direct_exact,
            "fused_count": len(fused_names), "fused_max_relative_l2": fused_max_relative,
            "original_checkpoint_sha256": sha(checkpoint), "native_model_sha256": sha(gguf_path)}


def wer(expected, actual):
    a = re.findall(r"[a-z]+", expected.lower())
    b = re.findall(r"[a-z]+", actual.lower())
    row = list(range(len(b) + 1))
    for word in a:
        nxt = [row[0] + 1]
        for j, other in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (word != other)))
        row = nxt
    return row[-1] / max(1, len(a))


def make_original(source, config, checkpoint, bert_snapshot):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Original Melo requires actual CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ensure_english_nltk_resources(WORK / "nltk-data")
    unused_requests = pin_original_english_bert(bert_snapshot)
    sys.path.insert(0, str(source))
    blocked_japanese = block_unused_japanese_cleaner(source)
    from melo.api import TTS
    from melo.text import japanese
    if japanese is not blocked_japanese:
        raise RuntimeError("Japanese cleaner guard was bypassed")
    if set(unused_requests) != EXPECTED_EAGER_TOKENIZER_IDS:
        raise RuntimeError("Pinned Melo source eager tokenizer inventory changed")
    started = time.perf_counter()
    tts = TTS(language="EN_V2", device="cuda", use_hf=False,
              config_path=str(config), ckpt_path=str(checkpoint))
    from melo.text import english_bert
    from transformers import AutoModelForMaskedLM
    if english_bert.model is None:
        english_bert.model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    if torch.cuda.memory_allocated(0) < 128 * 1024**2:
        raise RuntimeError("Original Melo/BERT did not allocate on GPU")
    speaker_map = tts.hps.data.spk2id
    speaker = speaker_map["EN-US"] if "EN-US" in speaker_map else speaker_map["en-us"]
    if speaker != 0 or tts.hps.data.sampling_rate != 44100:
        raise RuntimeError("Original Melo speaker/sample rate differs from qualification")
    probe_pcm = tts.tts_to_file("Warmup.", speaker, output_path=None,
                                sdp_ratio=0.0, noise_scale=0.667,
                                noise_scale_w=0.8, speed=1.0, quiet=True)
    torch.cuda.synchronize()
    if len(probe_pcm) <= 0:
        raise RuntimeError("Original Melo probe synthesis failed")
    model_infer = tts.model.infer
    model_times = []

    def timed_infer(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model_infer(*args, **kwargs)
        torch.cuda.synchronize()
        model_times.append(time.perf_counter() - t0)
        return output

    tts.model.infer = timed_infer

    def synth(text):
        model_times.clear()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.synchronize()
        started = time.perf_counter()
        pcm = tts.tts_to_file(text, speaker, output_path=None,
                              sdp_ratio=0.0, noise_scale=0.667,
                              noise_scale_w=0.8, speed=1.0, quiet=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if len(model_times) != 1:
            raise RuntimeError("Expected one original Melo model inference per phrase")
        return seconds, np.asarray(pcm, dtype=np.float32), {"model_infer_seconds": model_times[0]}
    return synth, {"load_seconds": load_seconds, "device": "cuda",
                   "eager_unused_tokenizer_ids": sorted(set(unused_requests)),
                   "torch_version": torch.__version__, "speaker_id": speaker}


def make_native(candidate, model, bert):
    import numpy as np
    lib = ct.CDLL(str(candidate / "libcrispasr.so"))

    class Params(ct.Structure):
        _fields_ = [("n_threads", ct.c_int), ("verbosity", ct.c_int), ("use_gpu", ct.c_bool),
                    ("noise_scale", ct.c_float), ("length_scale", ct.c_float),
                    ("noise_w", ct.c_float), ("sdp_ratio", ct.c_float),
                    ("speaker_id", ct.c_int), ("seed", ct.c_uint32)]

    fp = ct.POINTER(ct.c_float)
    lib.melotts_default_params.restype = Params
    lib.melotts_init_from_file.argtypes = [ct.c_char_p, Params]
    lib.melotts_init_from_file.restype = ct.c_void_p
    lib.melotts_load_bert.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.melotts_load_bert.restype = ct.c_bool
    lib.melotts_set_seed.argtypes = [ct.c_void_p, ct.c_uint32]
    lib.melotts_set_dump_dir.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.melotts_synthesize.argtypes = [ct.c_void_p, ct.c_char_p, ct.POINTER(fp), ct.POINTER(ct.c_int)]
    lib.melotts_synthesize.restype = ct.c_int
    lib.melotts_pcm_free.argtypes = [fp]
    lib.melotts_free.argtypes = [ct.c_void_p]
    params = lib.melotts_default_params()
    params.n_threads = 2
    params.verbosity = 1
    params.use_gpu = True
    params.noise_scale = 0.667
    params.length_scale = 1.0
    params.noise_w = 0.8
    params.sdp_ratio = 0.0
    params.speaker_id = 0
    params.seed = 42
    private_log = WORK / "native-stage-private.log"
    probe_dir = WORK / "native-cuda-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    old_stderr = os.dup(2)
    with private_log.open("wb") as log:
        os.dup2(log.fileno(), 2)
        try:
            started = time.perf_counter()
            ctx = lib.melotts_init_from_file(os.fsencode(model), params)
            if ctx and not lib.melotts_load_bert(ctx, os.fsencode(bert)):
                lib.melotts_free(ctx)
                ctx = None
            load_seconds = time.perf_counter() - started
            if ctx:
                lib.melotts_set_dump_dir(ctx, os.fsencode(probe_dir))
                probe_out = fp()
                probe_sr = ct.c_int()
                probe_samples = lib.melotts_synthesize(ctx, b"Warmup.", ct.byref(probe_out), ct.byref(probe_sr))
                if probe_out:
                    lib.melotts_pcm_free(probe_out)
                lib.melotts_set_dump_dir(ctx, b"")
                if probe_samples <= 0 or probe_sr.value != 44100:
                    lib.melotts_free(ctx)
                    ctx = None
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            shutil.rmtree(probe_dir, ignore_errors=True)
    init_log = private_log.read_text(errors="replace")
    private_log.write_bytes(b"")
    if not ctx or "BERT conditioning enabled" not in init_log:
        raise RuntimeError("Native F16 Melo/BERT failed to load")
    if "melotts: final activation backend CUDA" not in init_log:
        raise RuntimeError("Native Melo CUDA work was not observed")

    def synth(text):
        lib.melotts_set_seed(ctx, 42)
        output = fp()
        sample_rate = ct.c_int()
        old_stderr = os.dup(2)
        with private_log.open("ab") as log:
            os.dup2(log.fileno(), 2)
            try:
                started = time.perf_counter()
                n_samples = lib.melotts_synthesize(ctx, text.encode(), ct.byref(output), ct.byref(sample_rate))
                if n_samples <= 0 or not output:
                    raise RuntimeError("Native Melo synthesis failed")
                try:
                    pcm = np.ctypeslib.as_array(output, shape=(n_samples,)).copy()
                finally:
                    lib.melotts_pcm_free(output)
                seconds = time.perf_counter() - started
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
        if sample_rate.value != 44100:
            raise RuntimeError("Native Melo sample rate changed")
        log_text = private_log.read_text(errors="replace")
        if "melotts: synthesized " not in log_text:
            raise RuntimeError("Native Melo synthesis log missing")
        stages = {name: float(ms) for name, ms in re.findall(
            r"melotts_bench:\s+([a-z_]+)\s+([0-9.]+) ms", log_text)}
        if not {"synthesize", "text_encoder", "flow_inverse", "hifigan_decode"} <= stages.keys():
            raise RuntimeError("Native Melo stage counters missing")
        private_log.write_bytes(b"")
        return seconds, pcm, {"native_stages_ms": stages}

    return synth, {"load_seconds": load_seconds, "device": "CUDA backend with CPU components", "final_activation_backend": "CUDA"}, lambda: lib.melotts_free(ctx)


def main():
    if not Path("/kaggle/input").is_dir() or not OUT.is_dir():
        raise RuntimeError("Hosted private Kaggle GPU pilot only")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(WORK), HF_HOME=str(WORK / "hf"),
                      CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                      CRISPASR_MELOTTS_BENCH="1", HF_HUB_DISABLE_TELEMETRY="1",
                      USE_TORCH="1", USE_TF="0", USE_FLAX="0")
    receipt = {"version": VERSION, "complete": False, "acceptance": False,
               "scope": "Full original Melo API versus native F16 Melo C ABI; warm whole calls",
               "source": SOURCE, "melo_source": MELO_SOURCE,
               "original": {"repo": "myshell-ai/MeloTTS-English-v2", "revision": ORIGINAL_REV,
                            "checkpoint_sha256": ORIGINAL_CKPT_SHA, "bert_revision": BERT_REV,
                            "bert_safetensors_sha256": BERT_SHA},
               "native": {"repo": NATIVE_REPO, "revision": NATIVE_REV,
                          "model_sha256": NATIVE_MODEL_SHA, "bert_sha256": NATIVE_BERT_SHA,
                          "application_sha256": APPLICATION_SHA},
               "settings": {"speaker": "EN-US", "speaker_id": 0, "threads": 2, "seed": 42,
                            "sdp_ratio": 0.0, "noise_scale": 0.667,
                            "noise_scale_w": 0.8, "speed": 1.0},
               "limitations": ["Original wrapper appends 50 ms silence and calls torch.cuda.empty_cache",
                               "Original F32 model/BERT versus native F16 model/BERT with selected F32 tensors",
                               "Torch and ggml sampling RNGs differ despite seed 42",
                               "Original model.infer and native built-in stage counters are separate diagnostics"],
               "resource_checks": [], "cases": []}

    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def guard(label, *, first=False):
        receipt["resource_checks"].append(headroom(label, first=first))
        save()

    save()
    guard("admission", first=True)
    archive = find_exact("current-application-checkpoint.tar.gz", APPLICATION_SHA)
    candidate = prepare_application(archive)
    if os.environ.get("CRISPASR_MELO_LIB_READY") != "1":
        providers = sorted({str(path) for base in (Path("/usr/local/lib"), Path("/opt/conda/lib"))
                            for path in base.glob("python*/**/nvidia/*/lib") if path.is_dir()})
        providers += [str(path) for path in (Path("/usr/local/cuda/lib64"),
                                             Path("/usr/local/nvidia/lib64")) if path.is_dir()]
        env = dict(os.environ, CRISPASR_MELO_LIB_READY="1",
                   LD_LIBRARY_PATH=os.pathsep.join([str(candidate), *providers,
                                                    os.environ.get("LD_LIBRARY_PATH", "")]))
        os.execve(sys.executable, [sys.executable, "-B", __file__], env)
    linked = subprocess.check_output(["ldd", "-r", str(candidate / "libcrispasr.so")],
                                     text=True, stderr=subprocess.STDOUT)
    if "not found" in linked or "undefined symbol:" in linked:
        raise RuntimeError("Accepted native application has unresolved hosted dependencies")
    guard("before-dependencies")
    # Never replace Kaggle Torch/Torchaudio. Wheels or installed packages only;
    # a missing source dependency fails the pilot before any comparison.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:",
                    "faster-whisper==1.1.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf==0.17.1",
                    *ORIGINAL_IMPORT_DEPS], check=True)
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    from huggingface_hub import hf_hub_download, snapshot_download
    from scipy.signal import resample_poly
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("Original/native Melo requires an actual CUDA worker")
    receipt["torch_version"] = torch.__version__
    receipt["script_sha256"] = sha(__file__)
    save()
    guard("before-model-downloads")
    original_dir = Path(snapshot_download("myshell-ai/MeloTTS-English-v2", revision=ORIGINAL_REV,
                                          allow_patterns=["checkpoint.pth", "config.json", "README.md"],
                                          local_dir=WORK / "original-model"))
    original_ckpt = original_dir / "checkpoint.pth"
    if sha(original_ckpt) != ORIGINAL_CKPT_SHA:
        raise RuntimeError("Original Melo checkpoint hash changed")
    native_model = Path(hf_hub_download(NATIVE_REPO, "melotts-en-v2-f16.gguf",
                                        revision=NATIVE_REV, local_dir=WORK / "native-model"))
    native_bert = Path(hf_hub_download(NATIVE_REPO, "bert-base-uncased.gguf",
                                       revision=NATIVE_REV, local_dir=WORK / "native-bert"))
    if sha(native_model) != NATIVE_MODEL_SHA or sha(native_bert) != NATIVE_BERT_SHA:
        raise RuntimeError("Qualified native F16 artifacts changed")
    bert_dir = Path(snapshot_download("google-bert/bert-base-uncased", revision=BERT_REV,
                                      allow_patterns=["model.safetensors", "config.json", "tokenizer*",
                                                      "vocab.txt", "special_tokens_map.json"],
                                      local_dir=WORK / "original-bert"))
    if sha(bert_dir / "model.safetensors") != BERT_SHA:
        raise RuntimeError("Original BERT checkpoint changed")
    if sha(bert_dir / "vocab.txt") != BERT_VOCAB_SHA:
        raise RuntimeError("Original BERT tokenizer changed")
    guard("before-lineage-proof")
    receipt["checkpoint_lineage"] = verify_lineage(original_ckpt, native_model)
    save()
    guard("before-source-and-asr")
    source = WORK / "melo-source"
    clone_pin("https://github.com/myshell-ai/MeloTTS.git", MELO_SOURCE, source)
    asr_dir = Path(snapshot_download(ASR_REPO, revision=ASR_REV, local_dir=WORK / "asr"))
    asr = WhisperModel(str(asr_dir), device="cpu", compute_type="int8", cpu_threads=2)
    guard("before-native-load")
    native, native_info, native_free = make_native(candidate, native_model, native_bert)
    receipt["native"]["runtime"] = native_info
    save()
    try:
        guard("before-original-load")
        original, original_info = make_original(source, original_dir / "config.json",
                                                original_ckpt, bert_dir)
        receipt["original"]["runtime"] = original_info
        save()
        for index, text in enumerate(TEXTS):
            case = {"case": index, "text": text, "runs": []}
            receipt["cases"].append(case)
            save()
            for phase, count in (("warmup", 1), ("measured", 3)):
                for repeat in range(count):
                    order = ("native", "original") if repeat % 2 == 0 else ("original", "native")
                    for arm in order:
                        guard(f"case-{index}-{phase}-{repeat}-{arm}")
                        seconds, pcm, details = native(text) if arm == "native" else original(text)
                        if (pcm.ndim != 1 or not 22050 < len(pcm) < 44100 * 20 or
                                not np.isfinite(pcm).all()):
                            raise RuntimeError("Melo output rate/shape/finite gate failed")
                        audio16 = np.ascontiguousarray(resample_poly(pcm, 160, 441), dtype=np.float32)
                        segments, _ = asr.transcribe(audio16, language="en", beam_size=5)
                        transcript = " ".join(segment.text.strip() for segment in segments)
                        error = wer(text, transcript)
                        row = {"phase": phase, "repeat": repeat, "arm": arm,
                               "seconds": seconds, "samples": len(pcm),
                               "rate": 44100, "asr_wer": error, **details}
                        case["runs"].append(row)
                        save()
                        if error > .2:
                            raise RuntimeError("Unchanged tiny.en Melo content gate failed")
            case["median_whole_seconds"] = {
                arm: statistics.median(run["seconds"] for run in case["runs"]
                                       if run["phase"] == "measured" and run["arm"] == arm)
                for arm in ("original", "native")}
            case["whole_call_original_over_native"] = (
                case["median_whole_seconds"]["original"] / case["median_whole_seconds"]["native"])
            save()
    finally:
        native_free()
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("MELO_ORIGINAL_PAIRED_ACCEPTED", flush=True)


if __name__ == "__main__":
    main()
