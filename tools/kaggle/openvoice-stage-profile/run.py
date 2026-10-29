#!/usr/bin/env python3
"""Private hosted OpenVoice conversion timing on accepted F16-Melo-derived inputs.

This does not measure full original Melo generation. See README.md.
"""

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

VERSION = "openvoice-stage-profile-v1"
OUT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/openvoice-stage-profile")
CONDITIONING_SHA = "46badec89720409414dad5c61f3e5cc1c34928a3e0b8391a1cf4f92e4d90dec8"
APPLICATION_SHA = "ae93882f0a7854cfb59f519ebbf406d3c0854828d9f451c0ef308e7bc5269e71"
LIBRARY_SHA = "fffd43cd2536fd8d3876fdc7d9d3e4d8e0c730bd0b6f532e37a2123c2a560d57"
SOURCE = "97053abff8201870827027692dc25dc0c412e2a3"
UPSTREAM = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
TEXTS = ("The quick brown fox jumps over the lazy dog.",
         "We enjoy reading books and listening to music in the evening.")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def headroom(label, *, initial=False):
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    free = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,compute_cap",
         "--format=csv,noheader,nounits"], text=True, timeout=30).strip().split(",")
    row = {"phase": label, "available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
           "scratch_free_bytes": shutil.disk_usage(WORK).free,
           "gpu_free_mib": int(free[0].strip()), "compute_cap": free[1].strip(),
           "load": list(os.getloadavg()), "cpus": os.cpu_count() or 1}
    if (row["available_bytes"] < 3 * 1024**3 or row["scratch_free_bytes"] < 4 * 1024**3
            or row["gpu_free_mib"] < (8 * 1024 if initial else 1024)
            or row["load"][0] > 2 * row["cpus"]):
        raise RuntimeError("Insufficient hosted headroom before " + label)
    return row


def find_exact(name, expected):
    matches = [p for p in Path("/kaggle/input").rglob(name) if p.is_file() and sha(p) == expected]
    if len(matches) != 1:
        raise RuntimeError("Expected one pinned mounted " + name)
    return matches[0]


def extract_candidate(archive):
    root = WORK / "application"
    root.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(Path(m.name).is_absolute() or ".." in Path(m.name).parts or
               not (m.isfile() or m.isdir() or m.issym()) or
               (m.issym() and (Path(m.linkname).is_absolute() or ".." in Path(m.linkname).parts))
               for m in members):
            raise RuntimeError("Unsafe application checkpoint archive")
        bundle.extractall(root, filter="data")
    candidate = root / "candidate"
    if sha(candidate / "libcrispasr.so") != LIBRARY_SHA:
        raise RuntimeError("Accepted library digest changed")
    provenance = json.loads((candidate / "provenance.json").read_text())
    if provenance.get("source") != SOURCE:
        raise RuntimeError("Application source provenance changed")
    return candidate


def clone_source():
    upstream = WORK / "upstream"
    subprocess.run(["git", "init", str(upstream)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(upstream), "remote", "add", "origin",
                    "https://github.com/myshell-ai/OpenVoice.git"], check=True)
    subprocess.run(["git", "-C", str(upstream), "fetch", "--depth=1", "origin", UPSTREAM], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(upstream), "checkout", "FETCH_HEAD"], check=True,
                   stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise RuntimeError("Original OpenVoice source pin changed")
    return upstream


def cosine(actual, expected):
    import numpy as np
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(expected, dtype=np.float64).ravel()
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid converted PCM or sample count")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        raise RuntimeError("Silent conversion output")
    return float(np.dot(a, b) / denom)


def native_engine(candidate, model, use_gpu):
    import numpy as np
    lib = ct.CDLL(str(candidate / "libcrispasr.so"))
    class Params(ct.Structure):
        _fields_ = [("n_threads", ct.c_int), ("verbosity", ct.c_int),
                    ("use_gpu", ct.c_bool), ("tau", ct.c_float)]
    fp = ct.POINTER(ct.c_float)
    lib.openvoice2_context_default_params.restype = Params
    lib.openvoice2_init_from_file.argtypes = [ct.c_char_p, Params]
    lib.openvoice2_init_from_file.restype = ct.c_void_p
    lib.openvoice2_set_source_speaker.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.openvoice2_set_source_speaker.restype = ct.c_bool
    lib.openvoice2_convert.argtypes = [ct.c_void_p, fp, ct.c_int, ct.c_int,
                                       fp, ct.c_int, ct.c_int, ct.POINTER(fp), ct.POINTER(ct.c_int)]
    lib.openvoice2_convert.restype = ct.c_bool
    lib.openvoice2_free.argtypes = [ct.c_void_p]
    libc = ct.CDLL(None)
    libc.free.argtypes = [ct.c_void_p]
    params = lib.openvoice2_context_default_params()
    params.n_threads = 2
    params.verbosity = 1
    params.use_gpu = use_gpu
    params.tau = 0
    # C stderr contains only model stage logs and remains private scratch.
    old_stderr = os.dup(2)
    native_log = WORK / ("native-gpu-private.log" if use_gpu else "native-cpu-private.log")
    with native_log.open("wb") as log:
        os.dup2(log.fileno(), 2)
        try:
            ctx = lib.openvoice2_init_from_file(os.fsencode(model), params)
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    backend = "CUDA" if use_gpu else "CPU"
    if not ctx or "openvoice2: vocoder backend " + backend not in native_log.read_text(errors="replace"):
        if ctx:
            lib.openvoice2_free(ctx)
        raise RuntimeError("Native OpenVoice requested backend was not selected")

    def convert(source, reference, speaker):
        src = np.ascontiguousarray(source, dtype=np.float32).ravel()
        ref = np.ascontiguousarray(reference, dtype=np.float32).ravel()
        if not lib.openvoice2_set_source_speaker(ctx, speaker.encode()):
            raise RuntimeError("Accepted source speaker missing")
        output = fp()
        count = ct.c_int()
        # Native verbose diagnostics include target-embedding statistics.
        # Keep the C stream in private scratch, outside Kaggle's public log.
        offset = native_log.stat().st_size
        old_stderr = os.dup(2)
        with native_log.open("ab") as log:
            os.dup2(log.fileno(), 2)
            try:
                started = time.perf_counter()
                okay = lib.openvoice2_convert(ctx, src.ctypes.data_as(fp), len(src), 22050,
                                               ref.ctypes.data_as(fp), len(ref), 22050,
                                               ct.byref(output), ct.byref(count))
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
        if not okay or not output or count.value <= 0:
            raise RuntimeError("Native conversion failed")
        try:
            pcm = np.ctypeslib.as_array(output, shape=(count.value,)).copy()
        finally:
            libc.free(output)
        seconds = time.perf_counter() - started
        with native_log.open("rb") as log:
            log.seek(offset)
            text = log.read().decode(errors="replace")
        stages = {name: float(value) for name, value in re.findall(
            r"openvoice2_bench:\s+([a-z_]+)\s+([0-9.]+) ms", text)}
        required = {"convert", "resample", "stft", "ref_enc", "enc_q", "flow_forward", "flow_reverse", "hifigan_decode"}
        if not required <= stages.keys():
            raise RuntimeError("Missing native conversion stage timers")
        return seconds, pcm, stages
    return convert, lambda: lib.openvoice2_free(ctx)


def original_engine(upstream, model_dir):
    import numpy as np
    import torch
    sys.path.insert(0, str(upstream))
    from openvoice.models import SynthesizerTrn
    from openvoice.mel_processing import spectrogram_torch
    config = json.loads((model_dir / "converter/config.json").read_text())
    data = config["data"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = SynthesizerTrn(len(config.get("symbols", [])), data["filter_length"] // 2 + 1,
                          n_speakers=data["n_speakers"], **config["model"]).eval()
    checkpoint = torch.load(model_dir / "converter/checkpoint.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda()
    if torch.cuda.memory_allocated(0) < 64 * 1024**2:
        raise RuntimeError("Original OpenVoice model did not load on CUDA")
    # The native C API caches the reference embedding by PCM. Keep the same
    # cache boundary in the original arm after its first warmup call.
    reference_cache = {}

    def spec(pcm):
        audio = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).unsqueeze(0).cuda()
        return spectrogram_torch(audio, data["filter_length"], data["sampling_rate"],
                                 data["hop_length"], data["win_length"], center=False)

    def convert(source, reference, source_embedding):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            key = hashlib.sha256(np.ascontiguousarray(reference, dtype=np.float32).tobytes()).hexdigest()
            if key not in reference_cache:
                ref_spec = spec(reference)
                reference_cache[key] = model.ref_enc(ref_spec.transpose(1, 2)).unsqueeze(-1)
            target = reference_cache[key]
            src_spec = spec(source)
            source_se = torch.from_numpy(np.ascontiguousarray(source_embedding, dtype=np.float32)).reshape(1, -1, 1).cuda()
            lengths = torch.tensor([src_spec.shape[-1]], dtype=torch.long, device="cuda")
            wave = model.voice_conversion(src_spec, lengths, source_se, target, tau=0)[0][0, 0]
            pcm = wave.detach().float().cpu().numpy().copy()
            peak = float(np.max(np.abs(pcm)))
            if peak > .01:
                pcm *= np.float32(.95) / peak
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        return seconds, pcm
    return convert, torch.__version__


def main():
    if not Path("/kaggle/input").is_dir() or not OUT.is_dir():
        raise RuntimeError("Hosted private Kaggle GPU pilot only")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0",
                      CRISPASR_OV2_GPU="1", CRISPASR_OPENVOICE2_BENCH="1",
                      TMPDIR=str(WORK), HF_HOME=str(WORK / "hf"))
    receipt = {"version": VERSION, "complete": False, "acceptance": False,
               "scope": "Opt-in native stage profiling on accepted OpenVoice inputs; timer logging may perturb results; no full Melo original timing",
               "native_device": "CPU reference encoder and flows; CUDA HiFi-GAN",
               "native_cpu_device": "CPU reference encoder, flows and vocoder",
               "original_device": "PyTorch CUDA reference encoder, flows and HiFi-GAN",
               "source": SOURCE, "upstream": UPSTREAM, "tau": 0, "threads": 2,
               "precision": "Original F32 checkpoint versus native F32-sensitive GGUF; input Melo and BERT F16",
               "cases": [], "resource_checks": []}
    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    receipt["resource_checks"].append(headroom("admission", initial=True)); save()
    conditioning = find_exact("validation.json", CONDITIONING_SHA)
    inputs = json.loads(conditioning.read_text())
    if not inputs.get("complete") or not inputs.get("model_acceptance"):
        raise RuntimeError("Conditioning source was not accepted")
    data = conditioning.parent
    required = ["openvoice2-tcc-f32sens.gguf", "reference-0.gguf", "reference-1.gguf",
                "original-0.wav", "original-1.wav"]
    for name in required:
        row = inputs["artifacts"][name]
        if sha(data / name) != row["sha256"] or (data / name).stat().st_size != row["bytes"]:
            raise RuntimeError("Frozen conditioning artifact changed: " + name)
    archive = find_exact("current-application-checkpoint.tar.gz", APPLICATION_SHA)
    candidate = extract_candidate(archive)
    if os.environ.get("CRISPASR_OV2_LIB_READY") != "1":
        providers = sorted({str(path) for base in (Path("/usr/local/lib"), Path("/opt/conda/lib"))
                            for path in base.glob("python*/**/nvidia/*/lib") if path.is_dir()})
        providers += [str(path) for path in (Path("/usr/local/cuda/lib64"),
                                             Path("/usr/local/nvidia/lib64")) if path.is_dir()]
        env = dict(os.environ, CRISPASR_OV2_LIB_READY="1",
                   LD_LIBRARY_PATH=os.pathsep.join([str(candidate), *providers,
                                                    os.environ.get("LD_LIBRARY_PATH", "")]))
        os.execve(sys.executable, [sys.executable, "-B", __file__], env)
    linked = subprocess.check_output(["ldd", "-r", str(candidate / "libcrispasr.so")],
                                     text=True, stderr=subprocess.STDOUT)
    if "not found" in linked or "undefined symbol:" in linked:
        raise RuntimeError("Accepted application has unresolved hosted dependencies")
    receipt["resource_checks"].append(headroom("before-dependencies")); save()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "gguf==0.17.1", "librosa>=0.10", "soundfile", "huggingface_hub"], check=True)
    import numpy as np
    import soundfile as sf
    import torch
    from gguf import GGUFReader
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError("Original OpenVoice requires actual CUDA")
    references = [GGUFReader(data / f"reference-{i}.gguf") for i in range(2)]
    def field(reader, name):
        return reader.fields["crispasr.ref." + name].contents()
    revisions = {field(reader, "model_revision") for reader in references}
    source_commits = {field(reader, "upstream_commit") for reader in references}
    checkpoint_hashes = {field(reader, "ov2_checkpoint_sha256") for reader in references}
    if (len(revisions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", next(iter(revisions)))
            or source_commits != {UPSTREAM} or len(checkpoint_hashes) != 1):
        raise RuntimeError("Original reference provenance changed")
    model_rev = revisions.pop()
    checkpoint_hash = checkpoint_hashes.pop()
    receipt["original_model_revision"] = model_rev
    receipt["original_checkpoint_sha256"] = checkpoint_hash
    receipt["accepted_input_sha256"] = {name: sha(data / name) for name in required}
    receipt["application_checkpoint_sha256"] = APPLICATION_SHA
    receipt["script_sha256"] = sha(__file__)
    save()
    receipt["resource_checks"].append(headroom("before-original-model")); save()
    upstream = clone_source()
    original_dir = Path(snapshot_download("myshell-ai/OpenVoiceV2", revision=model_rev,
                                          allow_patterns=["converter/*", "README.md"],
                                          local_dir=WORK / "original"))
    if sha(original_dir / "converter/checkpoint.pth") != checkpoint_hash:
        raise RuntimeError("Original checkpoint does not match accepted reference")
    native, native_free = native_engine(candidate, data / required[0], True)
    native_cpu, native_cpu_free = native_engine(candidate, data / required[0], False)
    original, torch_version = original_engine(upstream, original_dir)
    receipt["torch_version"] = torch_version
    save()
    try:
        for index, reader in enumerate(references):
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            source = np.asarray(tensors["src_pcm"].data, dtype=np.float32).ravel().copy()
            reference = np.asarray(tensors["ref_pcm"].data, dtype=np.float32).ravel().copy()
            embedding = np.asarray(tensors["src_se"].data, dtype=np.float32).ravel().copy()
            speaker = field(reader, "ov2_source_speaker")
            truth, rate = sf.read(data / f"original-{index}.wav", dtype="float32")
            if (rate != 22050 or truth.ndim != 1 or speaker != "en-us" or
                    not len(source) or not len(reference) or embedding.size != 256):
                raise RuntimeError("Frozen case settings changed")
            case = {"case": index, "text": TEXTS[index], "speaker": speaker,
                    "input_samples": {"source": len(source), "reference": len(reference)},
                    "output_samples": len(truth), "runs": []}
            receipt["cases"].append(case); save()
            for phase, count in (("warmup", 1), ("measured", 3)):
                for repeat in range(count):
                    order = (("native", "native-cpu", "original") if repeat % 2 == 0
                             else ("original", "native-cpu", "native"))
                    for arm in order:
                        receipt["resource_checks"].append(headroom(f"case-{index}-{phase}-{repeat}-{arm}")); save()
                        stages = None
                        if arm == "native":
                            seconds, pcm, stages = native(source, reference, speaker)
                        elif arm == "native-cpu":
                            seconds, pcm, stages = native_cpu(source, reference, speaker)
                        else:
                            seconds, pcm = original(source, reference, embedding)
                        score = cosine(pcm, truth)
                        if score < .99:
                            raise RuntimeError("Accepted OpenVoice waveform gate failed")
                        row = {"phase": phase, "repeat": repeat, "arm": arm,
                               "seconds": seconds, "samples": len(pcm),
                               "cosine_to_accepted_original": score}
                        if stages is not None:
                            row["native_stages_ms"] = stages
                        case["runs"].append(row)
                        save()
            case["median_seconds"] = {
                arm: statistics.median(run["seconds"] for run in case["runs"]
                                       if run["phase"] == "measured" and run["arm"] == arm)
                for arm in ("native", "native-cpu", "original")}
            case["original_over_native_ratio"] = (case["median_seconds"]["original"] /
                                                   case["median_seconds"]["native"])
            save()
    finally:
        native_cpu_free()
        native_free()
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("OPENVOICE_PAIRED_TIMING_ACCEPTED", flush=True)


if __name__ == "__main__":
    main()
