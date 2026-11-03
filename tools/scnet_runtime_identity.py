"""Linux loaded-DSO identity for SCNet CPU diagnostics and qualification.

This enumerates the current process's main ELF link-map namespace. It binds
the file bytes named by loaded objects, not relocation/symbol-binding choices
or anonymous JIT code. Standalone ``ldd`` output has a different scope.
"""

from __future__ import annotations

from collections import Counter
import ctypes
import hashlib
import os
from pathlib import Path
import re
import sys


COLLECTOR = "linux-dl-iterate-phdr-v1"
GGML_NAMES = ("ggml", "ggml-base", "ggml-cpu", "ggml-blas")
OPENMP_NAME = r"lib(?:gomp|omp|iomp5)(?:[-.][^.]+)?\.so(?:\..+)?"
SCOPE = ("Current process main ELF link map; file-backed DSO identities, not relocation-binding proof. "
         "The main executable and anonymous/JIT mappings are outside this inventory.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_paths():
    """Copy loader names under its lock; hash files only after enumeration."""
    if not sys.platform.startswith("linux"):
        raise ValueError("loaded-DSO identity currently requires Linux")

    class Prefix(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("name", ctypes.c_char_p),
                    ("program_headers", ctypes.c_void_p), ("program_header_count", ctypes.c_ushort)]

    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(Prefix), ctypes.c_size_t, ctypes.c_void_p)
    paths, virtual, errors = [], [], []

    @callback_type
    def visit(info, size, unused):
        try:
            if size < ctypes.sizeof(Prefix):
                raise ValueError("unsupported dl_phdr_info prefix")
            name = os.fsdecode(info.contents.name or b"")
            if not name:
                return 0  # Main executable, not a DSO.
            if re.fullmatch(r"linux-(?:vdso|gate)\.so\.\d+", name):
                virtual.append(name)
            else:
                path = Path(name)
                if not path.is_absolute():
                    raise ValueError("loaded DSO has no absolute loader path")
                paths.append(path)
            return 0
        except BaseException as error:
            # ctypes callbacks cannot propagate Python exceptions themselves.
            errors.append(error)
            return 1

    iterator = ctypes.CDLL(None).dl_iterate_phdr
    iterator.argtypes, iterator.restype = [callback_type, ctypes.c_void_p], ctypes.c_int
    result = iterator(visit, None)
    if errors:
        raise ValueError("loaded-DSO enumeration failed") from errors[0]
    if result != 0 or not paths:
        raise ValueError("loaded-DSO enumeration was incomplete")
    return paths, sorted(set(virtual))


def display_path(path, build):
    if path.is_relative_to(build):
        return "{build}/" + path.relative_to(build).as_posix()
    torch = sys.modules.get("torch")
    origin = getattr(torch, "__file__", None)
    if origin:
        directory = Path(origin).resolve().parent
        if path.is_relative_to(directory):
            return "{torch}/" + path.relative_to(directory).as_posix()
    return "{runtime}/" + path.name


def capture_loaded_dsos(library, require_torch):
    library = Path(library).resolve(strict=True)
    build = library.parent.parent
    cache = build / "CMakeCache.txt"
    cache_text = cache.read_text(encoding="utf-8")
    if not re.search(r"^GGML_BACKEND_DL:BOOL=OFF$", cache_text, re.M):
        raise ValueError("loaded-DSO qualification requires GGML_BACKEND_DL=OFF")
    provider = re.search(r"^BLAS_openblas_LIBRARY:(?:FILEPATH|STRING)=(.+)$", cache_text, re.M)
    if not provider:
        raise ValueError("loaded-DSO qualification requires the configured OpenBLAS library")
    provider = Path(provider[1]).resolve(strict=True)
    paths, virtual = mapped_paths()
    counts = Counter(path.resolve(strict=True) for path in paths)
    files, identities = {}, {}
    for path, count in sorted(counts.items()):
        if not path.is_file():
            raise ValueError("a loaded DSO is missing or is not a regular file")
        identity = hashlib.sha256(os.fsencode(path)).hexdigest()
        identities[path] = identity
        files[identity] = {"file": display_path(path, build), "name": path.name,
                           "resolved_path_sha256": identity, "sha256": sha256(path),
                           "bytes": path.stat().st_size, "mappings": count}

    def one_named(pattern):
        found = [identity for identity, record in files.items() if re.fullmatch(pattern, record["name"])]
        if len(found) != 1:
            raise ValueError("a required loaded runtime is missing or ambiguous: " + pattern)
        return found[0]

    if library not in identities or provider not in identities:
        raise ValueError("native export or configured OpenBLAS dependency is not actually loaded")
    roles = {"native_library": identities[library], "configured_openblas": identities[provider],
             "ggml": {name: one_named(r"lib" + re.escape(name) + r"\.so(?:\..+)?") for name in GGML_NAMES},
             "openmp_candidates": sorted(identity for identity, record in files.items()
                                          if re.fullmatch(OPENMP_NAME, record["name"])),
             "torch_cpu": one_named(r"libtorch_cpu\.so") if require_torch else None}
    if not roles["openmp_candidates"]:
        raise ValueError("no loaded OpenMP runtime was observed")
    result = {"schema": 1, "collector": COLLECTOR, "process_id": os.getpid(),
              "scope": SCOPE,
              "require_torch": bool(require_torch), "files": files, "roles": roles,
              "virtual_objects_without_file_bytes": virtual, "cmake_cache_sha256": sha256(cache)}
    validate_snapshot(result)
    return result


def validate_snapshot(snapshot):
    def require(value, message):
        if not value:
            raise ValueError("loaded-DSO " + message)

    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)

    require(isinstance(snapshot, dict) and snapshot.get("schema") == 1 and snapshot.get("collector") == COLLECTOR and
            snapshot.get("scope") == SCOPE,
            "snapshot is missing or incompatible")
    require(type(snapshot.get("process_id")) is int and snapshot["process_id"] > 0 and
            type(snapshot.get("require_torch")) is bool and digest(snapshot.get("cmake_cache_sha256")),
            "process/configuration identity is incomplete")
    files = snapshot.get("files", {})
    require(isinstance(files, dict) and bool(files), "file inventory is empty")
    for identity, record in files.items():
        require(isinstance(record, dict) and digest(identity) and record.get("resolved_path_sha256") == identity and
                digest(record.get("sha256")) and type(record.get("bytes")) is int and record["bytes"] > 0 and
                type(record.get("mappings")) is int and record["mappings"] > 0 and
                isinstance(record.get("name"), str) and bool(record["name"]) and
                isinstance(record.get("file"), str) and record["file"].startswith(("{build}/", "{torch}/", "{runtime}/")),
                "file identity is malformed")
    roles = snapshot.get("roles", {})
    require(isinstance(roles, dict) and set(roles) ==
            {"native_library", "configured_openblas", "ggml", "openmp_candidates", "torch_cpu"}, "roles are incomplete")
    require(digest(roles.get("native_library")) and roles["native_library"] in files and
            digest(roles.get("configured_openblas")) and roles["configured_openblas"] in files,
            "native/provider identities are absent")
    ggml = roles.get("ggml", {})
    require(isinstance(ggml, dict) and set(ggml) == set(GGML_NAMES) and
            all(digest(identity) and identity in files and re.fullmatch(r"lib" + re.escape(name) + r"\.so(?:\..+)?", files[identity]["name"])
                for name, identity in ggml.items()), "ggml identities are absent or incompatible")
    openmp = roles.get("openmp_candidates")
    require(isinstance(openmp, list) and bool(openmp) and all(digest(identity) for identity in openmp) and
            openmp == sorted(set(openmp)) and
            all(identity in files and re.fullmatch(OPENMP_NAME, files[identity]["name"])
                for identity in openmp), "OpenMP candidates are absent or incompatible")
    require((digest(roles.get("torch_cpu")) and roles["torch_cpu"] in files and files[roles["torch_cpu"]]["name"] == "libtorch_cpu.so")
            if snapshot["require_torch"] else roles.get("torch_cpu") is None, "Torch CPU identity is absent or incompatible")
    return snapshot


def compare_snapshots(before, after):
    before, after = validate_snapshot(before), validate_snapshot(after)
    if before["process_id"] != after["process_id"] or before["require_torch"] != after["require_torch"]:
        raise ValueError("loaded-DSO snapshots belong to different process/scopes")
    first, last = before["files"], after["files"]
    common = set(first) & set(last)
    return {"common_file_bytes_unchanged": all(first[key]["sha256"] == last[key]["sha256"] and
                first[key]["bytes"] == last[key]["bytes"] for key in common),
            "mapping_set_unchanged": first == last, "roles_unchanged": before["roles"] == after["roles"],
            "configuration_unchanged": before["cmake_cache_sha256"] == after["cmake_cache_sha256"],
            "added_objects": sorted(set(last) - set(first)), "removed_objects": sorted(set(first) - set(last))}


def verify_timed_identity(evidence, native_sha256=None, native_files=None):
    if not isinstance(evidence, dict) or evidence.get("schema") != 1 or evidence.get("collector") != COLLECTOR or \
            evidence.get("complete") is not True or evidence.get("scope") != "after_warmups_through_all_timing_pairs":
        raise ValueError("loaded-DSO timed-phase evidence is missing or incomplete")
    initial, before, after = (validate_snapshot(evidence.get(key)) for key in ("initialization", "timed_before", "timed_after"))
    initialization = compare_snapshots(initial, before)
    timed = compare_snapshots(before, after)
    if evidence.get("initialization_transition") != initialization or evidence.get("timed_transition") != timed or \
            not initialization["common_file_bytes_unchanged"] or not initialization["configuration_unchanged"] or \
            not all(timed[key] for key in ("common_file_bytes_unchanged", "mapping_set_unchanged", "roles_unchanged", "configuration_unchanged")):
        raise ValueError("loaded-DSO identity changed or its recorded proof is inconsistent")
    if not before["require_torch"]:
        raise ValueError("paired loaded-DSO proof must include the original Torch CPU runtime")
    if native_sha256 is not None and before["files"][before["roles"]["native_library"]]["sha256"] != native_sha256:
        raise ValueError("loaded native export differs from the qualified library")
    if native_files is not None:
        # GOMP can legitimately come from Torch in this process; do not demand
        # that it match standalone ldd's OpenMP selection.
        required = [before["roles"]["native_library"], before["roles"]["configured_openblas"], *before["roles"]["ggml"].values()]
        for identity in required:
            record, expected = before["files"][identity], native_files.get(identity)
            if expected is None or any(record[key] != expected.get(key) for key in ("sha256", "bytes", "resolved_path_sha256")):
                raise ValueError("loaded native/ggml/OpenBLAS identity differs from the qualified artifacts")
    return after


class LoadedDSOTracker:
    def __init__(self, library):
        self.library = Path(library)
        self.evidence = {"schema": 1, "collector": COLLECTOR, "complete": False,
                         "scope": "after_warmups_through_all_timing_pairs",
                         "initialization": capture_loaded_dsos(self.library, require_torch=True)}

    def before_timing(self):
        self.evidence["timed_before"] = capture_loaded_dsos(self.library, require_torch=True)
        transition = compare_snapshots(self.evidence["initialization"], self.evidence["timed_before"])
        self.evidence["initialization_transition"] = transition
        if not transition["common_file_bytes_unchanged"] or not transition["configuration_unchanged"]:
            raise ValueError("loaded runtime bytes/configuration changed during initialization")

    def after_timing(self):
        self.evidence["timed_after"] = capture_loaded_dsos(self.library, require_torch=True)
        self.evidence["timed_transition"] = compare_snapshots(self.evidence["timed_before"], self.evidence["timed_after"])
        self.evidence["complete"] = True
        try:
            verify_timed_identity(self.evidence)
        except BaseException:
            self.evidence["complete"] = False
            raise
