"""Breeze TTS 2 reference backend — the LOCAL half of the diff harness (#412).

Breeze TTS 2 cannot be run locally: the upstream needs Linux + CUDA and
~7.7 GiB VRAM in eager mode, and the checkpoint is 6.97 GB bf16. The
reference activations are therefore produced ONCE on a Kaggle GPU by
`tools/kaggle/breeze-refdump/breeze_refdump.py` and published as .npy
fixtures under

    cstr/crispasr-regression-fixtures  (dataset)  breeze-tts-2/

This module is the consumer side. It has two entry points:

  dump(...)   the `tools/dump_reference.py` plug-in contract. Instead of
              running PyTorch it FETCHES the fixture set and returns it as
              {stage_name: ndarray}, so the shared GGUF writer produces a
              `breeze-tts-2-ref.gguf` that `crispasr-diff` loads exactly like
              any other backend's reference archive. The --audio argument is
              ignored (the reference clip is baked into the fixture).

              To wire it into the CLI, add one line to REGISTERED_BACKENDS in
              tools/dump_reference.py:
                  "breeze-tts-2": "reference_backends.breeze_tts_2",

  __main__    a standalone comparator:
                  python tools/reference_backends/breeze_tts_2.py \
                      --cpp-dump /path/to/crispasr/dump/dir
              loads the fixtures, pairs each stage with the C++ .npy of the
              same name, and prints cosine / max-abs / argmax-match per stage
              with the acceptance thresholds from the validation plan.

STAGE CONTRACT (must match breeze_refdump.py exactly)

  ref_audio                 (N,)          f32   24 kHz mono reference PCM
  ref_codes                 (T_ref, 16)   i32   codec codes of the ref clip
  prompt_input_ids          (L,)          i32
  prompt_text_ids_mask      (L,)          i32
  prompt_text_ids_len       (S,)          i32
  te_seg{K}_hidden          (len_K, 1152) f32   per-SEGMENT encoder output
  te_seg{K}_layer{J}        (len_K, 1152) f32   optional per-layer states
  te_proj_out               (n_text,2048) f32   text_encoder_proj output
  backbone_inputs_embeds    (L, 2048)     f32
  backbone_layer{J}_frame0  (2048,)       f32   J = 0..28 (0 = input embeds)
  backbone_hidden_frame0    (2048,)       f32
  backbone_logits_frame0    (2052,)       f32   lm_head; row 2051 = EOS class
  dd_logits_frame0_cb{C}    (2051,)       f32   C = 1..15
  dd_codes_frame0_stepwise  (16,)         i32   frame 0, per-codebook loop
  dd_codes_frame{F}         (16,)         i32   F = 0..4, from generate()
  codes                     (T, 16)       i32
  codec_audio               (N_out,)      f32

ACCEPTANCE (docs/breeze-tts-2-feasibility.md §4)
  every float stage         cos >= 0.999
  backbone_logits_frame0    argmax must match exactly
  dd_logits_frame0_cb*      argmax must match exactly
  codes / dd_codes_frame*   exact integer equality under greedy
  codec_audio               validated separately through the existing
                            qwen3-tts codec path, NOT here

ENV
  BREEZE_FIXTURE_DIR   local directory holding the fixtures (skips the
                       download entirely). Set this on the VPS to avoid
                       re-fetching.
  BREEZE_FIXTURE_REPO  override the dataset repo id
  HF_TOKEN             passed through to huggingface_hub if set
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

FIXTURE_REPO = os.environ.get("BREEZE_FIXTURE_REPO", "cstr/crispasr-regression-fixtures")
FIXTURE_PREFIX = "breeze-tts-2"

# Integer stages are compared for exact equality, never cosine.
INT_STAGES = {
    "ref_codes",
    "prompt_input_ids",
    "prompt_text_ids_mask",
    "prompt_text_ids_len",
    "codes",
}


def _is_int_stage(name: str) -> bool:
    return name in INT_STAGES or name.startswith("dd_codes_frame")


# Stage names that always exist. Per-segment / per-layer / per-codebook names
# are discovered from the fixture set (their counts depend on the prompt).
DEFAULT_STAGES = [
    "ref_audio",
    "ref_codes",
    "prompt_input_ids",
    "prompt_text_ids_mask",
    "prompt_text_ids_len",
    "te_proj_out",
    "backbone_inputs_embeds",
    "backbone_hidden_frame0",
    "backbone_logits_frame0",
    "dd_codes_frame0",
    "dd_codes_frame0_stepwise",
    "codes",
    "codec_audio",
]

# Acceptance thresholds, per the validation plan.
COS_THRESHOLD = 0.999
ARGMAX_STAGES = ("backbone_logits_frame0", "dd_logits_frame0_cb")


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def fixture_dir(download: bool = True) -> Path:
    """Locate the fixture directory, downloading it from HF if needed."""
    local = os.environ.get("BREEZE_FIXTURE_DIR")
    if local:
        p = Path(local)
        if not p.is_dir():
            raise FileNotFoundError(f"BREEZE_FIXTURE_DIR does not exist: {p}")
        return p
    if not download:
        raise FileNotFoundError(
            "no BREEZE_FIXTURE_DIR set and download disabled"
        )
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        FIXTURE_REPO,
        repo_type="dataset",
        allow_patterns=[f"{FIXTURE_PREFIX}/*"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    return Path(root) / FIXTURE_PREFIX


def load_fixtures(stages: Optional[Set[str]] = None,
                  download: bool = True) -> Dict[str, np.ndarray]:
    """Load every .npy in the fixture dir (optionally filtered by `stages`)."""
    d = fixture_dir(download=download)
    out: Dict[str, np.ndarray] = {}
    for p in sorted(d.glob("*.npy")):
        name = p.stem
        if stages and name not in stages:
            continue
        out[name] = np.load(p)
    if not out:
        raise FileNotFoundError(f"no .npy fixtures found in {d}")
    return out


def load_meta(download: bool = True) -> dict:
    p = fixture_dir(download=download) / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# tools/dump_reference.py plug-in contract
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 256) -> Dict[str, np.ndarray]:
    """Return the Kaggle-produced reference activations.

    `model_dir` and `audio` are ignored: the reference run is fixed (fixed
    seed, fixed text, samples/jfk.wav as the clone reference) and lives in
    the fixture dataset. Running the upstream locally is not possible on the
    8 GB VPS — see the module docstring.
    """
    del model_dir, audio, max_new_tokens  # fixed by the fixture
    want = set(stages) if stages else None
    fx = load_fixtures(stages=want)
    # Everything downstream wants float32; keep the integer stages as int32
    # so exact-equality checks stay meaningful.
    out: Dict[str, np.ndarray] = {}
    for k, v in fx.items():
        if _is_int_stage(k):
            out[k] = np.ascontiguousarray(v.astype(np.int32))
        else:
            out[k] = np.ascontiguousarray(v.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    dx, dy = np.linalg.norm(x), np.linalg.norm(y)
    if dx == 0.0 or dy == 0.0:
        return 1.0 if dx == dy else 0.0
    return float(np.dot(x, y) / (dx * dy))


def compare_stage(name: str, ref: np.ndarray, cpp: np.ndarray) -> dict:
    r = {"stage": name, "ref_shape": list(ref.shape), "cpp_shape": list(cpp.shape)}
    if ref.shape != cpp.shape:
        r["shape_mismatch"] = True
    if _is_int_stage(name):
        n = min(ref.size, cpp.size)
        eq = int(np.sum(ref.ravel()[:n] == cpp.ravel()[:n]))
        r["exact"] = eq == n and ref.shape == cpp.shape
        r["match_frac"] = eq / n if n else 0.0
        first_bad = np.flatnonzero(ref.ravel()[:n] != cpp.ravel()[:n])
        r["first_mismatch"] = int(first_bad[0]) if first_bad.size else None
        r["pass"] = bool(r["exact"])
        return r
    r["cos"] = cosine(ref, cpp)
    n = min(ref.size, cpp.size)
    d = np.abs(ref.astype(np.float64).ravel()[:n] - cpp.astype(np.float64).ravel()[:n])
    r["max_abs"] = float(d.max()) if n else 0.0
    r["mean_abs"] = float(d.mean()) if n else 0.0
    r["pass"] = r["cos"] >= COS_THRESHOLD and not r.get("shape_mismatch")
    if name.startswith(ARGMAX_STAGES):
        r["ref_argmax"] = int(np.argmax(ref))
        r["cpp_argmax"] = int(np.argmax(cpp))
        r["argmax_match"] = r["ref_argmax"] == r["cpp_argmax"]
        r["pass"] = bool(r["pass"] and r["argmax_match"])
    return r


def compare(cpp_dump: Path, download: bool = True,
            only: Optional[Set[str]] = None) -> int:
    """Diff a C++ dump directory against the reference fixtures.

    Returns a process exit code: 0 = every present stage passed.
    """
    ref = load_fixtures(download=download)
    meta = load_meta(download=download)
    if meta:
        print(f"fixture: seed={meta.get('seed')} greedy={meta.get('greedy')} "
              f"text={meta.get('syn_text')!r}")
        print(f"         template={meta.get('template')}")
    cpp_dump = Path(cpp_dump)
    if not cpp_dump.is_dir():
        raise NotADirectoryError(cpp_dump)

    rows, missing, failures = [], [], 0
    for name in sorted(ref):
        if only and name not in only:
            continue
        p = cpp_dump / f"{name}.npy"
        if not p.exists():
            missing.append(name)
            continue
        rows.append(compare_stage(name, ref[name], np.load(p)))

    width = max((len(r["stage"]) for r in rows), default=10)
    for r in rows:
        tag = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            failures += 1
        if "cos" in r:
            extra = ""
            if "argmax_match" in r:
                extra = (f"  argmax ref={r['ref_argmax']} cpp={r['cpp_argmax']}"
                         f" {'ok' if r['argmax_match'] else 'MISMATCH'}")
            print(f"[{tag}] {r['stage']:<{width}}  cos={r['cos']:.6f}  "
                  f"maxabs={r['max_abs']:.3e}  {tuple(r['ref_shape'])}{extra}")
        else:
            print(f"[{tag}] {r['stage']:<{width}}  exact={r['exact']}  "
                  f"match={r['match_frac']:.4f}  first_bad={r['first_mismatch']}  "
                  f"{tuple(r['ref_shape'])}")
        if r.get("shape_mismatch"):
            print(f"       shape mismatch: ref{tuple(r['ref_shape'])} "
                  f"vs cpp{tuple(r['cpp_shape'])}")

    if missing:
        print(f"\n{len(missing)} reference stage(s) not produced by the C++ dump: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    print(f"\n{len(rows) - failures}/{len(rows)} stages passed "
          f"(cos >= {COS_THRESHOLD}, exact ints, argmax match on logits)")
    return 1 if failures else 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare a CrispASR breeze-tts-2 dump against the Kaggle "
                    "reference fixtures")
    ap.add_argument("--cpp-dump", type=Path,
                    help="directory of C++-produced <stage>.npy files")
    ap.add_argument("--list", action="store_true",
                    help="list the fixture stages and their shapes, then exit")
    ap.add_argument("--stage", action="append", default=None,
                    help="restrict the comparison to this stage (repeatable)")
    ap.add_argument("--no-download", action="store_true",
                    help="require BREEZE_FIXTURE_DIR instead of hitting HF")
    args = ap.parse_args()

    download = not args.no_download
    if args.list or not args.cpp_dump:
        fx = load_fixtures(download=download)
        meta = load_meta(download=download)
        if meta:
            print(json.dumps({k: v for k, v in meta.items() if k != "shapes"},
                             indent=2))
        for k in sorted(fx):
            print(f"  {k:34s} {fx[k].shape}  {fx[k].dtype}")
        raise SystemExit(0)

    raise SystemExit(compare(args.cpp_dump, download=download,
                             only=set(args.stage) if args.stage else None))


if __name__ == "__main__":
    main()
