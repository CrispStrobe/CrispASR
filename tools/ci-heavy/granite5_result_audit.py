#!/usr/bin/env python3
"""Audit retained Granite5 outputs without loading or running model weights.

Exit status covers structural, numerical and normalized-content gates. Exact
token/text fidelity is reported separately for review, including for F16; a
quantization-induced greedy difference is not automatically a runtime bug.
This audit alone cannot establish model, CPU, CUDA or production acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


QUANTS = ("f16", "q8_0", "q4_k")


def words(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold())


def metrics(expected, actual):
    """The same gate is used for real outputs and negative controls."""
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: {expected.shape} != {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite logits")
    a, b = expected.astype(np.float64), actual.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        raise ValueError("zero logits norm")
    cosine = float(np.dot(a, b) / (na * nb))
    ratio = float(nb / na)
    return {"cosine": cosine, "norm_ratio": ratio,
            "relative_l2": float(np.linalg.norm(a - b) / na),
            "maxabs": float(np.max(np.abs(a - b))),
            "pass": cosine >= 0.99 and 0.99 <= ratio <= 1.01}


def rank(logits, pick):
    top = int(np.argmax(logits))
    return {"source_token": pick, "native_argmax": top,
            "source_token_rank": int(np.count_nonzero(logits > logits[pick])) + 1,
            "gap_to_native_max": float(float(logits[top]) - float(logits[pick])),
            "source_token_is_argmax": top == pick}


def token_comparison(expected, actual):
    common = min(len(expected), len(actual))
    differences = np.flatnonzero(expected[:common] != actual[:common])
    first = int(differences[0]) if len(differences) else (common if len(expected) != len(actual) else None)
    return {"exact": bool(np.array_equal(expected, actual)),
            "expected_count": len(expected), "actual_count": len(actual),
            "matching_prefix_count": common if first is None else first,
            "first_difference": first,
            "expected_token": int(expected[first]) if first is not None and first < len(expected) else None,
            "actual_token": int(actual[first]) if first is not None and first < len(actual) else None}


def text_comparison(expected, actual):
    return {"exact_stripped": actual.strip() == expected.strip(),
            "normalized_words_equal": words(actual) == words(expected),
            "text": actual}


def controls(expected):
    cases = {"identity": expected.copy(), "wrong_shape": expected[:-1],
             "twice_scale": expected * 2, "mismatched_values": -expected}
    nonfinite = expected.copy()
    nonfinite[0] = np.nan
    cases["nonfinite"] = nonfinite
    results = {}
    for name, actual in cases.items():
        try:
            passed = metrics(expected, actual)["pass"]
            results[name] = {"gate_pass": passed, "control_pass": passed == (name == "identity")}
        except ValueError as error:
            results[name] = {"gate_pass": False, "control_pass": name != "identity", "error": str(error)}
    if not all(result["control_pass"] for result in results.values()):
        raise ValueError("numerical negative controls failed")
    return results


def audit_case(directory, expected_logits, expected_ids, expected_text):
    def read(name, dtype):
        path = directory / name
        if path.stat().st_size % np.dtype(dtype).itemsize:
            raise ValueError(f"invalid byte length: {name}")
        return np.fromfile(path, dtype=dtype)

    result = {"directory": directory.name, "missing_checks": [], "check_errors": {}, "texts": {}}

    def check(name, operation):
        try:
            return operation()
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["check_errors"][name] = str(error).replace(str(directory), "${CASE}")
            if isinstance(error, FileNotFoundError):
                result["missing_checks"].append(name)
            return None

    def logits_check(name):
        logits = read(name, "<f4")
        numerical = metrics(expected_logits, logits)
        numerical.update(rank(logits, int(np.argmax(expected_logits))))
        return numerical

    result["teacher_prefill"] = check("teacher_prefill", lambda: logits_check("llm_prefill_logits.bin"))
    result["pipeline_prefill"] = check("pipeline_prefill", lambda: logits_check("pipeline_prefill_logits.bin"))

    def tokens(name):
        values = read(name, "<i4")
        if not len(values):
            raise ValueError("empty generated token sequence")
        return values

    bucket = check("bucketed_tokens", lambda: tokens("generated_ids.bin"))
    dynamic = check("dynamic_tokens", lambda: tokens("dynamic_generated_ids.bin"))
    result["original_vs_bucketed"] = token_comparison(expected_ids, bucket) if bucket is not None else None
    result["original_vs_dynamic"] = token_comparison(expected_ids, dynamic) if dynamic is not None else None
    paths_equal = bucket is not None and dynamic is not None and bool(np.array_equal(bucket, dynamic))
    result["bucketed_vs_dynamic_exact"] = paths_equal
    texts = result["texts"]
    texts["native"] = check("native_text", lambda: text_comparison(
        expected_text, (directory / "generated_text.txt").read_text()))
    def session_outputs():
        outputs = json.loads((directory / "session.json").read_text())
        if not isinstance(outputs, dict):
            raise ValueError("Session outputs must be a JSON object")
        return outputs

    sessions = check("session_outputs", session_outputs)

    def official_check():
        official = sessions["official"]
        if not isinstance(official, list) or len(official) != 2:
            raise ValueError("expected two official Session repetitions")
        return [text_comparison(expected_text, text) for text in official]

    if sessions is not None:
        texts["session_official"] = check("session_official", official_check)
        texts["session_legacy_default"] = check("session_legacy_default", lambda: text_comparison(
            expected_text, sessions["legacy_default"]))
    else:
        texts["session_official"] = None
        texts["session_legacy_default"] = None
    texts["cli"] = check("cli_text", lambda: text_comparison(
        expected_text, (directory / "cli-official.txt").read_text()))
    official_texts = [texts["native"], *(texts["session_official"] or []), texts["cli"]]
    all_texts = [*official_texts, texts["session_legacy_default"]]
    text_checks_complete = len(all_texts) == 5 and all(t is not None for t in all_texts)
    result["checks_complete"] = not result["check_errors"]
    result["all_text_exact_stripped"] = text_checks_complete and all(t["exact_stripped"] for t in all_texts)
    result["official_request_text_exact"] = (
        len(official_texts) == 4 and all(t is not None and t["exact_stripped"] for t in official_texts))
    result["legacy_default_note"] = "Different request: publisher's raw-transcript prompt, not the punctuation/case reference request"
    result["all_text_normalized_words_equal"] = (
        text_checks_complete and all(t["normalized_words_equal"] for t in all_texts))
    result["structural_numerical_content_pass"] = (
        result["checks_complete"] and result["teacher_prefill"]["pass"] and result["pipeline_prefill"]["pass"] and paths_equal
        and result["all_text_normalized_words_equal"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quants", nargs="+", choices=(*QUANTS, "q6_k"), default=QUANTS)
    args = parser.parse_args()
    from gguf import GGUFReader

    report = {"schema_version": 2, "model_free": True, "full_acceptance_established": False,
              "scope": "Retained frontend stages, teacher/pipeline logits, native decoder paths, and transcript fidelity",
              "fidelity_policy": "Exact token/case/punctuation differences require review; quantization can change greedy output",
              "reference": args.reference.name, "results_root": args.results_root.name,
              "requested_quants": list(args.quants), "cases": {}}
    try:
        report["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        ref = GGUFReader(args.reference)
        tensors = {t.name: t.data.reshape(-1).copy() for t in ref.tensors
                   if t.name in ("llm_prefill_logits", "llm_argmax", "prompt_ids", "mel_spectrogram",
                                 "encoder_out", "projector_out", "llm_inputs_embeds")}
        logits, ids, prompt = (tensors[name] for name in ("llm_prefill_logits", "llm_argmax", "prompt_ids"))
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(prompt.dtype, np.integer):
            raise ValueError("reference token tensors must be integer")
        if len(ids) <= len(prompt) or not np.array_equal(ids[:len(prompt)], prompt):
            raise ValueError("original generation does not begin with reference prompt")
        expected = ref.fields["crispasr.ref.generated_text"].contents()
        if not words(expected):
            raise ValueError("empty original transcript")
        report["numerical_controls"] = controls(logits)
        top = np.sort(logits.astype(np.float64))[-2:]
        report["original"] = {"text": expected, "prompt_tokens": len(prompt),
                              "generated_tokens": len(ids) - len(prompt),
                              "first_token": int(np.argmax(logits)),
                              "top_two_logit_margin": float(top[-1] - top[-2])}
        for quant in args.quants:
            try:
                report["cases"][quant] = audit_case(args.results_root / quant, logits, ids[len(prompt):], expected)
                report["cases"][quant]["quantized"] = quant != "f16"
                case = report["cases"][quant]
                case["stages"] = {}
                for name in ("mel_spectrogram", "encoder_out", "projector_out", "llm_inputs_embeds"):
                    try:
                        path = args.results_root / quant / (name + ".bin")
                        if path.stat().st_size != tensors[name].size * 4:
                            raise ValueError("stage byte length does not match reference")
                        actual = np.fromfile(path, dtype="<f4")
                        result = metrics(tensors[name], actual)
                        if name == "llm_inputs_embeds" and quant == "f16":
                            result["pass"] = result["pass"] and result["cosine"] >= .99999
                        case["stages"][name] = result
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        message = str(error).replace(str(args.results_root), "${RESULTS}")
                        case["stages"][name] = {"pass": False, "error": message}
                        case["check_errors"][name] = message
                        if isinstance(error, FileNotFoundError):
                            case["missing_checks"].append(name)
                ranks = [case["teacher_prefill"], case["pipeline_prefill"]]
                case["first_token_rank_pass"] = all(
                    value is not None and value["source_token_rank"] <= (1 if quant == "f16" else 2)
                    and value["gap_to_native_max"] < 1.0 for value in ranks)
                case["checks_complete"] = not case["check_errors"]
                case["structural_numerical_content_pass"] = (
                    case["structural_numerical_content_pass"] and case["checks_complete"]
                    and case["first_token_rank_pass"] and all(v["pass"] for v in case["stages"].values()))
            except (OSError, ValueError, KeyError, TypeError) as error:
                report["cases"][quant] = {"structural_numerical_content_pass": False, "error": str(error)}
        report["structural_numerical_content_pass"] = all(
            case["structural_numerical_content_pass"] for case in report["cases"].values())
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
        report["structural_numerical_content_pass"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Public receipts use logical input names rather than maintainer paths,
    # including error messages generated by the operating system.
    for case in report["cases"].values():
        if "error" in case:
            case["error"] = case["error"].replace(str(args.results_root), "${RESULTS_ROOT}")
    if "error" in report:
        report["error"] = report["error"].replace(str(args.reference), "${REFERENCE}")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Structural/numerical/content gates: {report['structural_numerical_content_pass']}; exact fidelity reported separately")
    return 0 if report["structural_numerical_content_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
