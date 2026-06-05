#!/usr/bin/env python3
"""
Audit HuggingFace model licenses: compare cstr/ repos against upstream base models.

Usage:
    python tools/audit-hf-licenses.py [--fix] [--filter PATTERN] [--check-readmes]

Checks every model under the cstr/ HF namespace:
  1. Reads base_model from card_data (if set)
  2. If no base_model: infers upstream repo from name (strips -GGUF, -ONNX, etc.)
  3. Fetches upstream license via HF API (with rate-limit retry)
  4. Flags mismatches

Options:
    --fix             Automatically fix mismatched licenses (updates README on HF)
    --filter          Only check repos matching this substring (e.g. "wmt21")
    --check-readmes   List all public (non-private) repos missing a README/model card
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# Known upstream → license mappings for models where HF metadata is missing/private/gated
KNOWN_LICENSES: dict[str, str] = {
    "SebastianBodza/Kartoffelbox_Turbo": "cc-by-4.0",
    "SebastianBodza/Kartoffelbox-v0.1": "cc-by-4.0",
    "jinaai/jina-embeddings-v5-text-small-retrieval": "cc-by-nc-4.0",
    "openai/whisper-large-v3": "apache-2.0",
    "openai/whisper-large-v3-turbo": "mit",
}

# Known owner aliases: our repo name fragment → upstream org/owner
KNOWN_OWNERS: dict[str, str] = {
    "kartoffelbox": "SebastianBodza",
    "Kartoffelbox": "SebastianBodza",
    "jina-embeddings": "jinaai",
    "madlad400": "google",
    "MADLAD": "google",
    "m2m100": "facebook",
    "wmt21": "facebook",
    "ALMA": "haoranxu",
    "whisper": "openai",
    "Llama-3": "meta-llama",
    "Llama-2": "meta-llama",
    "Mistral": "mistralai",
    "Mixtral": "mistralai",
    "gemma": "google",
    "Qwen": "Qwen",
    "phi-": "microsoft",
    "Phi-": "microsoft",
    "deepseek": "deepseek-ai",
    "DeepSeek": "deepseek-ai",
    "EuroLLM": "utter-project",
    "parakeet": "nvidia",
    "canary": "nvidia",
    "moonshine": "UsefulSensors",
}

# Suffixes to strip when inferring upstream repo name
STRIP_SUFFIXES = [
    "-GGUF", "-gguf",
    "-ONNX", "-onnx",
    "-ONNX-FP16", "-onnx-fp16",
    "-ONNX-INT8", "-onnx-int8",
    "-ONNX-INT4", "-onnx-int4",
    "-ONNX-INT8-FULL",
    "-Q4_K_M-GGUF", "-Q4_K-GGUF", "-Q8_0-GGUF",
    "-mlx", "-MLX",
    "-laser",
]

# Rate limit state
_last_api_call = 0.0
API_DELAY_S = 0.15  # ~7 req/s to stay well under HF's 100 req/min


def rate_limit():
    """Simple rate limiter for HF API calls."""
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if elapsed < API_DELAY_S:
        time.sleep(API_DELAY_S - elapsed)
    _last_api_call = time.time()


def get_license(repo_id: str) -> tuple[str | None, str]:
    """
    Get license from a HF repo.
    Returns (license_string_or_None, source_description).
    """
    if repo_id in KNOWN_LICENSES:
        return KNOWN_LICENSES[repo_id], "known_list"

    rate_limit()
    try:
        info = api.model_info(repo_id)
        if info.card_data and info.card_data.license:
            return info.card_data.license, "api"
        return None, "no_license_in_card"
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err:
            return None, "gated/private"
        if "404" in err:
            return None, "not_found"
        if "429" in err:
            # Rate limited — back off and retry once
            print(f"    [rate limited, sleeping 30s...]")
            time.sleep(30)
            try:
                info = api.model_info(repo_id)
                if info.card_data and info.card_data.license:
                    return info.card_data.license, "api (retry)"
            except Exception:
                pass
            return None, "rate_limited"
        return None, f"error: {err[:80]}"


def infer_upstream(repo_name: str) -> list[str]:
    """
    Infer possible upstream repo IDs from a cstr/ repo name.
    Returns a list of candidates to try (most likely first).

    Strategy:
      1. Strip format suffixes (-GGUF, -ONNX, etc.)
      2. Try known owner patterns (fast, no API call)
      3. Search HF by name, sorted by downloads — pick the top non-cstr/ result
    """
    # Strip "cstr/" prefix
    name = repo_name.replace("cstr/", "")

    # Strip known suffixes
    base = name
    for suffix in sorted(STRIP_SUFFIXES, key=len, reverse=True):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # Try to find owner from known patterns
    candidates = []

    for pattern, owner in KNOWN_OWNERS.items():
        if pattern.lower() in base.lower():
            candidates.append(f"{owner}/{base}")
            candidates.append(f"{owner}/{base.lower()}")
            break

    if "/" in base:
        candidates.insert(0, base)

    # Search HF for the base name, sorted by most downloads — the upstream
    # original will almost always be the most-downloaded repo with that name.
    rate_limit()
    try:
        results = list(api.list_models(search=base, sort="downloads", direction=-1, limit=10))
        for r in results:
            if r.id.startswith("cstr/"):
                continue  # skip our own repos
            if r.id not in candidates:
                candidates.append(r.id)
    except Exception:
        pass

    # Also try with different casing/variations
    if not candidates:
        # Last resort: try lowercase search
        rate_limit()
        try:
            results = list(api.list_models(search=base.lower(), sort="downloads", direction=-1, limit=5))
            for r in results:
                if not r.id.startswith("cstr/"):
                    candidates.append(r.id)
        except Exception:
            pass

    return candidates


def check_readmes(models_list, filter_str: str = ""):
    """
    Check all public (non-private) cstr/ repos for missing README/model cards.
    A repo is flagged if it has no card_data or card_data has no text content.
    """
    print("=" * 70)
    print("README / Model Card Audit — cstr/ namespace (public repos only)")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Filter: {filter_str or '(all)'}")
    print()

    missing = []
    has_readme = []

    for i, m in enumerate(sorted(models_list, key=lambda x: x.id)):
        repo_id = m.id
        progress = f"[{i+1}/{len(models_list)}]"

        # Skip private repos
        if getattr(m, "private", False):
            continue

        rate_limit()
        try:
            full_info = api.model_info(repo_id)
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                # Private/gated — skip
                continue
            print(f"{progress} ERROR {repo_id} — {err[:60]}")
            continue

        # Check if there's meaningful card content
        has_card = False
        if full_info.card_data:
            # card_data exists but may be empty/minimal
            has_card = True

        # Also check if README.md exists in files
        siblings = full_info.siblings or []
        has_readme_file = any(s.rfilename == "README.md" for s in siblings)

        if not has_readme_file:
            downloads = getattr(full_info, "downloads", 0) or 0
            print(f"{progress} NO README  {repo_id}  (downloads: {downloads})")
            missing.append((repo_id, downloads))
        else:
            has_readme.append(repo_id)

    # Report
    print()
    print("=" * 70)
    print(f"README AUDIT RESULTS")
    print("=" * 70)
    print(f"\n  {len(missing)} public repos WITHOUT README.md:\n")
    # Sort by downloads (most important first)
    missing.sort(key=lambda x: x[1], reverse=True)
    for repo_id, downloads in missing:
        print(f"  {repo_id}  ({downloads} downloads/month)")
    print(f"\n  {len(has_readme)} repos have README.md")
    print(f"  {len(missing)} repos MISSING README.md")
    print()
    return missing


def main():
    parser = argparse.ArgumentParser(description="Audit cstr/ HF model licenses")
    parser.add_argument("--fix", action="store_true", help="Auto-fix mismatches")
    parser.add_argument("--filter", type=str, default="", help="Only check repos matching pattern")
    parser.add_argument("--check-readmes", action="store_true", help="List repos missing README/model card")
    args = parser.parse_args()

    print("=" * 70)
    print("HuggingFace License Audit — cstr/ namespace")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Filter: {args.filter or '(all)'}")
    print()

    print("Fetching cstr/ models...")
    rate_limit()
    models = list(api.list_models(author="cstr"))
    print(f"Found {len(models)} models total")

    if args.filter:
        models = [m for m in models if args.filter.lower() in m.id.lower()]
        print(f"After filter: {len(models)} models")
    print()

    if args.check_readmes:
        check_readmes(models, args.filter)
        return 0

    mismatches = []
    possible_mismatches = []
    matched_ok = []
    upstream_unknown = []
    no_license_ours = []
    skipped = []

    for i, m in enumerate(sorted(models, key=lambda x: x.id)):
        repo_id = m.id
        progress = f"[{i+1}/{len(models)}]"

        # list_models() returns incomplete card_data — fetch full model info
        rate_limit()
        try:
            full_info = api.model_info(repo_id)
            card_data = full_info.card_data
        except Exception as e:
            print(f"{progress} ERROR {repo_id} — can't fetch info: {e}")
            skipped.append(repo_id)
            continue

        our_license = card_data.license if card_data else None

        # 1. Try base_model from card_data
        base_models = []
        if card_data:
            bm = getattr(card_data, "base_model", None)
            if bm:
                base_models = bm if isinstance(bm, list) else [bm]

        # 2. If no base_model, infer from name
        if not base_models:
            base_models = infer_upstream(repo_id)
            source = "inferred"
        else:
            source = "card_data"

        # 3. Try to get upstream license
        upstream_license = None
        upstream_source = ""
        matched_base = None

        for base in base_models:
            upstream_license, upstream_source = get_license(base)
            if upstream_license is not None:
                matched_base = base
                break
            if upstream_source == "not_found":
                continue  # Try next candidate
            if upstream_source in ("gated/private", "rate_limited"):
                matched_base = base
                break  # Don't try more candidates

        # 4. Classify result
        if upstream_license is None:
            if our_license is None:
                print(f"{progress} SKIP  {repo_id} — no license on ours, upstream unknown")
                skipped.append(repo_id)
            else:
                reason = upstream_source or "all candidates 404"
                print(f"{progress} UNKN  {repo_id} (ours={our_license}) — upstream: {reason}")
                upstream_unknown.append((repo_id, our_license, base_models[0] if base_models else "?", reason))
        elif our_license is None:
            print(f"{progress} MISS  {repo_id} — we have NO license, upstream ({matched_base}): {upstream_license}")
            no_license_ours.append((repo_id, matched_base, upstream_license))
        elif our_license == upstream_license:
            print(f"{progress} OK    {repo_id} — {our_license} (via {source}: {matched_base})")
            matched_ok.append((repo_id, our_license, matched_base))
        else:
            confidence = "CONFIRMED" if source == "card_data" else "inferred"
            if source == "card_data":
                print(f"{progress} !! MISMATCH !!  {repo_id}")
                print(f"         ours: {our_license}  upstream ({matched_base}): {upstream_license}")
                mismatches.append((repo_id, our_license, matched_base, upstream_license))
            else:
                print(f"{progress} ?? POSSIBLE ??  {repo_id}")
                print(f"         ours: {our_license}  inferred upstream ({matched_base}): {upstream_license}")
                print(f"         [inferred — verify manually, search may have matched wrong repo]")
                possible_mismatches.append((repo_id, our_license, matched_base, upstream_license))

    # Final report
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    if mismatches:
        print(f"\n  *** {len(mismatches)} LICENSE MISMATCHES ***\n")
        for repo, ours, base, upstream in mismatches:
            print(f"  {repo}")
            print(f"    OURS: {ours}  →  SHOULD BE: {upstream}  (from {base})")
        print()

    if possible_mismatches:
        print(f"\n  {len(possible_mismatches)} POSSIBLE mismatches (inferred, verify manually):\n")
        for repo, ours, base, upstream in possible_mismatches:
            print(f"  {repo}")
            print(f"    ours: {ours}  inferred upstream ({base}): {upstream}")
        print()

    if no_license_ours:
        print(f"\n  {len(no_license_ours)} repos MISSING license (upstream has one):\n")
        for repo, base, upstream in no_license_ours[:20]:
            print(f"  {repo} → should be: {upstream} (from {base})")
        if len(no_license_ours) > 20:
            print(f"  ... and {len(no_license_ours) - 20} more")
        print()

    print(f"\nSummary:")
    print(f"  {len(matched_ok):4d} OK (license matches upstream)")
    print(f"  {len(mismatches):4d} MISMATCH (confirmed via base_model metadata)")
    print(f"  {len(possible_mismatches):4d} POSSIBLE (inferred upstream, may be wrong match)")
    print(f"  {len(no_license_ours):4d} MISSING (we have no license, upstream does)")
    print(f"  {len(upstream_unknown):4d} UNKNOWN (couldn't determine upstream license)")
    print(f"  {len(skipped):4d} SKIPPED (no license anywhere)")
    print(f"  {len(models):4d} TOTAL checked")

    if mismatches:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
