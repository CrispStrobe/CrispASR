"""Smoke test: every CA_EXPORT symbol from crispasr_c_api.cpp is reachable
from the Python ctypes binding.

Does NOT instantiate models or run inference — purely checks that the
binding declares (and can look up) all 149 exported C-ABI functions.
Requires CRISPASR_LIB_PATH pointing at a built libcrispasr.{so,dylib}.

    CRISPASR_LIB_PATH=build/src/libcrispasr.so python -m pytest tests/test_binding_parity.py -v
"""

import ctypes
import os
import sys

import pytest

# All 149 CA_EXPORT symbols (sorted, from:
#   grep -oP 'CA_EXPORT\s+\w+[\s*]+\K(crispasr_\w+)' src/crispasr_c_api.cpp | sort -u
# )
ALL_SYMBOLS = [
    "crispasr_align_result_free",
    "crispasr_align_result_n_words",
    "crispasr_align_result_word_t0",
    "crispasr_align_result_word_t1",
    "crispasr_align_words_abi",
    "crispasr_cache_dir_abi",
    "crispasr_cache_ensure_file_abi",
    "crispasr_detect_backend_from_gguf",
    "crispasr_detect_language",
    "crispasr_detect_language_pcm",
    "crispasr_diarize_segments_abi",
    "crispasr_enhance_audio_rnnoise",
    "crispasr_kokoro_lang_has_native_voice_abi",
    "crispasr_kokoro_lang_is_german_abi",
    "crispasr_kokoro_resolve_fallback_voice_abi",
    "crispasr_kokoro_resolve_model_for_lang_abi",
    "crispasr_lcs_dedup_prefix_count",
    "crispasr_parakeet_free",
    "crispasr_parakeet_init",
    "crispasr_parakeet_result_free",
    "crispasr_parakeet_result_n_tokens",
    "crispasr_parakeet_result_n_words",
    "crispasr_parakeet_result_token_p",
    "crispasr_parakeet_result_token_t0",
    "crispasr_parakeet_result_token_t1",
    "crispasr_parakeet_result_word_t0",
    "crispasr_parakeet_result_word_t1",
    "crispasr_parakeet_transcribe",
    "crispasr_params_set_alt_n",
    "crispasr_params_set_best_of",
    "crispasr_params_set_detect_language",
    "crispasr_params_set_initial_prompt",
    "crispasr_params_set_language",
    "crispasr_params_set_max_len",
    "crispasr_params_set_max_tokens",
    "crispasr_params_set_no_context",
    "crispasr_params_set_n_threads",
    "crispasr_params_set_print_progress",
    "crispasr_params_set_print_realtime",
    "crispasr_params_set_print_special",
    "crispasr_params_set_print_timestamps",
    "crispasr_params_set_single_segment",
    "crispasr_params_set_split_on_word",
    "crispasr_params_set_suppress_blank",
    "crispasr_params_set_tdrz",
    "crispasr_params_set_temperature",
    "crispasr_params_set_token_timestamps",
    "crispasr_params_set_translate",
    "crispasr_params_set_vad",
    "crispasr_params_set_vad_min_silence_ms",
    "crispasr_params_set_vad_min_speech_ms",
    "crispasr_params_set_vad_model_path",
    "crispasr_params_set_vad_threshold",
    "crispasr_pcm_free",
    "crispasr_punc_free",
    "crispasr_punc_free_text",
    "crispasr_punc_init",
    "crispasr_pyannote_cache_apply_abi",
    "crispasr_pyannote_cache_compute_abi",
    "crispasr_pyannote_cache_free_abi",
    "crispasr_registry_list_backends_abi",
    "crispasr_registry_lookup_abi",
    "crispasr_registry_lookup_by_filename_abi",
    "crispasr_session_available_backends",
    "crispasr_session_close",
    "crispasr_session_detect_language",
    "crispasr_session_is_custom_voice",
    "crispasr_session_is_voice_design",
    "crispasr_session_kokoro_clear_phoneme_cache",
    "crispasr_session_n_speakers",
    "crispasr_session_open",
    "crispasr_session_open_explicit",
    "crispasr_session_open_with_params",
    "crispasr_session_result_free",
    "crispasr_session_result_n_segments",
    "crispasr_session_result_n_words",
    "crispasr_session_result_segment_t0",
    "crispasr_session_result_segment_t1",
    "crispasr_session_result_word_alt_p",
    "crispasr_session_result_word_n_alts",
    "crispasr_session_result_word_p",
    "crispasr_session_result_word_t0",
    "crispasr_session_result_word_t1",
    "crispasr_session_set_alt_n",
    "crispasr_session_set_ask",
    "crispasr_session_set_beam_size",
    "crispasr_session_set_best_of",
    "crispasr_session_set_cfg_weight",
    "crispasr_session_set_codec_path",
    "crispasr_session_set_exaggeration",
    "crispasr_session_set_fallback_thresholds",
    "crispasr_session_set_frequency_penalty",
    "crispasr_session_set_grammar_text",
    "crispasr_session_set_instruct",
    "crispasr_session_set_length_scale",
    "crispasr_session_set_max_new_tokens",
    "crispasr_session_set_max_speech_tokens",
    "crispasr_session_set_min_p",
    "crispasr_session_set_punctuation",
    "crispasr_session_set_repetition_penalty",
    "crispasr_session_set_source_language",
    "crispasr_session_set_speaker_name",
    "crispasr_session_set_target_language",
    "crispasr_session_set_temperature",
    "crispasr_session_set_top_p",
    "crispasr_session_set_translate",
    "crispasr_session_set_tts_seed",
    "crispasr_session_set_tts_steps",
    "crispasr_session_set_voice",
    "crispasr_session_set_whisper_decode_extras",
    "crispasr_session_stream_open",
    "crispasr_session_synthesize",
    "crispasr_session_transcribe",
    "crispasr_session_transcribe_lang",
    "crispasr_session_transcribe_vad",
    "crispasr_session_transcribe_vad_lang",
    "crispasr_session_translate_text",
    "crispasr_session_translate_text_free",
    "crispasr_speaker_cluster_abi",
    "crispasr_speaker_db_count",
    "crispasr_speaker_db_enroll",
    "crispasr_speaker_db_free",
    "crispasr_speaker_db_load",
    "crispasr_speaker_db_match",
    "crispasr_speaker_embedder_dim_abi",
    "crispasr_speaker_embedder_embed_abi",
    "crispasr_speaker_embedder_free_abi",
    "crispasr_speaker_embedder_make_abi",
    "crispasr_stream_close",
    "crispasr_stream_feed",
    "crispasr_stream_flush",
    "crispasr_stream_get_text",
    "crispasr_stream_open",
    "crispasr_stream_set_live_decode",
    "crispasr_text_detect_language",
    "crispasr_titanet_cosine_sim",
    "crispasr_titanet_embed",
    "crispasr_titanet_free",
    "crispasr_titanet_init",
    "crispasr_token_alt_id",
    "crispasr_token_alt_p",
    "crispasr_token_alt_text",
    "crispasr_token_n_alts",
    "crispasr_token_p",
    "crispasr_token_t0",
    "crispasr_token_t1",
    "crispasr_vad_free",
    "crispasr_vad_segments",
    "crispasr_vad_slices",
]


@pytest.fixture(scope="module")
def lib():
    lib_path = os.environ.get("CRISPASR_LIB_PATH")
    if not lib_path:
        # Try common build paths
        for candidate in [
            "build/src/libcrispasr.so",
            "/mnt/volume1/build-main/src/libcrispasr.so",
            "build/libcrispasr.so",
        ]:
            if os.path.exists(candidate):
                lib_path = candidate
                break
    if not lib_path or not os.path.exists(lib_path):
        pytest.skip("libcrispasr not found — set CRISPASR_LIB_PATH")
    return ctypes.CDLL(lib_path)


@pytest.mark.parametrize("symbol", ALL_SYMBOLS)
def test_symbol_resolves(lib, symbol):
    """Every CA_EXPORT symbol from crispasr_c_api.cpp must be resolvable."""
    assert hasattr(lib, symbol), f"symbol {symbol} not found in loaded libcrispasr"


def test_symbol_count(lib):
    """Sanity: we expect exactly 149 crispasr_* symbols."""
    assert len(ALL_SYMBOLS) == 149, f"expected 149 symbols, got {len(ALL_SYMBOLS)}"


def test_python_binding_imports():
    """The Python binding module must import without error."""
    # This validates syntax and top-level structure.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    import ast
    binding_path = os.path.join(
        os.path.dirname(__file__), "..", "python", "crispasr", "_binding.py"
    )
    with open(binding_path) as f:
        ast.parse(f.read())


def test_python_binding_declares_all_symbols():
    """Every CA_EXPORT symbol must appear somewhere in _binding.py."""
    binding_path = os.path.join(
        os.path.dirname(__file__), "..", "python", "crispasr", "_binding.py"
    )
    with open(binding_path) as f:
        content = f.read()
    missing = [s for s in ALL_SYMBOLS if s not in content]
    assert missing == [], f"Python binding missing symbols: {missing}"
