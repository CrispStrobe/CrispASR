#!/usr/bin/env python3
"""Unit tests for all generation-control session setters (PLAN §88 / commit c88306fa).

Two orthogonal test classes:

  TestSetterSymbols   — ctypes: every C-ABI symbol is exported and returns -1
                        on a null session handle.  No model, no network.
  TestBindingMethods  — imports python/crispasr and verifies the Session class
                        exposes all corresponding Python methods.

Run:
  python tests/test_session_setters.py
  pytest tests/test_session_setters.py -v
"""

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

LIB_PATH = os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for candidate in [
        os.path.join(REPO_ROOT, "build-ninja-compile", "src", "libwhisper.dylib"),
        os.path.join(REPO_ROOT, "build-ninja-compile", "src", "libwhisper.so"),
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.so"),
        os.path.join(REPO_ROOT, "build-shared", "src", "libwhisper.so"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break

# Each entry: (symbol_name, argtypes, call_args)
# All calls use a null handle (None) — expected return value is -1.
_SETTER_SPECS = [
    (
        "crispasr_session_set_temperature",
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_uint64],
        [None, 0.8, 42],
    ),
    (
        "crispasr_session_set_tts_seed",
        [ctypes.c_void_p, ctypes.c_uint64],
        [None, 12345],
    ),
    (
        "crispasr_session_set_tts_steps",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 20],
    ),
    (
        "crispasr_session_set_max_new_tokens",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 256],
    ),
    (
        "crispasr_session_set_frequency_penalty",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 0.4],
    ),
    (
        "crispasr_session_set_top_p",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 0.9],
    ),
    (
        "crispasr_session_set_min_p",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 0.05],
    ),
    (
        "crispasr_session_set_repetition_penalty",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 1.2],
    ),
    (
        "crispasr_session_set_cfg_weight",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 0.5],
    ),
    (
        "crispasr_session_set_exaggeration",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 0.5],
    ),
    (
        "crispasr_session_set_max_speech_tokens",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 1000],
    ),
    (
        "crispasr_session_set_length_scale",
        [ctypes.c_void_p, ctypes.c_float],
        [None, 1.0],
    ),
    (
        "crispasr_session_set_best_of",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 5],
    ),
    (
        "crispasr_session_set_beam_size",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 4],
    ),
    (
        "crispasr_session_set_alt_n",
        [ctypes.c_void_p, ctypes.c_int],
        [None, 3],
    ),
    (
        "crispasr_session_set_ask",
        [ctypes.c_void_p, ctypes.c_char_p],
        [None, b"hello"],
    ),
    (
        "crispasr_session_set_grammar_text",
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float],
        [None, None, None, 100.0],
    ),
    (
        "crispasr_session_set_fallback_thresholds",
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
        [None, 2.4, -1.0, 0.6, 0.2],
    ),
    (
        "crispasr_session_set_whisper_decode_extras",
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int],
        [None, 0, b"", 0],
    ),
]

# Python Session method names that must exist on the binding class.
_BINDING_METHODS = [
    "set_temperature",
    "set_tts_seed",
    "set_tts_steps",
    "set_max_new_tokens",
    "set_frequency_penalty",
    "set_top_p",
    "set_min_p",
    "set_repetition_penalty",
    "set_cfg_weight",
    "set_exaggeration",
    "set_max_speech_tokens",
    "set_length_scale",
    "set_best_of",
    "set_beam_size",
    "set_alt_n",
    "set_ask",
    "set_grammar_text",
    "set_fallback_thresholds",
    "set_whisper_decode_extras",
]


@unittest.skipUnless(LIB_PATH, "libwhisper/libcrispasr not built — set CRISPASR_LIB or build first")
class TestSetterSymbols(unittest.TestCase):
    """Every C-ABI setter must be exported and return -1 on a null handle."""

    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(LIB_PATH)

    def _call_setter(self, sym, argtypes, args):
        fn = getattr(self.lib, sym)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn(*args)

    def test_all_symbols_exported(self):
        for sym, _, _ in _SETTER_SPECS:
            self.assertTrue(
                hasattr(self.lib, sym),
                f"symbol not found in shared library: {sym}",
            )

    def test_temperature_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[0]), -1)

    def test_tts_seed_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[1]), -1)

    def test_tts_steps_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[2]), -1)

    def test_max_new_tokens_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[3]), -1)

    def test_frequency_penalty_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[4]), -1)

    def test_top_p_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[5]), -1)

    def test_min_p_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[6]), -1)

    def test_repetition_penalty_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[7]), -1)

    def test_cfg_weight_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[8]), -1)

    def test_exaggeration_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[9]), -1)

    def test_max_speech_tokens_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[10]), -1)

    def test_length_scale_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[11]), -1)

    def test_best_of_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[12]), -1)

    def test_beam_size_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[13]), -1)

    def test_alt_n_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[14]), -1)

    def test_ask_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[15]), -1)

    def test_grammar_text_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[16]), -1)

    def test_fallback_thresholds_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[17]), -1)

    def test_whisper_decode_extras_null_returns_neg1(self):
        self.assertEqual(self._call_setter(*_SETTER_SPECS[18]), -1)


class TestBindingMethods(unittest.TestCase):
    """The Python Session class must expose every setter as a callable method."""

    def setUp(self):
        try:
            from crispasr import Session
            self.Session = Session
        except ImportError as exc:
            self.skipTest(f"crispasr Python package not importable: {exc}")

    def test_all_methods_present(self):
        for method in _BINDING_METHODS:
            self.assertTrue(
                callable(getattr(self.Session, method, None)),
                f"Session.{method} not found or not callable",
            )

    def test_set_temperature_signature(self):
        import inspect
        sig = inspect.signature(self.Session.set_temperature)
        params = list(sig.parameters)
        self.assertIn("temperature", params)
        self.assertIn("seed", params)

    def test_set_grammar_text_has_defaults(self):
        import inspect
        sig = inspect.signature(self.Session.set_grammar_text)
        params = sig.parameters
        self.assertIn("root_rule", params)
        self.assertIn("penalty", params)
        self.assertIsNot(params["root_rule"].default, inspect.Parameter.empty)
        self.assertIsNot(params["penalty"].default, inspect.Parameter.empty)

    def test_set_fallback_thresholds_signature(self):
        import inspect
        sig = inspect.signature(self.Session.set_fallback_thresholds)
        params = list(sig.parameters)
        self.assertIn("entropy_thold", params)
        self.assertIn("logprob_thold", params)
        self.assertIn("no_speech_thold", params)
        self.assertIn("temperature_inc", params)

    def test_set_whisper_decode_extras_defaults(self):
        import inspect
        sig = inspect.signature(self.Session.set_whisper_decode_extras)
        params = sig.parameters
        for name in ("suppress_nst", "suppress_regex", "carry_initial_prompt"):
            self.assertIn(name, params, f"missing param: {name}")
            self.assertIsNot(
                params[name].default, inspect.Parameter.empty,
                f"param {name} has no default",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
