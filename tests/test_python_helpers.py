"""Constructor regressions for installed, overridden, and legacy libraries."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crispasr import _binding as binding


class HelperDiscoveryTests(unittest.TestCase):
    def check_constructor(self, system, mode):
        names = {
            "Linux": ("libcrispasr.so", "libcrispasr_helpers.so"),
            "Darwin": ("libcrispasr.dylib", "libcrispasr_helpers.dylib"),
            "Windows": ("crispasr.dll", "crispasr_helpers.dll"),
        }
        library_name, helper_name = names[system]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python" / "crispasr"
            package.mkdir(parents=True)
            library = package / library_name
            library.touch()
            adjacent = package / helper_name
            adjacent.touch()
            helper = adjacent
            kwargs = {}
            environment = {}
            if mode in ("explicit", "environment"):
                elsewhere = root / "native"
                elsewhere.mkdir()
                library = elsewhere / library_name
                library.touch()
                helper = elsewhere / helper_name
                helper.touch()
                if mode == "explicit":
                    kwargs["lib_path"] = str(library)
                else:
                    environment["CRISPASR_LIB_PATH"] = str(library)
            elif mode == "helper_override":
                helper = root / "custom-helper"
                helper.touch()
                kwargs["helpers_lib_path"] = str(helper)
            elif mode in ("build", "legacy"):
                adjacent.unlink()
                if mode == "build":
                    (root / "build").mkdir()
                    helper = root / "build" / helper_name
                    helper.touch()
                else:
                    helper = None

            native, helpers = Mock(), Mock()
            native.whisper_context_default_params_by_ref.return_value = 17
            native.whisper_init_from_file.return_value = 12
            helpers.whisper_init_from_file_ptr.return_value = 11
            loaded = []

            def load(path):
                loaded.append(str(path))
                if str(path) == str(library):
                    return native
                self.assertEqual(str(path), str(helper))
                return helpers

            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(binding, "__file__", str(package / "_binding.py")), \
                    patch.object(binding.platform, "system", return_value=system), \
                    patch.object(binding, "_register_dll_dir"), \
                    patch.object(binding.ctypes, "CDLL", side_effect=load):
                model = binding.CrispASR("model.gguf", **kwargs)
                if helper is not None:
                    self.assertEqual(loaded, [str(library), str(helper)])
                    helpers.whisper_init_from_file_ptr.assert_called_once_with(b"model.gguf", 17)
                    native.whisper_free_context_params.assert_called_once_with(17)
                    native.whisper_init_from_file.assert_not_called()
                    self.assertIs(model._helpers, helpers)
                    expected_context = 11
                else:
                    self.assertEqual(loaded, [str(library)])
                    native.whisper_init_from_file.assert_called_once_with(b"model.gguf")
                    self.assertIsNone(model._helpers)
                    expected_context = 12
                model.close()
                native.whisper_free.assert_called_once_with(expected_context)

    def test_constructor_routes(self):
        for system in ("Linux", "Darwin", "Windows"):
            for mode in ("automatic", "explicit", "environment", "helper_override", "build", "legacy"):
                with self.subTest(system=system, mode=mode):
                    self.check_constructor(system, mode)


if __name__ == "__main__":
    unittest.main()
