import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


class TestBackendConfig(unittest.TestCase):
    def test_backend_gpu_helper_exists(self) -> None:
        text = read("examples/cli/crispasr_backend_utils.h")
        self.assertIn("inline bool crispasr_backend_should_use_gpu", text)
        self.assertIn('params.gpu_backend != "cpu"', text)

    def test_wrappers_use_shared_gpu_helper(self) -> None:
        wrappers = [
            "examples/cli/crispasr_backend_canary.cpp",
            "examples/cli/crispasr_backend_cohere.cpp",
            "examples/cli/crispasr_backend_firered_asr.cpp",
            "examples/cli/crispasr_backend_glm_asr.cpp",
            "examples/cli/crispasr_backend_granite.cpp",
            "examples/cli/crispasr_backend_kyutai_stt.cpp",
            "examples/cli/crispasr_backend_omniasr.cpp",
            "examples/cli/crispasr_backend_parakeet.cpp",
            "examples/cli/crispasr_backend_qwen3.cpp",
            "examples/cli/crispasr_backend_voxtral.cpp",
            "examples/cli/crispasr_backend_voxtral4b.cpp",
        ]
        for relpath in wrappers:
            text = read(relpath)
            self.assertIn('#include "crispasr_backend_utils.h"', text, relpath)
            self.assertIn("crispasr_backend_should_use_gpu(", text, relpath)
            self.assertNotIn('use_gpu && p.gpu_backend != "cpu"', text, relpath)
            self.assertNotIn('use_gpu && params.gpu_backend != "cpu"', text, relpath)

    def test_glm_encoder_frame_helper_is_declared_and_used(self) -> None:
        header = read("src/glm_asr.h")
        impl = read("src/glm_asr.cpp")

        self.assertIn("int glm_asr_encoder_frames_from_mel_frames(int T_mel);", header)
        self.assertIn("int glm_asr_encoder_frames_from_mel_frames(int T_mel)", impl)
        self.assertIn("return (T_mel + 1) / 2;", impl)
        self.assertEqual(impl.count("glm_asr_encoder_frames_from_mel_frames(T_mel)"), 2)
        self.assertIn("if (T_mel != T_target) {", impl)
        self.assertIn("const int T_copy = std::min(T_mel, T_target);", impl)
        self.assertIn("const int T_pack = T_proj * 4;", impl)

    def test_cli_only_loads_all_backends_when_gpu_is_enabled(self) -> None:
        text = read("examples/cli/cli.cpp")
        self.assertIn('if (params.use_gpu && params.gpu_backend != "cpu") {', text)
        self.assertIn("ggml_backend_load_all();", text)

    def test_omniasr_keeps_explicit_cpu_fallback_backend(self) -> None:
        text = read("src/omniasr.cpp")
        self.assertIn("ctx->backend_cpu = ggml_backend_cpu_init();", text)
        self.assertIn("if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)", text)
        self.assertIn("ggml_backend_sched_new(backends, nullptr, n_backends, 65536, false, false);", text)

    def test_backend_default_params_enable_gpu(self) -> None:
        expected = {
            "src/canary.cpp": "p.use_gpu = true;",
            "src/cohere.cpp": ".use_gpu = true",
            "src/firered_asr.cpp": "/*use_gpu=*/true",
            "src/glm_asr.cpp": "/*use_gpu=*/true",
            "src/granite_speech.cpp": "/*use_gpu=*/true",
            "src/kyutai_stt.cpp": "/*use_gpu=*/true",
            "src/omniasr.cpp": "p.use_gpu = true;",
            "src/parakeet.cpp": "p.use_gpu = true;",
            "src/qwen3_asr.cpp": "p.use_gpu = true;",
            "src/voxtral.cpp": "p.use_gpu = true;",
            "src/voxtral4b.cpp": "/*use_gpu=*/true",
        }
        for relpath, needle in expected.items():
            self.assertIn(needle, read(relpath), relpath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
