"""Model-free failure controls for streamed Kaggle output recovery."""
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


path = Path(__file__).resolve().parents[1] / "tools/kaggle/recover_output.py"
spec = importlib.util.spec_from_file_location("recover_output", path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
URL = "https://storage.example/artifact?signature=private-signature"
PAYLOAD = b"first-second-third"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Response:
    def __init__(self, chunks=None, status=200, headers=None):
        self.chunks = chunks if chunks is not None else [b"first-", b"", b"second-", b"third"]
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    @property
    def content(self):
        raise AssertionError("must never buffer response.content")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, chunk_size):
        if chunk_size != 1024 * 1024:
            raise AssertionError("expected bounded chunk size")
        yield from self.chunks


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.destination = self.root / "artifact.tar"
        self.destination.write_bytes(b"existing-good-output")

    def run_download(self, response, **kwargs):
        get = Mock(return_value=response)
        result = recovery.stream_artifact(URL, self.destination, len(PAYLOAD), SHA, get=get, **kwargs)
        get.assert_called_once_with(URL, stream=True, timeout=(30, 120), allow_redirects=False,
                                    headers={"Accept-Encoding": "identity"})
        return result

    def assert_preserved(self):
        self.assertEqual(self.destination.read_bytes(), b"existing-good-output")
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_streams_verifies_then_replaces_atomically_with_unique_sibling(self):
        response = Response(headers={"Content-Length": str(len(PAYLOAD))})
        original_replace = os.replace
        def replace(source, destination):
            self.assertEqual(Path(source).parent, self.root)
            self.assertNotEqual(Path(source), self.destination)
            self.assertTrue(str(source).endswith(".part"))
            self.assertEqual(self.destination.read_bytes(), b"existing-good-output")
            self.assertEqual(Path(source).read_bytes(), PAYLOAD)
            original_replace(source, destination)
        with patch.object(recovery.os, "replace", side_effect=replace) as publish:
            self.assertEqual(self.run_download(response), (len(PAYLOAD), SHA))
        publish.assert_called_once()
        self.assertTrue(response.closed)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_zero_byte_placeholder_and_unrelated_partial_are_preserved_until_publish(self):
        self.destination.write_bytes(b"")
        other = self.root / ".artifact.tar.unrelated.part"
        other.write_bytes(b"unrelated")
        self.run_download(Response())
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(other.read_bytes(), b"unrelated")
        self.assertEqual(set(self.root.iterdir()), {self.destination, other})

    def test_failed_new_destination_does_not_leave_placeholder(self):
        self.destination.unlink()
        with self.assertRaises(recovery.RecoveryError):
            self.run_download(Response(chunks=[b"short"]))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_publishes_when_destination_absent_and_sha_uppercase(self):
        self.destination.unlink()
        recovery.stream_artifact(URL, self.destination, len(PAYLOAD), SHA.upper(), get=Mock(return_value=Response()))
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_rejects_http_errors_redirects_partial_and_no_content(self):
        for status in (206, 204, 301, 302, 403, 404, 500):
            with self.subTest(status=status), self.assertRaisesRegex(recovery.RecoveryError, "requires HTTP 200"):
                self.run_download(Response(status=status))
            self.assert_preserved()

    def test_rejects_incorrect_or_malformed_length_and_encoding(self):
        for headers in ({"Content-Length": "0"}, {"Content-Length": "unknown"},
                        {"Content-Encoding": "gzip"}):
            with self.subTest(headers=headers), self.assertRaises(recovery.RecoveryError):
                self.run_download(Response(headers=headers))
            self.assert_preserved()

    def test_rejects_truncated_oversized_and_wrong_hash_payloads(self):
        for chunks, message in (([PAYLOAD[:-1]], "byte count mismatch"),
                                ([PAYLOAD, b"extra"], "exceeds expected bytes"),
                                ([b"x" * len(PAYLOAD)], "SHA-256 mismatch")):
            with self.subTest(message=message), self.assertRaisesRegex(recovery.RecoveryError, message):
                self.run_download(Response(chunks=chunks))
            self.assert_preserved()

    def test_interrupted_stream_closes_response_removes_partial_and_redacts_error(self):
        def chunks():
            yield b"first-"
            raise OSError(URL + " private-owner")
        response = Response(chunks=chunks())
        with self.assertRaises(recovery.RecoveryError) as caught:
            self.run_download(response)
        self.assertNotIn("private", str(caught.exception))
        self.assertTrue(response.closed)
        self.assert_preserved()

    def test_request_failure_is_redacted(self):
        with self.assertRaisesRegex(recovery.RecoveryError, "download failed") as caught:
            recovery.stream_artifact(URL, self.destination, len(PAYLOAD), SHA,
                                     get=Mock(side_effect=RuntimeError(URL)))
        self.assertNotIn("signature", str(caught.exception))
        self.assert_preserved()

    def test_flush_and_publish_failures_preserve_destination_and_cleanup(self):
        for name in ("fsync", "replace"):
            with self.subTest(name=name), patch.object(recovery.os, name, side_effect=OSError("private-path")):
                with self.assertRaises(recovery.RecoveryError) as caught:
                    self.run_download(Response())
            self.assertNotIn("private", str(caught.exception))
            self.assert_preserved()

    def test_invalid_inputs_fail_before_request(self):
        for url, size, sha in (("http://storage.example/file", len(PAYLOAD), SHA),
                               ("https://user:secret@storage.example/file", len(PAYLOAD), SHA),
                               (URL, -1, SHA), (URL, len(PAYLOAD), "bad")):
            get = Mock()
            with self.subTest(url=url, size=size, sha=sha), self.assertRaises(recovery.RecoveryError):
                recovery.stream_artifact(url, self.destination, size, sha, get=get)
            get.assert_not_called()
            self.assert_preserved()

    def test_keyboard_interrupt_also_cleans_partial(self):
        def chunks():
            yield b"first-"
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_download(Response(chunks=chunks()))
        self.assert_preserved()


class ListingTests(unittest.TestCase):
    def list_with(self, pages, artifact="nested/artifact.tar"):
        endpoint = Mock(side_effect=pages)
        client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
            list_kernel_session_output=endpoint)))
        api = Mock()
        api.build_kaggle_client.return_value = contextlib.nullcontext(client)
        result = recovery.find_artifact(api, SimpleNamespace, "private-owner", "kernel", artifact)
        return result, endpoint.call_args_list

    @staticmethod
    def page(names, token=""):
        return SimpleNamespace(files=[SimpleNamespace(file_name=name, url=URL) for name in names],
                               next_page_token=token)

    def test_exact_selection_on_later_page_uses_latest_session(self):
        url, calls = self.list_with([self.page(["artifact.tar", "nested/artifact.tar.bak"], "next"),
                                     self.page(["nested/artifact.tar"])])
        self.assertEqual(url, URL)
        self.assertEqual(len(calls), 2)
        for index, call in enumerate(calls):
            request = call.args[0]
            self.assertEqual(request.user_name, "private-owner")
            self.assertEqual(request.kernel_slug, "kernel")
            self.assertFalse(hasattr(request, "version_label"))
            self.assertEqual(request.page_token, "next" if index else "")

    def test_missing_duplicate_or_repeating_page_is_rejected(self):
        cases = ([self.page([])], [self.page(["nested/artifact.tar"], "next"),
                                  self.page(["nested/artifact.tar"])],
                 [self.page([], "repeat"), self.page([], "repeat")])
        for pages in cases:
            with self.subTest(pages=len(pages)), self.assertRaises(recovery.RecoveryError):
                self.list_with(pages)

    def test_sdk_retry_diagnostics_are_suppressed(self):
        def noisy_failure(request):
            print(URL)
            print("private-owner", file=sys.stderr)
            raise RuntimeError(URL)
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(recovery.RecoveryError):
                self.list_with(noisy_failure)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(error.getvalue(), "")

    def test_listing_failure_does_not_expose_remote_exception(self):
        with self.assertRaises(recovery.RecoveryError) as caught:
            self.list_with([RuntimeError(URL + " private-owner")])
        self.assertNotIn("private", str(caught.exception))
        self.assertIn("check authentication", str(caught.exception))


class CLITests(unittest.TestCase):
    def test_token_alias_and_quiet_authentication(self):
        api = Mock()
        api.authenticate.side_effect = lambda: print("private-owner")
        fake_module = SimpleNamespace(KaggleApi=Mock(return_value=api))
        request_module = SimpleNamespace(ApiListKernelSessionOutputRequest=SimpleNamespace)
        with patch.dict(os.environ, {"KAGGLE_TOKEN": "private-token"}, clear=True), \
             patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": fake_module,
                                      "kagglesdk.kernels.types.kernels_api_service": request_module}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(recovery.authenticated_api(), (api, SimpleNamespace))
            self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "private-token")
        self.assertEqual(output.getvalue(), "")

    def test_unsupported_version_option_is_rejected_before_authentication(self):
        with patch.object(recovery, "authenticated_api") as authenticate, \
             contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                recovery.main(["kernel", "--artifact", "archive.tar", "--destination", "archive.tar",
                               "--expected-bytes", "1", "--sha256", SHA, "--version", "2"])
        self.assertEqual(caught.exception.code, 2)
        authenticate.assert_not_called()
        self.assertIn("unrecognized arguments: --version", error.getvalue())

    def test_cli_error_is_safe_and_nonzero(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as output:
            result = recovery.main(["kernel", "--artifact", "archive.tar", "--destination", "archive.tar",
                                    "--expected-bytes", "1", "--sha256", SHA])
        self.assertEqual(result, 1)
        self.assertIn("set KAGGLE_ACCOUNT", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
