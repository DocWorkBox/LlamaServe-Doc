import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.backend import (
    BackendInstallError,
    build_curl_download_command,
    find_server_executable,
    safe_extract_zip,
    select_release_assets,
    verify_file_hash,
)


class BackendTests(unittest.TestCase):
    def test_curl_download_command_supports_retry_and_resume(self):
        command = build_curl_download_command("https://example/model.zip", Path("C:/temp/model.zip"))

        self.assertEqual(command[0], "curl.exe")
        self.assertIn("--retry-all-errors", command)
        self.assertEqual(command[command.index("--continue-at") + 1], "-")
        self.assertEqual(command[command.index("--output") + 1], "C:/temp/model.zip")

    def test_selects_matching_cuda_runtime_and_binary_assets(self):
        release = {
            "tag_name": "b10218",
            "assets": [
                {
                    "name": "llama-b10218-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example/base.zip",
                    "digest": "sha256:" + "a" * 64,
                },
                {
                    "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example/cudart.zip",
                    "digest": "sha256:" + "b" * 64,
                },
            ],
        }

        selected = select_release_assets(release, "cuda12")

        self.assertEqual([asset.name for asset in selected], [
            "llama-b10218-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ])
        self.assertEqual(selected[0].sha256, "a" * 64)

    def test_rejects_release_asset_without_sha256_digest(self):
        release = {
            "tag_name": "b10218",
            "assets": [{
                "name": "llama-b10218-bin-win-cuda-12.4-x64.zip",
                "browser_download_url": "https://example/base.zip",
                "digest": None,
            }],
        }

        with self.assertRaises(BackendInstallError):
            select_release_assets(release, "cuda12")

    def test_safe_extract_rejects_parent_directory_escape(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("../escaped.txt", "bad")
        data.seek(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "bad.zip"
            archive_path.write_bytes(data.read())
            with self.assertRaises(BackendInstallError):
                safe_extract_zip(archive_path, Path(temp_dir) / "output")

    def test_verify_file_hash_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asset.zip"
            path.write_bytes(b"known content")

            self.assertTrue(verify_file_hash(path, "41277d8d0b0610e58f13bdc06b732c629a2fd3ff93c382f40af3f60cfe5e5c9e"))
            self.assertFalse(verify_file_hash(path, "0" * 64))

    def test_finds_server_executable_in_nested_release_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "release" / "llama-server.exe"
            expected.parent.mkdir()
            expected.write_bytes(b"exe")

            self.assertEqual(find_server_executable(Path(temp_dir)), expected)


if __name__ == "__main__":
    unittest.main()
