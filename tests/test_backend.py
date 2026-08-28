import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.backend import (
    BackendInstallError,
    _download,
    build_curl_download_command,
    detect_runtime_target,
    find_server_executable,
    safe_extract_archive,
    safe_extract_zip,
    select_release_assets,
    verify_file_hash,
)


class BackendTests(unittest.TestCase):
    def test_curl_download_command_supports_retry_and_resume(self):
        command = build_curl_download_command("https://example/model.zip", Path("C:/temp/model.zip"))

        self.assertEqual(command[0], "curl")
        self.assertIn("--retry-all-errors", command)
        self.assertEqual(command[command.index("--continue-at") + 1], "-")
        self.assertEqual(command[command.index("--output") + 1], "C:/temp/model.zip")

    def test_download_falls_back_to_python_when_curl_fails(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "7"}

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "asset.tar.gz"
            with patch(
                "comfyui_llama_server.backend.shutil.which",
                return_value="curl",
            ), patch(
                "comfyui_llama_server.backend.subprocess.run",
                side_effect=subprocess.CalledProcessError(35, ["curl"]),
            ), patch(
                "comfyui_llama_server.backend.urlopen",
                return_value=Response(b"payload"),
            ):
                _download("https://example/asset.tar.gz", destination)

            self.assertEqual(destination.read_bytes(), b"payload")

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

    def test_selects_linux_vulkan_archive(self):
        release = {
            "tag_name": "b10621",
            "assets": [{
                "name": "llama-b10621-bin-ubuntu-vulkan-x64.tar.gz",
                "browser_download_url": "https://example/linux.tar.gz",
                "digest": "sha256:" + "c" * 64,
            }],
        }

        selected = select_release_assets(release, "linux-x64-vulkan")

        self.assertEqual(
            [asset.name for asset in selected],
            ["llama-b10621-bin-ubuntu-vulkan-x64.tar.gz"],
        )

    def test_selects_macos_arm64_metal_archive(self):
        release = {
            "tag_name": "b10621",
            "assets": [{
                "name": "llama-b10621-bin-macos-arm64.tar.gz",
                "browser_download_url": "https://example/macos.tar.gz",
                "digest": "sha256:" + "d" * 64,
            }],
        }

        selected = select_release_assets(release, "macos-arm64-metal")

        self.assertEqual(selected[0].name, "llama-b10621-bin-macos-arm64.tar.gz")

    def test_detects_cross_platform_runtime_targets(self):
        self.assertEqual(
            detect_runtime_target("auto", system="Windows", machine="AMD64").key,
            "windows-x64-cuda12",
        )
        self.assertEqual(
            detect_runtime_target("auto", system="Darwin", machine="arm64").key,
            "macos-arm64-metal",
        )
        self.assertEqual(
            detect_runtime_target("auto", system="Darwin", machine="x86_64").key,
            "macos-x64-cpu",
        )
        self.assertEqual(
            detect_runtime_target(
                "auto",
                system="Linux",
                machine="x86_64",
                which=lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None,
            ).key,
            "linux-x64-vulkan",
        )
        self.assertEqual(
            detect_runtime_target(
                "auto",
                system="Linux",
                machine="aarch64",
                which=lambda _name: None,
            ).key,
            "linux-arm64-cpu",
        )

    def test_linux_auto_prefers_cuda_when_build_toolchain_is_available(self):
        tools = {
            "nvcc": "/usr/local/cuda/bin/nvcc",
            "cmake": "/usr/bin/cmake",
            "git": "/usr/bin/git",
            "vulkaninfo": "/usr/bin/vulkaninfo",
        }

        target = detect_runtime_target(
            "auto",
            system="Linux",
            machine="x86_64",
            which=tools.get,
        )

        self.assertEqual(target.key, "linux-x64-cuda")

    def test_linux_auto_does_not_treat_driver_only_install_as_cuda_toolkit(self):
        tools = {"nvidia-smi": "/usr/bin/nvidia-smi"}

        target = detect_runtime_target(
            "auto",
            system="Linux",
            machine="x86_64",
            which=tools.get,
        )

        self.assertEqual(target.key, "linux-x64-cpu")

    def test_linux_cuda_requires_a_build_toolchain_or_user_supplied_server(self):
        with self.assertRaisesRegex(BackendInstallError, "nvcc"):
            detect_runtime_target(
                "cuda",
                system="Linux",
                machine="x86_64",
                which=lambda _name: None,
                environ={},
            )

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

    def test_safe_extract_tar_rejects_parent_directory_escape(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            member = tarfile.TarInfo("../escaped.txt")
            payload = b"bad"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        data.seek(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "bad.tar.gz"
            archive_path.write_bytes(data.read())
            with self.assertRaises(BackendInstallError):
                safe_extract_archive(archive_path, Path(temp_dir) / "output")

    def test_safe_extract_tar_preserves_unix_executable_mode(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            member = tarfile.TarInfo("release/llama-server")
            payload = b"server"
            member.size = len(payload)
            member.mode = 0o755
            archive.addfile(member, io.BytesIO(payload))
        data.seek(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "server.tar.gz"
            archive_path.write_bytes(data.read())
            destination = Path(temp_dir) / "output"

            safe_extract_archive(archive_path, destination)

            executable = destination / "release" / "llama-server"
            self.assertTrue(executable.is_file())
            if os.name != "nt":
                self.assertNotEqual(executable.stat().st_mode & 0o111, 0)

    def test_safe_extract_tar_materializes_relative_link_chains(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            outer = tarfile.TarInfo("release/libggml.dylib")
            outer.type = tarfile.SYMTYPE
            outer.linkname = "libggml.0.dylib"
            archive.addfile(outer)
            inner = tarfile.TarInfo("release/libggml.0.dylib")
            inner.type = tarfile.SYMTYPE
            inner.linkname = "libggml.0.22.0.dylib"
            archive.addfile(inner)
            target = tarfile.TarInfo("release/libggml.0.22.0.dylib")
            payload = b"library"
            target.size = len(payload)
            archive.addfile(target, io.BytesIO(payload))
        data.seek(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "links.tar.gz"
            archive_path.write_bytes(data.read())
            destination = Path(temp_dir) / "output"

            safe_extract_archive(archive_path, destination)

            self.assertEqual((destination / "release" / "libggml.dylib").read_bytes(), b"library")
            self.assertEqual((destination / "release" / "libggml.0.dylib").read_bytes(), b"library")

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

            self.assertEqual(
                find_server_executable(Path(temp_dir), executable_name="llama-server.exe"),
                expected,
            )

    def test_finds_unix_server_executable_in_nested_release_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "release" / "llama-server"
            expected.parent.mkdir()
            expected.write_bytes(b"exe")

            self.assertEqual(
                find_server_executable(Path(temp_dir), executable_name="llama-server"),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
