import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.backend import BackendInstaller


class BackendInstallerTests(unittest.TestCase):
    def test_installs_verified_release_and_reuses_existing_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base.zip"
            cudart = root / "cudart.zip"
            with zipfile.ZipFile(base, "w") as archive:
                archive.writestr("release/llama-server.exe", b"exe")
                archive.writestr("release/ggml.dll", b"ggml")
            with zipfile.ZipFile(cudart, "w") as archive:
                archive.writestr("cudart64_12.dll", b"cuda")

            def asset(name, source):
                return {
                    "name": name,
                    "browser_download_url": source.as_uri(),
                    "digest": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                }

            release = {
                "tag_name": "b10218",
                "assets": [
                    asset("llama-b10218-bin-win-cuda-12.4-x64.zip", base),
                    asset("cudart-llama-bin-win-cuda-12.4-x64.zip", cudart),
                ],
            }
            downloads = []
            sources = {base.as_uri(): base, cudart.as_uri(): cudart}

            def downloader(url, destination, progress=None):
                downloads.append(url)
                destination.write_bytes(sources[url].read_bytes())

            installer = BackendInstaller(
                root / "runtime",
                fetch_release=lambda: release,
                downloader=downloader,
                system=lambda: "Windows",
                machine=lambda: "AMD64",
                which=lambda _name: None,
            )

            executable = installer.ensure_installed("cuda12")
            reused = installer.ensure_installed("cuda12")

            self.assertEqual(executable, reused)
            self.assertEqual(executable, executable.resolve())
            self.assertTrue(executable.is_file())
            self.assertTrue((executable.parent / "cudart64_12.dll").is_file())
            self.assertEqual(len(downloads), 2)
            manifest = json.loads((executable.parent / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["release"], "b10218")

    def test_installs_unix_tar_release_and_uses_recent_complete_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "macos.tar.gz"
            payload = b"server"
            with tarfile.open(package, "w:gz") as archive:
                member = tarfile.TarInfo("llama-b10621/llama-server")
                member.size = len(payload)
                member.mode = 0o755
                archive.addfile(member, io.BytesIO(payload))

            asset = {
                "name": "llama-b10621-bin-macos-arm64.tar.gz",
                "browser_download_url": package.as_uri(),
                "digest": "sha256:" + hashlib.sha256(package.read_bytes()).hexdigest(),
            }
            releases = [
                {"tag_name": "v0.3.0", "assets": []},
                {"tag_name": "b10621", "assets": [asset]},
            ]
            downloads = []

            def downloader(url, destination, progress=None):
                downloads.append(url)
                destination.write_bytes(package.read_bytes())

            installer = BackendInstaller(
                root / "runtime",
                fetch_release=lambda: releases,
                downloader=downloader,
                system=lambda: "Darwin",
                machine=lambda: "arm64",
                which=lambda _name: None,
            )

            executable = installer.ensure_installed("auto")

            self.assertEqual(executable.name, "llama-server")
            self.assertEqual(executable.read_bytes(), payload)
            self.assertEqual(downloads, [package.as_uri()])
            manifest = json.loads((executable.parent / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["target"], "macos-arm64-metal")
            if os.name != "nt":
                self.assertNotEqual(executable.stat().st_mode & 0o111, 0)

    def test_explicit_environment_server_skips_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "llama-server"
            executable.write_bytes(b"system server")
            fetches = []
            installer = BackendInstaller(
                root / "runtime",
                fetch_release=lambda: fetches.append(True),
                system=lambda: "Linux",
                machine=lambda: "x86_64",
                which=lambda _name: None,
                environ={"LLAMASERVE_DOC_SERVER": str(executable)},
            )

            resolved = installer.ensure_installed("cuda")

            self.assertEqual(resolved, executable.resolve())
            self.assertEqual(fetches, [])

    def test_linux_cuda_toolchain_builds_official_nightly_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = []
            tools = {
                "nvcc": "/usr/local/cuda/bin/nvcc",
                "cmake": "/usr/bin/cmake",
                "git": "/usr/bin/git",
                "vulkaninfo": "/usr/bin/vulkaninfo",
            }

            def command_runner(command, check):
                commands.append(command)
                if command[0] == tools["cmake"] and "--build" in command:
                    build_dir = Path(command[command.index("--build") + 1])
                    executable = build_dir / "bin" / "llama-server"
                    executable.parent.mkdir(parents=True, exist_ok=True)
                    executable.write_bytes(b"cuda server")
                    (executable.parent / "libggml-cuda.so").write_bytes(b"cuda library")

            installer = BackendInstaller(
                root / "runtime",
                fetch_release=lambda: [
                    {"tag_name": "v0.3.0", "assets": []},
                    {"tag_name": "b10675", "assets": []},
                ],
                system=lambda: "Linux",
                machine=lambda: "x86_64",
                which=tools.get,
                command_runner=command_runner,
            )

            executable = installer.ensure_installed("auto")
            reused = installer.ensure_installed("auto")

            self.assertEqual(executable, reused)
            self.assertEqual(executable, executable.resolve())
            self.assertEqual(executable.read_bytes(), b"cuda server")
            self.assertTrue((executable.parent / "libggml-cuda.so").is_file())
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][:5], [
                tools["git"],
                "clone",
                "--depth",
                "1",
                "--branch",
            ])
            self.assertIn("b10675", commands[0])
            self.assertIn("-DGGML_CUDA=ON", commands[1])
            self.assertIn("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", commands[1])
            self.assertIn("llama-server", commands[2])
            manifest = json.loads((executable.parent / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["target"], "linux-x64-cuda")
            self.assertEqual(manifest["source_release"], "b10675")


if __name__ == "__main__":
    unittest.main()
