import hashlib
import json
import sys
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
            )

            executable = installer.ensure_installed("cuda12")
            reused = installer.ensure_installed("cuda12")

            self.assertEqual(executable, reused)
            self.assertTrue(executable.is_file())
            self.assertTrue((executable.parent / "cudart64_12.dll").is_file())
            self.assertEqual(len(downloads), 2)
            manifest = json.loads((executable.parent / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["release"], "b10218")


if __name__ == "__main__":
    unittest.main()
