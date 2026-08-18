import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PackageMetadataTests(unittest.TestCase):
    def test_pyproject_uses_llamaserve_doc_identity(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))

        self.assertEqual(metadata["project"]["name"], "LlamaServe-Doc")
        self.assertEqual(metadata["tool"]["comfy"]["DisplayName"], "LlamaServe-Doc")


if __name__ == "__main__":
    unittest.main()
