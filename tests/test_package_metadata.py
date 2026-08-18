import json
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
        self.assertEqual(metadata["project"]["version"], "1.0.0")
        self.assertEqual(metadata["project"]["license"], {"file": "LICENSE"})
        self.assertEqual(metadata["project"]["dependencies"], [])
        self.assertIn(
            "Operating System :: Microsoft :: Windows",
            metadata["project"]["classifiers"],
        )
        self.assertIn(
            "Environment :: GPU :: NVIDIA CUDA",
            metadata["project"]["classifiers"],
        )
        self.assertEqual(metadata["tool"]["comfy"]["DisplayName"], "LlamaServe-Doc")
        self.assertEqual(metadata["tool"]["comfy"]["PublisherId"], "DocWorkBox")
        self.assertNotIn("Icon", metadata["tool"]["comfy"])

    def test_registry_archive_excludes_development_only_files(self):
        patterns = (ROOT / ".comfyignore").read_text("utf-8").splitlines()

        self.assertIn("tests/", patterns)
        self.assertIn(".github/", patterns)

    def test_official_registry_publish_workflow_is_manual_and_secret_backed(self):
        workflow = (ROOT / ".github" / "workflows" / "publish_action.yml").read_text("utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("Comfy-Org/publish-node-action@main", workflow)
        self.assertIn("secrets.REGISTRY_ACCESS_TOKEN", workflow)

    def test_default_example_workflow_uses_registered_nodes(self):
        workflow_path = ROOT / "example_workflows" / "Qwen3.6 H3 Prompt rewrite.json"
        workflow = json.loads(workflow_path.read_text("utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}

        self.assertIn("LlamaServeDocLoader", node_types)
        self.assertIn("LlamaServeDocGenerate", node_types)
        self.assertEqual(workflow["version"], 0.4)


if __name__ == "__main__":
    unittest.main()
