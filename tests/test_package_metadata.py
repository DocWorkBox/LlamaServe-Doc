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
        self.assertEqual(metadata["project"]["version"], "1.4.3")
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
        self.assertIn("Operating System :: POSIX :: Linux", metadata["project"]["classifiers"])
        self.assertIn("Operating System :: MacOS", metadata["project"]["classifiers"])
        self.assertEqual(metadata["tool"]["comfy"]["DisplayName"], "LlamaServe-Doc")
        self.assertEqual(metadata["tool"]["comfy"]["PublisherId"], "zhaoke1006")
        self.assertNotIn("Icon", metadata["tool"]["comfy"])

    def test_registry_archive_excludes_development_only_files(self):
        patterns = (ROOT / ".comfyignore").read_text("utf-8").splitlines()

        self.assertIn("tests/", patterns)
        self.assertIn(".github/", patterns)
        self.assertIn("requirements-test.txt", patterns)

    def test_ci_installs_lightweight_comfy_test_dependencies(self):
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text("utf-8")
        requirements_path = ROOT / "requirements-test.txt"

        self.assertTrue(requirements_path.is_file())
        requirements = requirements_path.read_text("utf-8").casefold()
        self.assertIn("python -m pip install -r requirements-test.txt", workflow)
        self.assertIn("numpy", requirements)
        self.assertIn("pillow", requirements)
        self.assertNotIn("torch", requirements)
        self.assertIn("windows-latest", workflow)
        self.assertIn("ubuntu-latest", workflow)
        self.assertIn("macos-14", workflow)

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

    def test_obsolete_media_chain_example_and_prompt_are_removed(self):
        workflow_path = ROOT / "example_workflows" / "Qwen2.5 Omni H3 Video Caption.json"
        prompt_path = ROOT / "prompts" / "video_caption_system.txt"

        self.assertFalse(workflow_path.exists())
        self.assertFalse(prompt_path.exists())

    def test_official_omni_preset_example_uses_one_combined_generate_node(self):
        workflow_path = ROOT / "example_workflows" / "Qwen2.5 Omni H3 Official Presets.json"
        self.assertTrue(workflow_path.is_file())
        workflow = json.loads(workflow_path.read_text("utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        loader = next(node for node in workflow["nodes"] if node["type"] == "LlamaServeDocLoader")

        self.assertEqual(
            node_types,
            {
                "LlamaServeDocLoader",
                "LlamaServeDocH3OmniGenerate",
                "LoadImage",
                "VHS_LoadVideo",
                "LoadAudio",
                "PreviewAny",
            },
        )
        self.assertNotIn("LlamaServeDocMedia", node_types)
        self.assertNotIn("LlamaServeDocH3OmniPreset", node_types)
        self.assertNotIn("LlamaServeDocGenerate", node_types)
        self.assertEqual(len(loader["widgets_values"]), 8)
        self.assertEqual(loader["widgets_values"][-1], "auto")
        self.assertNotIn("port", loader.get("widgets_values_named", {}))
        self.assertNotIn("port", {item["name"] for item in loader["inputs"]})
        self.assertEqual(workflow["version"], 0.4)

    def test_h3_reference_autocomplete_is_packaged_as_a_web_extension(self):
        package_init = (ROOT / "__init__.py").read_text("utf-8")
        extension = ROOT / "web" / "h3_reference_autocomplete.js"

        self.assertIn('WEB_DIRECTORY = "./web"', package_init)
        self.assertIn('"WEB_DIRECTORY"', package_init)
        self.assertTrue(extension.is_file())
        self.assertIn("app.registerExtension", extension.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
