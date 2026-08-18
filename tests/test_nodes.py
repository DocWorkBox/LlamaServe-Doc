import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "C:/models"
folder_paths.folder_names_and_paths = {}
folder_paths.get_filename_list = lambda category: [
    "model.gguf",
    "mmproj-F16.gguf",
    "legacy/model.safetensors",
]
folder_paths.get_full_path = lambda category, name: f"C:/models/LLM/{name}"
sys.modules.setdefault("folder_paths", folder_paths)

comfy = types.ModuleType("comfy")
model_management = types.ModuleType("comfy.model_management")
model_management.throw_exception_if_processing_interrupted = lambda: None
comfy.model_management = model_management
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.model_management", model_management)

from comfyui_llama_server.nodes import (
    LlamaServerGenerate,
    LlamaServerLoader,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    build_chat_request,
)


class NodeTests(unittest.TestCase):
    def test_registration_uses_llamaserve_doc_identity(self):
        self.assertEqual(
            set(NODE_CLASS_MAPPINGS),
            {"LlamaServeDocLoader", "LlamaServeDocGenerate"},
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocLoader"],
            "LlamaServe-Doc Loader",
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocGenerate"],
            "LlamaServe-Doc Generate",
        )

    def test_loader_lists_model_separately_from_mmproj(self):
        inputs = LlamaServerLoader.INPUT_TYPES()["required"]

        self.assertEqual(inputs["model"][0], ["model.gguf"])
        self.assertEqual(inputs["mmproj"][0], ["None", "mmproj-F16.gguf"])

    def test_reasoning_off_disables_thinking_in_request(self):
        request = build_chat_request(
            system_prompt="system",
            user_prompt="user",
            max_tokens=128,
            temperature=0.7,
            top_k=30,
            top_p=0.9,
            min_p=0.05,
            repeat_penalty=1.05,
            seed=1,
            reasoning="off",
        )

        self.assertEqual(request["reasoning_effort"], "none")
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])

    def test_generate_node_contains_integrated_stop_switch(self):
        required = LlamaServerGenerate.INPUT_TYPES()["required"]

        self.assertIn("stop_server_after_generate", required)
        self.assertFalse(required["stop_server_after_generate"][1]["default"])


if __name__ == "__main__":
    unittest.main()
