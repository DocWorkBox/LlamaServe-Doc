import sys
import inspect
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

folder_paths = sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
folder_paths.base_path = str(ROOT)
folder_paths.models_dir = "C:/models"
folder_paths.folder_names_and_paths = {}
folder_paths.get_filename_list = lambda category: [
    "model.gguf",
    "mmproj-F16.gguf",
    "legacy/model.safetensors",
]
folder_paths.get_full_path = lambda category, name: f"C:/models/LLM/{name}"

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
            {
                "LlamaServeDocLoader",
                "LlamaServeDocH3OmniPreset",
                "LlamaServeDocH3OmniGenerate",
                "LlamaServeDocGenerate",
            },
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocLoader"],
            "LlamaServe-Doc Loader",
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocGenerate"],
            "LlamaServe-Doc Generate",
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocH3OmniPreset"],
            "LlamaServe-Doc H3 Omni Preset",
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocH3OmniGenerate"],
            "LlamaServe-Doc H3 Omni Generate",
        )

    def test_loader_lists_model_separately_from_mmproj(self):
        inputs = LlamaServerLoader.INPUT_TYPES()["required"]

        self.assertEqual(inputs["model"][0], ["model.gguf"])
        self.assertEqual(inputs["mmproj"][0], ["None", "mmproj-F16.gguf"])

    def test_loader_hides_and_automatically_configures_media_root(self):
        inputs = LlamaServerLoader.INPUT_TYPES()
        parameters = inspect.signature(LlamaServerLoader.load).parameters

        self.assertNotIn("media_root", inputs.get("optional", {}))
        self.assertNotIn("media_root", parameters)

        config = LlamaServerLoader().load(
            "model.gguf",
            "None",
            4096,
            47,
            "on",
            "q8_0",
            "q8_0",
        )[0]
        self.assertEqual(
            config.media_root,
            Path(folder_paths.base_path).resolve(),
        )

    def test_loader_hides_and_automatically_configures_server_port(self):
        inputs = LlamaServerLoader.INPUT_TYPES()
        parameters = inspect.signature(LlamaServerLoader.load).parameters

        self.assertNotIn("port", inputs.get("required", {}))
        self.assertNotIn("port", inputs.get("optional", {}))
        self.assertNotIn("port", parameters)

        config = LlamaServerLoader().load(
            "model.gguf",
            "None",
            4096,
            47,
            "on",
            "q8_0",
            "q8_0",
        )[0]
        self.assertEqual(config.port, 0)

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

    def test_chat_request_emits_ordered_llama_cpp_media_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_dir = root / "input"
            media_dir.mkdir()
            video = media_dir / "clip.mp4"
            audio = media_dir / "sound.wav"
            video.write_bytes(b"video")
            audio.write_bytes(b"audio")
            media = (
                {"type": "video", "path": str(video)},
                {"type": "audio", "path": str(audio)},
            )

            request = build_chat_request(
                system_prompt="system",
                user_prompt="caption this",
                max_tokens=128,
                temperature=0.7,
                top_k=30,
                top_p=0.9,
                min_p=0.05,
                repeat_penalty=1.05,
                seed=1,
                reasoning="off",
                media=media,
                media_root=root,
            )

        self.assertEqual(
            request["messages"][1]["content"],
            [
                {"type": "text", "text": "caption this"},
                {
                    "type": "input_video",
                    "input_video": {"url": "file://input/clip.mp4"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"url": "file://input/sound.wav"},
                },
            ],
        )

    def test_chat_request_emits_llama_cpp_image_url_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"image")

            try:
                request = build_chat_request(
                    system_prompt="",
                    user_prompt="describe",
                    max_tokens=64,
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    min_p=0.0,
                    repeat_penalty=1.0,
                    seed=0,
                    reasoning="off",
                    media=({"type": "image", "path": str(image)},),
                    media_root=root,
                )
            except ValueError as error:
                self.fail(f"image media should be supported: {error}")

        self.assertEqual(
            request["messages"][0]["content"][1],
            {
                "type": "image_url",
                "image_url": {"url": "file://frame.png"},
            },
        )

    def test_generate_node_contains_integrated_stop_switch(self):
        required = LlamaServerGenerate.INPUT_TYPES()["required"]

        self.assertIn("stop_server_after_generate", required)
        self.assertFalse(required["stop_server_after_generate"][1]["default"])
        self.assertIn("idle_timeout_minutes", required)
        self.assertEqual(required["idle_timeout_minutes"][1]["default"], 5)
        self.assertEqual(required["idle_timeout_minutes"][1]["min"], 0)

    def test_generate_accepts_optional_ordered_media(self):
        inputs = LlamaServerGenerate.INPUT_TYPES()
        parameters = inspect.signature(build_chat_request).parameters

        self.assertIn("optional", inputs)
        optional = inputs["optional"]
        self.assertEqual(optional["media"][0], "LLAMASERVE_MEDIA")
        self.assertIn("media", parameters)
        self.assertIn("media_root", parameters)


if __name__ == "__main__":
    unittest.main()
