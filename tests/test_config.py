import sys
import unittest
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.config import ServerConfig


class ServerConfigTests(unittest.TestCase):
    def test_config_supports_restricted_local_media_root(self):
        self.assertIn("media_root", {field.name for field in fields(ServerConfig)})

    def test_build_command_uses_exact_gpu_layers_and_omits_mmproj(self):
        config = ServerConfig(
            model_path=Path("C:/models/model.gguf"),
            mmproj_path=None,
            context_length=4096,
            gpu_layers=47,
            flash_attention="on",
            cache_type_k="f16",
            cache_type_v="f16",
            host="127.0.0.1",
            port=8191,
        )

        command = config.build_command(Path("C:/runtime/llama-server.exe"))

        self.assertEqual(command[0], "C:/runtime/llama-server.exe")
        self.assertIn("C:/models/model.gguf", command)
        self.assertEqual(command[command.index("--gpu-layers") + 1], "47")
        self.assertEqual(command[command.index("--ctx-size") + 1], "4096")
        self.assertEqual(command[command.index("--flash-attn") + 1], "on")
        self.assertNotIn("--mmproj", command)

    def test_build_command_adds_mmproj_and_maps_minus_one_to_auto(self):
        config = ServerConfig(
            model_path=Path("C:/models/model.gguf"),
            mmproj_path=Path("C:/models/mmproj.gguf"),
            context_length=2048,
            gpu_layers=-1,
            flash_attention="auto",
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            port=8191,
            media_root=Path("C:/ComfyUI"),
        )

        command = config.build_command(Path("C:/runtime/llama-server.exe"))

        self.assertEqual(command[command.index("--gpu-layers") + 1], "auto")
        self.assertEqual(command[command.index("--mmproj") + 1], "C:/models/mmproj.gguf")
        self.assertIn("--media-path", command)
        self.assertEqual(command[command.index("--media-path") + 1], "C:/ComfyUI")

    def test_signature_changes_when_server_settings_change(self):
        base = ServerConfig(Path("C:/models/model.gguf"), None, 4096, 47, "on", "f16", "f16", port=8191)
        changed = ServerConfig(Path("C:/models/model.gguf"), None, 4096, 46, "on", "f16", "f16", port=8191)

        self.assertNotEqual(base.signature(), changed.signature())


if __name__ == "__main__":
    unittest.main()
