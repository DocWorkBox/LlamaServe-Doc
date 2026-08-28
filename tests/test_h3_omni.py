import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "C:/models"
folder_paths.folder_names_and_paths = {}
folder_paths.get_filename_list = lambda category: ["model.gguf", "mmproj-F16.gguf"]
folder_paths.get_full_path = lambda category, name: f"C:/models/LLM/{name}"
sys.modules.setdefault("folder_paths", folder_paths)

comfy = types.ModuleType("comfy")
model_management = types.ModuleType("comfy.model_management")
model_management.throw_exception_if_processing_interrupted = lambda: None
comfy.model_management = model_management
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.model_management", model_management)

from comfyui_llama_server.h3_omni import build_h3_omni_preset
from comfyui_llama_server.nodes import (
    LlamaServerH3OmniPreset,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    build_chat_request,
)


class H3OmniPresetTests(unittest.TestCase):
    def _media(self, directory, *kinds):
        result = []
        for index, kind in enumerate(kinds, start=1):
            suffix = {"image": ".png", "video": ".mp4", "audio": ".wav"}[kind]
            path = Path(directory) / f"asset-{index}{suffix}"
            path.write_bytes(kind.encode("ascii"))
            result.append({"type": kind, "path": str(path.resolve())})
        return tuple(result)

    def test_node_registers_all_five_official_modes(self):
        self.assertIn("LlamaServeDocH3OmniPreset", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocH3OmniPreset"],
            "LlamaServe-Doc H3 Omni Preset",
        )
        inputs = LlamaServerH3OmniPreset.INPUT_TYPES()
        self.assertEqual(
            inputs["required"]["mode"][0],
            ["T2AV", "I2AV", "L2AV", "FL2AV", "Ref2AV"],
        )
        self.assertEqual(inputs["required"]["duration"][1]["min"], 4)
        self.assertEqual(inputs["required"]["duration"][1]["max"], 15)
        self.assertEqual(
            inputs["required"]["resolution"][0],
            ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
        )
        self.assertEqual(inputs["optional"]["media"][0], "LLAMASERVE_MEDIA")
        self.assertEqual(
            LlamaServerH3OmniPreset.RETURN_TYPES,
            ("STRING", "STRING", "LLAMASERVE_MEDIA"),
        )

    def test_t2av_formats_official_request_without_media(self):
        system_prompt, user_prompt, media = build_h3_omni_preset(
            "T2AV", "A fox crosses a snowy forest.", 5, "16:9"
        )

        self.assertIn("T2AV, I2AV, FL2AV, and L2AV modes", system_prompt)
        self.assertEqual(
            user_prompt,
            "Rewrite request:\n"
            "task: T2AV\n"
            "resolution: 16:9\n"
            "effective_duration: 5.17s\n"
            "raw_prompt: A fox crosses a snowy forest.",
        )
        self.assertEqual(media, ())

    def test_keyframe_modes_enforce_image_roles_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            one_image = self._media(directory, "image")
            two_images = self._media(directory, "image", "image")

            _, i_prompt, i_media = build_h3_omni_preset(
                "I2AV", "Move forward.", 8, "adaptive", one_image
            )
            _, l_prompt, l_media = build_h3_omni_preset(
                "L2AV", "Reveal the field.", 11, "adaptive", one_image
            )
            _, fl_prompt, fl_media = build_h3_omni_preset(
                "FL2AV", "Connect both frames.", 5, "adaptive", two_images
            )

        self.assertEqual(i_media[0]["heading"], "<Picture 1> — exact first frame at 0.00 seconds:\n")
        self.assertIn("effective_duration: 8.00s", i_prompt)
        self.assertEqual(l_media[0]["heading"], "<Picture 1> — exact final frame at 11.54s:\n")
        self.assertEqual(
            [item["heading"] for item in fl_media],
            [
                "<Picture 1> — exact first frame at 0.00 seconds:\n",
                "<Picture 2> — exact final frame at 5.17s:\n",
            ],
        )

    def test_ref2av_labels_mixed_media_and_uses_six_section_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            source_media = self._media(directory, "image", "video", "audio")
            system_prompt, user_prompt, media = build_h3_omni_preset(
                "Ref2AV",
                "Use <Picture 1> for the subject, <Video 1> for motion, and <Audio 1> for sound.",
                12,
                "9:16",
                source_media,
            )

        self.assertIn("exactly these six sections", system_prompt)
        self.assertEqual(
            [item["heading"] for item in media],
            ["<Picture 1>:\n", "<Video 1>:\n", "<Audio 1>:\n"],
        )
        self.assertTrue(media[0]["preamble"].startswith("Ordered MiniMax-H3 references:"))
        self.assertIn("task: REF2AV", user_prompt)
        self.assertIn("resolution: 9:16", user_prompt)
        self.assertIn("effective_duration: 12.25s", user_prompt)

    def test_ref2av_normalizes_chinese_reference_aliases_to_official_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source_media = self._media(directory, "image", "video", "audio")
            _, user_prompt, _ = build_h3_omni_preset(
                "Ref2AV",
                "视频1中的人物替换成图片1的人物，不修改音频",
                8,
                "16:9",
                source_media,
            )

        self.assertIn("<Video 1>", user_prompt)
        self.assertIn("<Picture 1>", user_prompt)
        self.assertIn("<Audio 1>", user_prompt)
        self.assertNotIn("视频1", user_prompt)
        self.assertNotIn("图片1", user_prompt)

    def test_ref2av_ignores_connected_media_not_mentioned_in_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            source_media = self._media(directory, "image", "audio", "video")
            _, user_prompt, media = build_h3_omni_preset(
                "Ref2AV",
                "Replace the person in <Video 1> with <Picture 1>.",
                8,
                "16:9",
                source_media,
            )

        self.assertEqual(
            [(item["type"], item["label"]) for item in media],
            [("image", "<Picture 1>"), ("video", "<Video 1>")],
        )
        self.assertNotIn("<Audio 1>", user_prompt)

    def test_ref2av_does_not_guess_unnumbered_alias_with_multiple_same_type_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            source_media = self._media(directory, "audio", "audio")
            with self.assertRaisesRegex(
                ValueError, "must mention at least one supplied reference label"
            ):
                build_h3_omni_preset(
                    "Ref2AV",
                    "保持音频不变",
                    8,
                    "16:9",
                    source_media,
                )

    def test_ref2av_rejects_label_that_has_no_connected_media(self):
        with tempfile.TemporaryDirectory() as directory:
            source_media = self._media(directory, "image")
            with self.assertRaisesRegex(ValueError, "not supplied"):
                build_h3_omni_preset(
                    "Ref2AV",
                    "Use <Picture 2> for the subject.",
                    8,
                    "16:9",
                    source_media,
                )

    def test_invalid_media_shapes_and_ref_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            image = self._media(directory, "image")
            video = self._media(directory, "video")
            with self.assertRaisesRegex(ValueError, "T2AV requires exactly 0 image"):
                build_h3_omni_preset("T2AV", "prompt", 5, "16:9", image)
            with self.assertRaisesRegex(ValueError, "I2AV requires exactly 1 image"):
                build_h3_omni_preset("I2AV", "prompt", 5, "adaptive", video)
            with self.assertRaisesRegex(ValueError, "Ref2AV resolution"):
                build_h3_omni_preset("Ref2AV", "Use <Video 1>.", 5, "1:1", video)
            with self.assertRaisesRegex(ValueError, "must mention at least one supplied reference label"):
                build_h3_omni_preset("Ref2AV", "Use this motion.", 5, "16:9", video)

    def test_chat_request_interleaves_official_headings_and_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_media = self._media(directory, "image", "video")
            system_prompt, user_prompt, media = build_h3_omni_preset(
                "Ref2AV",
                "Use <Picture 1> for style and <Video 1> for motion.",
                10,
                "16:9",
                source_media,
            )
            request = build_chat_request(
                system_prompt,
                user_prompt,
                1024,
                0.8,
                30,
                0.9,
                0.05,
                1.05,
                0,
                "off",
                media=media,
                media_root=root,
            )

        content = request["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Ordered MiniMax-H3 references:\n"})
        self.assertEqual(content[1], {"type": "text", "text": "<Picture 1>:\n"})
        self.assertEqual(content[2]["type"], "image_url")
        self.assertEqual(content[3], {"type": "text", "text": "<Video 1>:\n"})
        self.assertEqual(content[4]["type"], "input_video")
        self.assertEqual(content[5], {"type": "text", "text": user_prompt})


if __name__ == "__main__":
    unittest.main()
