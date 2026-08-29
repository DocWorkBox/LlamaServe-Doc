import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


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

from comfyui_llama_server import h3_reference_inputs
from comfyui_llama_server.h3_reference_inputs import (
    collect_official_references,
    materialize_reference_media,
    pack_director_group,
)
from comfyui_llama_server.nodes import (
    LlamaServerH3OmniGenerate,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    generate_h3_omni,
)


class H3CombinedNodeTests(unittest.TestCase):
    def test_combined_node_is_registered(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["LlamaServeDocH3OmniGenerate"],
            LlamaServerH3OmniGenerate,
        )

    def test_combined_generation_returns_enhanced_prompt_stats_and_director_group(self):
        class FakeService:
            def __init__(self):
                self.request = None

            def generate(
                self,
                config,
                request,
                stop_after_generate,
                interrupt_check=None,
                idle_timeout_seconds=0,
            ):
                self.request = request
                self.stop_after_generate = stop_after_generate
                self.idle_timeout_seconds = idle_timeout_seconds
                return types.SimpleNamespace(
                    text="integrated_multimodal_description: enhanced",
                    tokens_per_second=12.5,
                    prompt_tokens_per_second=34.0,
                    raw_timings={"predicted_per_second": 12.5},
                )

        with tempfile.TemporaryDirectory() as directory:
            config = types.SimpleNamespace(media_root=Path(directory))
            service = FakeService()
            text, performance, group = generate_h3_omni(
                config,
                "T2AV",
                "A fox walks through snow.",
                5,
                "16:9",
                128,
                0.2,
                20,
                0.9,
                0.05,
                1.05,
                0,
                "off",
                True,
                generation_service=service,
                idle_timeout_minutes=7,
            )

        self.assertEqual(text, "integrated_multimodal_description: enhanced")
        self.assertIn('"tokens_per_second": 12.5', performance)
        self.assertEqual((group["family"], group["kind"]), ("i2v", "t2v"))
        self.assertEqual(group["prompt"], text)
        self.assertTrue(service.stop_after_generate)
        self.assertEqual(service.idle_timeout_seconds, 420)
        self.assertEqual(service.request["messages"][1]["content"].splitlines()[1], "task: T2AV")
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["LlamaServeDocH3OmniGenerate"],
            "LlamaServe-Doc H3 Omni Generate",
        )

    def test_reference_order_matches_official_minimax_h3_node(self):
        image0, image1 = object(), object()
        video0, video2 = object(), object()
        paired0, paired2, standalone0 = object(), object(), object()

        refs = collect_official_references(
            ref_images={"ref_image_1": image1, "ref_image_0": image0},
            ref_videos={"ref_video_2": video2, "ref_video_0": video0},
            ref_video_audios={
                "ref_video_audio_0": paired0,
                "ref_video_audio_1": object(),
                "ref_video_audio_2": paired2,
            },
            ref_audios={"ref_audio_0": standalone0},
        )

        self.assertEqual(
            [(item.source, item.media_type) for item in refs.ordered],
            [
                ("ref_image_0", "image"),
                ("ref_image_1", "image"),
                ("ref_video_audio_0", "audio"),
                ("ref_video_0", "video"),
                ("ref_video_audio_2", "audio"),
                ("ref_video_2", "video"),
                ("ref_audio_0", "audio"),
            ],
        )
        self.assertNotIn(1, refs.ref_video_audios)

    def test_director_group_matches_i2v_family_for_base_modes(self):
        first, last = object(), object()
        refs = collect_official_references(
            ref_images={"ref_image_0": first, "ref_image_1": last}
        )

        t2v = pack_director_group("T2AV", "enhanced", 5, collect_official_references())
        i2v = pack_director_group(
            "I2AV", "enhanced", 5, collect_official_references(ref_images={"ref_image_0": first})
        )
        l2v = pack_director_group(
            "L2AV", "enhanced", 5, collect_official_references(ref_images={"ref_image_0": last})
        )
        fl2v = pack_director_group("FL2AV", "enhanced", 5, refs)

        self.assertEqual((t2v["family"], t2v["kind"]), ("i2v", "t2v"))
        self.assertEqual((i2v["family"], i2v["kind"]), ("i2v", "i2v"))
        self.assertIs(i2v["first_frame"], first)
        self.assertEqual((l2v["family"], l2v["kind"]), ("i2v", "fl2v"))
        self.assertIs(l2v["last_frame"], last)
        self.assertEqual((fl2v["family"], fl2v["kind"]), ("i2v", "fl2v"))
        self.assertIs(fl2v["first_frame"], first)
        self.assertIs(fl2v["last_frame"], last)

    def test_director_group_matches_r2v_group_contract(self):
        image, video, paired_audio, audio = object(), object(), object(), object()
        refs = collect_official_references(
            ref_images={"ref_image_0": image},
            ref_videos={"ref_video_0": video},
            ref_video_audios={"ref_video_audio_0": paired_audio},
            ref_audios={"ref_audio_0": audio},
        )

        group = pack_director_group("Ref2AV", "enhanced", 8, refs)

        self.assertEqual(group["version"], 1)
        self.assertEqual(group["family"], "r2v")
        self.assertEqual(group["kind"], "r2v")
        self.assertEqual(group["prompt"], "enhanced")
        self.assertEqual(group["duration_sec"], 8.0)
        self.assertEqual(group["ref_images"], {0: image})
        self.assertEqual(group["ref_videos"], {0: video})
        self.assertEqual(group["ref_video_audios"], {0: paired_audio})
        self.assertEqual(group["ref_audios"], {0: audio})

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for reference videos")
    def test_generation_omits_unmentioned_reference_from_request_and_director_group(self):
        class FakeService:
            def __init__(self):
                self.request = None

            def generate(
                self,
                config,
                request,
                stop_after_generate,
                interrupt_check=None,
                idle_timeout_seconds=0,
            ):
                self.request = request
                return types.SimpleNamespace(
                    text="integrated_multimodal_description: enhanced",
                    tokens_per_second=12.5,
                    prompt_tokens_per_second=34.0,
                    raw_timings={},
                )

        image = np.zeros((1, 4, 6, 3), dtype=np.float32)
        video = np.zeros((5, 4, 6, 3), dtype=np.float32)
        audio = {"waveform": np.zeros((1, 1, 320), dtype=np.float32), "sample_rate": 32000}
        with tempfile.TemporaryDirectory() as directory:
            service = FakeService()
            _, _, group = generate_h3_omni(
                types.SimpleNamespace(media_root=Path(directory)),
                "Ref2AV",
                "Use <Picture 1> for identity and <Video 1> for motion.",
                8,
                "16:9",
                128,
                0.2,
                20,
                0.9,
                0.05,
                1.05,
                0,
                "off",
                False,
                ref_images={"ref_image_0": image},
                ref_videos={"ref_video_0": video},
                ref_video_audios={"ref_video_audio_0": audio},
                generation_service=service,
            )

        content_types = [item["type"] for item in service.request["messages"][1]["content"]]
        self.assertIn("image_url", content_types)
        self.assertIn("input_video", content_types)
        self.assertNotIn("input_audio", content_types)
        self.assertEqual(group["ref_images"], {0: image})
        self.assertEqual(group["ref_videos"], {0: video})
        self.assertEqual(group["ref_video_audios"], {})

    def test_generation_does_not_materialize_unmentioned_reference(self):
        class FakeService:
            def generate(
                self,
                config,
                request,
                stop_after_generate,
                interrupt_check=None,
                idle_timeout_seconds=0,
            ):
                return types.SimpleNamespace(
                    text="integrated_multimodal_description: enhanced",
                    tokens_per_second=12.5,
                    prompt_tokens_per_second=34.0,
                    raw_timings={},
                )

        image = np.zeros((1, 4, 6, 3), dtype=np.float32)
        invalid_but_unused_audio = object()
        with tempfile.TemporaryDirectory() as directory:
            _, _, group = generate_h3_omni(
                types.SimpleNamespace(media_root=Path(directory)),
                "Ref2AV",
                "Use <Picture 1> for the subject.",
                8,
                "16:9",
                128,
                0.2,
                20,
                0.9,
                0.05,
                1.05,
                0,
                "off",
                False,
                ref_images={"ref_image_0": image},
                ref_audios={"ref_audio_0": invalid_but_unused_audio},
                generation_service=FakeService(),
            )

        self.assertEqual(group["ref_images"], {0: image})
        self.assertEqual(group["ref_audios"], {})

    def test_materializer_writes_llama_cpp_media_and_cleans_temporary_files(self):
        image = np.zeros((1, 4, 6, 3), dtype=np.float32)
        audio = {"waveform": np.zeros((1, 1, 320), dtype=np.float32), "sample_rate": 32000}
        refs = collect_official_references(
            ref_images={"ref_image_0": image},
            ref_audios={"ref_audio_0": audio},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with materialize_reference_media(refs, root) as media:
                paths = [Path(item["path"]) for item in media]
                self.assertEqual([item["type"] for item in media], ["image", "audio"])
                self.assertTrue(all(path.is_file() for path in paths))
                temp_parent = paths[0].parent
                self.assertTrue(temp_parent.is_relative_to(root))
                self.assertEqual(temp_parent.parent, root / "llamaserve_doc")
            self.assertFalse(temp_parent.exists())

    def test_reference_video_plan_matches_qwen_omni_sampling_and_pixel_budget(self):
        self.assertTrue(hasattr(h3_reference_inputs, "plan_reference_video"))
        plan = h3_reference_inputs.plan_reference_video(
            frame_count=120,
            height=2160,
            width=4050,
        )

        self.assertEqual(len(plan.indices), 10)
        self.assertAlmostEqual(plan.fps, 2.0)
        self.assertLessEqual(plan.width * plan.height, 768 * 28 * 28)
        self.assertEqual(plan.width % 2, 0)
        self.assertEqual(plan.height % 2, 0)
        self.assertAlmostEqual(plan.width / plan.height, 4050 / 2160, delta=0.01)
        self.assertEqual(plan.indices[0], 0)
        self.assertEqual(plan.indices[-1], 119)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for reference videos")
    def test_materializer_encodes_image_batch_as_video(self):
        frames = np.zeros((5, 4, 6, 3), dtype=np.float32)
        frames[:, :, :, 2] = 1.0
        refs = collect_official_references(ref_videos={"ref_video_0": frames})
        with tempfile.TemporaryDirectory() as directory:
            with materialize_reference_media(refs, Path(directory)) as media:
                video = Path(media[0]["path"])
                self.assertEqual(media[0]["type"], "video")
                self.assertTrue(video.is_file())
                self.assertGreater(video.stat().st_size, 0)

    def test_schema_uses_official_autogrow_prefixes_when_comfy_v3_is_available(self):
        if not hasattr(LlamaServerH3OmniGenerate, "define_schema"):
            input_types = LlamaServerH3OmniGenerate.INPUT_TYPES()
            inputs = input_types["optional"]
            self.assertIn("ref_image_0", inputs)
            self.assertIn("ref_video_0", inputs)
            self.assertIn("ref_video_audio_0", inputs)
            self.assertIn("ref_audio_0", inputs)
            self.assertEqual(
                input_types["required"]["idle_timeout_minutes"][1]["default"],
                5,
            )
            return

        schema = LlamaServerH3OmniGenerate.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        expected = {
            "ref_images": ("ref_image_", 9),
            "ref_videos": ("ref_video_", 3),
            "ref_video_audios": ("ref_video_audio_", 3),
            "ref_audios": ("ref_audio_", 3),
        }
        for input_id, (prefix, maximum) in expected.items():
            template = inputs[input_id].template
            self.assertEqual(template.prefix, prefix)
            self.assertEqual(template.min, 0)
            self.assertEqual(template.max, maximum)
        self.assertEqual(schema.outputs[-1].id, "groups")
        self.assertEqual(inputs["idle_timeout_minutes"].default, 5)


if __name__ == "__main__":
    unittest.main()
