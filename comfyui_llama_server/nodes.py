from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted

from .backend import BackendInstaller
from .client import LlamaServerClient
from .config import ServerConfig
from .h3_omni import RATIOS, TASKS, build_h3_omni_preset
from .h3_reference_inputs import (
    collect_official_references,
    materialize_reference_media,
    pack_director_group,
    select_official_references,
)
from .lifecycle import install_comfy_memory_cleanup_hook, register_memory_cleanup_route
from .manager import ServerManager
from .media import build_media_content
from .service import GenerationService


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PLUGIN_ROOT / "runtime"


def _register_llm_folder() -> None:
    llm_path = str(Path(folder_paths.models_dir) / "LLM")
    extensions = {".gguf"}
    current = folder_paths.folder_names_and_paths.get("LLM")
    if current is None:
        folder_paths.folder_names_and_paths["LLM"] = ([llm_path], extensions)
        return
    paths, current_extensions = current
    paths = list(paths)
    if llm_path not in paths:
        paths.append(llm_path)
    folder_paths.folder_names_and_paths["LLM"] = (
        paths,
        set(current_extensions) | extensions,
    )


_register_llm_folder()


_installer = BackendInstaller(RUNTIME_ROOT)
_manager = ServerManager(
    executable_provider=lambda backend: _installer.ensure_installed(backend),
    log_dir=PLUGIN_ROOT / "logs",
)
_service = GenerationService(_manager, LlamaServerClient())
register_memory_cleanup_route(_manager)
install_comfy_memory_cleanup_hook(_manager)


def _all_gguf() -> list[str]:
    return sorted(
        (
            name
            for name in folder_paths.get_filename_list("LLM")
            if Path(name).suffix.casefold() == ".gguf"
        ),
        key=str.casefold,
    )


def _is_mmproj(name: str) -> bool:
    return Path(name).name.casefold().startswith("mmproj")


def _default_media_root() -> Path:
    candidates = []
    for getter_name in ("get_temp_directory", "get_input_directory"):
        getter = getattr(folder_paths, getter_name, None)
        if callable(getter):
            candidates.append((getter_name, getter))

    base_path = getattr(folder_paths, "base_path", None)
    candidates.append(
        (
            "ComfyUI base directory",
            lambda: base_path or Path(folder_paths.models_dir).parent,
        )
    )

    failures = []
    for label, getter in candidates:
        try:
            root = Path(getter()).expanduser().resolve()
            (root / "llamaserve_doc").mkdir(parents=True, exist_ok=True)
            return root
        except (OSError, TypeError, ValueError) as error:
            failures.append(f"{label}: {error}")

    raise NotADirectoryError(
        "No writable ComfyUI temp or input directory is available for llama-server media: "
        + "; ".join(failures)
    )


def build_chat_request(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    seed: int,
    reasoning: str,
    media=None,
    media_root=None,
) -> dict:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    user_content = (
        build_media_content(user_prompt, media, media_root)
        if media
        else user_prompt
    )
    messages.append({"role": "user", "content": user_content})
    request = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
    }
    if reasoning == "off":
        request["reasoning_effort"] = "none"
        request["chat_template_kwargs"] = {"enable_thinking": False}
    elif reasoning == "on":
        request["reasoning_effort"] = "medium"
        request["chat_template_kwargs"] = {"enable_thinking": True}
    return request


def generate_h3_omni(
    server_config,
    mode,
    raw_prompt,
    duration,
    resolution,
    max_tokens,
    temperature,
    top_k,
    top_p,
    min_p,
    repeat_penalty,
    seed,
    reasoning,
    stop_server_after_generate,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    generation_service=None,
    idle_timeout_minutes=5,
    **flat_references,
):
    references = collect_official_references(
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
        **flat_references,
    )
    media_root = getattr(server_config, "media_root", None)
    if media_root is None:
        raise ValueError("Internal ComfyUI media directory is unavailable")
    service = generation_service or _service
    described_media = tuple(
        {"type": slot.media_type, "source": slot.source}
        for slot in references.ordered
    )
    system_prompt, user_prompt, selected_media = build_h3_omni_preset(
        mode,
        raw_prompt,
        duration,
        resolution,
        described_media,
    )
    selected_references = select_official_references(
        references,
        {item["source"] for item in selected_media},
    )
    with materialize_reference_media(selected_references, media_root) as media:
        materialized_by_source = {item["source"]: item for item in media}
        formatted_media = tuple(
            {**item, **materialized_by_source[item["source"]]}
            for item in selected_media
        )
        request = build_chat_request(
            system_prompt,
            user_prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            seed,
            reasoning,
            media=formatted_media,
            media_root=media_root,
        )
        result = service.generate(
            server_config,
            request,
            stop_after_generate=stop_server_after_generate,
            interrupt_check=throw_exception_if_processing_interrupted,
            idle_timeout_seconds=max(0, idle_timeout_minutes) * 60,
        )
    stats = {
        "tokens_per_second": result.tokens_per_second,
        "prompt_tokens_per_second": result.prompt_tokens_per_second,
        "timings": result.raw_timings,
        "server_pid": _manager.owned_pid,
        "server_kept_running": not stop_server_after_generate,
        "idle_timeout_minutes": 0 if stop_server_after_generate else idle_timeout_minutes,
    }
    group = pack_director_group(mode, result.text, duration, selected_references)
    return result.text, json.dumps(stats, ensure_ascii=False, indent=2), group


class LlamaServerH3OmniPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(TASKS),),
                "raw_prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "resolution": (list(RATIOS), {"default": "16:9"}),
            },
            "optional": {"media": ("LLAMASERVE_MEDIA",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMASERVE_MEDIA")
    RETURN_NAMES = ("system_prompt", "user_prompt", "media")
    FUNCTION = "build"
    CATEGORY = "LlamaServe-Doc"
    DESCRIPTION = (
        "按 Lightx2v 官方格式生成 T2AV、I2AV、L2AV、FL2AV 或 Ref2AV 请求，"
        "并校验参考媒体数量、顺序、标签、时长和画幅。"
    )

    def build(self, mode, raw_prompt, duration, resolution, media=None):
        return build_h3_omni_preset(mode, raw_prompt, duration, resolution, media)


try:
    from comfy_api.latest import io as comfy_io
except ImportError:  # Older ComfyUI and lightweight unit-test environments.
    comfy_io = None


if comfy_io is not None:
    _ServerConfigType = comfy_io.Custom("LLAMA_SERVER_CONFIG")
    _DirectorGroupType = comfy_io.Custom("MMX_DIR_GROUP")

    class LlamaServerH3OmniGenerate(comfy_io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="LlamaServeDocH3OmniGenerate",
                display_name="LlamaServe-Doc H3 Omni Generate",
                category="LlamaServe-Doc",
                description=(
                    "Lightx2v H3 Omni official presets + llama-server generation. "
                    "Reference inputs use the same Autogrow names and ordering as the "
                    "official MiniMax H3 Reference to Video node."
                ),
                is_output_node=True,
                inputs=[
                    _ServerConfigType.Input("server_config"),
                    comfy_io.Combo.Input("mode", options=list(TASKS), default="Ref2AV"),
                    comfy_io.String.Input("raw_prompt", multiline=True, default=""),
                    comfy_io.Int.Input("duration", default=5, min=4, max=15, step=1),
                    comfy_io.Combo.Input("resolution", options=list(RATIOS), default="16:9"),
                    comfy_io.Int.Input("max_tokens", default=2048, min=1, max=32768),
                    comfy_io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.01),
                    comfy_io.Int.Input("top_k", default=20, min=0, max=200),
                    comfy_io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01),
                    comfy_io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    comfy_io.Float.Input("repeat_penalty", default=1.05, min=0.0, max=2.0, step=0.01),
                    comfy_io.Int.Input("seed", default=0, min=-1, max=0x7FFFFFFFFFFFFFFF),
                    comfy_io.Combo.Input("reasoning", options=["off", "auto", "on"], default="off"),
                    comfy_io.Boolean.Input("stop_server_after_generate", default=False),
                    comfy_io.Int.Input(
                        "idle_timeout_minutes",
                        default=5,
                        min=0,
                        max=1440,
                        step=1,
                        tooltip="0 disables automatic idle shutdown.",
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_images",
                        optional=True,
                        tooltip="Reference images → <Picture N>.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input("ref_image"),
                            prefix="ref_image_",
                            min=0,
                            max=9,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_videos",
                        optional=True,
                        tooltip="Reference video frame batches at 24 fps → <Video N>.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input("ref_video"),
                            prefix="ref_video_",
                            min=0,
                            max=3,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_video_audios",
                        optional=True,
                        tooltip="Soundtrack paired with the same-numbered ref_video_N.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Audio.Input("ref_video_audio"),
                            prefix="ref_video_audio_",
                            min=0,
                            max=3,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_audios",
                        optional=True,
                        tooltip="Standalone reference audio → <Audio N>.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Audio.Input("ref_audio"),
                            prefix="ref_audio_",
                            min=0,
                            max=3,
                        ),
                    ),
                ],
                outputs=[
                    comfy_io.String.Output("text"),
                    comfy_io.String.Output("performance_json"),
                    _DirectorGroupType.Output("groups"),
                ],
            )

        @classmethod
        def execute(
            cls,
            server_config,
            mode,
            raw_prompt,
            duration,
            resolution,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            seed,
            reasoning,
            stop_server_after_generate,
            idle_timeout_minutes,
            ref_images=None,
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
        ):
            return comfy_io.NodeOutput(
                *generate_h3_omni(
                    server_config,
                    mode,
                    raw_prompt,
                    duration,
                    resolution,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repeat_penalty,
                    seed,
                    reasoning,
                    stop_server_after_generate,
                    idle_timeout_minutes=idle_timeout_minutes,
                    ref_images=ref_images,
                    ref_videos=ref_videos,
                    ref_video_audios=ref_video_audios,
                    ref_audios=ref_audios,
                )
            )

else:
    class LlamaServerH3OmniGenerate:
        @classmethod
        def INPUT_TYPES(cls):
            optional = {}
            for index in range(9):
                optional[f"ref_image_{index}"] = ("IMAGE",)
            for index in range(3):
                optional[f"ref_video_{index}"] = ("IMAGE",)
                optional[f"ref_video_audio_{index}"] = ("AUDIO",)
                optional[f"ref_audio_{index}"] = ("AUDIO",)
            return {
                "required": {
                    "server_config": ("LLAMA_SERVER_CONFIG",),
                    "mode": (list(TASKS), {"default": "Ref2AV"}),
                    "raw_prompt": ("STRING", {"default": "", "multiline": True}),
                    "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                    "resolution": (list(RATIOS), {"default": "16:9"}),
                    "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                    "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                    "top_k": ("INT", {"default": 20, "min": 0, "max": 200}),
                    "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                    "seed": ("INT", {"default": 0, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                    "reasoning": (["off", "auto", "on"], {"default": "off"}),
                    "stop_server_after_generate": ("BOOLEAN", {"default": False}),
                    "idle_timeout_minutes": (
                        "INT",
                        {"default": 5, "min": 0, "max": 1440, "step": 1},
                    ),
                },
                "optional": optional,
            }

        RETURN_TYPES = ("STRING", "STRING", "MMX_DIR_GROUP")
        RETURN_NAMES = ("text", "performance_json", "groups")
        FUNCTION = "generate"
        CATEGORY = "LlamaServe-Doc"
        OUTPUT_NODE = True
        DESCRIPTION = "H3 Omni official presets, reference inputs, generation, and Director groups."

        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return float("nan")

        def generate(self, **kwargs):
            return generate_h3_omni(**kwargs)


class LlamaServerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in _all_gguf() if not _is_mmproj(name)]
        mmproj = ["None"] + [name for name in _all_gguf() if _is_mmproj(name)]
        return {
            "required": {
                "model": (models or ["No GGUF models found in models/LLM"],),
                "mmproj": (mmproj,),
                "context_length": ("INT", {"default": 4096, "min": 512, "max": 262144, "step": 512}),
                "gpu_layers": ("INT", {"default": 47, "min": -1, "max": 999, "step": 1}),
                "flash_attention": (["on", "auto", "off"], {"default": "on"}),
                "cache_type_k": (["q8_0", "f16", "q4_0"], {"default": "q8_0"}),
                "cache_type_v": (["q8_0", "f16", "q4_0"], {"default": "q8_0"}),
                "backend": (["auto", "cuda", "vulkan", "metal", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("LLAMA_SERVER_CONFIG",)
    RETURN_NAMES = ("server_config",)
    FUNCTION = "load"
    CATEGORY = "LlamaServe-Doc"
    DESCRIPTION = "配置原生 llama-server。后端在首次执行时自动下载，不需要安装 LM Studio。"

    def load(
        self,
        model,
        mmproj,
        context_length,
        gpu_layers,
        flash_attention,
        cache_type_k,
        cache_type_v,
        backend="auto",
    ):
        model_path = folder_paths.get_full_path("LLM", model)
        if not model_path:
            raise FileNotFoundError(f"GGUF model not found: {model}")
        mmproj_path = None
        if mmproj != "None":
            mmproj_path = folder_paths.get_full_path("LLM", mmproj)
            if not mmproj_path:
                raise FileNotFoundError(f"mmproj not found: {mmproj}")
        resolved_media_root = _default_media_root()
        if not resolved_media_root.is_dir():
            raise NotADirectoryError(f"ComfyUI media directory is not a directory: {resolved_media_root}")
        return (
            ServerConfig(
                model_path=Path(model_path),
                mmproj_path=Path(mmproj_path) if mmproj_path else None,
                context_length=context_length,
                gpu_layers=gpu_layers,
                flash_attention=flash_attention,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                port=0,
                media_root=resolved_media_root,
                backend=backend,
            ),
        )


class LlamaServerGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_config": ("LLAMA_SERVER_CONFIG",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "reasoning": (["off", "auto", "on"], {"default": "off"}),
                "stop_server_after_generate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "生成后释放显存",
                        "label_off": "保持服务器以便下次复用",
                    },
                ),
                "idle_timeout_minutes": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 1440,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "media": ("LLAMASERVE_MEDIA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "performance_json")
    FUNCTION = "generate"
    CATEGORY = "LlamaServe-Doc"
    OUTPUT_NODE = True
    DESCRIPTION = "通过独立 llama-server 高效推理；停止开关已集成在本节点中。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def generate(
        self,
        server_config,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        seed,
        reasoning,
        stop_server_after_generate,
        idle_timeout_minutes=5,
        media=None,
    ):
        request = build_chat_request(
            system_prompt,
            user_prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            seed,
            reasoning,
            media=media,
            media_root=getattr(server_config, "media_root", None),
        )
        result = _service.generate(
            server_config,
            request,
            stop_after_generate=stop_server_after_generate,
            interrupt_check=throw_exception_if_processing_interrupted,
            idle_timeout_seconds=max(0, idle_timeout_minutes) * 60,
        )
        stats = {
            "tokens_per_second": result.tokens_per_second,
            "prompt_tokens_per_second": result.prompt_tokens_per_second,
            "timings": result.raw_timings,
            "server_pid": _manager.owned_pid,
            "server_kept_running": not stop_server_after_generate,
            "idle_timeout_minutes": 0 if stop_server_after_generate else idle_timeout_minutes,
        }
        return result.text, json.dumps(stats, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "LlamaServeDocLoader": LlamaServerLoader,
    "LlamaServeDocH3OmniPreset": LlamaServerH3OmniPreset,
    "LlamaServeDocH3OmniGenerate": LlamaServerH3OmniGenerate,
    "LlamaServeDocGenerate": LlamaServerGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaServeDocLoader": "LlamaServe-Doc Loader",
    "LlamaServeDocH3OmniPreset": "LlamaServe-Doc H3 Omni Preset",
    "LlamaServeDocH3OmniGenerate": "LlamaServe-Doc H3 Omni Generate",
    "LlamaServeDocGenerate": "LlamaServe-Doc Generate",
}
