from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted

from .backend import BackendInstaller
from .client import LlamaServerClient
from .config import ServerConfig
from .manager import ServerManager
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
    executable_provider=lambda: _installer.ensure_installed("cuda12"),
    log_dir=PLUGIN_ROOT / "logs",
)
_service = GenerationService(_manager, LlamaServerClient())


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
) -> dict:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
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
                "port": ("INT", {"default": 8191, "min": 1024, "max": 65535}),
            }
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
        port,
    ):
        model_path = folder_paths.get_full_path("LLM", model)
        if not model_path:
            raise FileNotFoundError(f"GGUF model not found: {model}")
        mmproj_path = None
        if mmproj != "None":
            mmproj_path = folder_paths.get_full_path("LLM", mmproj)
            if not mmproj_path:
                raise FileNotFoundError(f"mmproj not found: {mmproj}")
        return (
            ServerConfig(
                model_path=Path(model_path),
                mmproj_path=Path(mmproj_path) if mmproj_path else None,
                context_length=context_length,
                gpu_layers=gpu_layers,
                flash_attention=flash_attention,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                port=port,
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
            }
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
        )
        result = _service.generate(
            server_config,
            request,
            stop_after_generate=stop_server_after_generate,
            interrupt_check=throw_exception_if_processing_interrupted,
        )
        stats = {
            "tokens_per_second": result.tokens_per_second,
            "prompt_tokens_per_second": result.prompt_tokens_per_second,
            "timings": result.raw_timings,
            "server_pid": _manager.owned_pid,
            "server_kept_running": not stop_server_after_generate,
        }
        return result.text, json.dumps(stats, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "LlamaServeDocLoader": LlamaServerLoader,
    "LlamaServeDocGenerate": LlamaServerGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaServeDocLoader": "LlamaServe-Doc Loader",
    "LlamaServeDocGenerate": "LlamaServe-Doc Generate",
}
