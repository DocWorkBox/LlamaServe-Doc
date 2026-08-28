from __future__ import annotations

import logging


ROUTE_PATH = "/llamaserve_doc/stop"
_ROUTE_MARKER = "_llamaserve_doc_cleanup_route_registered"
_HOOK_MARKER = "_llamaserve_doc_memory_cleanup_hook"


def install_comfy_memory_cleanup_hook(manager, *, model_management=None) -> bool:
    if model_management is None:
        try:
            from comfy import model_management
        except ImportError:
            return False

    existing = getattr(model_management, _HOOK_MARKER, None)
    if existing is not None:
        existing["manager"] = manager
        return False

    original = getattr(model_management, "unload_all_models", None)
    if not callable(original):
        return False

    state = {"manager": manager, "original": original}

    def unload_all_models_and_stop_llama(*args, **kwargs):
        try:
            return state["original"](*args, **kwargs)
        finally:
            try:
                state["manager"].stop()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not stop llama-server during ComfyUI memory cleanup"
                )

    setattr(model_management, "unload_all_models", unload_all_models_and_stop_llama)
    setattr(model_management, _HOOK_MARKER, state)
    return True


def register_memory_cleanup_route(
    manager,
    *,
    prompt_server=None,
    response_factory=None,
) -> bool:
    if prompt_server is None:
        try:
            from server import PromptServer
        except ImportError:
            return False
        prompt_server = PromptServer.instance
    if prompt_server is None or getattr(prompt_server, _ROUTE_MARKER, False):
        return False

    if response_factory is None:
        from aiohttp import web

        response_factory = web.json_response

    @prompt_server.routes.post(ROUTE_PATH)
    async def stop_owned_llama_server(_request):
        pid = manager.owned_pid
        manager.stop()
        return response_factory({"stopped": pid is not None, "pid": pid})

    setattr(prompt_server, _ROUTE_MARKER, True)
    return True
