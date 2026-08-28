import asyncio
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server import lifecycle
from comfyui_llama_server.lifecycle import register_memory_cleanup_route


class FakeRoutes:
    def __init__(self):
        self.handlers = {}

    def post(self, path):
        def register(handler):
            self.handlers[path] = handler
            return handler

        return register


class FakePromptServer:
    def __init__(self):
        self.routes = FakeRoutes()


class FakeManager:
    def __init__(self):
        self.owned_pid = 1234
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        self.owned_pid = None


class FakeModelManagement:
    def __init__(self):
        self.unload_calls = 0

    def unload_all_models(self):
        self.unload_calls += 1
        return "unloaded"


class CleanupRouteTests(unittest.TestCase):
    def test_cleanup_route_stops_only_the_injected_manager(self):
        prompt_server = FakePromptServer()
        manager = FakeManager()
        registered = register_memory_cleanup_route(
            manager,
            prompt_server=prompt_server,
            response_factory=lambda payload: payload,
        )

        response = asyncio.run(
            prompt_server.routes.handlers["/llamaserve_doc/stop"](None)
        )

        self.assertTrue(registered)
        self.assertEqual(manager.stopped, 1)
        self.assertEqual(response, {"stopped": True, "pid": 1234})

    def test_comfy_backend_model_cleanup_stops_owned_llama_server(self):
        self.assertTrue(
            hasattr(lifecycle, "install_comfy_memory_cleanup_hook"),
            "backend cleanup hook is missing",
        )
        manager = FakeManager()
        model_management = FakeModelManagement()

        installed = lifecycle.install_comfy_memory_cleanup_hook(
            manager,
            model_management=model_management,
        )
        result = model_management.unload_all_models()

        self.assertTrue(installed)
        self.assertEqual(result, "unloaded")
        self.assertEqual(model_management.unload_calls, 1)
        self.assertEqual(manager.stopped, 1)


if __name__ == "__main__":
    unittest.main()
