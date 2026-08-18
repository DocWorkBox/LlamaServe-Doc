import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.service import GenerationService


class FakeManager:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def ensure_started(self, config, interrupt_check=None):
        self.started += 1
        self.interrupt_check = interrupt_check
        return "http://127.0.0.1:8191"

    def stop(self):
        self.stopped += 1


class FakeClient:
    def __init__(self, error=None):
        self.error = error

    def generate(self, base_url, request, interrupt_check=None, on_text=None):
        if self.error:
            raise self.error
        return "result"


class GenerationServiceTests(unittest.TestCase):
    def test_keeps_server_running_by_default(self):
        manager = FakeManager()
        service = GenerationService(manager, FakeClient())

        result = service.generate("config", {"messages": []}, stop_after_generate=False)

        self.assertEqual(result, "result")
        self.assertEqual(manager.started, 1)
        self.assertEqual(manager.stopped, 0)

    def test_stops_owned_server_after_success_when_enabled(self):
        manager = FakeManager()
        service = GenerationService(manager, FakeClient())

        service.generate("config", {"messages": []}, stop_after_generate=True)

        self.assertEqual(manager.stopped, 1)

    def test_stops_owned_server_after_error_when_enabled(self):
        manager = FakeManager()
        service = GenerationService(manager, FakeClient(RuntimeError("boom")))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            service.generate("config", {"messages": []}, stop_after_generate=True)

        self.assertEqual(manager.stopped, 1)

    def test_passes_interrupt_check_to_startup_and_generation(self):
        manager = FakeManager()
        client = FakeClient()
        service = GenerationService(manager, client)
        interrupt_check = lambda: None

        service.generate("config", {"messages": []}, False, interrupt_check=interrupt_check)

        self.assertIs(manager.interrupt_check, interrupt_check)


if __name__ == "__main__":
    unittest.main()
