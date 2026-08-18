import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.config import ServerConfig
from comfyui_llama_server.manager import ServerManager


class FakeProcess:
    next_pid = 100

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed += 1
        self.returncode = -9


def config(gpu_layers=47):
    return ServerConfig(Path("C:/model.gguf"), None, 4096, gpu_layers, "on", "f16", "f16", port=8191)


class ServerManagerTests(unittest.TestCase):
    def setUp(self):
        self.processes = []
        self.temp_dir = tempfile.TemporaryDirectory()

        def process_factory(command, **kwargs):
            process = FakeProcess(command, **kwargs)
            self.processes.append(process)
            return process

        self.manager = ServerManager(
            executable_provider=lambda: Path("C:/runtime/llama-server.exe"),
            process_factory=process_factory,
            health_check=lambda _url: True,
            port_available=lambda _host, _port: True,
            sleep=lambda _seconds: None,
            startup_timeout=1,
            log_dir=Path(self.temp_dir.name),
        )

    def tearDown(self):
        self.manager.stop()
        self.temp_dir.cleanup()

    def test_reuses_owned_healthy_server_for_same_config(self):
        first_url = self.manager.ensure_started(config())
        second_url = self.manager.ensure_started(config())

        self.assertEqual(first_url, second_url)
        self.assertEqual(len(self.processes), 1)

    def test_restarts_owned_server_when_config_changes(self):
        self.manager.ensure_started(config(47))
        first_process = self.processes[0]

        self.manager.ensure_started(config(46))

        self.assertEqual(first_process.terminated, 1)
        self.assertEqual(len(self.processes), 2)

    def test_stop_only_terminates_process_created_by_manager(self):
        self.manager.stop()
        self.assertEqual(len(self.processes), 0)

        self.manager.ensure_started(config())
        owned = self.processes[0]
        self.manager.stop()

        self.assertEqual(owned.terminated, 1)
        self.assertIsNone(self.manager.owned_pid)


if __name__ == "__main__":
    unittest.main()
