from __future__ import annotations

import atexit
from datetime import datetime
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


class ServerStartError(RuntimeError):
    pass


def _health_check(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/health", timeout=1) as response:
            return 200 <= response.status < 300
    except OSError:
        return False


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


class ServerManager:
    def __init__(
        self,
        executable_provider,
        process_factory=subprocess.Popen,
        health_check=_health_check,
        port_available=_port_available,
        sleep=time.sleep,
        startup_timeout: float = 180,
        log_dir: Path | None = None,
    ):
        self._executable_provider = executable_provider
        self._process_factory = process_factory
        self._health_check = health_check
        self._port_available = port_available
        self._sleep = sleep
        self.startup_timeout = startup_timeout
        self.log_dir = Path(log_dir or Path.cwd() / "logs")
        self._process = None
        self._signature = None
        self._log_handle = None
        atexit.register(self.stop)

    @property
    def owned_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def ensure_started(self, config, interrupt_check=None) -> str:
        base_url = f"http://{config.host}:{config.port}"
        signature = config.signature()
        if (
            self._process is not None
            and self._process.poll() is None
            and self._signature == signature
            and self._health_check(base_url)
        ):
            return base_url

        if self._process is not None:
            self.stop()
        elif not self._port_available(config.host, config.port):
            raise ServerStartError(
                f"Port {config.port} is already in use by a process not owned by this node."
            )

        executable = Path(self._executable_provider())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"llama-server-{datetime.now():%Y%m%d-%H%M%S}.log"
        self._log_handle = log_path.open("a", encoding="utf-8")
        kwargs = {
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
            "cwd": str(executable.parent),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._process = self._process_factory(config.build_command(executable), **kwargs)
        self._signature = signature

        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if interrupt_check is not None:
                    interrupt_check()
                returncode = self._process.poll()
                if returncode is not None:
                    raise ServerStartError(
                        f"llama-server exited during startup with code {returncode}. See {log_path}"
                    )
                if self._health_check(base_url):
                    return base_url
                self._sleep(0.25)
            raise ServerStartError(
                f"llama-server did not become healthy within {self.startup_timeout:g} seconds. See {log_path}"
            )
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._signature = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
