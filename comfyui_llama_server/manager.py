from __future__ import annotations

import atexit
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import platform
import socket
import subprocess
import threading
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


def _select_available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _prepend_environment_path(environment: dict[str, str], name: str, directory: str) -> None:
    current = environment.get(name)
    environment[name] = directory if not current else f"{directory}{os.pathsep}{current}"


def _runtime_environment(executable: Path) -> dict[str, str]:
    environment = os.environ.copy()
    runtime_directory = str(executable.parent)
    _prepend_environment_path(environment, "PATH", runtime_directory)
    system = platform.system()
    if system == "Linux":
        _prepend_environment_path(environment, "LD_LIBRARY_PATH", runtime_directory)
    elif system == "Darwin":
        _prepend_environment_path(environment, "DYLD_LIBRARY_PATH", runtime_directory)
    return environment


class ServerManager:
    def __init__(
        self,
        executable_provider,
        process_factory=subprocess.Popen,
        health_check=_health_check,
        port_available=_port_available,
        port_selector=_select_available_port,
        sleep=time.sleep,
        startup_timeout: float = 180,
        log_dir: Path | None = None,
        timer_factory=threading.Timer,
    ):
        self._executable_provider = executable_provider
        self._process_factory = process_factory
        self._health_check = health_check
        self._port_available = port_available
        self._port_selector = port_selector
        self._sleep = sleep
        self.startup_timeout = startup_timeout
        self.log_dir = Path(log_dir or Path.cwd() / "logs")
        self._timer_factory = timer_factory
        self._process = None
        self._signature = None
        self._base_url = None
        self._log_handle = None
        self._idle_timer = None
        atexit.register(self.stop)

    @property
    def owned_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def ensure_started(self, config, interrupt_check=None) -> str:
        self.cancel_idle_stop()
        signature = config.signature()
        if (
            self._process is not None
            and self._process.poll() is None
            and self._signature == signature
            and self._base_url is not None
            and self._health_check(self._base_url)
        ):
            return self._base_url

        if self._process is not None:
            self.stop()
        if config.port == 0:
            config = replace(config, port=self._port_selector(config.host))
        elif not self._port_available(config.host, config.port):
            raise ServerStartError(
                f"Port {config.port} is already in use by a process not owned by this node."
            )

        base_url = f"http://{config.host}:{config.port}"

        executable = Path(self._executable_provider(config.backend))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"llama-server-{datetime.now():%Y%m%d-%H%M%S}.log"
        self._log_handle = log_path.open("a", encoding="utf-8")
        kwargs = {
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
            "cwd": str(executable.parent),
            "env": _runtime_environment(executable),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._process = self._process_factory(config.build_command(executable), **kwargs)
        self._signature = signature
        self._base_url = base_url

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

    def cancel_idle_stop(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def schedule_idle_stop(self, timeout_seconds: float) -> None:
        self.cancel_idle_stop()
        process = self._process
        if timeout_seconds <= 0 or process is None or process.poll() is not None:
            return

        timer = None

        def stop_if_still_idle() -> None:
            if self._idle_timer is not timer:
                return
            self._idle_timer = None
            if self._process is process and process.poll() is None:
                self.stop()

        timer = self._timer_factory(float(timeout_seconds), stop_if_still_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def stop(self) -> None:
        self.cancel_idle_stop()
        process = self._process
        self._process = None
        self._signature = None
        self._base_url = None
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
