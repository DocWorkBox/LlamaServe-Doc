from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    model_path: Path
    mmproj_path: Path | None
    context_length: int
    gpu_layers: int
    flash_attention: str
    cache_type_k: str
    cache_type_v: str
    host: str = "127.0.0.1"
    port: int = 0
    media_root: Path | None = None

    def build_command(self, executable: Path) -> list[str]:
        gpu_layers = "auto" if self.gpu_layers == -1 else str(self.gpu_layers)
        command = [
            executable.as_posix(),
            "--model",
            self.model_path.as_posix(),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_length),
            "--gpu-layers",
            gpu_layers,
            "--flash-attn",
            self.flash_attention,
            "--cache-type-k",
            self.cache_type_k,
            "--cache-type-v",
            self.cache_type_v,
            "--parallel",
            "1",
            "--jinja",
            "--metrics",
            "--reasoning",
            "auto",
        ]
        if self.mmproj_path is not None:
            command.extend(["--mmproj", self.mmproj_path.as_posix()])
        if self.media_root is not None:
            command.extend(["--media-path", self.media_root.as_posix()])
        return command

    def signature(self) -> tuple[object, ...]:
        return (
            self.model_path.resolve(),
            self.mmproj_path.resolve() if self.mmproj_path else None,
            self.context_length,
            self.gpu_layers,
            self.flash_attention,
            self.cache_type_k,
            self.cache_type_v,
            self.host,
            self.port,
            self.media_root.resolve() if self.media_root else None,
        )
