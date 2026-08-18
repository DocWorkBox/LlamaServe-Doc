from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
import zipfile


class BackendInstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    sha256: str


def verify_file_hash(path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.lower()


def find_server_executable(root: Path) -> Path:
    matches = sorted(root.rglob("llama-server.exe"))
    if not matches:
        raise BackendInstallError(f"llama-server.exe was not found under {root}")
    return matches[0]


def _asset_from_json(asset: dict) -> ReleaseAsset:
    digest = asset.get("digest") or ""
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise BackendInstallError(f"Release asset has no SHA-256 digest: {asset.get('name', '<unknown>')}")
    return ReleaseAsset(
        name=asset["name"],
        url=asset["browser_download_url"],
        sha256=digest.removeprefix("sha256:").lower(),
    )


def select_release_assets(release: dict, backend: str) -> list[ReleaseAsset]:
    assets = release.get("assets") or []
    if backend != "cuda12":
        raise BackendInstallError(f"Unsupported backend: {backend}")

    base_pattern = re.compile(r"^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$")
    runtime_name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
    base = next((asset for asset in assets if base_pattern.match(asset.get("name", ""))), None)
    runtime = next((asset for asset in assets if asset.get("name") == runtime_name), None)
    if base is None or runtime is None:
        raise BackendInstallError(f"Release {release.get('tag_name', '<unknown>')} has no complete CUDA 12 Windows backend")
    return [_asset_from_json(base), _asset_from_json(runtime)]


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise BackendInstallError(f"Archive entry escapes destination: {member.filename}")
        archive.extractall(destination)


def build_curl_download_command(url: str, destination: Path) -> list[str]:
    return [
        "curl.exe",
        "--fail",
        "--location",
        "--show-error",
        "--progress-bar",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "15",
        "--continue-at",
        "-",
        "--output",
        destination.as_posix(),
        url,
    ]


def _fetch_latest_release() -> dict:
    if os.name == "nt" and shutil.which("curl.exe"):
        try:
            completed = subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "60",
                    "--header",
                    "Accept: application/vnd.github+json",
                    "--header",
                    "User-Agent: ComfyUI-LlamaServer",
                    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(completed.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            raise BackendInstallError(f"Could not read the latest llama.cpp release: {error}") from error
    request = Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ComfyUI-LlamaServer",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _download(url: str, destination: Path, progress=None) -> None:
    if os.name == "nt" and shutil.which("curl.exe"):
        try:
            subprocess.run(build_curl_download_command(url, destination), check=True)
            return
        except subprocess.CalledProcessError as error:
            raise BackendInstallError(f"Backend download failed with curl exit code {error.returncode}") from error
    request = Request(url, headers={"User-Agent": "ComfyUI-LlamaServer"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if progress is not None:
                progress(received, total)


class BackendInstaller:
    def __init__(self, runtime_dir: Path, fetch_release=_fetch_latest_release, downloader=_download):
        self.runtime_dir = Path(runtime_dir)
        self._fetch_release = fetch_release
        self._downloader = downloader

    def _current_executable(self, backend: str) -> Path | None:
        pointer = self.runtime_dir / backend / "current.json"
        if not pointer.is_file():
            return None
        try:
            data = json.loads(pointer.read_text("utf-8"))
            executable = (pointer.parent / data["executable"]).resolve()
            backend_root = pointer.parent.resolve()
            if backend_root != executable and backend_root not in executable.parents:
                return None
            return executable if executable.is_file() else None
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def ensure_installed(self, backend: str = "cuda12", progress=None) -> Path:
        existing = self._current_executable(backend)
        if existing is not None:
            return existing

        backend_root = self.runtime_dir / backend
        releases_root = backend_root / "releases"
        downloads_root = backend_root / "downloads"
        releases_root.mkdir(parents=True, exist_ok=True)
        downloads_root.mkdir(parents=True, exist_ok=True)
        release = self._fetch_release()
        assets = select_release_assets(release, backend)
        release_tag = str(release.get("tag_name") or "unknown")

        staging = Path(tempfile.mkdtemp(prefix=".install-", dir=releases_root))
        try:
            extracted = staging / "extracted"
            extracted.mkdir()
            for asset in assets:
                archive = downloads_root / asset.name
                if not archive.is_file() or not verify_file_hash(archive, asset.sha256):
                    self._downloader(asset.url, archive, progress)
                if not verify_file_hash(archive, asset.sha256):
                    raise BackendInstallError(f"SHA-256 verification failed: {asset.name}")
                safe_extract_zip(archive, extracted)

            find_server_executable(extracted)
            payload = staging / "payload"
            payload.mkdir()
            for source in sorted(path for path in extracted.rglob("*") if path.is_file()):
                target = payload / source.name
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise BackendInstallError(f"Conflicting files in release archives: {source.name}")
                shutil.copy2(source, target)

            executable = payload / "llama-server.exe"
            if not executable.is_file():
                raise BackendInstallError("The extracted backend has no llama-server.exe")
            manifest = {
                "release": release_tag,
                "backend": backend,
                "assets": [
                    {"name": asset.name, "sha256": asset.sha256}
                    for asset in assets
                ],
            }
            (payload / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            destination = releases_root / f"{release_tag}-{os.getpid()}"
            suffix = 1
            while destination.exists():
                destination = releases_root / f"{release_tag}-{os.getpid()}-{suffix}"
                suffix += 1
            payload.replace(destination)

            relative_executable = (destination / "llama-server.exe").relative_to(backend_root)
            pointer_staging = backend_root / ".current.json.tmp"
            pointer_staging.write_text(
                json.dumps({"executable": relative_executable.as_posix()}, indent=2),
                encoding="utf-8",
            )
            os.replace(pointer_staging, backend_root / "current.json")
            for asset in assets:
                (downloads_root / asset.name).unlink(missing_ok=True)
            return destination / "llama-server.exe"
        except Exception:
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
