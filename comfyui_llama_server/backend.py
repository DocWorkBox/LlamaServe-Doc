from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import tarfile
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


@dataclass(frozen=True)
class RuntimeTarget:
    key: str
    executable_name: str
    asset_patterns: tuple[str, ...] = ()
    legacy_keys: tuple[str, ...] = ()
    source_build: bool = False


@dataclass(frozen=True)
class CudaToolchain:
    nvcc: str
    cmake: str
    git: str


RUNTIME_TARGETS = {
    "windows-x64-cuda12": RuntimeTarget(
        key="windows-x64-cuda12",
        executable_name="llama-server.exe",
        asset_patterns=(
            r"llama-b\d+-bin-win-cuda-12\.4-x64\.zip",
            r"cudart-llama-bin-win-cuda-12\.4-x64\.zip",
        ),
        legacy_keys=("cuda12",),
    ),
    "windows-x64-vulkan": RuntimeTarget(
        key="windows-x64-vulkan",
        executable_name="llama-server.exe",
        asset_patterns=(r"llama-b\d+-bin-win-vulkan-x64\.zip",),
    ),
    "windows-x64-cpu": RuntimeTarget(
        key="windows-x64-cpu",
        executable_name="llama-server.exe",
        asset_patterns=(r"llama-b\d+-bin-win-cpu-x64\.zip",),
    ),
    "windows-arm64-cpu": RuntimeTarget(
        key="windows-arm64-cpu",
        executable_name="llama-server.exe",
        asset_patterns=(r"llama-b\d+-bin-win-cpu-arm64\.zip",),
    ),
    "linux-x64-vulkan": RuntimeTarget(
        key="linux-x64-vulkan",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-ubuntu-vulkan-x64\.tar\.gz",),
    ),
    "linux-arm64-vulkan": RuntimeTarget(
        key="linux-arm64-vulkan",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-ubuntu-vulkan-arm64\.tar\.gz",),
    ),
    "linux-x64-cpu": RuntimeTarget(
        key="linux-x64-cpu",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-ubuntu-x64\.tar\.gz",),
    ),
    "linux-arm64-cpu": RuntimeTarget(
        key="linux-arm64-cpu",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-ubuntu-arm64\.tar\.gz",),
    ),
    "linux-x64-cuda": RuntimeTarget(
        key="linux-x64-cuda",
        executable_name="llama-server",
        source_build=True,
    ),
    "linux-arm64-cuda": RuntimeTarget(
        key="linux-arm64-cuda",
        executable_name="llama-server",
        source_build=True,
    ),
    "macos-arm64-metal": RuntimeTarget(
        key="macos-arm64-metal",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-macos-arm64\.tar\.gz",),
    ),
    "macos-x64-cpu": RuntimeTarget(
        key="macos-x64-cpu",
        executable_name="llama-server",
        asset_patterns=(r"llama-b\d+-bin-macos-x64\.tar\.gz",),
    ),
}


def find_cuda_toolchain(which=shutil.which, environ=None) -> CudaToolchain | None:
    environment = os.environ if environ is None else environ
    nvcc = which("nvcc")
    if not nvcc:
        cuda_root = environment.get("CUDA_HOME") or environment.get("CUDA_PATH")
        candidates = []
        if cuda_root:
            candidates.append(Path(cuda_root) / "bin" / "nvcc")
        candidates.extend((Path("/usr/local/cuda/bin/nvcc"), Path("/opt/cuda/bin/nvcc")))
        nvcc = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
    cmake = which("cmake")
    git = which("git")
    if not nvcc or not cmake or not git:
        return None
    return CudaToolchain(nvcc=str(nvcc), cmake=str(cmake), git=str(git))


def _normalized_system(system: str) -> str:
    names = {"windows": "windows", "linux": "linux", "darwin": "macos", "macos": "macos"}
    try:
        return names[system.casefold()]
    except KeyError as error:
        raise BackendInstallError(f"Unsupported operating system: {system}") from error


def _normalized_architecture(machine: str) -> str:
    names = {
        "amd64": "x64",
        "x86_64": "x64",
        "x64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    try:
        return names[machine.casefold()]
    except KeyError as error:
        raise BackendInstallError(f"Unsupported CPU architecture: {machine}") from error


def detect_runtime_target(
    backend: str = "auto",
    *,
    system: str | None = None,
    machine: str | None = None,
    which=shutil.which,
    environ=None,
) -> RuntimeTarget:
    requested = backend.casefold()
    if requested in RUNTIME_TARGETS:
        return RUNTIME_TARGETS[requested]
    system_name = _normalized_system(system or platform.system())
    architecture = _normalized_architecture(machine or platform.machine())
    cuda_toolchain = find_cuda_toolchain(which=which, environ=environ) if system_name == "linux" else None

    if requested == "cuda12":
        requested = "cuda"
    if requested == "auto":
        if system_name == "windows" and architecture == "x64":
            requested = "cuda"
        elif system_name == "macos" and architecture == "arm64":
            requested = "metal"
        elif system_name == "linux" and cuda_toolchain is not None:
            requested = "cuda"
        elif system_name == "linux" and which("vulkaninfo"):
            requested = "vulkan"
        else:
            requested = "cpu"

    if requested == "cuda":
        if system_name == "windows" and architecture == "x64":
            key = "windows-x64-cuda12"
        elif system_name == "linux":
            if cuda_toolchain is None:
                raise BackendInstallError(
                    "Linux CUDA auto-build requires nvcc, cmake, and git. Install the CUDA Toolkit and build tools, put an existing llama-server on PATH, or set LLAMASERVE_DOC_SERVER."
                )
            key = f"linux-{architecture}-cuda"
        else:
            raise BackendInstallError(f"CUDA is not available for {system_name}-{architecture}")
    elif requested == "metal":
        if system_name != "macos" or architecture != "arm64":
            raise BackendInstallError("The packaged Metal backend requires macOS on Apple Silicon")
        key = "macos-arm64-metal"
    elif requested == "vulkan":
        key = f"{system_name}-{architecture}-vulkan"
    elif requested == "cpu":
        key = f"{system_name}-{architecture}-cpu"
    else:
        raise BackendInstallError(f"Unsupported backend: {backend}")

    try:
        return RUNTIME_TARGETS[key]
    except KeyError as error:
        raise BackendInstallError(f"No packaged {requested} backend is available for {system_name}-{architecture}") from error


def verify_file_hash(path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.lower()


def find_server_executable(root: Path, executable_name: str | None = None) -> Path:
    name = executable_name or ("llama-server.exe" if os.name == "nt" else "llama-server")
    matches = sorted(root.rglob(name))
    if not matches:
        raise BackendInstallError(f"{name} was not found under {root}")
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


def _runtime_target(target: RuntimeTarget | str) -> RuntimeTarget:
    if isinstance(target, RuntimeTarget):
        return target
    key = "windows-x64-cuda12" if target == "cuda12" else target
    try:
        return RUNTIME_TARGETS[key]
    except KeyError as error:
        raise BackendInstallError(f"Unsupported runtime target: {target}") from error


def select_release_assets(release: dict, target: RuntimeTarget | str) -> list[ReleaseAsset]:
    runtime_target = _runtime_target(target)
    assets = release.get("assets") or []
    selected = []
    for pattern in runtime_target.asset_patterns:
        matcher = re.compile(f"^(?:{pattern})$")
        asset = next((item for item in assets if matcher.match(item.get("name", ""))), None)
        if asset is None:
            raise BackendInstallError(
                f"Release {release.get('tag_name', '<unknown>')} has no complete {runtime_target.key} backend"
            )
        selected.append(_asset_from_json(asset))
    return selected


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise BackendInstallError(f"Archive entry escapes destination: {member.filename}")
        archive.extractall(destination)


def _safe_tar_target(destination: Path, name: str) -> Path:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or (normalized.parts and ":" in normalized.parts[0]):
        raise BackendInstallError(f"Archive entry escapes destination: {name}")
    target = (destination / Path(*normalized.parts)).resolve()
    if target != destination and destination not in target.parents:
        raise BackendInstallError(f"Archive entry escapes destination: {name}")
    return target


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        targets = {member: _safe_tar_target(destination, member.name) for member in members}
        for member in members:
            target = targets[member]
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise BackendInstallError(f"Could not read archive entry: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            elif member.issym() or member.islnk():
                links.append(member)
            else:
                raise BackendInstallError(f"Unsupported archive entry type: {member.name}")

        pending = links
        while pending:
            unresolved = []
            for member in pending:
                target = targets[member]
                if member.issym():
                    source = _safe_tar_target(destination, str(PurePosixPath(member.name).parent / member.linkname))
                else:
                    source = _safe_tar_target(destination, member.linkname)
                if not source.is_file():
                    unresolved.append(member)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if len(unresolved) == len(pending):
                names = ", ".join(member.name for member in unresolved)
                raise BackendInstallError(f"Archive link targets are unavailable: {names}")
            pending = unresolved


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    name = archive_path.name.casefold()
    if name.endswith(".zip"):
        safe_extract_zip(archive_path, destination)
    elif name.endswith((".tar.gz", ".tgz")):
        safe_extract_tar(archive_path, destination)
    else:
        raise BackendInstallError(f"Unsupported backend archive: {archive_path.name}")


def build_curl_download_command(url: str, destination: Path, executable: str = "curl") -> list[str]:
    return [
        executable,
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


def _fetch_recent_releases() -> list[dict]:
    request = Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ComfyUI-LlamaServer",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            releases = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise BackendInstallError(f"Could not read recent llama.cpp releases: {error}") from error
    if not isinstance(releases, list):
        raise BackendInstallError("GitHub returned an invalid llama.cpp release list")
    return releases


def _download(url: str, destination: Path, progress=None) -> None:
    curl_name = "curl.exe" if os.name == "nt" else "curl"
    curl = shutil.which(curl_name)
    curl_error = None
    if curl:
        try:
            subprocess.run(build_curl_download_command(url, destination, curl), check=True)
            return
        except subprocess.CalledProcessError as error:
            curl_error = error
    request = Request(url, headers={"User-Agent": "ComfyUI-LlamaServer"})
    temporary = destination.with_name(f"{destination.name}.part")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
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
        os.replace(temporary, destination)
    except OSError as error:
        prefix = f"curl exited with code {curl_error.returncode}; " if curl_error else ""
        raise BackendInstallError(f"Backend download failed: {prefix}{error}") from error


def _select_complete_release(releases: list[dict], target: RuntimeTarget) -> tuple[dict, list[ReleaseAsset]]:
    for release in releases:
        try:
            return release, select_release_assets(release, target)
        except BackendInstallError:
            continue
    raise BackendInstallError(f"No recent llama.cpp release has a complete {target.key} backend with SHA-256 digests")


def _select_source_release(releases: list[dict]) -> dict:
    release = next(
        (item for item in releases if re.fullmatch(r"b\d+", str(item.get("tag_name") or ""))),
        None,
    )
    if release is None:
        raise BackendInstallError("No recent llama.cpp nightly release is available for the Linux CUDA build")
    return release


class BackendInstaller:
    def __init__(
        self,
        runtime_dir: Path,
        fetch_release=_fetch_recent_releases,
        downloader=_download,
        system=platform.system,
        machine=platform.machine,
        which=shutil.which,
        environ=None,
        command_runner=subprocess.run,
    ):
        self.runtime_dir = Path(runtime_dir)
        self._fetch_release = fetch_release
        self._downloader = downloader
        self._system = system
        self._machine = machine
        self._which = which
        self._environ = os.environ if environ is None else environ
        self._command_runner = command_runner

    def _external_executable(self) -> Path | None:
        configured = self._environ.get("LLAMASERVE_DOC_SERVER")
        if configured:
            executable = Path(configured).expanduser().resolve()
            if not executable.is_file():
                raise BackendInstallError(f"LLAMASERVE_DOC_SERVER does not point to a file: {executable}")
            return executable
        executable_name = "llama-server.exe" if _normalized_system(self._system()) == "windows" else "llama-server"
        discovered = self._which(executable_name)
        return Path(discovered).resolve() if discovered else None

    def _current_executable(self, backend_key: str) -> Path | None:
        pointer = self.runtime_dir / backend_key / "current.json"
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

    def _copy_runtime_files(self, source_root: Path, payload: Path) -> None:
        payload.mkdir()
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            destination_file = payload / source.name
            if destination_file.exists() and destination_file.read_bytes() != source.read_bytes():
                raise BackendInstallError(f"Conflicting runtime files: {source.name}")
            shutil.copy2(source, destination_file)

    def _publish_payload(
        self,
        payload: Path,
        target: RuntimeTarget,
        backend_root: Path,
        releases_root: Path,
        release_tag: str,
        manifest: dict,
    ) -> Path:
        executable = payload / target.executable_name
        if not executable.is_file():
            raise BackendInstallError(f"The backend has no {target.executable_name}")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (payload / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        safe_release_tag = re.sub(r"[^A-Za-z0-9._-]", "_", release_tag)
        destination = releases_root / f"{safe_release_tag}-{os.getpid()}"
        suffix = 1
        while destination.exists():
            destination = releases_root / f"{safe_release_tag}-{os.getpid()}-{suffix}"
            suffix += 1
        payload.replace(destination)

        relative_executable = (destination / target.executable_name).relative_to(backend_root)
        pointer_staging = backend_root / ".current.json.tmp"
        pointer_staging.write_text(
            json.dumps({"executable": relative_executable.as_posix()}, indent=2),
            encoding="utf-8",
        )
        os.replace(pointer_staging, backend_root / "current.json")
        return (destination / target.executable_name).resolve()

    def _build_cuda_backend(
        self,
        target: RuntimeTarget,
        backend: str,
        releases: list[dict],
        backend_root: Path,
        releases_root: Path,
    ) -> Path:
        toolchain = find_cuda_toolchain(which=self._which, environ=self._environ)
        if toolchain is None:
            raise BackendInstallError(
                "Linux CUDA auto-build requires nvcc, cmake, and git. Install the CUDA Toolkit and build tools, put an existing llama-server on PATH, or set LLAMASERVE_DOC_SERVER."
            )
        release = _select_source_release(releases)
        release_tag = str(release["tag_name"])
        staging = Path(tempfile.mkdtemp(prefix=".build-", dir=releases_root))
        try:
            source_root = staging / "source"
            build_root = staging / "build"
            commands = [
                [
                    toolchain.git,
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    release_tag,
                    "--single-branch",
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(source_root),
                ],
                [
                    toolchain.cmake,
                    "-S",
                    str(source_root),
                    "-B",
                    str(build_root),
                    "-DGGML_CUDA=ON",
                    "-DGGML_BACKEND_DL=ON",
                    "-DGGML_CPU_ALL_VARIANTS=ON",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_SERVER=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_CUDA_COMPILER={toolchain.nvcc}",
                    "-DCMAKE_INSTALL_RPATH=$ORIGIN",
                    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                ],
                [
                    toolchain.cmake,
                    "--build",
                    str(build_root),
                    "--config",
                    "Release",
                    "--target",
                    "llama-server",
                    "-j",
                    str(max(1, os.cpu_count() or 1)),
                ],
            ]
            try:
                for command in commands:
                    self._command_runner(command, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise BackendInstallError(f"Linux CUDA llama-server build failed: {error}") from error

            built_executable = find_server_executable(build_root, target.executable_name)
            payload = staging / "payload"
            self._copy_runtime_files(built_executable.parent, payload)
            manifest = {
                "source_release": release_tag,
                "source_repository": "https://github.com/ggml-org/llama.cpp",
                "target": target.key,
                "backend": backend,
                "cuda_compiler": toolchain.nvcc,
            }
            return self._publish_payload(
                payload,
                target,
                backend_root,
                releases_root,
                release_tag,
                manifest,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def ensure_installed(self, backend: str = "auto", progress=None) -> Path:
        external = self._external_executable()
        if external is not None:
            return external

        target = detect_runtime_target(
            backend,
            system=self._system(),
            machine=self._machine(),
            which=self._which,
            environ=self._environ,
        )
        for backend_key in (target.key, *target.legacy_keys):
            existing = self._current_executable(backend_key)
            if existing is not None:
                return existing

        backend_root = self.runtime_dir / target.key
        releases_root = backend_root / "releases"
        downloads_root = backend_root / "downloads"
        releases_root.mkdir(parents=True, exist_ok=True)
        downloads_root.mkdir(parents=True, exist_ok=True)
        release_data = self._fetch_release()
        releases = release_data if isinstance(release_data, list) else [release_data]
        if target.source_build:
            return self._build_cuda_backend(
                target,
                backend,
                releases,
                backend_root,
                releases_root,
            )
        release, assets = _select_complete_release(releases, target)
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
                safe_extract_archive(archive, extracted)

            find_server_executable(extracted, target.executable_name)
            payload = staging / "payload"
            self._copy_runtime_files(extracted, payload)
            manifest = {
                "release": release_tag,
                "target": target.key,
                "backend": backend,
                "assets": [{"name": asset.name, "sha256": asset.sha256} for asset in assets],
            }
            executable = self._publish_payload(
                payload,
                target,
                backend_root,
                releases_root,
                release_tag,
                manifest,
            )
            for asset in assets:
                (downloads_root / asset.name).unlink(missing_ok=True)
            return executable
        finally:
            shutil.rmtree(staging, ignore_errors=True)
