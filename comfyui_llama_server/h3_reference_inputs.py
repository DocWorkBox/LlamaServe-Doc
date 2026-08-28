from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import math
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image


REFERENCE_VIDEO_SOURCE_FPS = 24.0
REFERENCE_VIDEO_TARGET_FPS = 2.0
REFERENCE_VIDEO_MIN_FRAMES = 4
REFERENCE_VIDEO_MAX_PIXELS = 768 * 28 * 28


@dataclass(frozen=True)
class ReferenceSlot:
    source: str
    media_type: str
    value: object


@dataclass(frozen=True)
class OfficialReferences:
    ref_images: dict[int, object]
    ref_videos: dict[int, object]
    ref_video_audios: dict[int, object]
    ref_audios: dict[int, object]
    ordered: tuple[ReferenceSlot, ...]


@dataclass(frozen=True)
class VideoTranscodePlan:
    indices: tuple[int, ...]
    width: int
    height: int
    fps: float


def plan_reference_video(
    frame_count: int,
    height: int,
    width: int,
    source_fps: float = REFERENCE_VIDEO_SOURCE_FPS,
    target_fps: float = REFERENCE_VIDEO_TARGET_FPS,
    max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
) -> VideoTranscodePlan:
    if frame_count < 1 or height < 1 or width < 1 or source_fps <= 0:
        raise ValueError("reference video dimensions, frame count, and fps must be positive")
    duration = frame_count / source_fps
    target_count = min(
        frame_count,
        max(REFERENCE_VIDEO_MIN_FRAMES, round(duration * target_fps)),
    )
    indices = tuple(
        int(index)
        for index in np.linspace(0, frame_count - 1, target_count).round()
    )

    output_width, output_height = width, height
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        output_width = max(2, int(width * scale) // 2 * 2)
        output_height = max(2, int(height * scale) // 2 * 2)
    output_fps = target_count / duration
    return VideoTranscodePlan(
        indices=indices,
        width=output_width,
        height=output_height,
        fps=output_fps,
    )


def _autogrow_index_map(raw, prefix: str) -> dict[int, object]:
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(key, int):
            index = key
        else:
            name = str(key)
            if not name.startswith(prefix):
                continue
            try:
                index = int(name[len(prefix) :])
            except ValueError:
                continue
        if index >= 0:
            result[index] = value
    return dict(sorted(result.items()))


def collect_official_references(
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    **flat_inputs,
) -> OfficialReferences:
    images = _autogrow_index_map(ref_images, "ref_image_")
    videos = _autogrow_index_map(ref_videos, "ref_video_")
    video_audios = _autogrow_index_map(ref_video_audios, "ref_video_audio_")
    audios = _autogrow_index_map(ref_audios, "ref_audio_")

    for name, value in flat_inputs.items():
        if value is None:
            continue
        for prefix, target in (
            ("ref_image_", images),
            ("ref_video_audio_", video_audios),
            ("ref_video_", videos),
            ("ref_audio_", audios),
        ):
            if name.startswith(prefix):
                suffix = name[len(prefix) :]
                if suffix.isdigit():
                    target.setdefault(int(suffix), value)
                break

    images = dict(sorted(images.items()))
    videos = dict(sorted(videos.items()))
    audios = dict(sorted(audios.items()))
    # Same-numbered video audio only exists when its reference video exists.
    video_audios = {
        index: video_audios[index]
        for index in sorted(video_audios)
        if index in videos
    }

    ordered = []
    for index, value in images.items():
        ordered.append(ReferenceSlot(f"ref_image_{index}", "image", value))
    for index, value in videos.items():
        soundtrack = video_audios.get(index)
        if soundtrack is not None:
            ordered.append(
                ReferenceSlot(f"ref_video_audio_{index}", "audio", soundtrack)
            )
        ordered.append(ReferenceSlot(f"ref_video_{index}", "video", value))
    for index, value in audios.items():
        ordered.append(ReferenceSlot(f"ref_audio_{index}", "audio", value))

    return OfficialReferences(
        ref_images=images,
        ref_videos=videos,
        ref_video_audios=video_audios,
        ref_audios=audios,
        ordered=tuple(ordered),
    )


def select_official_references(
    references: OfficialReferences,
    selected_sources: set[str],
) -> OfficialReferences:
    selected = set(selected_sources)
    return OfficialReferences(
        ref_images={
            index: value
            for index, value in references.ref_images.items()
            if f"ref_image_{index}" in selected
        },
        ref_videos={
            index: value
            for index, value in references.ref_videos.items()
            if f"ref_video_{index}" in selected
        },
        ref_video_audios={
            index: value
            for index, value in references.ref_video_audios.items()
            if f"ref_video_audio_{index}" in selected
        },
        ref_audios={
            index: value
            for index, value in references.ref_audios.items()
            if f"ref_audio_{index}" in selected
        },
        ordered=tuple(slot for slot in references.ordered if slot.source in selected),
    )


def _tensor_to_uint8(value) -> np.ndarray:
    tensor = value.detach().cpu() if hasattr(value, "detach") else value
    array = np.asarray(tensor)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C] or [H,W,C], got {array.shape}")
    return np.clip(array * 255.0, 0, 255).round().astype(np.uint8)


def _save_image(value, path: Path) -> None:
    Image.fromarray(_tensor_to_uint8(value)).save(path, format="PNG")


def _save_video(value, path: Path, work_dir: Path) -> None:
    tensor = value.detach().cpu() if hasattr(value, "detach") else value
    frames = np.asarray(tensor)
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError(f"Expected reference video IMAGE batch [N,H,W,C], got {frames.shape}")
    plan = plan_reference_video(
        frame_count=frames.shape[0],
        height=frames.shape[1],
        width=frames.shape[2],
    )
    frame_dir = work_dir / f"{path.stem}-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for output_index, source_index in enumerate(plan.indices):
        frame = frames[source_index]
        image = np.clip(frame * 255.0, 0, 255).round().astype(np.uint8)
        pil_image = Image.fromarray(image)
        if pil_image.size != (plan.width, plan.height):
            pil_image = pil_image.resize(
                (plan.width, plan.height),
                resample=Image.Resampling.LANCZOS,
            )
        pil_image.save(frame_dir / f"{output_index:06d}.png", format="PNG")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to send ref_video_N inputs to llama-server")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{plan.fps:.8g}",
        "-i",
        str(frame_dir / "%06d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    kwargs = {"check": True, "capture_output": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.run(command, **kwargs)
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
        raise RuntimeError(f"FFmpeg could not encode reference video: {message}") from error


def _save_audio(value, path: Path) -> None:
    if not isinstance(value, dict) or value.get("waveform") is None:
        raise ValueError("Expected AUDIO with waveform and sample_rate")
    waveform = value["waveform"]
    waveform = waveform.detach().cpu() if hasattr(waveform, "detach") else waveform
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim == 3:
        samples = samples[0]
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError(f"Expected AUDIO waveform [B,C,L] or [C,L], got {samples.shape}")
    pcm = (np.clip(samples, -1.0, 1.0).T * 32767.0).round().astype("<i2")
    sample_rate = int(value.get("sample_rate") or 32000)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[0])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


@contextmanager
def materialize_reference_media(references: OfficialReferences, media_root: Path):
    root = Path(media_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"media_root is not a directory: {root}")
    parent = root / "temp" / "llamaserve_doc"
    parent.mkdir(parents=True, exist_ok=True)
    request_dir = Path(tempfile.mkdtemp(prefix="h3-omni-", dir=parent))
    try:
        media = []
        for slot in references.ordered:
            if slot.media_type == "image":
                path = request_dir / f"{slot.source}.png"
                _save_image(slot.value, path)
            elif slot.media_type == "video":
                path = request_dir / f"{slot.source}.mp4"
                _save_video(slot.value, path, request_dir)
            elif slot.media_type == "audio":
                path = request_dir / f"{slot.source}.wav"
                _save_audio(slot.value, path)
            else:
                raise ValueError(f"Unsupported reference media type: {slot.media_type}")
            media.append(
                {"type": slot.media_type, "path": str(path), "source": slot.source}
            )
        yield tuple(media)
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


def pack_director_group(
    mode: str,
    enhanced_prompt: str,
    duration: int | float,
    references: OfficialReferences,
) -> dict:
    normalized = str(mode).strip().lower()
    common = {
        "version": 1,
        "prompt": (enhanced_prompt or "").strip(),
        "duration_sec": float(duration),
    }
    if normalized == "ref2av":
        return {
            **common,
            "family": "r2v",
            "kind": "r2v",
            "first_frame": None,
            "last_frame": None,
            "ref_images": dict(references.ref_images),
            "ref_videos": dict(references.ref_videos),
            "ref_video_audios": dict(references.ref_video_audios),
            "ref_audios": dict(references.ref_audios),
        }

    images = list(references.ref_images.values())
    first_frame = images[0] if normalized in {"i2av", "fl2av"} and images else None
    last_frame = images[-1] if normalized in {"l2av", "fl2av"} and images else None
    kind = "t2v" if normalized == "t2av" else "i2v" if normalized == "i2av" else "fl2v"
    return {
        **common,
        "family": "i2v",
        "kind": kind,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "ref_images": {},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {},
    }
