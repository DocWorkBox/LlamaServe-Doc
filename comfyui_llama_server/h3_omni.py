from __future__ import annotations

import math
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

from .h3_omni_prompts import REF2AV_SYSTEM_PROMPT, SYSTEM_PROMPT


TASKS = ("T2AV", "I2AV", "L2AV", "FL2AV", "Ref2AV")
RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
MEDIA_TYPES = {"image", "video", "audio"}
REFERENCE_LABEL_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)
CHINESE_REFERENCE_ALIASES = {
    "image": ("图片", "图像"),
    "video": ("视频",),
    "audio": ("音频",),
}


def h3_effective_duration(requested_duration: int) -> tuple[int, float]:
    if isinstance(requested_duration, bool) or not isinstance(requested_duration, int):
        raise ValueError("duration must be an integer from 4 through 15")
    if not 4 <= requested_duration <= 15:
        raise ValueError("duration must be an integer from 4 through 15")
    frames = math.ceil((24 * requested_duration - 5) / 17) * 17 + 5
    return frames, frames / 24.0


def _format_duration(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _normalized_task(task: str) -> str:
    aliases = {
        "t2v": "t2av",
        "t2va": "t2av",
        "t2av": "t2av",
        "i2v": "i2av",
        "i2va": "i2av",
        "i2av": "i2av",
        "l2v": "l2av",
        "l2va": "l2av",
        "l2av": "l2av",
        "fl2v": "fl2av",
        "fl2va": "fl2av",
        "fl2av": "fl2av",
        "flf2v": "fl2av",
        "flf2va": "fl2av",
        "flf2av": "fl2av",
        "ref2v": "ref2av",
        "ref2va": "ref2av",
        "ref2av": "ref2av",
    }
    try:
        return aliases[task.strip().lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"Unsupported task {task!r}; use T2AV, I2AV, L2AV, FL2AV, or Ref2AV"
        ) from error


def _validate_resolution(task: str, resolution: str) -> str:
    resolution = resolution.strip() if isinstance(resolution, str) else ""
    if resolution not in RATIOS:
        raise ValueError(f"resolution must be one of {RATIOS}; got {resolution!r}")
    if task == "ref2av" and resolution not in {"16:9", "9:16"}:
        raise ValueError("Ref2AV resolution must be 16:9 or 9:16")
    return resolution


def _canonical_media(task: str, media, formatted_duration: str) -> tuple[dict, ...]:
    source_media = tuple(media or ())
    kinds = [item.get("type") for item in source_media]
    invalid = [kind for kind in kinds if kind not in MEDIA_TYPES]
    if invalid:
        raise ValueError(f"Unsupported media type(s): {invalid}")

    expected_images = {"t2av": 0, "i2av": 1, "l2av": 1, "fl2av": 2}
    if task in expected_images:
        expected = expected_images[task]
        if kinds != ["image"] * expected:
            raise ValueError(
                f"{task.upper()} requires exactly {expected} image reference(s) "
                f"and no other media; got {kinds}"
            )
    elif not source_media:
        raise ValueError("Ref2AV requires at least one reference asset")

    counts = Counter(kinds)
    if (
        counts["image"] > 9
        or counts["video"] > 3
        or counts["audio"] > 3
        or len(source_media) > 12
    ):
        raise ValueError(
            "Ref2AV reference limit exceeded; at most 9 images, 3 videos, "
            "3 audio files, and 12 total assets are supported"
        )

    counters = Counter()
    canonical = []
    for index, item in enumerate(source_media, start=1):
        kind = item["type"]
        counters[kind] += 1
        label_prefix = {"image": "Picture", "video": "Video", "audio": "Audio"}[kind]
        label = f"<{label_prefix} {counters[kind]}>"
        if task == "i2av":
            heading = f"{label} — exact first frame at 0.00 seconds:\n"
        elif task == "l2av":
            heading = f"{label} — exact final frame at {formatted_duration}:\n"
        elif task == "fl2av" and index == 1:
            heading = f"{label} — exact first frame at 0.00 seconds:\n"
        elif task == "fl2av":
            heading = f"{label} — exact final frame at {formatted_duration}:\n"
        else:
            heading = f"{label}:\n"
        enriched = dict(item)
        enriched.update({"order": index, "label": label, "heading": heading})
        if task == "ref2av" and index == 1:
            enriched["preamble"] = "Ordered MiniMax-H3 references:\n"
        canonical.append(enriched)
    return tuple(canonical)


def _select_ref2av_media(prompt: str, media: tuple[dict, ...]) -> tuple[dict, ...]:
    expected = {item["label"].casefold() for item in media}
    mentioned = {
        f"<{match.group(1).capitalize()} {int(match.group(2))}>".casefold()
        for match in REFERENCE_LABEL_RE.finditer(prompt)
    }
    unknown = mentioned - expected
    if unknown:
        raise ValueError(
            "Ref2AV prompt mentions reference labels that were not supplied; "
            f"available {sorted(expected)}, unknown {sorted(unknown)}"
        )
    if not mentioned:
        raise ValueError("Ref2AV prompt must mention at least one supplied reference label")

    selected = []
    for item in media:
        if item["label"].casefold() not in mentioned:
            continue
        selected_item = dict(item)
        selected_item.pop("preamble", None)
        selected.append(selected_item)
    selected[0]["preamble"] = "Ordered MiniMax-H3 references:\n"
    return tuple(selected)


def _normalize_ref2av_mentions(prompt: str, media: tuple[dict, ...]) -> str:
    counts = Counter(item["type"] for item in media)
    prefixes = {"image": "Picture", "video": "Video", "audio": "Audio"}
    normalized = prompt
    for kind, aliases in CHINESE_REFERENCE_ALIASES.items():
        alias_group = "|".join(map(re.escape, aliases))
        normalized = re.sub(
            rf"(?:{alias_group})\s*(\d+)",
            lambda match, prefix=prefixes[kind]: f"<{prefix} {int(match.group(1))}>",
            normalized,
        )
        if counts[kind] == 1:
            normalized = re.sub(
                rf"(?:{alias_group})(?!\s*\d)",
                f"<{prefixes[kind]} 1>",
                normalized,
            )
    return normalized


def build_h3_omni_preset(
    task: str,
    raw_prompt: str,
    duration: int,
    resolution: str,
    media=None,
) -> tuple[str, str, tuple[dict, ...]]:
    normalized_task = _normalized_task(task)
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("raw_prompt must be a non-empty string")
    resolution = _validate_resolution(normalized_task, resolution)
    _, effective_duration = h3_effective_duration(duration)
    formatted_duration = f"{_format_duration(effective_duration)}s"
    canonical_media = _canonical_media(normalized_task, media, formatted_duration)
    if normalized_task == "ref2av":
        raw_prompt = _normalize_ref2av_mentions(raw_prompt, canonical_media)
        canonical_media = _select_ref2av_media(raw_prompt, canonical_media)

    user_prompt = (
        "Rewrite request:\n"
        f"task: {normalized_task.upper()}\n"
        f"resolution: {resolution}\n"
        f"effective_duration: {formatted_duration}\n"
        f"raw_prompt: {raw_prompt.strip()}"
    )
    system_prompt = REF2AV_SYSTEM_PROMPT if normalized_task == "ref2av" else SYSTEM_PROMPT
    return system_prompt, user_prompt, canonical_media
