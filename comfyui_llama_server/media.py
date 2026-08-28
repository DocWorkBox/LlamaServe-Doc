from __future__ import annotations

from pathlib import Path


MEDIA_CONTENT_KEYS = {
    "image": ("image_url", "image_url"),
    "video": ("input_video", "input_video"),
    "audio": ("input_audio", "input_audio"),
}


def build_media_content(user_prompt: str, media, media_root) -> list[dict]:
    if media_root is None:
        raise ValueError("media_root is required when media inputs are connected")
    root = Path(media_root).expanduser().resolve()
    uses_structured_prompt = any("heading" in item or "preamble" in item for item in media)
    content = [] if uses_structured_prompt else [{"type": "text", "text": user_prompt}]
    for item in media:
        if item.get("preamble"):
            content.append({"type": "text", "text": item["preamble"]})
        if item.get("heading"):
            content.append({"type": "text", "text": item["heading"]})
        media_path = Path(item["path"]).expanduser().resolve()
        if not media_path.is_file():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        try:
            relative_path = media_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Media file must be inside media_root ({root}): {media_path}"
            ) from error
        try:
            content_type, payload_key = MEDIA_CONTENT_KEYS[item["type"]]
        except KeyError as error:
            raise ValueError(f"Unsupported media type: {item['type']}") from error
        content.append(
            {
                "type": content_type,
                payload_key: {"url": f"file://{relative_path.as_posix()}"},
            }
        )
    if uses_structured_prompt:
        content.append({"type": "text", "text": user_prompt})
    return content
