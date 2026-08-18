from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable, Iterator
from urllib.request import Request, urlopen


class StreamInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    text: str
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    raw_timings: dict | None = None


def iter_sse_data(
    lines: Iterable[bytes | str],
    interrupt_check: Callable[[], None] | None = None,
) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        if interrupt_check is not None:
            interrupt_check()
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


def parse_chat_stream(
    lines: Iterable[bytes | str],
    interrupt_check: Callable[[], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> ChatResult:
    parts: list[str] = []
    timings: dict = {}
    for data in iter_sse_data(lines, interrupt_check=interrupt_check):
        if data == "[DONE]":
            break
        payload = json.loads(data)
        choices = payload.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                parts.append(content)
                if on_text is not None:
                    on_text("".join(parts))
        if isinstance(payload.get("timings"), dict):
            timings.update(payload["timings"])
    return ChatResult(
        text="".join(parts),
        tokens_per_second=timings.get("predicted_per_second"),
        prompt_tokens_per_second=timings.get("prompt_per_second"),
        raw_timings=timings or None,
    )


class LlamaServerClient:
    def __init__(self, opener=urlopen, timeout: float = 600):
        self._opener = opener
        self.timeout = timeout

    def generate(
        self,
        base_url: str,
        payload: dict,
        interrupt_check: Callable[[], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatResult:
        body = dict(payload)
        body["stream"] = True
        request = Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self.timeout) as response:
            return parse_chat_stream(
                response,
                interrupt_check=interrupt_check,
                on_text=on_text,
            )
