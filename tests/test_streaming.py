import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.client import StreamInterrupted, iter_sse_data, parse_chat_stream


class StreamingTests(unittest.TestCase):
    def test_iter_sse_data_joins_multiline_data_and_ignores_comments(self):
        lines = [
            b": keepalive\n",
            b"data: {\"first\":\n",
            b"data: 1}\n",
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        self.assertEqual(list(iter_sse_data(lines)), ['{"first":\n1}', "[DONE]"])

    def test_parse_chat_stream_collects_content_and_performance_stats(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
            b'\n',
            b'data: {"choices":[{"delta":{"content":"world"}}],"timings":{"predicted_per_second":5.25,"prompt_per_second":81.0}}\n',
            b'\n',
            b'data: [DONE]\n',
            b'\n',
        ]

        result = parse_chat_stream(lines)

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.tokens_per_second, 5.25)
        self.assertEqual(result.prompt_tokens_per_second, 81.0)

    def test_interrupt_check_stops_stream_between_chunks(self):
        checks = 0

        def interrupt_check():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise StreamInterrupted("stop")

        lines = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n', b'\n', b'data: [DONE]\n', b'\n']

        with self.assertRaises(StreamInterrupted):
            parse_chat_stream(lines, interrupt_check=interrupt_check)


if __name__ == "__main__":
    unittest.main()
