import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_llama_server.client import LlamaServerClient


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True

    def __iter__(self):
        return iter(self.lines)


class HttpClientTests(unittest.TestCase):
    def test_posts_streaming_chat_request_and_closes_response(self):
        captured = {}
        response = FakeResponse([
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b'\n',
            b'data: [DONE]\n',
            b'\n',
        ])

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return response

        client = LlamaServerClient(opener=opener, timeout=600)
        result = client.generate("http://127.0.0.1:8191", {"messages": []})

        self.assertEqual(captured["url"], "http://127.0.0.1:8191/v1/chat/completions")
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(result.text, "ok")
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
