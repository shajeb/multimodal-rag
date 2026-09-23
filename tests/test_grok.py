import json
import httpx
import pytest
from rag.config import Settings
from rag.providers import Grok, provider


def test_grok_stream_uses_xai_and_skips_reasoning(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "xai-test-only")
    monkeypatch.setenv("GROK_MODEL", "grok-4.6")
    model = provider("grok")
    assert isinstance(model, Grok)
    assert Settings(provider="grok").provider == "grok"
    assert str(model.client.base_url) == "https://api.x.ai/v1/"
    assert model.client.headers["Authorization"] == "Bearer xai-test-only"
    model.client.close()
    def handler(request):
        assert str(request.url) == "https://api.x.ai/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "grok-4.6" and payload["stream"] is True
        assert payload["messages"][0]["role"] == "system"
        events = [
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world"}}]},
            {"choices": [], "usage": {"total_tokens": 10}},
        ]
        body = ": heartbeat\n\n" + "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)
    with httpx.Client(base_url="https://api.x.ai/v1/", transport=httpx.MockTransport(handler)) as client:
        model.client = client
        assert list(model.stream("policy", "question")) == ["Hello ", "world"]


@pytest.mark.parametrize("status,body,error", [
    (401, "Unauthorized", httpx.HTTPStatusError),
    (200, 'data: {"error":{"message":"failed"}}\n\n', RuntimeError),
])
def test_grok_errors_are_not_silently_successful(status, body, error):
    model = Grok.__new__(Grok)
    model.model = "grok-4.6"
    with httpx.Client(base_url="https://api.x.ai/v1/", transport=httpx.MockTransport(lambda _: httpx.Response(status, text=body))) as client:
        model.client = client
        with pytest.raises(error):
            list(model.stream("policy", "question"))
