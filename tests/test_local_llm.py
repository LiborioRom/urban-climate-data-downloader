from agent.local_llm import ollama_num_ctx, ollama_status


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": "qwen3:8b"}]}


def test_status_does_not_confuse_different_qwen_sizes(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr("agent.local_llm.requests.get", lambda *args, **kwargs: FakeResponse())
    assert ollama_status()["model_available"] is False


def test_invalid_context_uses_conservative_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "not-a-number")
    assert ollama_num_ctx() == 4096
