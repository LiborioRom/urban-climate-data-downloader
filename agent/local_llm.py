from __future__ import annotations

import json
import os
from typing import TypeVar

import requests
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:4b")


def ollama_num_ctx() -> int:
    try:
        return max(1024, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
    except ValueError:
        return 4096


def ollama_status(timeout: float = 1.0) -> dict[str, object]:
    try:
        response = requests.get(f"{ollama_host()}/api/tags", timeout=timeout)
        response.raise_for_status()
        installed = [item.get("name", "") for item in response.json().get("models", [])]
        target = ollama_model()
        available = target in installed or (
            ":" not in target and any(name.split(":")[0] == target for name in installed)
        )
        return {"running": True, "model": target, "model_available": available, "installed": installed}
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return {"running": False, "model": ollama_model(), "model_available": False, "installed": []}


def structured_chat(
    response_model: type[T],
    *,
    system_prompt: str,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    timeout: float = 120.0,
) -> T:
    messages = [{"role": "system", "content": system_prompt}]
    for item in (history or [])[-8:]:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item.get("content", "")})
    messages.append({"role": "user", "content": user_message})
    response = requests.post(
        f"{ollama_host()}/api/chat",
        json={
            "model": ollama_model(),
            "messages": messages,
            "stream": False,
            "think": False,
            "format": response_model.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": ollama_num_ctx()},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content")
    if not content:
        raise ValueError("Ollama no devolvió contenido estructurado.")
    return response_model.model_validate(json.loads(content))
