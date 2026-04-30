"""Thin Ollama HTTP client. Handles text + multimodal (image) requests."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from config import LLM_CONFIG


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model or LLM_CONFIG["model"]
        self.base_url = (base_url or LLM_CONFIG["base_url"]).rstrip("/")
        self.temperature = temperature if temperature is not None else LLM_CONFIG["temperature"]
        self.timeout = timeout or LLM_CONFIG["timeout"]
        self.num_ctx = num_ctx or LLM_CONFIG["num_ctx"]

    def _options(self, **overrides: Any) -> dict[str, Any]:
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return opts

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        images: list[str] | None = None,
        format: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature=temperature),
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = [self._encode_image(p) for p in images]
        if format:
            payload["format"] = format
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask the model for a JSON object and parse it. Falls back to extraction
        if the model wraps JSON in prose or fences."""
        text = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            images=images,
            format="json",
        )
        return _safe_json_loads(text)

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"_raw": text}
