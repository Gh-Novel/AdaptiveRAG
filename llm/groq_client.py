"""Groq API client — drop-in replacement for OllamaClient when running hosted.

Free tier: 14,400 requests/day, ~500 req/min.
Text model  : llama-3.1-8b-instant  (fast, cheap)
Vision model: llama-3.2-11b-vision-preview (images)
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests

_TEXT_MODEL = "llama-3.1-8b-instant"
_VISION_MODEL = "llama-3.2-11b-vision-preview"
_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.temperature = 0.1
        self.timeout = 60

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            f"{_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _build_messages(
        self,
        prompt: str,
        system: str | None,
        images: list[str] | None,
    ) -> tuple[list[dict], str]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img_path in images:
                b64 = self._encode_image(img_path)
                suffix = Path(img_path).suffix.lower().lstrip(".")
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp"}.get(suffix, "image/png")
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:{mime};base64,{b64}"}})
            msgs.append({"role": "user", "content": content})
            return msgs, _VISION_MODEL
        msgs.append({"role": "user", "content": prompt})
        return msgs, _TEXT_MODEL

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        images: list[str] | None = None,
        format: str | None = None,
    ) -> str:
        msgs, model = self._build_messages(prompt, system, images)
        return self._chat(
            msgs, model,
            temperature if temperature is not None else self.temperature,
            json_mode=(format == "json"),
        )

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        text = self.generate(prompt=prompt, system=system,
                             temperature=temperature, images=images, format="json")
        return _safe_json(text)

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            r = requests.get(f"{_BASE_URL}/models", headers=self._headers(), timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False


def _safe_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_raw": text}
