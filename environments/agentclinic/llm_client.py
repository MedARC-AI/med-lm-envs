"""
Unified LLM Client for AgentClinic
Supports multiple backends: OpenAI, Anthropic, Gemini, Mistral, vLLM
"""
from __future__ import annotations
from typing import Dict, List, Optional
import json
import os
import re
import urllib.request
import urllib.error


def _normalize_ws(x: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", x or "").strip()


class LLMClient:


    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: str = "auto",
        temperature: float = 0.05,
        max_tokens: int = 200,
    ):
        """
        Initialize LLM client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "mistral-large-latest")
            api_base: API base URL (optional, auto-detected from backend)
            api_key: API key (optional, read from environment if not provided)
            backend: Backend type - "openai", "anthropic", "gemini", "mistral", "vllm", or "auto"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Auto-detect backend from model name if not specified
        if backend == "auto":
            backend = self._detect_backend(model)

        self.backend = backend.lower()

        # Set API base URL
        if api_base:
            self.api_base = api_base.rstrip("/")
        else:
            self.api_base = self._get_default_api_base(self.backend)

        # Get API key from environment or parameter
        self.api_key = api_key or self._get_api_key(self.backend)

        if not self.api_key and self.backend != "vllm":
            raise RuntimeError(f"API key not set for backend: {self.backend}")

    def _detect_backend(self, model: str) -> str:
        """Auto-detect backend from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "gemini"
        elif "mistral" in model_lower or "codestral" in model_lower or "pixtral" in model_lower:
            return "mistral"
        else:
            # Default to vLLM for unknown models (likely local models)
            return "vllm"

    def _get_default_api_base(self, backend: str) -> str:
        """Get default API base URL for backend."""
        defaults = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "mistral": "https://api.mistral.ai/v1",
            "vllm": "http://localhost:8000/v1",  # Default vLLM server
        }
        return defaults.get(backend, "http://localhost:8000/v1")

    def _get_api_key(self, backend: str) -> Optional[str]:
        """Get API key from environment for backend."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "vllm": "VLLM_API_KEY",  # Optional for vLLM
        }
        env_var = env_vars.get(backend)
        return os.environ.get(env_var) if env_var else None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with "role" and "content" keys

        Returns:
            Generated text response
        """
        if self.backend in ["openai", "vllm", "mistral"]:
            # OpenAI-compatible APIs
            return self._generate_openai_compatible(messages)
        elif self.backend == "anthropic":
            return self._generate_anthropic(messages)
        elif self.backend == "gemini":
            return self._generate_gemini(messages)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        """Generate using OpenAI-compatible API (OpenAI, Mistral, vLLM)."""
        url = f"{self.api_base}/chat/completions"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")

        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return _normalize_ws(body["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"[LLMClient/{self.backend}] Error: {e}")
            return ""

    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Generate using Anthropic API."""
        url = f"{self.api_base}/messages"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", self.api_key)
        req.add_header("anthropic-version", "2023-06-01")

        # Convert messages format (Anthropic expects system separate)
        system_msg = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system_msg:
            payload["system"] = system_msg

        try:
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return _normalize_ws(body["content"][0]["text"])
        except Exception as e:
            print(f"[LLMClient/anthropic] Error: {e}")
            return ""

    def _generate_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Generate using Google Gemini API."""
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")

        # Convert messages format for Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return _normalize_ws(body["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            print(f"[LLMClient/gemini] Error: {e}")
            return ""
