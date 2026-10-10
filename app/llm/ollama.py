"""Ollama HTTP implementation of the provider-neutral LLM contract."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.llm.exceptions import ProviderConnectionError, ProviderTimeoutError, LLMError
from app.llm.provider import LLMRequest, LLMResponse

#: The timeout a request carries when its caller does not choose one. Used to
#: tell "the caller asked for something specific" from "nobody said anything".
DEFAULT_REQUEST_TIMEOUT = 120.0


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2",
                 timeout_seconds: float = 120.0, keep_alive: str = "30m"):
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        #: Timeout applied to every request unless the request overrides it.
        #: Recorded in assessment records so a timeout can be read against the
        #: limit that was actually in force.
        self.timeout_seconds = timeout_seconds
        #: How long Ollama keeps the model resident between requests.
        self.keep_alive = keep_alive

    def describe(self) -> dict:
        """Identify this provider explicitly.

        The audit layer reads identity from here rather than guessing at
        attributes. Guessing works right up until something wraps the provider,
        at which point the attributes vanish and records silently lose their
        model name.
        """
        from app.jobs.assessment import is_local_endpoint, is_local_model

        return {
            "provider": "local-ollama",
            "model": self.default_model,
            "endpoint": self.base_url,
            "local_only": bool(
                is_local_endpoint(self.base_url)
                and is_local_model(self.default_model)
            ),
            "timeout_seconds": self.timeout_seconds,
            "keep_alive": self.keep_alive,
        }

    def health_check(self) -> bool:
        try:
            self._request("/api/version", {}, timeout_seconds=5.0)
        except LLMError:
            return False
        return True

    def list_models(self) -> list[str]:
        payload = self._request("/api/tags", {}, timeout_seconds=10.0)
        return [item["name"] for item in payload.get("models", []) if item.get("name")]

    def effective_timeout(self, request: LLMRequest) -> float:
        """The timeout actually applied.

        A provider configured with an explicit timeout wins over the request
        default, so raising the limit is a deliberate configuration change
        rather than something each caller has to remember to pass.
        """
        if request.timeout_seconds != DEFAULT_REQUEST_TIMEOUT:
            return request.timeout_seconds
        return self.timeout_seconds

    def preload(self, timeout_seconds: float = 600.0) -> float:
        """Load the model into memory and keep it resident.

        Returns the seconds the load took, so a cold start can be reported
        separately from generation. Without this, every request pays the load
        again and a slow model looks like an incapable one.
        """
        started = time.monotonic()
        self._request(
            "/api/generate",
            {
                "model": self.default_model,
                "prompt": "",
                "stream": False,
                # Ask Ollama to hold the model resident instead of evicting it
                # when the (empty) request finishes.
                "keep_alive": self.keep_alive,
            },
            timeout_seconds=timeout_seconds,
        )
        return time.monotonic() - started

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        prompt = f"{request.system_prompt}\n\n{request.user_prompt}"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.response_format == "json":
            payload["format"] = "json"

        started = time.monotonic()
        result = self._request(
            "/api/generate",
            payload,
            timeout_seconds=self.effective_timeout(request),
        )
        text = result.get("response")
        if not isinstance(text, str):
            raise LLMError("Ollama response did not contain a text response")
        return LLMResponse(
            text=text,
            model=model,
            latency_ms=round((time.monotonic() - started) * 1000),
            raw=result,
        )

    def _request(self, path: str, payload: dict, timeout_seconds: float) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8") if payload else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload else "GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderConnectionError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(f"Ollama request timed out: {exc}") from exc
        except URLError as exc:
            raise ProviderConnectionError(f"Ollama endpoint is unreachable: {exc}") from exc
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise LLMError("Ollama returned a non-object JSON response")
        if "error" in value:
            raise LLMError(f"Ollama error: {value['error']}")
        return value