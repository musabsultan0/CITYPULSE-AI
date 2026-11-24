from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field

from backend.app.core.config import settings

T = TypeVar("T", bound=BaseModel)


class SupervisorLLMOutput(BaseModel):
    priority_domains: list[str] = Field(default_factory=list, max_length=6)
    coordination_brief: str = Field(min_length=5, max_length=600)
    data_quality_watchouts: list[str] = Field(default_factory=list, max_length=5)


class RiskSynthesisLLMOutput(BaseModel):
    cross_domain_relationships: list[str] = Field(default_factory=list, max_length=6)
    cascading_risk_narrative: str = Field(min_length=5, max_length=700)
    uncertainty_note: str = Field(default="Verify synthetic/live data provenance before action.", max_length=400)


class LLMActionItem(BaseModel):
    priority: Literal["high", "medium", "low"] = "medium"
    action: str = Field(min_length=5, max_length=400)
    rationale: str = Field(default="", max_length=300)
    source_citation: str = Field(default="citypulse_analysis", max_length=200)


class ResponsePlanLLMOutput(BaseModel):
    actions: list[LLMActionItem] = Field(min_length=1, max_length=8)
    coordination_summary: str = Field(min_length=5, max_length=500)


class CommunicationLLMOutput(BaseModel):
    report_summary: str = Field(min_length=5, max_length=500)
    executive_briefing: str = Field(min_length=10, max_length=1200)
    citizen_advisory_draft: str = Field(default="", max_length=600)


def get_llm_status() -> dict[str, Any]:
    provider = settings.llm_provider.lower().strip()
    key_configured = bool(settings.llm_api_key.strip())
    requires_key = provider not in {"ollama", "disabled"}
    ready = bool(settings.llm_enabled and provider != "disabled" and (not requires_key or key_configured))
    return {
        "llm_enabled": settings.llm_enabled,
        "provider": provider,
        "model": settings.llm_model,
        "api_key_configured": key_configured,
        "base_url_configured": bool(settings.llm_base_url.strip()),
        "ready": ready,
        "fallback_mode": "deterministic_rule_based",
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON root must be an object.")
    return parsed


def _invoke_provider(
    system_prompt: str,
    user_content: str,
    http_client: httpx.Client | None = None,
) -> str:
    provider = settings.llm_provider.lower().strip()
    api_key = settings.llm_api_key.strip()
    model = settings.llm_model.strip()
    timeout = settings.llm_timeout_seconds

    def _post(url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, json_body: dict[str, Any]) -> dict[str, Any]:
        if http_client is not None:
            resp = http_client.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, params=params, json=json_body)
        resp.raise_for_status()
        return resp.json()

    if provider in {"openai", "openai_compatible"}:
        base = settings.llm_base_url.strip().rstrip("/") or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        data = _post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json_body={
                "model": model,
                "temperature": settings.llm_temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        return str(data["choices"][0]["message"]["content"])

    if provider == "gemini":
        base = settings.llm_base_url.strip().rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{model}:generateContent"
        data = _post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json_body={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                "generationConfig": {
                    "temperature": settings.llm_temperature,
                    "responseMimeType": "application/json",
                },
            },
        )
        return str(data["candidates"][0]["content"]["parts"][0]["text"])

    if provider == "ollama":
        base = settings.llm_base_url.strip().rstrip("/") or "http://localhost:11434"
        url = f"{base}/api/chat"
        data = _post(
            url,
            headers={"Content-Type": "application/json"},
            json_body={
                "model": model,
                "stream": False,
                "format": "json",
                "options": {"temperature": settings.llm_temperature},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        return str(data["message"]["content"])

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'")


def call_llm_json(
    system_prompt: str,
    payload: dict[str, Any],
    schema_cls: type[T],
    use_llm: bool | None = None,
    http_client: httpx.Client | None = None,
) -> tuple[T | None, dict[str, Any]]:
    """Invoke configured LLM with strict JSON schema validation and automatic deterministic fallback."""
    effective_enabled = settings.llm_enabled if use_llm is None else bool(use_llm)
    provider = settings.llm_provider.lower().strip()
    model = settings.llm_model.strip()

    if not effective_enabled or provider == "disabled":
        return None, {
            "reasoning_mode": "deterministic_fallback",
            "provider": provider,
            "model": model,
            "fallback_reason": "LLM reasoning disabled; using deterministic baseline.",
        }

    if provider != "ollama" and not settings.llm_api_key.strip():
        return None, {
            "reasoning_mode": "deterministic_fallback",
            "provider": provider,
            "model": model,
            "fallback_reason": "LLM_API_KEY not configured in environment; fell back to deterministic reasoning.",
        }

    try:
        user_content = json.dumps(payload, default=str)
        raw_text = _invoke_provider(system_prompt, user_content, http_client=http_client)
        parsed = _extract_json_object(raw_text)
        validated = schema_cls.model_validate(parsed)
        return validated, {
            "reasoning_mode": "llm",
            "provider": provider,
            "model": model,
            "fallback_reason": None,
        }
    except Exception as exc:
        return None, {
            "reasoning_mode": "deterministic_fallback",
            "provider": provider,
            "model": model,
            "fallback_reason": f"LLM call failed ({type(exc).__name__}: {exc}); fell back to deterministic reasoning.",
        }
