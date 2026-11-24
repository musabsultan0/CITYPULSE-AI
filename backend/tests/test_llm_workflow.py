import json
import httpx

from backend.app.core.config import settings
from backend.app.llm import (
    CommunicationLLMOutput,
    ResponsePlanLLMOutput,
    RiskSynthesisLLMOutput,
    SupervisorLLMOutput,
    call_llm_json,
    get_llm_status,
)
from backend.app.main import demo_readings, system_status
from backend.app.workflow import run_workflow


def test_llm_fallback_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    parsed, meta = call_llm_json("System", {"hello": "world"}, SupervisorLLMOutput)
    assert parsed is None
    assert meta["reasoning_mode"] == "deterministic_fallback"


def test_llm_fallback_when_key_missing(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "")
    parsed, meta = call_llm_json("System", {"hello": "world"}, SupervisorLLMOutput)
    assert parsed is None
    assert meta["reasoning_mode"] == "deterministic_fallback"
    assert "LLM_API_KEY" in meta["fallback_reason"]


def test_openai_provider_structured_json(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test-dummy")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer sk-test-dummy"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "priority_domains": ["flood", "traffic"],
                                "coordination_brief": "Prioritize Zone A flood drainage and arterial corridor review.",
                                "data_quality_watchouts": ["Synthetic demo readings in use."],
                            })
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        parsed, meta = call_llm_json("System", {"zone": "Zone A"}, SupervisorLLMOutput, http_client=client)

    assert parsed is not None
    assert meta["reasoning_mode"] == "llm"
    assert parsed.priority_domains == ["flood", "traffic"]


def test_gemini_and_ollama_providers(monkeypatch):
    # Gemini test
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_model", "gemini-2.0-flash")
    monkeypatch.setattr(settings, "llm_api_key", "gemini-dummy")

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        assert "generateContent" in str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps({
                                        "cross_domain_relationships": ["Heavy rain degrades road throughput."],
                                        "cascading_risk_narrative": "Flooding and traffic congestion compound in Zone A.",
                                        "uncertainty_note": "Synthetic ML model outputs.",
                                    })
                                }
                            ]
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(gemini_handler)) as client:
        parsed, meta = call_llm_json("System", {"zone": "Zone A"}, RiskSynthesisLLMOutput, http_client=client)
    assert parsed is not None and meta["reasoning_mode"] == "llm"

    # Ollama local test (no key required)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "llama3.2")
    monkeypatch.setattr(settings, "llm_api_key", "")

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        assert "/api/chat" in str(request.url)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps({
                        "report_summary": "Draft summary for Zone A.",
                        "executive_briefing": "Zone A shows elevated combined flood and traffic risk awaiting officer approval.",
                        "citizen_advisory_draft": "Draft advisory pending municipal sign-off.",
                    })
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(ollama_handler)) as client:
        parsed2, meta2 = call_llm_json("System", {"zone": "Zone A"}, CommunicationLLMOutput, http_client=client)
    assert parsed2 is not None and meta2["reasoning_mode"] == "llm"


def test_llm_malformed_json_falls_back_safely(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test-dummy")

    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "NOT VALID JSON"}}]})

    with httpx.Client(transport=httpx.MockTransport(bad_handler)) as client:
        parsed, meta = call_llm_json("System", {"zone": "Zone A"}, SupervisorLLMOutput, http_client=client)

    assert parsed is None
    assert meta["reasoning_mode"] == "deterministic_fallback"


def test_full_langgraph_workflow_with_mocked_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test-dummy")

    def fake_invoke(system_prompt: str, user_content: str, http_client=None) -> str:
        if "City Supervisor Agent" in system_prompt:
            return json.dumps({
                "priority_domains": ["flood", "traffic", "infrastructure"],
                "coordination_brief": "Coordinating storm surge and arterial traffic analysis for Zone A.",
                "data_quality_watchouts": ["Demo readings used; verify official telemetry."],
            })
        if "Risk Assessment Agent" in system_prompt:
            return json.dumps({
                "cross_domain_relationships": [
                    "Heavy rainfall + 45% drainage capacity raises street-level waterlogging risk.",
                    "Waterlogging slows arterial traffic and delays field inspection vehicles.",
                ],
                "cascading_risk_narrative": "High flood probability and severe congestion overlap in Zone A.",
                "uncertainty_note": "Both ML models trained on synthetic datasets.",
            })
        if "Response Planning Agent" in system_prompt:
            return json.dumps({
                "actions": [
                    {
                        "priority": "high",
                        "action": "Verify rainfall and water-level sensors against official municipal gauges.",
                        "rationale": "Prevents acting on unverified or synthetic readings.",
                        "source_citation": "knowledge_base/sample_flood_sop.md",
                    },
                    {
                        "priority": "high",
                        "action": "Review arterial corridor diversions avoiding low-lying underpasses.",
                        "rationale": "Prevents routing traffic into flood-prone roads.",
                        "source_citation": "knowledge_base/sample_traffic_sop.md",
                    },
                ],
                "coordination_summary": "Prioritize verification, drainage inspection, and flood-safe traffic diversions.",
            })
        return json.dumps({
            "report_summary": "Zone A heavy rain incident assessed at high risk with 2 SOP-grounded recommendations awaiting human review.",
            "executive_briefing": "Traffic RF and Flood ML indicate high overlapping risk in Zone A. Human operator review is required before taking any action.",
            "citizen_advisory_draft": "[DRAFT] Avoid low-lying corridors in Zone A pending official municipal updates.",
        })

    monkeypatch.setattr("backend.app.llm._invoke_provider", fake_invoke)

    result = run_workflow({
        "scenario": "heavy_rain",
        "zone": "Zone A",
        "readings": demo_readings("heavy_rain"),
        "use_llm": True,
    })

    assert len(result["workflow_trace"]) == 9
    assert result["supervisor_finding"]["reasoning_mode"] == "llm"
    assert result["risk_finding"]["reasoning_mode"] == "llm"
    assert result["response_finding"]["reasoning_mode"] == "llm"
    assert result["communication_finding"]["reasoning_mode"] == "llm"
    assert result["response_plan"]["requires_human_approval"] is True
    assert result["response_plan"]["execution_status"] == "recommendation_only"
    assert result["communication_finding"]["findings"]["notification_status"] == "draft_only"
    assert any("[llm]" in step for step in result["workflow_trace"])
    assert any("faiss_embeddings" in step for step in result["workflow_trace"])

    sys_stat = system_status()
    assert sys_stat["status"] == "ok"
    assert sys_stat["models"]["traffic_model"]["loaded"] is True
    assert sys_stat["models"]["flood_model"]["loaded"] is True
    assert sys_stat["rag"]["active_mode"] == "faiss"
    assert get_llm_status()["ready"] is True
