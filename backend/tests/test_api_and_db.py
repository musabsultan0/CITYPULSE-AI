import re
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.core.config import ROOT_DIR, settings
from backend.app.db import Approval, Incident, SessionLocal, init_db
from backend.app.main import (
    app,
    approve,
    demo_readings,
    health,
    incidents,
    knowledge_search,
    knowledge_status,
    llm_status,
    models_status,
    root,
    scenario,
    system_status,
)
from backend.app.rag import KB_DIR, rebuild_knowledge_index, search_knowledge
from backend.app.schemas import (
    AnalyzeRequest,
    ApprovalRequest,
    FloodPredictionRequest,
    KnowledgeSearchRequest,
    TrafficPredictionRequest,
)
from backend.app.workflow import run_workflow


def test_core_endpoints_contract():
    assert "CITYPULSE" in root()["message"]
    assert health()["status"] == "ok"
    assert scenario()["scenario"] == "heavy_rain"
    assert models_status()["traffic_model"]["loaded"] is True
    assert models_status()["flood_model"]["loaded"] is True
    assert knowledge_status()["active_mode"] == "faiss"
    assert "provider" in llm_status()
    assert system_status()["status"] == "ok"


def test_cors_allows_ports_3000_and_3001():
    allowed = set(settings.cors_origin_list)
    for origin in [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]:
        assert origin in allowed
        assert re.match(settings.cors_origin_regex, origin) is not None

    cors_middlewares = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert len(cors_middlewares) == 1
    opts = cors_middlewares[0].kwargs
    assert "http://localhost:3001" in opts["allow_origins"]
    assert "http://localhost:3000" in opts["allow_origins"]
    assert opts["allow_origin_regex"] is not None


def test_knowledge_search_and_multiformat_indexing():
    res = knowledge_search(KnowledgeSearchRequest(query="traffic corridor diversion", top_k=3))
    assert res["rag_mode"] == "faiss"
    assert len(res["results"]) >= 1
    assert "sample_traffic_sop" in res["results"][0]["source"]

    # Verify TXT and MD ingestion indexing roundtrip
    tmp_txt = KB_DIR / "_test_bridge_inspection.txt"
    tmp_md = KB_DIR / "_test_grid_outage.md"
    try:
        tmp_txt.write_text(
            "Bridge Structural Sensor Checklist: Inspect suspension cable vibration gauges after gale warnings.",
            encoding="utf-8",
        )
        tmp_md.write_text(
            "# Substation Power Outage Checklist\nCoordinate backup generators for traffic signal intersections.\n",
            encoding="utf-8",
        )
        status = rebuild_knowledge_index()
        assert status["indexed_documents"] >= 6
        hits_txt = search_knowledge("suspension cable vibration gauges bridge", top_k=2)
        assert any("_test_bridge_inspection.txt" in h["source"] for h in hits_txt)
        hits_md = search_knowledge("substation power outage backup generators traffic signal", top_k=2)
        assert any("_test_grid_outage.md" in h["source"] for h in hits_md)
    finally:
        for p in (tmp_txt, tmp_md):
            if p.exists():
                p.unlink()
        rebuild_knowledge_index()


def test_sqlite_incident_and_approval_lifecycle():
    init_db()
    db = SessionLocal()
    incident_id = str(uuid4())
    try:
        wf = run_workflow({
            "scenario": "heavy_rain",
            "zone": "Zone B",
            "readings": demo_readings("heavy_rain"),
        })
        db.add(
            Incident(
                id=incident_id,
                zone="Zone B",
                scenario="heavy_rain",
                severity=wf["risk_assessment"]["severity"],
                risk_score=wf["risk_assessment"]["overall_score"],
                status="awaiting_review",
                summary=wf["communication_finding"]["findings"]["report_summary"],
                result_json=wf,
            )
        )
        db.commit()

        rows = incidents(limit=20, db=db)
        assert any(r["id"] == incident_id for r in rows)

        approval_res = approve(
            incident_id,
            ApprovalRequest(decision="approved", reviewer="Test Officer", comment="Verified."),
            db=db,
        )
        assert approval_res["status"] == "approved"

        updated = db.get(Incident, incident_id)
        assert updated is not None and updated.status == "approved"

        with pytest.raises(HTTPException) as exc:
            approve(
                "non-existent-id",
                ApprovalRequest(decision="rejected", reviewer="Test Officer"),
                db=db,
            )
        assert exc.value.status_code == 404
    finally:
        db.query(Approval).filter(Approval.incident_id == incident_id).delete()
        db.query(Incident).filter(Incident.id == incident_id).delete()
        db.commit()
        db.close()


def test_schema_validation_guards():
    with pytest.raises(ValidationError):
        TrafficPredictionRequest(hour=25)
    with pytest.raises(ValidationError):
        TrafficPredictionRequest(road_capacity=0)
    with pytest.raises(ValidationError):
        FloodPredictionRequest(rainfall_mm_hour=-5)
    with pytest.raises(ValidationError):
        ApprovalRequest(decision="auto_dispatch", reviewer="Ops")  # type: ignore[arg-type]
    req = AnalyzeRequest(scenario="custom", zone="Zone C", use_live_traffic=True, use_llm=False)
    assert req.use_live_traffic is True


def test_deployment_files_present_and_valid():
    for rel in [
        "backend/Dockerfile",
        "frontend/Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "frontend/.env.example",
        "backend/requirements.txt",
        "pytest.ini",
        "README.md",
    ]:
        p = ROOT_DIR / rel
        assert p.exists(), f"Missing deployment file: {rel}"

    compose_text = (ROOT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    assert "http://localhost:3001" in compose_text
    assert "backend/Dockerfile" in compose_text
    assert "frontend/Dockerfile" in compose_text
