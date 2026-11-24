"""End-to-end live HTTP verification script for CITYPULSE AI backend & frontend."""

from __future__ import annotations

import sys
import httpx

BACKEND = "http://127.0.0.1:8000/api/v1"
FRONTEND = "http://localhost:3001"
ORIGIN = "http://localhost:3001"


def main() -> int:
    headers = {"Origin": ORIGIN}
    with httpx.Client(timeout=20.0) as client:
        # 1. Health + CORS header check for port 3001
        r = client.get(f"{BACKEND}/health", headers=headers)
        assert r.status_code == 200, f"health failed: {r.status_code} {r.text}"
        acao = r.headers.get("access-control-allow-origin")
        assert acao == ORIGIN, f"CORS header mismatch: got {acao!r}, expected {ORIGIN!r}"
        print(f"[PASS] GET /health -> 200 OK (CORS ACAO={acao})")

        # 2. CORS preflight OPTIONS check
        r_opt = client.options(
            f"{BACKEND}/analyze",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r_opt.status_code == 200, f"CORS preflight failed: {r_opt.status_code}"
        assert r_opt.headers.get("access-control-allow-origin") == ORIGIN
        print("[PASS] OPTIONS /analyze CORS preflight -> 200 OK")

        # 3. System status
        r_sys = client.get(f"{BACKEND}/system/status", headers=headers)
        assert r_sys.status_code == 200, f"system/status failed: {r_sys.status_code} {r_sys.text}"
        sys_data = r_sys.json()
        assert sys_data["models"]["traffic_model"]["loaded"] is True
        assert sys_data["models"]["flood_model"]["loaded"] is True
        assert sys_data["rag"]["active_mode"] == "faiss"
        print(
            f"[PASS] GET /system/status -> TrafficML=loaded, FloodML=loaded, "
            f"RAG={sys_data['rag']['active_mode']} ({sys_data['rag']['indexed_chunks']} chunks), "
            f"LLM_ready={sys_data['llm']['ready']}"
        )

        # 4. Traffic RF ML prediction
        r_tr = client.post(
            f"{BACKEND}/predict/traffic",
            headers=headers,
            json={
                "hour": 18,
                "day_of_week": 2,
                "vehicle_count": 950,
                "average_speed": 25.0,
                "road_capacity": 600,
                "rainfall_mm": 12.0,
                "special_event": 1,
                "zone": "Zone A",
            },
        )
        assert r_tr.status_code == 200, f"predict/traffic failed: {r_tr.status_code} {r_tr.text}"
        tr = r_tr.json()
        print(f"[PASS] POST /predict/traffic -> {tr['predicted_congestion_pct']}% ({tr['traffic_status']})")

        # 5. Flood ML prediction
        r_fl = client.post(
            f"{BACKEND}/predict/flood",
            headers=headers,
            json={
                "rainfall_mm_hour": 85.0,
                "water_level_m": 1.35,
                "drainage_capacity_pct": 40.0,
                "previous_24h_rainfall_mm": 120.0,
                "soil_moisture_pct": 78.0,
                "elevation_m": 12.0,
                "zone": "Zone A",
            },
        )
        assert r_fl.status_code == 200, f"predict/flood failed: {r_fl.status_code} {r_fl.text}"
        fl = r_fl.json()
        print(f"[PASS] POST /predict/flood -> {fl['flood_risk_pct']}% ({fl['risk_level']})")

        # 6. Live traffic API with simulated fallback
        r_live = client.get(f"{BACKEND}/traffic/live", params={"zone": "Zone A", "use_live": True}, headers=headers)
        assert r_live.status_code == 200, f"traffic/live failed: {r_live.status_code} {r_live.text}"
        live = r_live.json()
        print(
            f"[PASS] GET /traffic/live -> origin={live['data_origin']}, "
            f"speed={live['metrics']['current_speed_kmph']}km/h, "
            f"congestion={live['metrics']['congestion_pct']}%"
        )

        # 7. FAISS Knowledge Search
        r_rag = client.post(
            f"{BACKEND}/knowledge/search",
            headers=headers,
            json={"query": "flood water level sensor verification and corridor diversion", "top_k": 3},
        )
        assert r_rag.status_code == 200, f"knowledge/search failed: {r_rag.status_code} {r_rag.text}"
        rag = r_rag.json()
        assert rag["rag_mode"] == "faiss"
        assert len(rag["results"]) >= 1
        print(
            f"[PASS] POST /knowledge/search -> mode={rag['rag_mode']}, "
            f"top={rag['results'][0]['source']} (score={rag['results'][0]['score']})"
        )

        # 8. Full 9-agent LangGraph workflow + SQLite persistence
        r_an = client.post(
            f"{BACKEND}/analyze",
            headers=headers,
            json={"scenario": "heavy_rain", "zone": "Zone A", "use_live_traffic": True, "use_live_weather": False},
        )
        assert r_an.status_code == 200, f"analyze failed: {r_an.status_code} {r_an.text}"
        an = r_an.json()
        incident_id = an["incident_id"]
        assert len(an["agent_findings"]) == 9
        assert len(an["workflow_trace"]) == 9
        print(
            f"[PASS] POST /analyze -> incident={incident_id[:8]}..., "
            f"score={an['risk_assessment']['overall_score']}/100 ({an['risk_assessment']['severity']}), "
            f"agents={len(an['agent_findings'])}, llm_mode={an['llm_summary']['reasoning_mode']}"
        )

        # 9. Human approval & incidents list
        r_app = client.post(
            f"{BACKEND}/approvals/{incident_id}",
            headers=headers,
            json={"decision": "approved", "reviewer": "E2E Verifier", "comment": "Validated live."},
        )
        assert r_app.status_code == 200 and r_app.json()["status"] == "approved"
        r_inc = client.get(f"{BACKEND}/incidents?limit=10", headers=headers)
        assert r_inc.status_code == 200
        assert any(row["id"] == incident_id and row["status"] == "approved" for row in r_inc.json())
        print(f"[PASS] POST /approvals + GET /incidents -> incident {incident_id[:8]}... approved & persisted")

        # 10. Frontend check (port 3001 or 3000)
        for url in [FRONTEND, "http://localhost:3000"]:
            try:
                rf = client.get(url)
                if rf.status_code == 200 and "CITYPULSE" in rf.text:
                    print(f"[PASS] Frontend reachable at {url} (HTTP 200)")
                    break
            except Exception:
                continue
        else:
            print("[INFO] Frontend dev server not currently listening on 3001/3000 (or needs restart after next build).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
